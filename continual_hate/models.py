from pprint import pprint as pp

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import get_peft_model

from torch.nn.parallel import DistributedDataParallel as DDP

from .continual_learning_techniques import CLTechniques
from .utils import DistTrainingConfig, set_seed
from .constants import cl_hyperparameters, generation_config, llama_guard_equivalence
from .processing import ProcessingConfig

set_seed(42)

class AutoContinualLearner(nn.Module):
    
    def __init__(self, 
                model_id, 
                model_type,
                cl_technique, 
                objective,
                distributed_config_object:DistTrainingConfig, 
                cl_hyperparams:dict=cl_hyperparameters, 
                generation_config:dict=generation_config, 
                quantization_config:dict=False, 
                lora_config=False, 
                torch_dtype=torch.bfloat16):

        super().__init__()

        self.model_id = model_id
        self.model_type = model_type

        self.cl_technique = cl_technique
        self.cl_hyperparams = cl_hyperparams
        self.cl = None # will only change if there is a cl technique init, if there is no cl technique it will stay None

        if self.cl_technique in self.cl_hyperparams.keys():
            self.hyperparam_str = "_".join([str(k) + "=" + str(v) for k, v in self.cl_hyperparams[self.cl_technique].items()])
        else: 
            self.hyperparam_str = ""

        self.objective = objective
        self.generation_config = generation_config
        self.quantization_config = quantization_config
        self.lora_config = lora_config

        self.device = distributed_config_object.device
        self.distributed_config = distributed_config_object

        if self.objective == "CAUSAL_LM":
            if self.quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=self.quantization_config, torch_dtype=torch_dtype)
            else: 
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch_dtype)

        elif self.objective == "SEQ_CLS":

            if self.quantization_config:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=2, quantization_config=self.quantization_config, torch_dtype=torch_dtype)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=2, torch_dtype=torch_dtype)

        self.n_initial_params = sum(t.numel() for t in self.model.parameters())
        self.n_trainable_params = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        # self.linear_layers = [name for name, param in self.model.named_parameters() if "linear" in name]


        if "guard" in str(self.model_id).lower():
            print("Guard detected, overriding the generation config and setting do_sample to False")
            self.generation_config = {"do_sample": False} # in case i try other generation strategies

    def init_LORA_model(self):

        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model = get_peft_model(self.model, self.lora_config).to(self.device)
        self.model_lora_rank = self.lora_config.to_dict()["r"]
        self.n_trainable_params = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        self.model.print_trainable_parameters()

    def init_CL_model(self):

        self.cl = CLTechniques(self.model, self.device, self.cl_technique, self.cl_hyperparams)
        self.model.to(self.device)

    def init_DDP_model(self):

        self.model = DDP(self.model, 
                        device_ids=[self.distributed_config.local_rank], 
                        output_device=self.distributed_config.local_rank, 
                        )

    def prep_model(self):

        if self.model_type == "PLM": # if the model is a plm we dont do loras

            if self.cl_technique in self.cl_hyperparams.keys(): # for a cl technique to be accepted, it have to be the key of the cl params dictionary
                self.init_CL_model()
            
            self.model.to(self.device)

        elif self.model_type == "LLM" and self.cl_technique != "ZERO_SHOT_TESTING" and self.cl_technique != "FEW_SHOT_TESTING": # if the model is an llm we do loras but not the traditional cl methods
            self.cl = False
            self.init_LORA_model()
            print("total_parameters_after prepping the model")
            print()


        elif self.model_type == "LLM" and (self.cl_technique == "ZERO_SHOT_TESTING" or self.cl_technique == "FEW_SHOT_TESTING"): # if the model is an llm we do loras but not the traditional cl methods
            self.cl = False
            print(f"Zero Shot testing - Loading the model as is and sending it to the device {self.device}.")
            self.model.to(self.device)

        if self.distributed_config.distributed_training: # if we are doing ditributed training, the dpp has to go in the end
            self.init_DDP_model()
            

    def forward(self, **kwargs):
        return self.model(**kwargs)