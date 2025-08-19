import dis
from typing import List, Dict

from pprint import pprint as pp

import pandas as pd

from peft import LoraConfig
from transformers import BitsAndBytesConfig

import torch
from torch.optim import AdamW

from .datasets import TaskDataset
from .datastreams import DataStream
from .processing import ProcessingConfig
from .utils import DistTrainingConfig, translate_prediction_to_label, EarlyStopperAndCheckpointer, set_seed
from .models import AutoContinualLearner
from .constants import base_prompt_few_shot, llama_guard_equivalence, translation_dict_from_label_to_verbalizer, translation_dict_from_label_to_int, tokenizer_args, data_collator_args, base_prompt, label2int, int2label, generation_config
from .trainer import Trainer

set_seed(42)

def setup_continual_experiment(df:pd.DataFrame,
                                tasks:List[str]|str,
                                zero_shot_tasks:List[str]|bool|str,
                                experiment_name:str,
                                processing_config:ProcessingConfig,
                                dist_training_config:DistTrainingConfig,
                                batch_size:int=16,
                                **few_shot_args) -> DataStream:
    
    print("Starting Experiment SETUP")
    print("Datasets part of the datastream: ", tasks)

    task_datasets = []
    zero_shot_datasets = []

    if zero_shot_tasks == False:
        zero_shot_tasks = []
    
    merged_datasets = False
    # print(task_datasets)
    for i, task_d in enumerate(tasks):
        # print("Checking the type of the tasks")
        if type(task_d) == list:
            print("Task is a list of tasks!")
            print("Processing the list of tasks into their individual testing splits")
            merged_datasets = True
            zero_shot = True
            for task in task_d: # add every test set individually
                print("\tAdding task:", task)
                task_dataset = TaskDataset(df=df, 
                                        task=task, 
                                        task_column=processing_config.task_column, 
                                        text_column=processing_config.text_column, 
                                        label_column=processing_config.label_column,
                                        split_column=processing_config.split_column, 
                                        label_equivalence_general=processing_config.label_equivalence_general, 
                                        label_equivalence_int=processing_config.label_equivalence_int,
                                        zero_shot=zero_shot)    

                if processing_config.training_obj == "SEQ_CLS":
                    task_dataset.transform_label_to_int()
                    task_dataset.transform_df_to_hf_dataset()

                elif processing_config.training_obj == "CAUSAL_LM":
                    if "guard" in str(processing_config.tokenizer_id.lower()):
                        print("We are only passing the messaege to llama guard so we are skipping the turning message to prompt step!!!!!")
                        
                    if few_shot_args: # if we are doing few shot, process the few shot way
                        task_dataset.transform_text_to_few_shot_prompt(**few_shot_args)
                        # print(task_dataset.texts.to_list()[2])
                        # print("----FEW SHOT PROCESSED----")

                    else:
                        task_dataset.transform_text_to_prompt(prompt=processing_config.base_prompt)
                        

                    task_dataset.transform_label_to_verbalizer()
                    task_dataset.transform_df_to_hf_dataset()

                task_dataset.process_hf_dataset(processing_function=processing_config.processing_function, 
                                    processing_function_args=processing_config.tokenizer_args,
                                    input_columns=processing_config.hf_dataset_processing_args["input_columns"], 
                                    columns_to_keep=processing_config.hf_dataset_processing_args["cols_to_keep"], 
                                    data_collator=processing_config.data_collator, 
                                    batch_size=batch_size, 
                                    distributed_training=dist_training_config.distributed_training,
                                    world_size=dist_training_config.world_size,
                                    local_rank=dist_training_config.local_rank,
                                    with_rank=False, # these go into the hf_dataset.map
                                    num_proc=None, # these go into the hf_dataset.map
                                    shuffle=processing_config.data_shuffle) # IF im doing distributed trianing, this is false withing the data_loader of the process_hf_dataset itself

                zero_shot_datasets.append(task_dataset)
                print("----TASK ADDED TO ZEROSHOT DATASETS----")

    # print("Type of tasks: ", type(tasks))
    # print("Type of zero shot tasks: ", type(zero_shot_tasks))

    if type(tasks) == str:
        print("Changing type of task to list")
        tasks = [tasks]
    if type(zero_shot_tasks) == str:
        print("Changing type of zero shot tasks to list")
        zero_shot_tasks = [zero_shot_tasks]
    
    print("Tasks to train: ", tasks)
    print("Zero shot tasks: ", zero_shot_tasks)

    all_tasks = tasks + zero_shot_tasks

    for task in all_tasks:

        print("Processing task: ", task)

        zero_shot = False

        if task in zero_shot_tasks:
            print("\tTask is part of the Zero Shot")
            zero_shot = True            

        task_dataset = TaskDataset(df=df, 
                                    task=task, 
                                    task_column=processing_config.task_column, 
                                    text_column=processing_config.text_column, 
                                    label_column=processing_config.label_column,
                                    split_column=processing_config.split_column, 
                                    label_equivalence_general=processing_config.label_equivalence_general, 
                                    label_equivalence_int=processing_config.label_equivalence_int,
                                    zero_shot=zero_shot)                


        if processing_config.training_obj == "SEQ_CLS":
            task_dataset.transform_label_to_int()
            task_dataset.transform_df_to_hf_dataset()

        elif processing_config.training_obj == "CAUSAL_LM":
            
            if few_shot_args:
                task_dataset.transform_text_to_few_shot_prompt(**few_shot_args)
                # print(task_dataset.texts.to_list()[2])
                # print("----FEW SHOT PROCESSED----")
                
            else:
                task_dataset.transform_text_to_prompt(prompt=processing_config.base_prompt)
                
            task_dataset.transform_label_to_verbalizer()
            task_dataset.transform_df_to_hf_dataset()

        task_dataset.process_hf_dataset(processing_function=processing_config.processing_function, 
                                processing_function_args=processing_config.tokenizer_args,
                                input_columns=processing_config.hf_dataset_processing_args["input_columns"], 
                                columns_to_keep=processing_config.hf_dataset_processing_args["cols_to_keep"], 
                                data_collator=processing_config.data_collator, 
                                batch_size=batch_size, 
                                distributed_training=dist_training_config.distributed_training,
                                world_size=dist_training_config.world_size,
                                local_rank=dist_training_config.local_rank,
                                with_rank=False, # these go into the hf_dataset.map
                                num_proc=None, # these go into the hf_dataset.map
                                shuffle=processing_config.data_shuffle) # IF im doing distributed trianing, this is false withing the data_loader of the process_hf_dataset itself
        
        if zero_shot:
            zero_shot_datasets.append(task_dataset)
        else:
            task_datasets.append(task_dataset)
        
        print("----TASK ADDED TO TASK DATASETS----")

    # print("Merged datasets: ", merged_datasets)
    print()
    print("-Creating the Datastream Object-")
    print()
    datastream = DataStream(datastream=task_datasets, experiment_name=experiment_name, zero_shot_datasets=zero_shot_datasets, merged_datasets=merged_datasets)
    print("-DATASTREAM CREATED-")
    return datastream

def few_shot_test(
                df_path, 
                zero_shot_tasks,
                base_prompt,
                model_id,
                generation_config=generation_config,
                training_obj = "CAUSAL_LM",
                model_type="LLM",
                mode="train",
                distributed_training=False, 
                label_column = "class", 
                split_column = "split", 
                text_column = "clean_post", 
                task_column = "task",
                tokenizer_args={"max_length": 512, "loss_function_over_prompt":False, "padding":"max_length"}, 
                data_collator_args={"padding":"longest", "return_tensors":"pt"},
                quantization=False,
                translation_dict_from_label_to_int=translation_dict_from_label_to_int,
                translation_dict_from_label_to_verbalizer=translation_dict_from_label_to_verbalizer,
                **few_shot_args):


        df = pd.read_csv(df_path)

        dist_config = DistTrainingConfig(distributed_training=distributed_training)

        processing_config_causal_lm = ProcessingConfig( 
                                            tokenizer_id=model_id, 
                                            training_obj=training_obj, 
                                            base_prompt=base_prompt,
                                            task_column=task_column, 
                                            text_column=text_column, 
                                            label_column=label_column,
                                            split_column=split_column,
                                            label_equivalence_general=translation_dict_from_label_to_verbalizer, 
                                            label_equivalence_int=translation_dict_from_label_to_int,
                                            label_equivalence_llama_guard=llama_guard_equivalence,
                                            tokenizer_args=tokenizer_args, 
                                            data_collator_args=data_collator_args)

        processing_config_causal_lm.set_processing_function()

        experiment_name = "_".join([model_id.replace("/", "_"), "FEW_SHOTS"])

        datastream_causal_lm = setup_continual_experiment(df=df,
                                        tasks=[],
                                        zero_shot_tasks=zero_shot_tasks,
                                        experiment_name=experiment_name,
                                        processing_config=processing_config_causal_lm,
                                        dist_training_config=dist_config,
                                        **few_shot_args)

        if quantization:
                quantization = BitsAndBytesConfig(  
                                load_in_4bit= True,
                                bnb_4bit_quant_type= "nf4",
                                bnb_4bit_compute_dtype= torch.bfloat16,
                                bnb_4bit_use_double_quant= True,
                                        )

        model =  AutoContinualLearner(model_id=model_id, 
                                model_type=model_type,
                                cl_technique="FEW_SHOT_TESTING",
                                objective=training_obj, 
                                distributed_config_object=dist_config, 
                                quantization_config=quantization,
                                generation_config=generation_config)

        model.prep_model()

        trainer = Trainer(model=model, 
                        mode=mode, 
                        objective=training_obj,
                        distributed_config_object=dist_config,
                        learning_rate=False,
                        optimizer=False,
                        early_stopper=False)

        test_results = trainer.few_shot_testing(datastream=datastream_causal_lm, 
                                                processing_config=processing_config_causal_lm)
        return test_results

def zero_shot_test(df_path, 
                        zero_shot_tasks,
                        base_prompt,
                        model_id,
                        generation_config=generation_config,
                        training_obj = "CAUSAL_LM",
                        model_type="LLM",
                        mode="train",
                        distributed_training=False, 
                        label_column = "class", 
                        split_column = "split", 
                        text_column = "clean_post", 
                        task_column = "task",
                        tokenizer_args={"max_length": 512, "loss_function_over_prompt":False, "padding":"max_length"}, 
                        data_collator_args={"padding":"longest", "return_tensors":"pt"},
                        quantization=False,
                        translation_dict_from_label_to_int=translation_dict_from_label_to_int,
                        translation_dict_from_label_to_verbalizer=translation_dict_from_label_to_verbalizer):


        df = pd.read_csv(df_path)

        dist_config = DistTrainingConfig(distributed_training=distributed_training)

        processing_config_causal_lm = ProcessingConfig( 
                                            tokenizer_id=model_id, 
                                            training_obj=training_obj, 
                                            base_prompt=base_prompt,
                                            task_column=task_column, 
                                            text_column=text_column, 
                                            label_column=label_column,
                                            split_column=split_column,
                                            label_equivalence_general=translation_dict_from_label_to_verbalizer, 
                                            label_equivalence_int=translation_dict_from_label_to_int,
                                            label_equivalence_llama_guard=llama_guard_equivalence,
                                            tokenizer_args=tokenizer_args, 
                                            data_collator_args=data_collator_args)

        processing_config_causal_lm.set_processing_function()

        experiment_name = "_".join([model_id.replace("/", "_"), "ZERO_SHOTS"])

        datastream_causal_lm = setup_continual_experiment(df=df,
                                        tasks=[],
                                        zero_shot_tasks=zero_shot_tasks,
                                        experiment_name=experiment_name,
                                        processing_config=processing_config_causal_lm,
                                        dist_training_config=dist_config)

        if quantization:
                quantization = BitsAndBytesConfig(  
                                load_in_4bit= True,
                                bnb_4bit_quant_type= "nf4",
                                bnb_4bit_compute_dtype= torch.bfloat16,
                                bnb_4bit_use_double_quant= True,
                                        )

        model =  AutoContinualLearner(model_id=model_id, 
                                model_type=model_type,
                                cl_technique="ZERO_SHOT_TESTING",
                                objective=training_obj, 
                                distributed_config_object=dist_config, 
                                quantization_config=quantization,
                                generation_config=generation_config)

        model.prep_model()

        trainer = Trainer(model=model, 
                        mode=mode, 
                        objective=training_obj,
                        distributed_config_object=dist_config,
                        learning_rate=False,
                        optimizer=False,
                        early_stopper=False)

        test_results = trainer.zero_shot_testing(datastream=datastream_causal_lm, 
                                                processing_config=processing_config_causal_lm)
        return test_results


def fine_tune_LLM_LoRA(df_path, 
                        task:list,
                        zero_shot_tasks,
                        base_prompt,
                        model_id,
                        generation_config,
                        experiment_name,
                        training_obj = "CAUSAL_LM",
                        mode="train",
                        batch_size = 16,
                        n_epochs=8,
                        early_stopper_patience=4,
                        learning_rate=1e-5,
                        distributed_training=False, 
                        label_column = "class", 
                        split_column = "split", 
                        text_column = "clean_post", 
                        task_column = "task",
                        tokenizer_args={"max_length": 512, "loss_function_over_prompt":False, "padding":"max_length"}, 
                        data_collator_args={"padding":"longest", "return_tensors":"pt"},
                        translation_dict_from_label_to_int=translation_dict_from_label_to_int,
                        translation_dict_from_label_to_verbalizer=translation_dict_from_label_to_verbalizer,
                        data_shuffle=False,
                        quantization=False,
                        lora_r = 8):


        df = pd.read_csv(df_path)

        dist_config = DistTrainingConfig(distributed_training=distributed_training)

        processing_config = ProcessingConfig( 
                                            tokenizer_id=model_id, 
                                            training_obj=training_obj, 
                                            base_prompt=base_prompt,
                                            task_column=task_column, 
                                            text_column=text_column, 
                                            label_column=label_column,
                                            split_column=split_column,
                                            label_equivalence_general=translation_dict_from_label_to_verbalizer, 
                                            label_equivalence_int=translation_dict_from_label_to_int,
                                            label_equivalence_llama_guard=llama_guard_equivalence,
                                            tokenizer_args=tokenizer_args, 
                                            data_collator_args=data_collator_args,
                                            data_shuffle=data_shuffle)

        processing_config.set_processing_function()

        # experiment_name = "_".join([model_id.replace("/", "_"), task[0]])

        if dist_config.distributed_training:
                batch_size = batch_size // dist_config.world_size
                print("Adjusting Batch size to the world size: ", batch_size)

        datastream = setup_continual_experiment(df=df,
                                        tasks=task,
                                        zero_shot_tasks=zero_shot_tasks,
                                        experiment_name=experiment_name,
                                        processing_config=processing_config,
                                        dist_training_config=dist_config,
                                        batch_size=batch_size)

        if quantization:
                quantization = BitsAndBytesConfig(  
                                load_in_4bit= True,
                                bnb_4bit_quant_type= "nf4",
                                bnb_4bit_compute_dtype= torch.bfloat16,
                                bnb_4bit_use_double_quant= True,
                                        )

        if training_obj == "CAUSAL_LM":
                modules_to_save = ["lm_head"]

        elif training_obj == "SEQ_CLS":
                modules_to_save = ["score"]

        lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_r*2, # the scaling of the weights will be one
                target_modules="all-linear", # the transformer block layers and the output
                modules_to_save=[], # will also train the final lm head that maps the distribution
                task_type=training_obj,
                lora_dropout=0.1,
                bias="none")

        model =  AutoContinualLearner(model_id=model_id, 
                                model_type="LLM",
                                cl_technique="BASIC_LORA_all-linear",
                                objective=training_obj, 
                                distributed_config_object=dist_config, 
                                quantization_config=quantization, # only pass if the model is big
                                lora_config=lora_config,
                                generation_config=generation_config)

        model.prep_model()
        early_stopper = EarlyStopperAndCheckpointer(patience=early_stopper_patience)

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        trainer = Trainer( model=model, 
                        mode=mode, 
                        objective=training_obj,
                        learning_rate=learning_rate,
                        optimizer=optimizer,
                        distributed_config_object=dist_config,
                        early_stopper=early_stopper)

        test_results, train_logs = trainer.continual_learning(datastream=datastream, 
                                                        n_epochs=n_epochs, 
                                                        processing_config=processing_config)
        return test_results, train_logs

def fine_tune_PLM_CONTINUAL_LEARNING(df_path, 
                                    tasks:list,
                                    cl_technique,
                                    zero_shot_tasks,
                                    model_id,
                                    cl_hyperparams = {
                                                        "ewc": {"ewc_lambda":1500},
                                                        "agem": {"mem_size":100},
                                                        "lwf": {"lwf_lambda":1,
                                                                "temperature":2},
                                                        "mas": {"mas_lambda":1000} },
                                    training_obj = "SEQ_CLS",
                                    mode="train",
                                    batch_size = 16,
                                    n_epochs=8,
                                    early_stopper_patience=4,
                                    learning_rate=1e-5,
                                    distributed_training=False, 
                                    label_column = "class", 
                                    split_column = "split", 
                                    text_column = "clean_post", 
                                    task_column = "task",
                                    tokenizer_args={"max_length": 512, "padding":True},
                                    data_collator_args={"padding":"longest", "return_tensors":"pt"},
                                    data_shuffle=False,
                                    translation_dict_from_label_to_int=translation_dict_from_label_to_int,
                                    translation_dict_from_label_to_verbalizer=translation_dict_from_label_to_verbalizer):

    df = pd.read_csv(df_path)
    dist_config = DistTrainingConfig(distributed_training)
    processing_config = ProcessingConfig( 
                                            tokenizer_id=model_id, 
                                            training_obj=training_obj, 
                                            base_prompt=None,
                                            task_column=task_column, 
                                            text_column=text_column, 
                                            label_column=label_column,
                                            split_column=split_column,
                                            label_equivalence_general=translation_dict_from_label_to_verbalizer, 
                                            label_equivalence_int=translation_dict_from_label_to_int,
                                            label_equivalence_llama_guard=llama_guard_equivalence,
                                            tokenizer_args=tokenizer_args, 
                                            data_collator_args=data_collator_args,
                                            data_shuffle=data_shuffle)

    processing_config.set_processing_function()

    experiment_name = "_".join([model_id.replace("/", "_"), "-MERGED-".join(tasks[:1])])
    
    if dist_config.distributed_training:
        batch_size = batch_size // dist_config.world_size
        print("Adjusting Batch size to the world size: ", batch_size)
        
    if early_stopper_patience:
        early_stopper = EarlyStopperAndCheckpointer(patience=early_stopper_patience)
    else:
        early_stopper = False
        experiment_name = experiment_name + "NO-ES"
        print("No early stopping")


    datastream = setup_continual_experiment(df=df,
                                    tasks=tasks,
                                    zero_shot_tasks=zero_shot_tasks,
                                    experiment_name=experiment_name,
                                    processing_config=processing_config,
                                    dist_training_config=dist_config,
                                    batch_size=batch_size)
    if type(cl_technique) == str:

        print("CL TECHNIQUE: ", cl_technique)

        model =  AutoContinualLearner(model_id=model_id,
                                    model_type="PLM", # or LLM, will affect if i init lora or not
                                    cl_technique=cl_technique,
                                    cl_hyperparams=cl_hyperparams,
                                    objective=training_obj, 
                                    distributed_config_object=dist_config)

        model.prep_model()

            
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        trainer = Trainer(  model=model, 
                            mode=mode, 
                            objective=training_obj,
                            learning_rate=learning_rate,
                            optimizer=optimizer,
                            distributed_config_object=dist_config,
                            early_stopper=early_stopper)

        test_results, train_logs = trainer.continual_learning(datastream=datastream, processing_config=processing_config, n_epochs=n_epochs)

        return test_results, train_logs

    elif type(cl_technique) == list:

        list_cl_techniques = cl_technique
        list_test_results, list_train_logs = [], []
        for cl_technique in list_cl_techniques:

            print("CL TECHNIQUE: ", cl_technique)

            model =  AutoContinualLearner(model_id=model_id,
                                    model_type="PLM", # or LLM, will affect if i init lora or not
                                    cl_technique=cl_technique,
                                    cl_hyperparams=cl_hyperparams,
                                    objective=training_obj, 
                                    distributed_config_object=dist_config)

            model.prep_model()

            if early_stopper_patience:
                early_stopper = EarlyStopperAndCheckpointer(patience=early_stopper_patience)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

            trainer = Trainer(  model=model, 
                                mode=mode, 
                                objective=training_obj,
                                learning_rate=learning_rate,
                                optimizer=optimizer,
                                distributed_config_object=dist_config,
                                early_stopper=early_stopper)

            test_results, train_logs = trainer.continual_learning(datastream=datastream, processing_config=processing_config, n_epochs=n_epochs)
            list_test_results.append(test_results)
            list_train_logs.append(train_logs)
        
        return list_test_results, list_train_logs

def fine_tune_PLM_MERGED_dataset(df_path, 
                                    tasks:list,
                                    zero_shot_tasks,
                                    model_id,
                                    training_obj = "SEQ_CLS",
                                    mode="train",
                                    batch_size = 16,
                                    n_epochs=8,
                                    early_stopper_patience=4,
                                    learning_rate=1e-5,
                                    distributed_training=False, 
                                    label_column = "class", 
                                    split_column = "split", 
                                    text_column = "clean_post", 
                                    task_column = "task",
                                    tokenizer_args={"max_length": 512, "padding":True},
                                    data_collator_args={"padding":"longest", "return_tensors":"pt"},
                                    data_shuffle=False,
                                    translation_dict_from_label_to_int=translation_dict_from_label_to_int,
                                    translation_dict_from_label_to_verbalizer=translation_dict_from_label_to_verbalizer):

    df = pd.read_csv(df_path)
    dist_config = DistTrainingConfig(distributed_training)
    processing_config = ProcessingConfig( 
                                            tokenizer_id=model_id, 
                                            training_obj=training_obj, 
                                            base_prompt=None,
                                            task_column=task_column, 
                                            text_column=text_column, 
                                            label_column=label_column,
                                            split_column=split_column,
                                            label_equivalence_general=translation_dict_from_label_to_verbalizer, 
                                            label_equivalence_int=translation_dict_from_label_to_int,
                                            label_equivalence_llama_guard=llama_guard_equivalence,
                                            tokenizer_args=tokenizer_args, 
                                            data_collator_args=data_collator_args,
                                            data_shuffle=data_shuffle)

    processing_config.set_processing_function()

    merged_names = "-MERGED-".join(tasks[0])
    experiment_name = "_".join([model_id.replace("/", "_"), merged_names])
    
    if dist_config.distributed_training:
        batch_size = batch_size // dist_config.world_size
        print("Adjusting Batch size to the world size: ", batch_size)

    datastream = setup_continual_experiment(df=df,
                                    tasks=tasks,
                                    zero_shot_tasks=zero_shot_tasks,
                                    experiment_name=experiment_name,
                                    processing_config=processing_config,
                                    dist_training_config=dist_config,
                                    batch_size=batch_size)

    model =  AutoContinualLearner(model_id=model_id,
                                model_type="PLM", # or LLM, will affect if i init lora or not
                                cl_technique="MERGED_FT", 
                                objective=training_obj, 
                                distributed_config_object=dist_config)

    model.prep_model()

    if early_stopper_patience:
        early_stopper = EarlyStopperAndCheckpointer(patience=early_stopper_patience)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    trainer = Trainer(  model=model, 
                        mode=mode, 
                        objective=training_obj,
                        learning_rate=learning_rate,
                        optimizer=optimizer,
                        distributed_config_object=dist_config,
                        early_stopper=early_stopper)

    test_results, train_logs = trainer.continual_learning(datastream=datastream, processing_config=processing_config, n_epochs=n_epochs)

    return test_results, train_logs

def fine_tune_PLM_individual_dataset(df_path, 
                                    task:list, # require to be a list by the Dataset task class
                                    zero_shot_tasks,
                                    model_id,
                                    training_obj = "SEQ_CLS",
                                    mode="train",
                                    batch_size = 16,
                                    n_epochs=8,
                                    early_stopper_patience=4,
                                    learning_rate=1e-5,
                                    distributed_training=False, 
                                    label_column = "class", 
                                    split_column = "split", 
                                    text_column = "clean_post", 
                                    task_column = "task",
                                    tokenizer_args={"max_length": 512, "padding":True},
                                    data_collator_args={"padding":"longest", "return_tensors":"pt"},
                                    data_shuffle=False,
                                    translation_dict_from_label_to_int=translation_dict_from_label_to_int,
                                    translation_dict_from_label_to_verbalizer=translation_dict_from_label_to_verbalizer):

    df = pd.read_csv(df_path)
    dist_config = DistTrainingConfig(distributed_training)
    processing_config = ProcessingConfig( 
                                            tokenizer_id=model_id, 
                                            training_obj=training_obj, 
                                            base_prompt=None,
                                            task_column=task_column, 
                                            text_column=text_column, 
                                            label_column=label_column,
                                            split_column=split_column,
                                            label_equivalence_general=translation_dict_from_label_to_verbalizer, 
                                            label_equivalence_int=translation_dict_from_label_to_int,
                                            label_equivalence_llama_guard=llama_guard_equivalence,
                                            tokenizer_args=tokenizer_args, 
                                            data_collator_args=data_collator_args,
                                            data_shuffle=data_shuffle)

    processing_config.set_processing_function()

    experiment_name = "VANILLA-FT"
    experiment_name += "_".join([model_id.replace("/", "_"), task[0]])
    
    if dist_config.distributed_training:
        batch_size = batch_size // dist_config.world_size
        print("Adjusting Batch size to the world size: ", batch_size)

    datastream = setup_continual_experiment(df=df,
                                    tasks=task,
                                    zero_shot_tasks=zero_shot_tasks,
                                    experiment_name=experiment_name,
                                    processing_config=processing_config,
                                    dist_training_config=dist_config,
                                    batch_size=batch_size)

    model =  AutoContinualLearner(model_id=model_id,
                                model_type="PLM", # or LLM, will affect if i init lora or not
                                cl_technique="BASIC_FT", 
                                objective=training_obj, 
                                distributed_config_object=dist_config)

    model.prep_model()

    if early_stopper_patience:
        early_stopper = EarlyStopperAndCheckpointer(patience=early_stopper_patience)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    trainer = Trainer(  model=model, 
                        mode=mode, 
                        objective=training_obj,
                        learning_rate=learning_rate,
                        optimizer=optimizer,
                        distributed_config_object=dist_config,
                        early_stopper=early_stopper)

    test_results, train_logs = trainer.continual_learning(datastream=datastream, processing_config=processing_config, n_epochs=n_epochs)

    return test_results, train_logs

#---------------------------------------------------------------------------------
def fine_tune_LLM_QLoRA(df_path, 
                        task:list,
                        zero_shot_tasks,
                        base_prompt,
                        model_id,
                        generation_config,
                        experiment_name,
                        training_obj = "CAUSAL_LM",
                        mode="train",
                        batch_size = 8,
                        n_epochs=8,
                        early_stopper_patience=5,
                        learning_rate=1e-5,
                        distributed_training=False, 
                        label_column = "class", 
                        split_column = "split", 
                        text_column = "clean_post", 
                        task_column = "task",
                        tokenizer_args={"max_length": 512, "loss_function_over_prompt":False, "padding":"max_length"}, 
                        data_collator_args={"padding":"longest", "return_tensors":"pt"},
                        translation_dict_from_label_to_int=translation_dict_from_label_to_int,
                        translation_dict_from_label_to_verbalizer=translation_dict_from_label_to_verbalizer,
                        data_shuffle=False,
                        lora_r = 8):


        df = pd.read_csv(df_path)

        if distributed_training == False:
            dist_config = DistTrainingConfig(distributed_training=distributed_training)
        else:
            print("Distributed obj passed")

        processing_config = ProcessingConfig( 
                                            tokenizer_id=model_id, 
                                            training_obj=training_obj, 
                                            base_prompt=base_prompt,
                                            task_column=task_column, 
                                            text_column=text_column, 
                                            label_column=label_column,
                                            split_column=split_column,
                                            label_equivalence_general=translation_dict_from_label_to_verbalizer, 
                                            label_equivalence_int=translation_dict_from_label_to_int,
                                            label_equivalence_llama_guard=llama_guard_equivalence,
                                            tokenizer_args=tokenizer_args, 
                                            data_collator_args=data_collator_args,
                                            data_shuffle=data_shuffle)

        processing_config.set_processing_function()

        # experiment_name = "_".join([model_id.replace("/", "_"), task[0]])

        if dist_config.distributed_training:
                batch_size = batch_size // dist_config.world_size
                print("Adjusting Batch size to the world size: ", batch_size)

        datastream = setup_continual_experiment(df=df,
                                        tasks=task,
                                        zero_shot_tasks=zero_shot_tasks,
                                        experiment_name=experiment_name,
                                        processing_config=processing_config,
                                        dist_training_config=dist_config,
                                        batch_size=batch_size)

        quantization = BitsAndBytesConfig(  
                                load_in_4bit= True,
                                bnb_4bit_quant_type= "nf4",
                                bnb_4bit_compute_dtype= torch.bfloat16,
                                bnb_4bit_use_double_quant= True,
                                        )

        lora_config = LoraConfig(
                            r=lora_r,
                            lora_alpha=lora_r*2, # the scaling of the weights will be one
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"], # the transformer block layers and the output
                            modules_to_save=[], # will also train the final lm head that maps the distribution
                            task_type="CAUSAL_LM",
                            lora_dropout=0.1,
                            bias="none")


        model =  AutoContinualLearner(model_id=model_id, 
                                model_type="LLM",
                                cl_technique="BASIC_QLORA",
                                objective=training_obj, 
                                distributed_config_object=dist_config, 
                                quantization_config=quantization, # only pass if the model is big
                                lora_config=lora_config,
                                generation_config=generation_config)
        
        # print("Params right after init the continual learner")
        # model.model.print_trainable_parameters()

        model.prep_model()
        if early_stopper_patience:
            early_stopper = EarlyStopperAndCheckpointer(patience=early_stopper_patience)
        else:
            early_stopper = False


        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        n_params = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
        print("Params in optimizer:", n_params) 
        n_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Params in the mode: ", n_params_model)


        trainer = Trainer(  model=model, 
                        mode=mode, 
                        objective=training_obj,
                        learning_rate=learning_rate,
                        optimizer=optimizer,
                        distributed_config_object=dist_config,
                        early_stopper=early_stopper)

        test_results, train_logs = trainer.continual_learning(datastream=datastream, 
                                                        n_epochs=n_epochs, 
                                                        processing_config=processing_config)
        return test_results, train_logs
