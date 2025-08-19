
import json

from pprint import pprint as pp

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

import numpy as np

from tqdm import tqdm
import gc

from .utils import translate_label_to_int, EarlyStopperAndCheckpointer, DistTrainingConfig, translate_prediction_to_label, set_seed, translate_llama_guard_labels, squeeze_notneeded_dimension
from .constants import label2int
from .metrics import get_metrics
from .processing import ProcessingConfig
from .datastreams import DataStream

set_seed(42)

class Trainer():

    def __init__(self, 
                model, 
                mode, 
                objective,
                learning_rate,
                optimizer,
                distributed_config_object:DistTrainingConfig,
                early_stopper:EarlyStopperAndCheckpointer):

        self.model = model # has to include a trainable_params method and model_id
        self.mode = mode
        self.distributed_config = distributed_config_object
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.objective = objective
        
        if early_stopper:
            self.early_stopper = early_stopper

    def validate(self, 
                validation_dataloader):

        self.model.eval()

        validation_losses = []

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(validation_dataloader)):
                if self.mode == "test":
                    if idx >= 1:
                        break

                batch = {key: tensor.to(self.distributed_config.device) for key, tensor in batch.items()}
                
                if self.distributed_config.distributed_training:
                    batch = {key: tensor.to(self.distributed_config.local_rank) for key, tensor in batch.items()}

                # print(self.model.device, self.distributed_config.device)
                # for key, tensor in batch.items():
                #     print(key, tensor.device)

                output = self.model(**batch)
                loss = output.loss

                if self.distributed_config.distributed_training:
                    loss_tensor = torch.tensor(loss, device=self.distributed_config.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss = loss_tensor/self.distributed_config.world_size

                validation_losses.append(loss.detach().item())

            epoch_validation_loss = np.mean(validation_losses)

        gc.collect()
        torch.cuda.empty_cache()

        return epoch_validation_loss

    def test(self, 
            test_dataloader,
            tokenizer,
            label_equivalence = label2int):

        self.model.eval()

        prompts_fed = []
        generations = []
        unfiltered_outputs = []
        verbalized_labels = []
        y_preds = []
        targets = []

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_dataloader)):
                if self.mode == "test":
                    if idx >= 1:
                        break

                if "guard" in str(self.model.model_id).lower():       
                    batch = {key: squeeze_notneeded_dimension(tensor) for key, tensor in batch.items()} # sometimes the collators include a useless dimension
                    
                # dims batch_size x embd_dim
                labels = batch["labels"]
                # print(labels)
                test_batch = {key: tensor.to(self.distributed_config.device) for key, tensor in batch.items() if key != "labels"}

                if self.objective == "CAUSAL_LM":
                    
                    prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                    prompts_fed.extend(prompts)
                    
                    print("---------------------")
                    print("Prompt example fed to the model:")
                    pp(prompts[0])
                    
                    outputs = self.model.model.generate(**test_batch, **self.model.generation_config)
                    unfiltered_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    unfiltered_outputs.extend(unfiltered_output)
                    
                    print("---------------------")
                    print("Unfiltered  Outputs")
                    pp(unfiltered_output[0])
                    
                    prompt_len = batch["input_ids"].shape[1]
                    assistant_response = outputs[:, prompt_len:]
                    assistant_output = tokenizer.batch_decode(assistant_response, skip_special_tokens=True) # after LABEL:
                    generations.extend(assistant_output)

                    print("---------------------")
                    print("Batch Assistant Output")
                    pp(assistant_output[0].strip())
                    print("---------------------")
                    
                    if "guard" in str(self.model.model_id).lower():
                        
                        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                        labels = [translate_llama_guard_labels(label) for label in labels]
                        pred = [translate_llama_guard_labels(label) for label in assistant_output]

                        targets.extend(labels)
                        y_preds.extend(pred)

                    else:
                        pred =  [translate_label_to_int(response.strip()) for response in assistant_output]
                        print("---------------------")
                        print("Prediction after parsing assistant output")
                        pp(pred[0])
                        print("---------------------")
                        # need to go from token labels to text labels
                        if type(labels[0]) == str: # this would mean that the label strings were never converted into token ids
                            # labels = [label.replace("<|begin_of_text|>", "") for label in labels]
                            verbalized_labels.extend(labels)
                            labels_int = [label2int[label.strip()] for label in labels] # label2int is the dictionary of label - int equivalence

                        elif type(labels) == torch.Tensor: # taking the label input_ids back to the verbalizer to get the categorical int
                            labels = [label[label != -100] for label in labels] # in case we passed -100 to ignore the loss over the prompt
                            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                            verbalized_labels.extend(labels)
                            labels_int = [label2int[label.strip()] for label in labels]
                        else:
                            print("Label Type not recognized: ", type(labels))
                            print(labels)
                            
                        print("True label verbalizer after decoding")
                        pp(labels[0])
                        print("---------------------")
                        
                        print("True label int after parsing the label2int dictionary")
                        pp(labels_int[0])
                        print("---------------------")

                        y_preds.extend(pred)
                        targets.extend(labels_int)
                    
                elif self.objective == "SEQ_CLS":
                    labels = [int(label) for label in labels]
                    outputs = self.model(**test_batch)
                    pred = outputs.logits.argmax(dim=-1) # the last dimension -> will get the max of each pair fo 2 logits
                    
                    y_preds.extend(list(pred.cpu()))
                    targets.extend(labels)
                    

        # assert type(targets[0]) == int        
        # assert type(y_preds[0]) == int
        # assert type(generations[0]) == str

        test_results = get_metrics(targets, y_preds)
        test_results["generations"] = generations
        test_results["unfiltered_outputs"] = unfiltered_outputs
        test_results["verbalized_labels"] = verbalized_labels

        gc.collect()
        torch.cuda.empty_cache()

        return test_results

    def train(  self, 
                train_dataloader, 
                validation_dataloader, 
                n_epochs,
                num_training_samples,
                train_sampler=None):

        validation_losses = []
        training_losses = []

        self.model.train()

        for epoch in tqdm(range(n_epochs)):

            if self.distributed_config.distributed_training:
                train_sampler.set_epoch(epoch)
                
            print("EPOCH")
            print(epoch)
            print()
            print("-------------------Training---------------------------")

            batch_losses = []

            for idx, batch in tqdm(enumerate(train_dataloader)):
                if self.mode == "test":
                    if idx >= 1:
                        break

                batch = {key: tensor.to(self.model.device) for key, tensor in batch.items()}

                if self.distributed_config.distributed_training:
                    batch = {key: tensor.to(self.distributed_config.local_rank) for key, tensor in batch.items()}

                # validating the devices
                # print(self.model.device, self.distributed_config.device)
                # for key, tensor in batch.items():
                #     print(key, tensor.device)

                output = self.model(**batch)
                loss = output.loss
                logits = output.logits

####################### CL PRE BACKWARD HERE####################################
                if self.model.cl:
                    batch['logits'] = logits.to(self.distributed_config.device) # needed for LwF
                    loss += self.model.cl.compute_regularization(batch, self.mode)
                    self.model.cl.pre_backward(batch, self.mode)

                loss.backward()
                
####################### CL POST BACKWARD HERE####################################
                if self.model.cl:
                    self.model.cl.post_backward(self.mode)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # no need to reduce the loss across ranks because DPP takes care of it
                if self.distributed_config.distributed_training:
                    loss_tensor = torch.tensor(loss, device=self.distributed_config.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss = loss_tensor/self.distributed_config.world_size

                batch_losses.append(loss.item())

                torch.cuda.empty_cache()
                gc.collect()

            epoch_loss = np.mean(batch_losses)
            training_losses.append(epoch_loss)

            print("-------------------Validation---------------------------")
            epoch_validation_loss = self.validate(validation_dataloader) # no validation sampler
            validation_losses.append(epoch_validation_loss)

            print("EPOCH FINISHED")
            print()       

            if hasattr(self, "early_stopper"):
                stop_training = False # have to define it before rank==0
                if self.distributed_config.distributed_training: 
                    if self.distributed_config.local_rank == 0:
                        stop_training = self.early_stopper.stop_training(epoch_validation_loss, epoch + 1, self.model) # since epoch starts in 0, we gotta add 1
                    else: 
                        torch.distributed.barrier()

                elif not self.distributed_config.distributed_training:
                    stop_training = self.early_stopper.stop_training(epoch_validation_loss, epoch + 1, self.model)

                if stop_training:
                    print("-----------EARLY STOPPING--------------")
                    print("Loading the best model")
                    print("Best Epoch:", self.early_stopper.best_epoch)
                    print("Best Loss:", self.early_stopper.best_loss)
                    print()
                    self.model = self.early_stopper.best_model
                    break

###################### CL POST TASK HERE ######################################
        if self.model.cl:
            self.model.cl.set_memory_size(num_training_samples)
            self.model.cl.post_task_update(train_dataloader, self.mode)

        torch.cuda.empty_cache()
        gc.collect()

        return training_losses, validation_losses

    def continual_learning(self, 
                            datastream:DataStream, 
                            processing_config:ProcessingConfig,
                            n_epochs:int): # modify with the actual way I am doing it

        self.test_results = []
        self.train_logs = []

        experiment_name = "_".join([self.model.model_id.replace("/", "-"), 
                                    self.model.cl_technique.upper(), 
                                    datastream.datastream_name, 
                                    self.model.hyperparam_str])
        
        if not hasattr(self, "early_stopper"):
            experiment_name = experiment_name + "NO-ES"
            
        if hasattr(self.model, "model_lora_rank"):
            experiment_name = experiment_name + f"rank-{self.model.model_lora_rank}_lr{(str(self.learning_rate))}_{str(n_epochs)}_epochs"

        print("Experiment name: ", experiment_name)
        print("Length of datastream: ", datastream.datastream_length)

        for t in tqdm(range(datastream.datastream_length)):

            torch.cuda.empty_cache()
            gc.collect()

            print("TIME")
            print(t)
            print()

            if hasattr(self, "early_stopper"):
                self.early_stopper.reset_stopper() # doing it outside the main train function so that the best_epoch attribute can be saved

            info_training_time_t = {}
            if self.distributed_config.distributed_training:
                dataset_at_t_train_loader, dataset_at_t_val_loader, dataset_at_t_name = datastream.get_data_at_time_t(t)
                training_losses, validation_losses = self.train(train_dataloader=dataset_at_t_train_loader, validation_dataloader=dataset_at_t_val_loader, 
                                                                n_epochs=n_epochs, num_training_samples=datastream.datastream_samples_per_time[t],
                                                                train_sampler=datastream.datastream[t].train_sampler)
                if self.distributed_config.local_rank == 0:
                    info_training_time_t = { "model": str(self.model.model_id),
                        "type_experiment": str(self.mode),
                        "n_trainable_params": int(self.model.n_trainable_params), 
                        "cl_technique": str(self.model.cl_technique), 
                        "hyperparams": str(self.model.hyperparam_str), 
                        "experiment_name": str(datastream.datastream_name),
                        "time": int(t),
                        "dataset_currently_training": str(dataset_at_t_name),
                        "target_epochs": int(n_epochs),
                        "best_epochs": int(n_epochs),
                        "learning_rate": float(self.learning_rate),
                        "batch_size": int(datastream.datastream[t].batch_size),
                        "current_num_samples_training": int(datastream.datastream_samples_per_time[t]),
                        "cumulative_samples_trained": int(datastream.datastream_cumulative_samples[t])}
                
                    if hasattr(self, "early_stopper"):
                        info_training_time_t["best_epochs"] = int(self.early_stopper.best_epoch),
                

                    info_training_time_t.update({"training_losses": list(training_losses), "validation_losses": list(validation_losses)})
                    self.train_logs.append(info_training_time_t)

            
            else:
                dataset_at_t_train_loader, dataset_at_t_val_loader, dataset_at_t_name = datastream.get_data_at_time_t(t)
                training_losses, validation_losses = self.train(train_dataloader=dataset_at_t_train_loader, validation_dataloader=dataset_at_t_val_loader, 
                                                                n_epochs=n_epochs, num_training_samples=datastream.datastream_samples_per_time[t])

                info_training_time_t = { "model": str(self.model.model_id),
                                        "type_experiment": str(self.mode),
                                        "n_trainable_params": int(self.model.n_trainable_params), 
                                        "cl_technique": str(self.model.cl_technique), 
                                        "hyperparams": str(self.model.hyperparam_str), 
                                        "experiment_name": str(datastream.datastream_name),
                                        "time": int(t),
                                        "dataset_currently_training": str(dataset_at_t_name),
                                        "target_epochs": int(n_epochs),
                                        "best_epochs": int(n_epochs),
                                        "learning_rate": float(self.learning_rate),
                                        "batch_size": int(datastream.datastream[t].batch_size),
                                        "current_num_samples_training": int(datastream.datastream_samples_per_time[t]),
                                        "cumulative_samples_trained": int(datastream.datastream_cumulative_samples[t])}
                
                if hasattr(self, "early_stopper"):
                    info_training_time_t["best_epochs"] = int(self.early_stopper.best_epoch),
                

                info_training_time_t.update({"training_losses": list(training_losses), "validation_losses": list(validation_losses)})
                self.train_logs.append(info_training_time_t)

            print(f"Training completed at time {t}")
            pp(info_training_time_t)

            test_datastream, test_names = datastream.get_datastream_testing_splits() # not need to change this in the merged tasks case since the function will check if the dfs were merged within the datastream.merged blablabla object
            print()
            print("Testing: ", test_names)
            print("Number of datasets: ", len(test_datastream))
            print()

            if self.distributed_config.distributed_training:
                if self.distributed_config.local_rank == 0:
                    for test_data, test_name in zip(test_datastream, test_names):
                        print()
                        print("Currently testing:", test_name)
                        print()
                        dataset_position, dataset_wrt_time = datastream.get_dataset_position_wrt_current_time(test_name, t)
                        test_results_time_t = self.test(test_dataloader=test_data, tokenizer=processing_config.tokenizer)

                        info_testing_time_t = { "model": str(self.model.model_id),
                                            "type_experiment": str(self.mode),
                                            "n_trainable_params": int(self.model.n_trainable_params), 
                                            "cl_technique": str(self.model.cl_technique), 
                                            "hyperparams": str(self.model.hyperparam_str), 
                                            "experiment_name": str(datastream.datastream_name),
                                            "time": int(t),
                                            "dataset_currently_testing": str(test_name),
                                            "dataset_currently_training": str(dataset_at_t_name),
                                            "dataset_wrt_training_datasets": str(dataset_wrt_time),
                                            "target_epochs": int(n_epochs),
                                            "best_epochs": int(n_epochs),
                                            "learning_rate": float(self.learning_rate),
                                            "batch_size": int(datastream.datastream[t].batch_size),
                                            "current_num_samples_training": int(datastream.datastream_samples_per_time[t]),
                                            "cumulative_samples_trained":int(datastream.datastream_cumulative_samples[t])}
                        
                        if hasattr(self, "early_stopper"):
                            info_testing_time_t["best_epoch"] = int(self.early_stopper.best_epoch)
                    
                        if hasattr(self.model, "model_lora_rank"):
                            info_testing_time_t["lora_rank"] = self.model.model_lora_rank

                        info_testing_time_t.update(test_results_time_t)
                        self.test_results.append(info_testing_time_t)
                        print(f"Test results for {test_name}:")
                        pp(info_testing_time_t)
                else:
                    dist.barrier() # the rest of the ranks will wait for the testing


            elif not self.distributed_config.distributed_training: 
                for test_data, test_name in zip(test_datastream, test_names):
                    print("Currently testing:", test_name)
                    dataset_position, dataset_wrt_time = datastream.get_dataset_position_wrt_current_time(test_name, t)
                    test_results_time_t = self.test(test_dataloader=test_data, tokenizer=processing_config.tokenizer)

                    info_testing_time_t = { "model": str(self.model.model_id),
                                            "type_experiment": str(self.mode),
                                            "n_trainable_params": int(self.model.n_trainable_params), 
                                            "cl_technique": str(self.model.cl_technique), 
                                            "hyperparams": str(self.model.hyperparam_str), 
                                            "experiment_name": str(datastream.datastream_name),
                                            "time": int(t),
                                            "dataset_currently_testing": str(test_name),
                                            "dataset_currently_training": str(dataset_at_t_name),
                                            "dataset_wrt_training_datasets": str(dataset_wrt_time),
                                            "target_epochs": int(n_epochs),
                                            "best_epochs": int(n_epochs),
                                            "learning_rate": float(self.learning_rate),
                                            "batch_size": int(datastream.datastream[t].batch_size),
                                            "current_num_samples_training": int(datastream.datastream_samples_per_time[t]),
                                            "cumulative_samples_trained":int(datastream.datastream_cumulative_samples[t])}
                    
                    if hasattr(self, "early_stopper"):
                        info_testing_time_t["best_epoch"] = int(self.early_stopper.best_epoch)
                            
                    if hasattr(self.model, "model_lora_rank"):
                        info_testing_time_t["lora_rank"] = self.model.model_lora_rank

                    
                    info_testing_time_t.update(test_results_time_t)
                    self.test_results.append(info_testing_time_t)

                    print(f"Test results for {test_name}:")
                    pp(info_testing_time_t)


        if self.distributed_config.distributed_training:
            if self.distributed_config.local_rank == 0:
                with open(f"{experiment_name}_RESULTS.json", "w") as f:
                    json.dump(self.test_results, f, indent=4)
                with open(f"{experiment_name}_TRAIN_LOGS.json", "w") as f:
                    json.dump(self.train_logs, f, indent=4)
            else:
                dist.barrier()
        
        else:
            with open(f"{experiment_name}_RESULTS.json", "w") as f:
                json.dump(self.test_results, f, indent=4)
            with open(f"{experiment_name}_TRAIN-LOGS.json", "w") as f:
                json.dump(self.train_logs, f, indent=4)

        # saving the adapters
        if self.model.model_type == "LLM":
            self.model.model.save_pretrained(f"./Adapter_{self.model.model_id.replace('/', '-')}_LoRA_{str(self.model.model_lora_rank)}_{dataset_at_t_name}")

        print(f"Number of test results should be {len(test_datastream)} per each time step. {datastream.datastream_length} time steps -> {len(test_datastream)*datastream.datastream_length}.\t Actual number: {len(self.test_results)}")
        print(f"Number of train logs should be {datastream.datastream_length}.\t\t Actual number: {len(self.train_logs)}")

        
        print("-----------EXPERIMENT COMPLETED-----------")


        return self.test_results, self.train_logs

    def zero_shot_testing(self, 
                            datastream, 
                            processing_config): # modify with the actual way I am doing it

        print("Initializing Zero Shot Testing")

        self.test_results = []
        self.train_logs = []
        dataset_wrt_time = "zero_shot"

        test_datastream, test_names = datastream.get_datastream_testing_splits()

        experiment_name = "_".join([self.model.model_id.replace("/", "-"), 
                                    self.objective, 
                                    self.model.cl_technique.upper()])

        test_names_string = "-".join(test_names)

        # experiment_name = "_".join([experiment_name, test_names_string])

        print("Experiment name: ", experiment_name)
        print("Length of testing datastream: ", len(datastream.zero_shot_datasets))
        print()


        for test_data, test_name in zip(test_datastream, test_names):
            
            if self.distributed_config.distributed_training:
                if self.distributed_config.local_rank == 0:
                    print()
                    print("Currently testing:", test_name)
                    print()
                    print("Type test data: ", type(test_data))
                    
                    test_results = self.test(test_dataloader=test_data, tokenizer=processing_config.tokenizer)

                    info_testing = { "model": str(self.model.model_id),
                                        "type_experiment": str(self.mode),
                                        "n_trainable_params": int(self.model.n_trainable_params), 
                                        "cl_technique": str(self.model.cl_technique), 
                                        "hyperparams": "NA", 
                                        "experiment_name": str(datastream.datastream_name),
                                        "time":"NA",
                                        "dataset_currently_testing": str(test_name),
                                        "dataset_currently_training": "NA",
                                        "dataset_wrt_training_datasets": str(dataset_wrt_time),
                                        "target_epochs": "NA",
                                        "best_epochs": "NA",
                                        "learning_rate": "NA",
                                        "batch_size": "NA",
                                        "current_num_samples_training": "NA",
                                        "cumulative_samples_trained":"NA"}
                
                    info_testing.update(test_results)
                    self.test_results.append(info_testing)
                    print(f"Test results for {test_name}:")
                    pp(test_results)
                else:
                    torch.distributed.barrier() # the rest of the ranks will wait for the testing


            elif not self.distributed_config.distributed_training: 
                print("Currently testing:", test_name)
                
                test_results = self.test(test_dataloader=test_data, tokenizer=processing_config.tokenizer)

                info_testing = { "model": str(self.model.model_id),
                                    "type_experiment": str(self.mode),
                                    "n_trainable_params": int(self.model.n_trainable_params), 
                                    "cl_technique": str(self.model.cl_technique), 
                                    "hyperparams": "NA", 
                                    "experiment_name": str(datastream.datastream_name),
                                    "time":"NA",
                                    "dataset_currently_testing": str(test_name),
                                    "dataset_currently_training": "NA",
                                    "dataset_wrt_training_datasets": str(dataset_wrt_time),
                                    "target_epochs": "NA",
                                    "best_epochs": "NA",
                                    "learning_rate": "NA",
                                    "batch_size": "NA",
                                    "current_num_samples_training": "NA",
                                    "cumulative_samples_trained":"NA"}
            
                info_testing.update(test_results)
                self.test_results.append(info_testing)
                print(f"Test results for {test_name}:")
                pp(test_results)
                
                with open(f"{experiment_name}_{str(test_name)}_RESULTS.json", "w") as f:
                    json.dump(info_testing, f, indent=4)



        if self.distributed_config.distributed_training:
            if self.distributed_config.local_rank == 0:
                with open(f"{experiment_name}_RESULTS.json", "w") as f:
                    json.dump(self.test_results, f, indent=4)
            else:
                dist.barrier()
        
        else:
            with open(f"{experiment_name}_RESULTS.json", "w") as f:
                json.dump(self.test_results, f, indent=4)

        print(f"Number of test results should be {len(test_datastream)} .\t Actual number: {len(self.test_results)}")        
        print("-----------ZERO SHOT EXPERIMENT COMPLETED-----------")


        return self.test_results

    def few_shot_testing(self, 
                            datastream, 
                            processing_config): # modify with the actual way I am doing it

        self.test_results = []
        self.train_logs = []
        dataset_wrt_time = "few_shot"

        test_datastream, test_names = datastream.get_datastream_testing_splits()

        experiment_name = "_".join([self.model.model_id.replace("/", "-"), 
                                    self.objective, 
                                    self.model.cl_technique.upper()])

        # test_names_string = "-".join(test_names)

        # experiment_name = "_".join([experiment_name, test_names_string])

        print("Experiment name: ", experiment_name)
        print("Length of testing datastream: ", len(datastream.zero_shot_datasets))
        print()


        for test_data, test_name in zip(test_datastream, test_names):
            
            if self.distributed_config.distributed_training:
                if self.distributed_config.local_rank == 0:
                    print()
                    print("Currently testing:", test_name)
                    print()
                    print("Type test data: ", type(test_data))
                    
                    test_results = self.test(test_dataloader=test_data, tokenizer=processing_config.tokenizer)

                    info_testing = { "model": str(self.model.model_id),
                                        "type_experiment": str(self.mode),
                                        "n_trainable_params": int(self.model.n_trainable_params), 
                                        "cl_technique": str(self.model.cl_technique), 
                                        "hyperparams": "NA", 
                                        "experiment_name": str(datastream.datastream_name),
                                        "time":"NA",
                                        "dataset_currently_testing": str(test_name),
                                        "dataset_currently_training": "NA",
                                        "dataset_wrt_training_datasets": str(dataset_wrt_time),
                                        "target_epochs": "NA",
                                        "best_epochs": "NA",
                                        "learning_rate": "NA",
                                        "batch_size": "NA",
                                        "current_num_samples_training": "NA",
                                        "cumulative_samples_trained":"NA",
                                        "n_shots": int(datastream.n_shots),
                                        "iteration": int(datastream.iteration)}
                
                    info_testing.update(test_results)
                    self.test_results.append(info_testing)
                    print(f"Test results for {test_name}:")
                    pp(test_results)
                else:
                    torch.distributed.barrier() # the rest of the ranks will wait for the testing


            elif not self.distributed_config.distributed_training: 

                print("Currently testing:", test_name)
                
                test_results = self.test(test_dataloader=test_data, tokenizer=processing_config.tokenizer)

                info_testing = { "model": str(self.model.model_id),
                                    "type_experiment": str(self.mode),
                                    "n_trainable_params": int(self.model.n_trainable_params), 
                                    "cl_technique": str(self.model.cl_technique), 
                                    "hyperparams": "NA", 
                                    "experiment_name": str(datastream.datastream_name),
                                    "time":"NA",
                                    "dataset_currently_testing": str(test_name),
                                    "dataset_currently_training": "NA",
                                    "dataset_wrt_training_datasets": str(dataset_wrt_time),
                                    "target_epochs": "NA",
                                    "best_epochs": "NA",
                                    "learning_rate": "NA",
                                    "batch_size": "NA",
                                    "current_num_samples_training": "NA",
                                    "cumulative_samples_trained":"NA",
                                    "n_shots": int(datastream.n_shots),
                                    "iteration": int(datastream.iteration)}
                
                info_testing.update(test_results)
                self.test_results.append(info_testing)
                print(f"Test results for {test_name}:")
                pp(test_results)
                
                with open(f"{experiment_name}_iter-{str(datastream.iteration)}_shots-{str(datastream.n_shots)}_{str(test_name)}_RESULTS.json", "w") as f:
                    json.dump(info_testing, f, indent=4)


        if self.distributed_config.distributed_training:
            if self.distributed_config.local_rank == 0:
                with open(f"{experiment_name}_iter-{str(datastream.iteration)}_shots-{str(datastream.n_shots)}_RESULTS.json", "w") as f:
                    json.dump(self.test_results, f, indent=4)
            else:
                dist.barrier()
        
        else:
            with open(f"{experiment_name}_iter-{str(datastream.iteration)}_shots-{str(datastream.n_shots)}_RESULTS.json", "w") as f:
                json.dump(self.test_results, f, indent=4)

        print(f"Number of test results should be {len(test_datastream)} .\t Actual number: {len(self.test_results)}")        
        print("-----------FEW SHOT EXPERIMENT COMPLETED-----------")


        return self.test_results

