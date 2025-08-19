import torch
import torch.nn as nn
import gc

from torch.utils.data import DataLoader, RandomSampler 

from copy import deepcopy
from torch.nn import KLDivLoss
import torch.nn.functional as F
from torch import autograd

from .constants import cl_hyperparameters
from .utils import set_seed

set_seed(42)

class CLTechniques():

    def __init__(self, model, device, technique="none", hyperparams:dict=cl_hyperparameters):

        self.model = model
        self.device = device
        self.technique = technique.lower()
        self.hyperparams = hyperparams

        # Initialize selected technique
        if self.technique == "ewc":
            print("INITIALIZING EWC")
            self._init_ewc(**self.hyperparams[self.technique])
        elif self.technique == "agem":
            print("INITIALIZING AGEM")
            self._init_agem(**self.hyperparams[self.technique])
        elif self.technique == "lwf":
            print("INITIALIZING LwF")
            self._init_lwf(**self.hyperparams[self.technique])
        elif self.technique == "mas":
            print("INITIALIZING MAS")
            self._init_mas(**self.hyperparams[self.technique])

    def _init_ewc(self, ewc_lambda):

        self.ewc_lambda = ewc_lambda

        # store the parameters
        self.params = {n: p.detach().clone().to(self.device)
                    for n, p in self.model.named_parameters()
                    if p.requires_grad}

        # start the fisher matrix with 0 because at time 0 there should be no regularization applied
        self.fisher = {n: torch.zeros_like(p, device=self.device)
                    for n, p in self.model.named_parameters()
                    if p.requires_grad}

        print("-EWCFisher initialized-") 
        print("-EWCLambda: ", self.ewc_lambda)
        print("-EWC FISHER and PARAMS initialized-")


    def _init_agem(self, mem_size_proportion):

        self.mem_size_proportion = mem_size_proportion
        self.mem_size = None
        self.memory = []

        print("-AGEG Memory initialized-") 
        print("-AGEG Proportion: ", self.mem_size_proportion)
        print("-AGEG Empty Memory initialized-")


    def _init_lwf(self, lwf_lambda, temperature):

        self.lwf_lambda = lwf_lambda
        self.temperature = temperature
        self.old_model = None

        print("-LwF initialized-")
        print("-LwF Lambda: ", self.lwf_lambda)
        print("-LwF Temperature: ", self.temperature)

    def _init_mas(self, mas_lambda, mas_variation):

        self.mas_lambda = mas_lambda
        self.mas_variation = mas_variation

        # start the importance matrix with 0 because at time 0 there should be no regularization applied
        self.importance = {n: torch.zeros_like(p, device=self.device)
                        for n, p in self.model.named_parameters()
                        if p.requires_grad}
        # old importances
        self.old_params = {n: p.detach().clone().to(self.device)
                    for n, p in self.model.named_parameters()
                    if p.requires_grad}

        print("-MAS initialized-")
        print("-MAS Lambda: ", self.mas_lambda)
        print("-MAS IMPORTANCE and OLD PARAMS initialized-")

    def set_memory_size(self, num_training_samples):

        if self.technique == "agem":
            self.mem_size = int(self.mem_size_proportion * num_training_samples)
            print("-AGEM- Setting the Memory size for")
            print("-AGEM- Memory size: ", self.mem_size)

    def compute_regularization(self, inputs=None, mode="train"): # in case i restric it for testing purposes

        gc.collect()

        if self.technique == "ewc":

            print("-EWC- Computing Regularization with FISHER MATRIX and PARAMS-")
            print("-EWC- Adjusting the penalty-")

            penalty = 0
            for n, p in self.model.named_parameters():

                # p is the CURRENT parameter value
                if p.requires_grad and p.grad is not None:

                    # get the importance of the parameter by the fisher matrix we already computed (will be 0 at time 0)
                    fisher_param = self.fisher[n]

                    # get the value of the param itself
                    param = self.params[n]

                    # the penalty is the sum of the importance of the parameter * (current parameter - old parameter)^2
                    penalty += (fisher_param * (p - param).pow(2)).sum()

            # we have to apply the lambda/2 to scale the effect of the penalty - some people don't scale the weight/2
            scaled_penalty = (self.ewc_lambda/2) * penalty

            torch.cuda.empty_cache()

            return scaled_penalty

        elif self.technique == "lwf" and self.old_model: # will skip if there was no old model (at time 0)

            print("-LwF- Computing Regularization with Old Model -")

            with torch.no_grad():

                # we get the logits from the current model and batch
                logits = inputs['logits']

                # we get the inputs of the current batch
                actual_inputs = {k:v for k,v in inputs.items() if k != "logits"}

                # pass the current inputs to the old model
                old_outputs = self.old_model(**actual_inputs)

                # save the logtis of the current batch but computed by the old model
                logits_old_model = old_outputs.logits

            student_ = torch.log_softmax(logits/self.temperature, dim=1)
            teacher_ = torch.softmax(logits_old_model/self.temperature, dim=1)
            kdiv_loss = KLDivLoss(reduction='batchmean')(student_, teacher_) * (self.temperature ** 2)

            # final loss scaled by the lambda param
            loss = self.lwf_lambda * kdiv_loss 

            torch.cuda.empty_cache()

            return loss

        elif self.technique == "mas":

            print("-MAS- Computing Regularization with IMPORTANCE AND OLD PARAMS")
            print("-MAS- Adjusting the penalty")


            penalty = 0

            for n, p in self.model.named_parameters():

                # get the imporance of the parameter
                importance_param = self.importance[n]

                # get the param value itself
                param = self.old_params[n]

                # the penalty is the sum of the importance of the parameter * (current parameter - old parameter)^2
                if p.requires_grad:
                    penalty += (importance_param * (p - param).pow(2)).sum()
                    
            # the finnal penaly is penalty scaled by that lambda param we set
            scaled_penalty = self.mas_lambda * penalty

            torch.cuda.empty_cache()

            return scaled_penalty

        return 0

    def pre_backward(self, inputs=None, mode="train"):

        # the reference gradients are computed for each training step
        if self.technique == "agem" and self.memory: # will skip at time 0

            print("-AGEM- Getting Reference Gradients")

            self.model.zero_grad() # this way, the gradients can be accumulated across batches

            n_batches = len(self.memory)

            self.ref_grad = {n: torch.zeros_like(p, device=self.device)
                        for n, p in self.model.named_parameters()
                        if p.requires_grad}

            # replay the memory to get the models gradients with a sample of batches of the old tasks
            for inputs_mem, labels_mem in self.memory:

                self.model.zero_grad() # not accumulate gradients

                torch.cuda.empty_cache()

                inputs_mem = {k:v for k,v in inputs_mem.items()}
                labels_mem = labels_mem
                outputs = self.model(**inputs_mem, labels=labels_mem)
                loss = outputs.loss

                loss.backward() # get the referent gradients 

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        ref_g = p.grad / n_batches
                        # update the reference gradients
                        self.ref_grad[n] += ref_g.detach().clone()

            self.model.zero_grad() # clear the gradients before computing the backward of the current task in the training loops
            torch.cuda.empty_cache()
            gc.collect()

    def post_backward(self, mode="train"):

        gc.collect()

        # after the backward passes of the current task's data and the data in memory, we perform the gradient projection
        if self.technique == "agem" and hasattr(self, 'ref_grad'): 
            
            print("-AGEM- Projecting the Gradients")

            # dot product of the gradient of the current parameter and the referent gradient
            dot_product = sum(torch.sum(p.grad * self.ref_grad[n])
                        for n, p in self.model.named_parameters() 
                        if p.requires_grad and p.grad is not None)

            # compute the squared L2 norm of the each referent gradient. it is squared bc if the reference gradient is negative, you would be changing the sign to positive and the direction would be the opposite
            ref_norm = sum(torch.sum(g_ref * g_ref)
                        for g_ref in self.ref_grad.values())

            # if the dot product is negative, we project the gradient because it means that the gradients are going in different directions
            if dot_product < 0:
                scale = dot_product / (ref_norm + 1e-8) # prevent division by 0
                for n, p in self.model.named_parameters():
                    if p.grad is not None and p.requires_grad:
                        # the referent gradient is scaled and then the current parameter gradient is equal to he current parameter gradient minus the scaled referent gradient
                        p.grad -= scale * self.ref_grad[n]
            
            torch.cuda.empty_cache()
            gc.collect()

    def post_task_update(self, dataloader=None, mode="train"):

        # update the fisher matrix using the current task's data and save the parameters for reference for the future task
        if self.technique == "ewc":

            print("-EWC- Updating the FISHER MATRIX AND PARAMS")

            self.model.eval()
            self.model.zero_grad() 

            # log_likelihoods = []

            for idx, batch in enumerate(dataloader):
                # for testing purposes
                if mode == "test" and idx >= 1:
                    break
            
                self.model.zero_grad()
                print("updated")
                batch = {k:v.to(self.device) for k, v in batch.items()}
                labels = batch['labels']

                outputs = self.model(**batch)
                logits = outputs.logits

                log_probs = F.log_softmax(logits, dim=1)
                # taking the log prob of just the correct classes
                correct_log_probs = log_probs[torch.arange(len(labels)), labels]
                loss = -correct_log_probs.mean()
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.fisher[n] += (p.grad.detach().clone() ** 2) / len(dataloader)   

                # log_likelihoods.append(correct_log_probs) -> if we werent averaging it
                torch.cuda.empty_cache()

            # CAUSES CUDA OUT OF MEMORY ISSUES -> TRYNG TO NOT KEEP ALL THE GRAD AND JUST DO IT BY BATCH
            # mean_log_likelihoods = torch.cat(log_likelihoods).mean()
            # # gradients of each param for the log likelihood of the correct class
            # grad_log_likelihood = autograd.grad(mean_log_likelihoods, self.model.parameters())
            
            # # updating the fisher matrix
            # for (n, p), g in zip(self.model.named_parameters(), grad_log_likelihood):
            #     if p.requires_grad and g is not None and p is not None:

            #         g = g.to(self.device)
            #         self.fisher[n] = self.fisher[n].to(self.device)

            #         self.fisher[n] += (g.detach() ** 2)      

            torch.cuda.empty_cache()

            # save the parameters of this tasks model
            self.params = {n: p.detach().clone()
                        for n, p in self.model.named_parameters()
                        if p.requires_grad}

        elif self.technique == "agem":

            print("-AGEM- Updating the MEMORY")

            # first in, first out memory strategy wrt to the datasets. samples from previous tasks are not kept
            self.memory.clear()

            # if we are going to sample sequentially, we need to create a random loader
            random_loader = DataLoader(
                                    dataset=dataloader.dataset,
                                    sampler=RandomSampler(dataloader.dataset),
                                    batch_size=dataloader.batch_size,
                                    shuffle=False)

            num_samples_in_memory = 0

            for idx, batch in enumerate(random_loader):

                # for testing purposes
                if mode == "test" and idx >= 2:
                    break

                inputs = {k:v.to(self.device) for k, v in batch.items()
                        if k in ['input_ids', 'attention_mask']}

                labels = batch['labels'].to(self.device)

                self.memory.append((inputs, labels)) # this is batched
                num_samples_in_memory += len(batch)

                if num_samples_in_memory > self.mem_size: 

                    n_samples_to_remove = num_samples_in_memory - self.mem_size # how mcuh did i go over for
                    
                    last_batch = self.memory.pop(-1) # remove the last batch
                    last_inputs = last_batch[0] # is a dict
                    last_labels = last_batch[1] # is a list
                    
                    # sanity check if the number samples to remove are the same number of samples in a batch, we just drop it
                    if n_samples_to_remove >= len(last_labels):
                        print("N samples to remove == N samples in the batch. Dropping them.")
                        num_samples_in_memory -= len(last_labels)
                        break

                    last_inputs_minus_n = {k: v[:-n_samples_to_remove] for k, v in last_inputs.items()} # remove the last n samples from the batch
                    last_labels_minus_n = last_labels[:-n_samples_to_remove] # remove the last n samples from the labels

                    num_samples_in_memory -= n_samples_to_remove #update the number of samples in memory

                    self.memory.append((last_inputs_minus_n, last_labels_minus_n))

                    break

                # for i, (mem_inputs, mem_labels) in enumerate(self.memory):
                #     # print(f"Memory example {i}:")
                #     for k, v in mem_inputs.items():
                #         print(f"{k}: {v.shape} | dtype: {v.dtype}")
                #     print(f"labels: {mem_labels.shape} | dtype: {mem_labels.dtype}")
                # print(f"Total memory stored: {len(self.memory)} examples")


        elif self.technique == "lwf":

            print("-LwF- Saving the Old Model")
            # saving the current model for the next time t
            self.old_model = deepcopy(self.model)
            self.old_model.eval()

        elif self.technique == "mas":

            print("-MAS- Updating the IMPORTANCE AND OLD PARAMS")

            self.model.zero_grad()
            self.model.eval()

            for idx, batch in enumerate(dataloader):

                self.model.zero_grad() # beacuse we don't want to accumulate gradients and I didnt pass the optimizer. since we are passing batches and we are adding them to the importance within the loop

                if mode == "test" and idx >= 2:
                    break

                batch = {k:v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits

                if self.mas_variation == "global":
                    l2_norm_logits = torch.norm(logits, p=2, dim=1) # global version of the paper, in the local version they square the logits and then do the mean - they justify this linking it with hebbian learning theory
                    l2_norm_logits = l2_norm_logits.pow(2).mean()
                elif self.mas_variation == "local":
                    l2_norm_logits = torch.norm(logits, p=2, dim=1).pow(2).mean()
                    l2_norm_logits = l2_norm_logits.pow(2).mean()

                # doing the backward pass will give us the sensitivity of the parameters with respect to the l2 normalization
                l2_norm_logits.backward()

                for n, p in self.model.named_parameters():
                    
                    if p.requires_grad and p.grad is not None:

                        self.importance[n] += p.grad.detach().clone().abs() / len(dataloader)
                
                torch.cuda.empty_cache()

            # saving the old params for the regularization of the future task
            self.old_params = {n: p.detach().clone()
                            for n, p in self.model.named_parameters()
                            if p.requires_grad and p is not None}

            self.model.zero_grad() # clear gradients