
import os

from typing import Dict, List

from pprint import pprint as pp
from dotenv import load_dotenv


from datetime import timedelta
from copy import deepcopy

import random
import re
import numpy as np
import pandas as pd 

from huggingface_hub import whoami, HfFolder

import torch.distributed as dist
import torch

from .constants import translation_dict_from_label_to_int, translation_dict_from_label_to_verbalizer, int2label, clean_up_mappings


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class DistTrainingConfig():

    def __init__(self, distributed_training:bool=False, time_out=1000000000000):

        self.distributed_training = distributed_training
        
        if self.distributed_training:
            print("Initializing Distributed Training")
            self.local_rank = int(os.environ.get("LOCAL_RANK"))
            self.rank = int(os.environ.get("RANK"))
            self.world_size = int(os.environ.get("WORLD_SIZE"))

            torch.cuda.set_device(self.local_rank)                 # rank i -> GPU i
            self.device = torch.device("cuda", self.local_rank)

            print("self.local_rank")
            print(self.local_rank)
            print("self.rank")
            print(self.rank)
            print("self.world_size")
            print(self.world_size)  

            dist.init_process_group(
                backend="nccl",
                init_method="env://",                         # uses MASTER_ADDR/PORT from your PBS script
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=time_out)
            )
            
            # dist.init_process_group("nccl", timeout=timedelta(seconds=18000))
            # local_rank = int(os.environ["LOCAL_RANK"])
            # torch.cuda.set_device(local_rank)
            # return local_rank

        
        else:
            self.world_size = None
            self.local_rank = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # safeguard
    def destroy_process(self):

        if self.distributed_training:
            dist.destroy_process_group()

        return print("Distributed training destroyed")

class EarlyStopperAndCheckpointer():

    def __init__(self, patience=4, min_epochs=0, delta=0):

        self.patience = patience
        self.min_epochs = min_epochs

        self.delta = delta

        self.counter = 0

        self.best_loss = float("inf")
        self.best_epoch = None
        self.best_model = None

    def stop_training(self, val_loss, epoch, model):

        stop = False

        if val_loss <= self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_model = deepcopy(model)

        else:
            self.counter += 1
            # safe guard if the min epoch number hasn't been reached but the counter has already reached the patience
            if epoch < self.min_epochs:
                pass # stop is still False

            elif self.counter >= self.patience:
                self.counter = 0
                stop = True
                print("----------------EARLY STOPPING TRAINING----------------")
                print("Best Epoch: ", self.best_epoch)
                print("Best Loss: ", self.best_loss)
                print("---------------------------------------------------------")
                self.best_loss = float("inf")

        return stop

    def reset_stopper(self):
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = None
        self.best_model = None

def log_hf():
    
    load_dotenv("env_vars.env")
    hf_token = os.environ.get("HF_ACCESS_TOKEN")
    HfFolder.save_token(hf_token)
    return print(whoami()["name"])

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_cl_name(cl_name):

    regex = r'<(?:[\w\.]+)?\.([\w]+) object at'
    matches =   re.findall(regex, cl_name)
    clean_string = " + ".join(matches)
    return clean_string

def clean_metric_name(metric_name):

    reg = r"\s([a-z_1]+)\s"
    match_ = re.search(reg, metric_name)
    clean_str = match_.group().strip()

    return clean_str

def translate_prediction_to_label(text):
    if "NOT HATEFUL" in text:
        text_clean = text.replace("NOT HATEFUL", "")
        if "HATEFUL" in text_clean or "HATEFUAL" in text_clean:
            return 2
        else:
            return 0
    elif "NOT_HATEFUL" in text:
        text_clean = text.replace("NOT_HATEFUL", "")
        if "HATEFUL" in text_clean or "HATEFUAL" in text_clean:
            return 2
        else: 
            return 0
    elif "HATEFUL" in text:
        text_clean = text.replace("HATEFUL", "")
        if "NOT_HATEFUL" in text_clean or "NOT HATEFUL" in text_clean:
            return 2
        else:
            return 1
    else:
        return 2

def transform_original_label(label:str|int, 
                            translation_dict:Dict[bool, str|int]=translation_dict_from_label_to_int,
                            int2label:Dict[int, str]=int2label):
    if type(label) == int: #already translated to int - useful for the domain transfer experiments that are already in hateful/not hateful
        return int2label[label]

    bool_label = label == "HATEFUL" or label != "not_hate"
    translation_label = translation_dict[bool_label]

    return translation_label

def squeeze_notneeded_dimension(x):
    x = x.squeeze(1) if x.dim() == 3 and x.size(1) == 1 else x
    return x

def translate_llama_guard_labels(label):
    if label == "\n\nsafe":
        # print("found safe label")
        return 0
    elif label == "\n\nunsafe\nS10":
        # print("found unsafe label")
        return 1
    else:
        # print("Label not recognized: ", label)
        return 2


def filter_few_shots_df(few_shot_df:pd.DataFrame, dataset:str, k:int, iteration:int):
    few_shot_df = few_shot_df[few_shot_df['source'] == dataset]
    few_shot_df = few_shot_df[few_shot_df['shots'] == k]
    few_shot_df = few_shot_df[few_shot_df['iteration'] == iteration]
    few_shot_df = few_shot_df.sample(frac=1, random_state=42).reset_index(drop=True)
    few_shot_examples = []
    for text, label in zip(few_shot_df['text'], few_shot_df['label']):
        few_shot_examples.append({"text":text, "label":label})
    return few_shot_examples


def convert_fewshot_examples_to_str(few_shot_examples:list):

    shot_string = """Example MESSAGE: {}
Example LABEL: {}
"""
    k_shot_list = []    

    for example_dict in few_shot_examples:
        formatted_string = shot_string.format(example_dict['text'], example_dict['label'])
        k_shot_list.append(formatted_string)

    formatted_shots_string = "\n".join(k_shot_list)

    return formatted_shots_string

def clean_up_df(df, mappings=clean_up_mappings):
    for mapping in mappings:
        df.replace(mapping, inplace=True)
        df.rename(columns=mapping, inplace=True)
        df.rename(index=mapping, inplace=True)
    return df

def translate_label_to_int(label):
    
    if re.search(r"not\b", label, re.IGNORECASE):
        # print("LABEL")
        # print(label)
        # print()
        # print("0")
        return 0
    
    elif re.search(r"hateful", label, re.IGNORECASE) and not re.search(r"not\b", label, re.IGNORECASE):
        # print("LABEL")
        # print(label)
        # print()
        # print("1")
        return 1
    
    else:
        return 2

def log_hf():
    load_dotenv("env_vars.env")
    hf_token = os.environ.get("HF_ACCESS_TOKEN")
    HfFolder.save_token(hf_token)
    return print(whoami()["name"])
