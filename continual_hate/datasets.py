from typing import Dict

import itertools

import pandas as pd
import numpy as np
from pprint import pprint as pp

from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from functools import partial

from .constants import translation_dict_from_label_to_verbalizer, translation_dict_from_label_to_int, int2label, base_prompt_few_shot
from .utils import transform_original_label, set_seed, filter_few_shots_df, convert_fewshot_examples_to_str

set_seed(42)

class TaskDataset():

    def __init__(self, 
                df:pd.DataFrame, 
                task:str|list[str], 
                task_column:str="task", 
                text_column:str="text", 
                label_column:str="labels",
                split_column:str="split", 
                label_equivalence_general:Dict[bool, str]=translation_dict_from_label_to_verbalizer, 
                label_equivalence_int:Dict[bool, int]=translation_dict_from_label_to_int,
                zero_shot:bool=False):

        self.zero_shot = zero_shot
        print("Creating TaskDataset Object")
        print()
        print("Dataset: ", task)
        print()
        print(df.columns)
        # to get the Merged-FT Baseline
        if isinstance(task, list): # in case i may want to do mixtures like all racism types -> all misogyny types
            print("\tA list was passed to the task argument so the tasks will be merged. This is expected behaviour for Merged FT Baselines.")
            list_tasks = task # just for readibility
            self.df = df[df[task_column].isin(list_tasks)]
            self.task = "merged" + "_".join(list_tasks)
            self.dfs = [df[df[task_column] == t] for t in list_tasks] # individual df for each task 


        elif isinstance(task, str):
            self.df = df[df[task_column] == task]
            self.task = task
            self.dfs = False

        self.text_column = text_column
        self.label_column = label_column
        self.split_column = split_column

        self.num_samples = len(self.df)
        self.num_training_samples = len(self.df[self.df[self.split_column] == "train"])
        self.num_classes = len(self.df[self.label_column].unique())

        self.task = task

        self.texts = self.df[self.text_column]
        self.labels = self.df[self.label_column]
        self.splits = self.df[self.split_column]

        self.text_type = "tweet"
        self.label_type = "original"

        self.label_equivalence_general = label_equivalence_general
        self.label_equivalence_int = label_equivalence_int
        
        self.split_labels = list(self.df[self.split_column].unique())
        self.unique_labels = list(self.df[self.label_column].unique())

        print("Unique Labels: ", self.unique_labels)
        print()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def get_info_about_label_splits(self):

        self.info_label_splits = {}
        self.info_label_splits["len_total"] = self.num_samples

        combinations = itertools.product(self.split_labels, self.unique_labels)
        for split, label in combinations:

            n_combination = len(self.df[(self.df[self.split_column] == split) & (self.df[self.label_column] == label)])
            n_label = len(self.df[self.df[self.label_column] == label])
            n_split = len(self.df[self.df[self.split_column] == split])
            label_proportion = n_combination / n_split

            self.info_label_splits[split + "_" + label] = {"split": split, 
                                "label": label, 
                                "total_n_label": n_label,
                                "total_n_split": n_split,
                                "total_n_label+split": n_combination,
                                "split_label_proportion": label_proportion, 
                                "split_proportion": n_split / self.num_samples}

            self.info_label_splits["total_len_" + split] = n_split
            self.info_label_splits["total_len_" + label] = n_label

        return self.info_label_splits


    def transform_text_to_prompt(self, prompt:str):
        """
        The prompt string should have a {} placed already.
        """
        self.prompt = prompt

        # in the preprocessing function, we are inputting the label separately from the prompt to make the message list so the label should not be part of the prompt
        self.texts = self.texts.apply(lambda original_tweet: self.prompt.format(original_tweet))
        self.text_type = "formatted_prompt_with_tweet"
        
        return self.texts

    def transform_text_to_few_shot_prompt(self, 
                                            base_prompt_few_shot:str, 
                                            few_shot_df:pd.DataFrame, 
                                            n_shots:int, 
                                            iteration:int):
        
        print("Processing the few shot prompt.")
        print("Texts before inserting the messages into the prompts.")
        print(self.texts.to_list()[2])
        self.prompt = base_prompt_few_shot
        self.n_shots = n_shots
        self.iteration = iteration
        self.few_shot_examples = filter_few_shots_df(few_shot_df=few_shot_df, dataset=self.task, k=self.n_shots, iteration=self.iteration)
        few_shot_string = convert_fewshot_examples_to_str(self.few_shot_examples)

        self.texts = self.texts.apply(lambda original_tweet: self.prompt.format(few_shot_string, original_tweet))
        print("Texts after constructing the few shot prompt.")
        pp(self.texts.to_list()[2])

        self.text_type = "few_shot_formatted_prompt_with_tweet"
        
        return self.texts


    def reset_texts(self):
        self.texts = self.df[self.text_column]
        self.text_type = "tweet"
        return self.texts

    def reset_labels(self):
        self.labels = self.df[self.label_column]
        self.label_type = "original"
        return self.labels
    
    def transform_label_to_int(self):

        print("\tTransforming labels to int - > Labels BEFORE transforming: ", self.unique_labels)
        if type(self.unique_labels[0]) == int or type(self.unique_labels[0]) == np.int64:
            print("\tLabels are already ints")
            self.label_type = "int"
        else:
            self.labels = self.labels.apply(lambda label: transform_original_label(label, self.label_equivalence_int))
            self.label_type = "int"
            self.unique_labels = list(self.labels.unique())
        print("\tLabels AFTER transforming: ", self.unique_labels)
        return self.labels

    def transform_label_to_verbalizer(self):
        
        if type(self.unique_labels[0]) == int or type(self.unique_labels[0]) == np.int64:
            print("\tLabels are already in ints")
            self.labels = self.labels.apply(lambda label: transform_original_label(label, int2label=int2label))
            self.label_type = "verbalizer"
            self.unique_labels = list(self.labels.unique())
            print("\tLabels AFTER transforming: ", self.unique_labels)

        else:
            self.labels = self.labels.apply(lambda label: transform_original_label(label, self.label_equivalence_general))
            self.label_type = "verbalizer"
            self.unique_labels = list(self.labels.unique())
            print("\tLabels AFTER transforming: ", self.unique_labels)
        return self.labels

    def transform_df_to_hf_dataset(self) -> DatasetDict: # can also use it to reset the hf_Dataset if i modify it outside of the class

        for split in ["train", "validation", "test"]: assert split in self.split_labels or self.dfs # either all the splits are in the ds or we made a list of the dfs so the testing would fail the assertion

        hf_dictionary = {"texts": self.texts.to_list(), "labels": self.labels.to_list(), "split": self.splits.to_list()} # using the class attributes because I am modifying those instead of the underlying dataframe
        hf_dataset = Dataset.from_dict(hf_dictionary)

        if self.dfs and not self.zero_shot: # not tests to process
            print("Transforming DF into train and validation splits merged into one")
            self.hf_dataset = DatasetDict({"train": hf_dataset.filter(lambda ds: ds["split"] == "train"), # don't really need to do self.split_column: "" since i am redefining the name of the column within the dictionary i am passing to .from_dcit
                                        "validation": hf_dataset.filter(lambda ds: ds["split"] == "validation")
                                        # "test": hf_dataset.filter(lambda ds: ds["split"] == "test")
                                        })
                                        
        elif self.zero_shot: # if its zero shot, i only need the test split
            print("Transforming DF into test split")
            self.hf_dataset = DatasetDict({"test": hf_dataset.filter(lambda ds: ds["split"] == "test")})

        else:
            print("Transforming DF into all the splits")
            self.hf_dataset = DatasetDict({"train": hf_dataset.filter(lambda ds: ds["split"] == "train"), # don't really need to do self.split_column: "" since i am redefining the name of the column within the dictionary i am passing to .from_dcit
                                        "validation": hf_dataset.filter(lambda ds: ds["split"] == "validation"),
                                        "test": hf_dataset.filter(lambda ds: ds["split"] == "test")})
        return self.hf_dataset

    def process_hf_dataset(self, 
                            processing_function:callable, 
                            processing_function_args:dict,
                            input_columns:list, 
                            columns_to_keep:list, 
                            data_collator:DataCollatorWithPadding=None, 
                            batch_size:int=None, 
                            distributed_training:bool=False,
                            world_size=None,
                            local_rank=None,
                            with_rank:bool=False,
                            num_proc:int=None,
                            shuffle:bool=True):

        assert self.hf_dataset
        
        self.batch_size = batch_size

        # partial allows to prefill some of the arguments of the function so we can use different preprocess fucntions for caulsa ml/sequence classification which have different number of args
        processing_function_pre_filled = partial(processing_function, **processing_function_args)

        for split in self.hf_dataset: # in the case of Causal LM, the test is left padded? - check that thingie

            processing_function_pre_filled = partial(processing_function, split=split, **processing_function_args)

            if distributed_training:

                # with_rank=True,
                # num_proc=torch.cuda.device_count()

                # wrap the processing in barriers so that the process is not duplicated in every gpu
                if local_rank > 0:
                    dist.barrier()

                if "labels" not in self.hf_dataset[split].column_names:
                    self.hf_dataset = self.hf_dataset.rename_column("label", "labels")

                columns_to_remove = [column for column in self.hf_dataset[split].column_names if column not in columns_to_keep]
                # self.hf_dataset[split] = self.hf_dataset[split].map(processing_function_pre_filled, input_columns=input_columns, with_rank=with_rank, num_proc=num_proc)
                self.hf_dataset[split] = self.hf_dataset[split].map(processing_function_pre_filled, input_columns=input_columns)

                self.hf_dataset[split] = self.hf_dataset[split].remove_columns(columns_to_remove)
                self.hf_dataset[split].set_format("torch")
                distributer = DistributedSampler(self.hf_dataset[split], num_replicas=world_size, rank=local_rank, shuffle=shuffle)
                # shuffle has to be false for the dataloader since the data distribution is handled by the sampler
                data_loader = DataLoader(self.hf_dataset[split], collate_fn=data_collator, batch_size=batch_size, sampler=distributer, shuffle=False)

                if split == "train":
                    self.train_data_loader = data_loader
                    self.train_sampler = distributer
                elif split == "validation":
                    self.validation_data_loader = data_loader
                    self.validation_sampler = distributer
                elif split == "test":
                    self.test_data_loader = data_loader
                    self.test_sampler = distributer

                if local_rank == 0:
                    dist.barrier()

            else: 

                if "labels" not in self.hf_dataset[split].column_names:
                    self.hf_dataset = self.hf_dataset.rename_column("label", "labels")

                columns_to_remove = [column for column in self.hf_dataset[split].column_names if column not in columns_to_keep]
                self.hf_dataset[split] = self.hf_dataset[split].map(processing_function_pre_filled, input_columns=input_columns)
                self.hf_dataset[split] = self.hf_dataset[split].remove_columns(columns_to_remove)
                self.hf_dataset[split].set_format("torch")
                # shuffle has to be false for the dataloader since the data distribution is handled by the sampler
                data_loader = DataLoader(self.hf_dataset[split], collate_fn=data_collator, batch_size=batch_size, shuffle=shuffle)

                if split == "train":
                    self.train_data_loader = data_loader
                elif split == "validation":
                    self.validation_data_loader = data_loader
                elif split == "test":
                    self.test_data_loader = data_loader

        return self.hf_dataset
