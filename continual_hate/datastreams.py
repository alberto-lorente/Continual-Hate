from typing import List

import numpy as np

from torch.utils.data import DataLoader

from .datasets import TaskDataset
from .utils import set_seed

set_seed(42)

class DataStream():

    def __init__(self, datastream:List[TaskDataset], 
                experiment_name:str, 
                zero_shot_datasets:bool|List[TaskDataset]=False, 
                merged_datasets:bool=False):

        self.datastream = datastream
        self.datastream_length = len(self.datastream)
        self.zero_shot_datasets = zero_shot_datasets
        self.merged_datasets = merged_datasets

        if zero_shot_datasets:
            if hasattr(self.zero_shot_datasets[0], "iteration") and hasattr(self.zero_shot_datasets[0], "n_shots"):
                self.iteration = self.zero_shot_datasets[0].iteration
                self.n_shots = self.zero_shot_datasets[0].n_shots

        self.dataset_names = [task_dataset.task for task_dataset in datastream]

        if not self.merged_datasets:
            self.datastream_name = "-TO-".join(self.dataset_names)
            print("-Datastream name: " + self.datastream_name + "-")

        else:
            self.datastream_name = "_"
            for dataset in self.dataset_names:
                self.datastream_name += "merged" + "_".join(dataset)
            print("-Datastream name: " + self.datastream_name + "-")

        self.experiment_name = experiment_name

        self.datastream_samples_per_time = [ds.num_training_samples for ds in self.datastream]
        self.datastream_cumulative_samples = np.cumsum(self.datastream_samples_per_time)


    def get_data_at_time_t(self, t:int, distributed_training:bool=False):

        if distributed_training:

            return (self.datastream[t].train_data_loader,
                    self.datastream[t].train_sampler,
                    self.datastream[t].validation_data_loader, 
                    self.datastream[t].validation_sampler, 
                    self.dataset_names[t])

        else:
            
            return (self.datastream[t].train_data_loader, 
                    self.datastream[t].validation_data_loader, 
                    self.dataset_names[t])

    def get_datastream_testing_splits(self) -> (List[DataLoader], List[str]):

        if self.merged_datasets: # the tasks in the datastreams will not have the test split, it is exclusively in the zero shot
            print("Merged datasets. Getting the test splits saved on the ZERO SHOT attributes")
            total_test_data = [task_data.test_data_loader for task_data in self.zero_shot_datasets]
            total_test_names = [task_data.task for task_data in self.zero_shot_datasets]
        
        else: # datasets were not merged -> every dataset in the stream has its test split inside the object
            print("Not merged datasets. Getting the test splits saved on the DATASTREAM attributes")
            total_test_data = [task_data.test_data_loader for task_data in self.datastream]
            total_test_names = [task_data.task for task_data in self.datastream]


            if self.zero_shot_datasets: # will get here if there are no merged datasets
                print("Adding the zero shot test splits")
                zero_shot_test_data = [task_data.test_data_loader for task_data in self.zero_shot_datasets]
                zero_shot_test_names = [task_data.task for task_data in self.zero_shot_datasets]
                total_test_data += zero_shot_test_data
                total_test_names += zero_shot_test_names

        return total_test_data, total_test_names

    def get_dataset_position_wrt_current_time(self, dataset_name:str, current_time:int) -> (int, str):

        if dataset_name not in self.dataset_names:
            dataset_wrt_time = "ZERO_SHOT_not_in_training_stream"
            dataset_position = -1
            return dataset_position, dataset_wrt_time
        
        dataset_position = self.dataset_names.index(dataset_name)
        if dataset_position == current_time:
            dataset_wrt_time = "IN_TRAINING"
        elif dataset_position < current_time:
            dataset_wrt_time = "TRAINING_PASSED"
        elif dataset_position > current_time:
            dataset_wrt_time = "ZERO_SHOT"

        return dataset_position, dataset_wrt_time
