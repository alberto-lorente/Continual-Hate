from typing import Dict
from pprint import pprint as pp

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from .constants import dict_name_experiments_domain_transfer, dict_name_experiments_explicit_implicit
from .utils import clean_metric_name, set_seed

set_seed(42)

def get_metrics(targets, y_pred, metrics=[f1_score, precision_score, recall_score, accuracy_score]) -> Dict[str, float|list]:
    
    result = {clean_metric_name(str(score)): float(score(targets, y_pred, average='macro')) for score in metrics if score != accuracy_score}
    result["accuracy"] = float(accuracy_score(targets, y_pred))

    results_hate_class = {"HATE_" + clean_metric_name(str(score)): float(score(targets, y_pred, labels=[1], average='macro')) for score in metrics if score  in [f1_score, precision_score, recall_score]}
    results_nohate_class = {"NoHATE_" + clean_metric_name(str(score)): float(score(targets, y_pred, labels=[0], average='macro')) for score in metrics if score  in [f1_score, precision_score, recall_score]}

    result.update(results_hate_class)
    result.update(results_nohate_class)

    result["predictions"] = [int(pred_label) for pred_label in y_pred]
    result["labels"] = [int(actual_y) for actual_y in targets]
    
    return result

def compute_last(matrix):
    last = float(np.mean(matrix[-1]))
    return last

def compute_bwt(matrix):

    num_tasks = matrix.shape[1]
    total_sum = 0
    T_idx = num_tasks - 1

    for i in range(num_tasks-1): # don't have to do -2 bc the range already gives me -1
    # print(i) # is go from 0 to 3 -> 4 idx
        # print(T_idx)
        # print(i)
        # print()
        total_sum += matrix[T_idx, i] - matrix[i, i]

    avg = float(total_sum / (num_tasks-1))
    return avg

def compute_fwt(matrix, zero_shot_array):

    num_tasks = matrix.shape[1]
    total_sum = 0

    i = 1 # the index starts shifted to the right by 1
    while i  < num_tasks: # skips the first item and then goes incrementally to T
        # print(i-1)
        # print(i)
        # print()
        total_sum += matrix[i-1 , i] - zero_shot_array[i]
        i += 1

    avg = float(total_sum / (num_tasks-1))
    return avg

def compute_aia(matrix): # version without the difference wrt to a baseline, some authors do it that way

    num_tasks = matrix.shape[1]
        
    total_sum = 0

    for i in range(num_tasks): # for every task test
        # print(i)
        sum_row = 0
        for j in range(i+1):
            # print("\t", j)
            sum_row += matrix[i, j] # skip over the zero shot
        # print()
        avg_sum_row = sum_row/ (i+1)
        total_sum += avg_sum_row

    avg = float(total_sum / num_tasks)

    return avg

def compute_transfer(matrix):
    num_tasks = matrix.shape[1]
    total_sum = 0
    i = 1 # starting with the idx shifted
    while i < num_tasks: # for every task test
        # print(i)
        sum_row = 0
        for j in range(i): # because the index starts at 1, this will already go from 0
            # print("\t", j)
            sum_row += matrix[i, j]
        # print()
        avg_sum_row = sum_row/(i) # because the index starts at 0 -> when our index is one=the second position, we divide by 1
        total_sum += avg_sum_row
        i += 1

    avg = float(total_sum / (num_tasks - 1) )
    return avg

def evaluate_cl_metrics(matrix, zero_shot_array):
    
    last = compute_last(matrix)
    aia = compute_aia(matrix)
    transfer = compute_transfer(matrix)
    bwt  = compute_bwt(matrix)
    fw_transfer = compute_fwt(matrix, zero_shot_array)
    
    cl_metrics = {
        "last":last,
        "avg_incremental_f1":aia,
        "transfer": transfer,
        "bwt": bwt,
        "fw_transfer": fw_transfer
    }
    
    return cl_metrics

def get_df_metadata(df, time_column="time", cols=["model", "cl_technique", "experiment_name", "time", "best_epochs", "current_num_samples_training", "learning_rate", "hyperparams"]):
    
    metadata = {}

    for col in cols: assert col in df.columns, f"Column {col} not in df!!!" # if the column is not, the for loop will fail

    for col in cols:
        if col == time_column:
            metadata[col] = df[col].max()
            continue
        md = df[col].unique()[0]
        metadata[col] = md   
    print("Metadata for current DF")
    print(metadata.keys())
    pp(metadata)
    return metadata


def compute_performance_matrix(df, 
                                performance_column="f1_score", 
                                experiment_name_column="type_experiment", 
                                dataset_test_results_column="dataset", 
                                dataset_current_training_column="curr_train", 
                                time_column="time", 
                                dataset_dict_trainings=dict_name_experiments_domain_transfer, 
                                zero_shot=False):

    experiment = df[experiment_name_column].unique()[0]
    print("Experiment -> ", experiment)
    train_list = dataset_dict_trainings[experiment]
    print("Datasets trained ->")
    print(train_list)
    never_trained = set(df[dataset_test_results_column].unique()) - set(df[dataset_current_training_column].unique()) # either the dataset is in train list or it is not in never trained
    df_filter = df[df[dataset_test_results_column].isin(train_list)]

    t_id_to_data = {int(t): df_filter[df_filter[time_column] == t][dataset_current_training_column].unique()[0] for t in df_filter[time_column].unique()}
    # print(t_id_to_data)
    data_to_t_id = {v:k for k, v in t_id_to_data.items()}
    # print(data_to_t_id)
    num_ds = len(data_to_t_id.keys())


    matrix = np.full((num_ds, num_ds), np.nan)
    for dataset, time in data_to_t_id.items():
        f1_list = list(df_filter[(df_filter[dataset_current_training_column]==dataset) & (df_filter[time_column] == time)][performance_column])
        j = 0
        for f1 in f1_list:
            matrix[time, j] = f1
            j += 1
    print()
    print("Matrix Computed")
    print(matrix)

    return matrix, num_ds, train_list

def get_zero_shot_filtered(metadata, 
                            zero_shots_df, 
                            in_training,
                            performance_column="f1_score", 
                            model_name_column="model", 
                            experiment_name_column="experiment_name", 
                            dataset_test_results_column="dataset_currently_testing", 
                            dataset_current_training_column="dataset_currently_training", 
                            time_column="time"):
    
    print("Getting the zero shot array")
    model = metadata[model_name_column]
    type_experiment = metadata[experiment_name_column]
    print(model)
    print(type_experiment)
    print(in_training)
    print(zero_shots_df.columns)
    
    model_condition = (zero_shots_df[model_name_column] == model)
    part_of_training_condition = (zero_shots_df[dataset_test_results_column].isin(in_training))
    # i dont think i need ->  (zero_shots_df[experiment_name_column] == type_experiment) &
    filtered_zero_shots = zero_shots_df[ model_condition & part_of_training_condition]
    filtered_zero_shots[dataset_test_results_column] = filtered_zero_shots[dataset_test_results_column].astype("category")
    filtered_zero_shots[dataset_test_results_column] = filtered_zero_shots[dataset_test_results_column].cat.set_categories(in_training)
    filtered_zero_shots.sort_values([dataset_test_results_column], inplace=True)

    print("ZERO SHOT ARRAY")
    print(filtered_zero_shots)

    return filtered_zero_shots

def transform_df_to_cl_metrics(df, 
                                zero_shots_df, 
                                performance_column="f1_score", 
                                experiment_name_column="type_experiment", 
                                dataset_test_results_column="dataset", 
                                dataset_current_training_column="curr_train", 
                                time_column="time",
                                model_name_column="model", 
                                dataset_dict_trainings=dict_name_experiments_domain_transfer, 
):

    metadata = get_df_metadata(df)
    matrix, num_ds, in_training = compute_performance_matrix(df,  
                                                            performance_column=performance_column, 
                                                            experiment_name_column=experiment_name_column, 
                                                            dataset_test_results_column=dataset_test_results_column, 
                                                            dataset_current_training_column=dataset_current_training_column, 
                                                            time_column=time_column, 
                                                            dataset_dict_trainings=dataset_dict_trainings, 
                                                            zero_shot=False)

    zero_shot_df = get_zero_shot_filtered(metadata=metadata, 
                                        zero_shots_df=zero_shots_df, 
                                        in_training=in_training,
                                        performance_column=performance_column, 
                                        model_name_column=model_name_column, 
                                        experiment_name_column=experiment_name_column, 
                                        dataset_test_results_column=dataset_test_results_column, 
                                        dataset_current_training_column=dataset_current_training_column, 
                                        time_column=time_column)

    zero_shot_array = zero_shot_df[performance_column].to_numpy()
    cl_metrics = evaluate_cl_metrics(matrix, zero_shot_array)
    metadata.update(cl_metrics)
    df_results = pd.DataFrame([metadata], columns=metadata.keys(), index=None)
    df_results["metric"] = performance_column
    
    return df_results