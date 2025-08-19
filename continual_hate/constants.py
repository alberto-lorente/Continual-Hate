base_prompt = """You are a social media content moderator.
INSTRUCTION: The following is a social media message that needs to be classified with the label HATEFUL or NOT HATEFUL.
MESSAGE: {}
OUTPUT AND FORMAT: your output should be just the HATEFUL or NOT HATEFUL label."""

translation_dict_from_label_to_verbalizer = {True: "HATEFUL",
                                            False: "NOT HATEFUL"}

translation_dict_from_label_to_int = {True: 1,
                                    False: 0}

label2int = {"NOT HATEFUL": 0, "HATEFUL": 1}
int2label = {0: "NOT HATEFUL", 1: "HATEFUL"}

llama_guard_equivalence = {"NOT HATEFUL": "\n\nsafe", "HATEFUL": "\n\nunsafe\nS10"}

# starting point
cl_hyperparameters = {"ewc": {"ewc_lambda":1500},
                    "agem": {"mem_size_proportion":0.025},
                    "lwf": {"lwf_lambda":1,
                            "temperature":2},
                    "mas": {"mas_lambda":1000,
                            "mas_variation":"global"}}

# more regularization and lwf favors old tasks
cl_hyperparameters_2 = {"ewc": {"ewc_lambda":2000},
                    "agem": {"mem_size_proportion":0.05},
                    "lwf": {"lwf_lambda":1.5,
                            "temperature":2},
                    "mas": {"mas_lambda":1500,
                            "mas_variation":"global"}}

# less regularization and lwf favors new tasks
cl_hyperparameters_3 = {"ewc": {"ewc_lambda":1000},
                    "agem": {"mem_size_proportion":0.01},
                    "lwf": {"lwf_lambda":0.5,
                            "temperature":2},
                    "mas": {"mas_lambda":500,
                            "mas_variation":"global"}}

# generation_config = {"max_new_tokens":16, 
#                     "top_p":1,
#                     "temperature":0.6,
#                     "length_penalty":-0.25, 
#                     "no_repeat_ngram_size":3}

generation_config = {"do_sample":False, "max_new_tokens":16}


tokenizer_args = {"max_length": 512, "loss_function_over_prompt":False, "padding":"max_length"}

data_collator_args={"padding":"longest", "return_tensors":"pt"}

dict_name_experiments_domain_transfer = {
        
        "alternate_mis_raci": [
                "evalita",
                "waseem-racism",
                "ibereval",
                "hateval-immigrant"],
        
        "from_rac_to_mis": [
                "hateval-immigrant",
                "waseem-racism",
                "ibereval"],

        "from_general_to_alternating_miso_raci": ["davidson",
                                                "founta_hateful_57k",
                                                "ibereval",
                                                "hateval-immigrant",
                                                "hateval-women",
                                                "waseem-racism"], 
        
        "evalita-TO-waseem-racism-TO-ibereval-TO-hateval-immigrant": [
                                                                "evalita",
                                                                "waseem-racism",
                                                                "ibereval",
                                                                "hateval-immigrant"],
        
        "hateval-immigrant-TO-waseem-racism-TO-ibereval": [
                                                        "hateval-immigrant",
                                                        "waseem-racism",
                                                        "ibereval"],

        "davidson-TO-founta_hateful_57k-TO-ibereval-TO-hateval-immigrant-TO-hateval-women-TO-waseem-racism": [
                                                                                                                "davidson",
                                                                                                                "founta_hateful_57k",
                                                                                                                "ibereval",
                                                                                                                "hateval-immigrant",
                                                                                                                "hateval-women",
                                                                                                                "waseem-racism"]}

dict_name_experiments_explicit_implicit = {"from_expl_to_impl": ["explicit_hs", "implicit_hs"]}

models_string_equivalences = {"diptanu/fBERT": "fBERT",
                            "FacebookAI/roberta-base": "RoBERTa",
                            'GroNLP/hateBERT': "HateBERT"}

cl_techniques_equivalence = {"vanilla_ft": "SEQ-FT",
                            "BASIC_FT": "FT",
                            "MERGED_FT": "M-FT",
                            "agem":"A-GEM",
                            "ewc": "EWC",
                            "lwf": "LwF",
                            "mas": "MAS"}
experiments_equivalence = {
    "davidson-TO-founta_hateful_57k-TO-ibereval-TO-hateval-immigrant-TO-hateval-women-TO-waseem-racism":"Gen. -> Alt. Mis. & Rac.",
    "evalita-TO-waseem-racism-TO-ibereval-TO-hateval-immigrant":"Alt. Mis. & Rac.",
    "hateval-immigrant-TO-waseem-racism-TO-ibereval":"Rac. -> Mis."
}

datasets_equivalence= {"davidson":"Davidson", 
                        "founta_hateful_57k":"Founta", 
                        "ibereval":"IberEval", 
                        "hateval-immigrant":"HatEval_immigrant",
                        "hateval-women":"HatEval_women",
                        "waseem-racism":"Waseem_racism",
                        "evalita":"Evalita",
                        "waseem-sexism":"Waseem_sexism"}

hyper_param_equivalence = {'mem_size_proportion=0.025':"Proportion of train data kept in Memory: 2.5%", 
                            'ewc_lambda=1500':"EWC Lambda: 1500",
                            'lwf_lambda=1_temperature=2': "LwF Lambda: 1\nTemperature: 2", 
                            'mas_lambda=1000_mas_variation=global': "MAS Lambda: 1000",
                            "mem_size_proportion=0.01": "Proportion of train data kept in Memory: 1%",
                            "mas_lambda=1500_mas_variation=global": "MAS Lambda: 1500",
                            "ewc_lambda=2000": "EWC Lambda: 2000",
                            "lwf_lambda=1.5_temperature=2": "LwF Lambda: 1.5\nTemperature: 2",
                            "mem_size_proportion=0.05": "Proportion of train data kept in Memory: 5%",
                            "mas_lambda=2000_mas_variation=global": "MAS Lambda: 2000",
                            "ewc_lambda=2500": "EWC Lambda: 2500",
                            "ewc_lambda=1000": "EWC Lambda: 1000",
                            "ewc_lambda=2000": "EWC Lambda: 2000",
                            "mem_size_proportion=0.05": "Proportion of train data kept in Memory: 5%",
                            "lwf_lambda=0.5_temperature=2": "LwF Lambda: 0.5\nTemperature: 2",
                            "mas_lambda=500_mas_variation=global": "MAS Lambda: 500",}

columns_equivalence = {
                "model": "Model",
                "type_experiment": "Type of Experiment",
                "n_trainable_params": "Number of Trainable Parameters",
                "cl_technique": "Continual Learning Technique",
                "hyperparams": "Hyperparameters",
                "experiment_name": "Experiment Name",
                "time": "Time",
                "dataset_currently_testing": "Testing Dataset",
                "dataset_currently_training": "Training Dataset",
                "dataset_wrt_training_datasets": "Position of the Dataset wrt the Datastream",
                "target_epochs": "Epochs",
                "best_epochs": "Best Epoch",
                "learning_rate": "Learning Rate",
                "batch_size": "Batch Size",
                "current_num_samples_training": "Current Number of Samples Training",
                "cumulative_samples_trained": "Cumulative Samples Trained (including current Dataset)",
                "f1_score": "F1 Score",
                "precision_score": "Precision",
                "recall_score": "Recall",
                "accuracy": "Accuracy",
                "HATE_f1_score": "Hate Class F1 Score",
                "HATE_precision_score": "Hate Class Precision",
                "HATE_recall_score": "Hate Class Recall",
                "NoHATE_f1_score": "Non-Hate Class F1 Score",
                "NoHATE_precision_score": "Non-Hate Class Precision",
                "NoHATE_recall_score": "Non-Hate Class Recall",
                "last": "LAST",
                "avg_incremental_f1": "Average Incremental F1",
                "transfer": "Transfer",
                "bwt": "BWT",
                "fw_transfer": "FWT",
                "metric": "Metric"
}

position_equivalence = {'IN_TRAINING': "Current Dataset Training", 
                        'ZERO_SHOT':"Dataset not trained yet", 
                        'ZERO_SHOT_not_in_training_stream': "Dataset not in the training stream",
                        'TRAINING_PASSED': "Already trained"}

clean_up_mappings = [cl_techniques_equivalence, models_string_equivalences, position_equivalence, columns_equivalence, hyper_param_equivalence, datasets_equivalence, experiments_equivalence]

base_prompt_few_shot = """You are a social media content moderator.
INSTRUCTION: The following is a set of social media messages examples already classified with the label HATEFUL or NOT HATEFUL followed by the social media message you have to classify with the label HATEFUL or NOT HATEFUL.

{}
YOUR MESSAGE: {}
OUTPUT AND FORMAT: your output should be just the HATEFUL or NOT HATEFUL label classification for YOUR MESSAGE."""
