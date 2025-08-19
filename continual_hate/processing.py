from typing import Dict

from pprint import pprint as pp

import torch
from transformers import AutoTokenizer, DataCollatorWithPadding

from .constants import llama_guard_equivalence, translation_dict_from_label_to_verbalizer, translation_dict_from_label_to_int, tokenizer_args, data_collator_args, base_prompt
from .utils import set_seed

set_seed(42)

class ProcessingConfig():

    def __init__(self, 
                tokenizer_id:str, 
                training_obj:str, 
                task_column:str,
                text_column:str,
                label_column:str,
                split_column:str,
                base_prompt:str=base_prompt,
                label_equivalence_general:Dict[bool, str]=translation_dict_from_label_to_verbalizer, 
                label_equivalence_int:Dict[bool, int]=translation_dict_from_label_to_int,
                label_equivalence_llama_guard:Dict[str, str]=llama_guard_equivalence,
                tokenizer_args:Dict=tokenizer_args, 
                data_collator_args:Dict=data_collator_args,
                data_shuffle=False):

        self.tokenizer_id = tokenizer_id
        self.training_obj = training_obj

        self.data_shuffle = data_shuffle

        self.task_column = task_column
        self.text_column = text_column
        self.label_column = label_column
        self.split_column = split_column

        self.label_equivalence_general = label_equivalence_general
        self.label_equivalence_int = label_equivalence_int
        self.label_equivalence_llama_guard = label_equivalence_llama_guard

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, **data_collator_args)

        self.base_prompt = base_prompt
        self.tokenizer_args = tokenizer_args
        self.tokenizer.model_max_length = self.tokenizer_args["max_length"] # have to change it

        if not self.tokenizer.chat_template and self.training_obj == "CAUSAL_LM":
            self.tokenizer_id = "Models/" + self.tokenizer_id
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id) # reload the local one
            self.tokenizer.chat_template = open(self.tokenizer_id + "/chat_template.jinja").read() # loading it automatically was throwing an error with CALMIP

        self.data_collator_args = data_collator_args

        if self.training_obj == "SEQ_CLS":
            self.hf_dataset_processing_args = {
                                                "processing_function_args": self.tokenizer_args,
                                                "input_columns":["texts"], 
                                                "cols_to_keep":["labels"]
                                                }


        elif self.training_obj == "CAUSAL_LM":
            self.hf_dataset_processing_args = {
                                                "processing_function_args": self.tokenizer_args,
                                                "input_columns":["texts", "labels"], 
                                                "cols_to_keep":["labels"],
                                                }
    def set_processing_function(self):

        if self.training_obj == "SEQ_CLS":

            def tokenize_function(text:str,
                                split, # not used for leaving it here to not alter the processing pipeline
                                tokenizer=self.tokenizer, 
                                padding = self.tokenizer_args["padding"],
                                max_length=self.tokenizer_args["max_length"]):

                if tokenizer.pad_token is None and "llama" in str(self.tokenizer_id).lower(): tokenizer.pad_token = '<|finetune_right_pad_id|>'
                elif tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

                tokenized = tokenizer(text=text, padding=padding, max_length=max_length, truncation=True, return_tensors="pt")
                
                for key, tensor in tokenized.items():
                    if isinstance(tensor, list) and len(tensor) == 1: # there is an extra unneeded dimension
                        tokenized[key] = tensor[0]
                    elif isinstance(tensor, torch.Tensor) and tensor.dim() == 2 and tensor.size(0) == 1:
                        tokenized[key] = tensor.squeeze(0)

                return tokenized

            self.processing_function = tokenize_function
            
        elif self.training_obj == "CAUSAL_LM":

            def preprocess_chat_template_and_tokenize(formatted_prompt, 
                                                        label,
                                                        split,
                                                        tokenizer=self.tokenizer,
                                                        loss_function_over_prompt=self.tokenizer_args["loss_function_over_prompt"], 
                                                        max_length=self.tokenizer_args["max_length"], 
                                                        padding=self.tokenizer_args["padding"]
                                                        ):
    
                if tokenizer.pad_token is None and "llama" in str(self.tokenizer_id).lower(): tokenizer.pad_token = '<|finetune_right_pad_id|>'
                elif tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

                continue_final_message = False
                tokenizer.padding_side = "right"
                messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": formatted_prompt + "\nLABEL: "},
                    {"role": "assistant", "content": label.rstrip().strip("\n")} # stripping trailing newlines and spaces
                ]



                if split == "test":# when testing the tokenizer should pad to the left and not include the label obvs
                    continue_final_message = True
                    tokenizer.padding_side = "left"
                    messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": formatted_prompt + "\nLABEL: "}
                    ]

                if "guard" in str(self.tokenizer_id).lower():
                    # print("label before: ", label)
                    
                    dict_equivalence = {"NOT HATEFUL": "\n\nsafe", "HATEFUL": "\n\nunsafe\nS10"}
                    label = dict_equivalence[label.strip().rstrip()]
                    # print("label after: ", label)
                    messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text", 
                                            "text":  f"{formatted_prompt}" # in the caseof llama guard, we will pass just the message without the prompt to the model
                                        },
                                    ],
                                }
                            ]
                    # print(label)
                    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, max_length=512, padding="max_length", truncation=True)
                    label = tokenizer(label, return_tensors="pt", add_special_tokens=False, max_length=512, padding="max_length", truncation=True)["input_ids"]
                    # print(inputs)
                    # print(inputs["input_ids"].shape)
                    # print(inputs["attention_mask"].shape)
                    # print(inputs)
                    # print(label)
                    # print(label.shape)
                    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": label}
                    
                if "qwen" in str(self.tokenizer_id).lower():
                    chat_template = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=continue_final_message, add_special_tokens=False, enable_thinking=False).rstrip() # sometimes there is a \n character added
                else:
                    chat_template = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=continue_final_message, add_special_tokens=False).rstrip() # sometimes there is a \n character added
                    
                tokenized = tokenizer(chat_template, return_tensors="pt", add_special_tokens=False, padding=padding, max_length=max_length)
                input_ids_tokenized = tokenized["input_ids"]
                attention_mask = input_ids_tokenized != tokenizer.pad_token_id
                tokenized["attention_mask"] = attention_mask
                
                if loss_function_over_prompt == False: # if i want the loss to be calculated JUST from the label and not for how well the model predicts the prompt

                    # auxiliary messages to determine where the label starts/ends with respect to the normal messages
                    messages_no_assistant_response = [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": formatted_prompt + "\nLABEL: "},
                        {"role": "assistant", "content": ""} 
                                                            ]

                    # getting the prompt without the assistant's response to know how much we need to add to the left as -100 and right as pad token
                    chat_template_no_assistant_response = tokenizer.apply_chat_template(messages_no_assistant_response, tokenize=False, continue_final_message=False, add_special_tokens=False).rstrip() # sometimes there is a \n character added
                    # the padding has to be false in order to get the actual length
                    input_ids_shape = tokenizer(chat_template_no_assistant_response.rstrip(tokenizer.eos_token), return_tensors="pt", add_special_tokens=False, padding=False)["input_ids"]

                    # getting the label target to only predict the actual label and ignore the prompt
                    labels_tokenized = tokenizer(label + tokenizer.eos_token, add_special_tokens=True, return_tensors="pt")["input_ids"]

                    # MASKING THE LABELS PRIOR TO THE LABEL
                    shape = input_ids_shape.shape[1] # how many tokens are to the left of the actual label
                    # print(shape)
                    zeros = torch.zeros((1, shape), dtype=labels_tokenized.dtype, device=labels_tokenized.device) # fill the left with zeros
                    zeros.fill_(-100) # then fill the left with -100 for the cross entropy loss
                    labels_left_padded = torch.cat([zeros, labels_tokenized], dim=1) # concatenate the -100s with the label
                    # print(input_ids_tokenized.shape)
                    # print(labels_left_padded.shape)
                    # print(labels_left_padded)

                    eos_n = input_ids_tokenized.shape[1] - labels_left_padded.shape[1] # how many tokens are to the right of the actual label that need to be padded
                    eos_n_tensor = torch.zeros((1, eos_n), dtype=labels_tokenized.dtype, device=labels_tokenized.device)
                    eos_n_tensor.fill_(tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]) # fill the right with pad tokens
                    labels_padded = torch.cat([labels_left_padded, eos_n_tensor], dim=1) # concatenate the -100s + label with the right pad tokens

                    tokenized = {
                                    "input_ids": input_ids_tokenized,
                                    "labels": labels_padded,
                                    "attention_mask": input_ids_tokenized != tokenizer.pad_token_id
                                                                                                    }
                else:
                    # if split == "test":
                    #     tokenized["labels"] = label
                    #     print(tokenized["labels"])
                    # else:
                    #     tokenized["labels"] = input_ids_tokenized
                    if split != "test":
                        tokenized["labels"] = input_ids_tokenized

                    else:
                        # to truncate or not to truncate
                        label_tokenized = tokenizer(label, return_tensors="pt", add_special_tokens=False, padding=padding, max_length=max_length)["input_ids"]
                        tokenized["labels"] = label_tokenized


                    # # reconstruct the label and input ids to check that the masking and padding worked as expected
                    # labels_tokens = labels_padded[labels_padded != -100]
                    # decoded_labels_processed = tokenizer.decode(token_ids=labels_tokens)
                    # decoded_input_ids = tokenizer.decode(token_ids=input_ids_shape[0])
                    # recode_inputs = tokenizer.decode(token_ids=input_ids_tokenized[0], skip_special_tokens=False)
                
                # if split != "": # model.generate necesitates the input to be of shape (batch_size, seq_length)
                for key, tensor in tokenized.items():
                    if isinstance(tensor, list) and len(tensor) == 1: # there is an extra unneeded dimension
                        tokenized[key] = tensor[0]
                    elif isinstance(tensor, torch.Tensor) and tensor.dim() == 2 and tensor.size(0) == 1:
                        tokenized[key] = tensor.squeeze(0)

                # for key in ["input_ids", "labels", "attention_mask"]: assert key in list(tokenized.keys())
                # print(tokenized.keys())
                return tokenized

            self.processing_function = preprocess_chat_template_and_tokenize
        
        return self.processing_function
