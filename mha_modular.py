#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
from math import sqrt
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import BertForSequenceClassification, AutoConfig
from low_rank_modules.distilbert import MultiHeadSelfAttentionLowRank 
import psutil

check_min_version("4.41.0.dev0")

#python3 isolated_sa.py --encoder_idx=0 && python3 isolated_sa.py --encoder_idx=1 && python3 isolated_sa.py --encoder_idx=2 && python3 isolated_sa.py --encoder_idx=3 && python3 isolated_sa.py --encoder_idx=4 && python3 isolated_sa.py --encoder_idx=5 && python3 isolated_sa.py --encoder_idx=6 && python3 isolated_sa.py --encoder_idx=7 && python3 isolated_sa.py --encoder_idx=8 && python3 isolated_sa.py --encoder_idx=9 && python3 isolated_sa.py --encoder_idx=10 && python3 isolated_sa.py --encoder_idx=11
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

'''
    Stuff I am doing to make it pull BERT from a local copy:
    export PYTHONPATH="./transformers:$PYTHONPATH"

'''


from transformers import BertModel
import transformers
import torch
import time

import torch.optim as optim
import torch.nn as nn
# Print the location of the BertModel class definition

'''

linear_projection: 1.06% of total time
prepare_query_key_value: 0.10% of total time
handle_cross_attention_and_past_key: 0.00% of total time
attention_score_calculation: 33.12% of total time
normalize_and_mask_attention_scores: 19.71% of total time
finalize_attention_output: 46.01% of total time
Average runtime for original module: 0.048446 seconds
Average runtime for shared module: 0.033481 seconds

'''

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024)} MB")  # RSS memory in MB
#python3 isolated_sa.py --encoder_idx=0 --model_name=google-bert/bert-base-uncased --name=test --num_labels=2 --epochs=50

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--encoder_idx")
parser.add_argument("--job_name")
parser.add_argument("--epochs")
parser.add_argument("--model_name")
parser.add_argument("--num_labels",type=int)
parser.add_argument("--task")
parser.add_argument("--threshold_scale",type=float)
parser.add_argument("--compression",type=int)
parser.add_argument("--random_seed",type=int)
args = parser.parse_args()

def main():
    set_seed(42)
    save_dir = "./downloads"

    config_path = os.path.join(save_dir, f"{args.task}_config")
    tokenizer_path = os.path.join(save_dir, f"{args.task}_tokenizer")
    model_path = os.path.join(save_dir, f"{args.task}_model")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    if not os.path.exists(config_path):
        config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=args.num_labels,
            finetuning_task=args.task,
            cache_dir=None,
            revision='main',
            token=None,
            trust_remote_code=False,
        )
        config.save_pretrained(config_path)
    else:
        print("LOAD FROM SAVE")
        config = AutoConfig.from_pretrained(config_path)

    # Load or save tokenizer
    if not os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=None,
            use_fast=True,
            revision='main',
            token=None,
            trust_remote_code=False
        )
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load or save model
    if not os.path.exists(model_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            from_tf=bool(".ckpt" in args.model_name),
            config=config,
            cache_dir=None,
            revision='main',
            token=None,
            trust_remote_code=False,
            ignore_mismatched_sizes=False,
        )
        model.save_pretrained(model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Configuration and model setup code remains unchanged


    encoder_idx = int(args.encoder_idx)
    print(f"Encoder idx = {encoder_idx}")

    # attention_layer = model.bert.encoder.layer[0].attention.self
    attention_layer =  MultiHeadSelfAttentionLowRank(config,compression=args.compression)
    # attention_layer = BertSelfAttention   (config)
	
    original_sa = model.distilbert.transformer.layer[encoder_idx].attention
    #Load saved inputs
    
    input_save_folder = f"./saves/{args.model_name}/{args.task}/mha/inputs/encoder_{encoder_idx}/"
    output_save_folder = f"./saves/{args.model_name}/{args.task}/mha/outputs/encoder_{encoder_idx}/"


    #print(f"Dimensions of generated input {inputs.shape}")
    #print(f"Dimensions of generated output {outputs.shape}")

    # Define an optimizer
    optimizer = optim.Adam(attention_layer.parameters(), lr=1e-4)
    # Number of epochs
    num_epochs = int(args.epochs)

    # Training mode
    device = torch.device("cuda:0")
    attention_layer = attention_layer.to(device)
    original_sa = original_sa.to(device)
    original_sa.eval()
    attention_layer.train()

    def add_noise_and_check(input_tensor, threshold=5.0):
        noisy_tensor = input_tensor.clone()
        distance = 0.0
    
        while distance < threshold:
            noise = torch.randn_like(input_tensor)  # Adjust the noise level if necessary
            noisy_tensor = input_tensor + 0.1 * noise
            distance = torch.norm(input_tensor - noisy_tensor).item()
    
        return noisy_tensor
    
    def create_directory_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")

    def dropout_tensor(input_tensor, percentage):
        """
        Applies dropout to a batched input tensor by setting a specified percentage of elements to zero.
        
        Parameters:
            input_tensor (torch.Tensor): The input tensor with shape (batch_size, *dimensions).
            percentage (float): The percentage of elements to set to zero, between 0 and 1.
        
        Returns:
            torch.Tensor: A tensor of the same shape as input_tensor with elements randomly set to zero.
        """
        # Ensure percentage is in the valid range [0, 1]
        if not (0 <= percentage <= 1):
            raise ValueError(f"Percentage must be between 0 and 1 but was {p}")
        
        # Create a mask of the same shape as the input_tensor
        # Each element of the mask is 0 with the probability of 'percentage', otherwise 1
        mask = torch.bernoulli((1 - percentage) * torch.ones_like(input_tensor)).to(input_tensor.device)
        
        # Apply the mask to the input_tensor
        return input_tensor * mask
    def augment_tensor(input_tensor, multiplier=1):
        # Check that input_tensor has at least one dimension (batch size)
        if input_tensor.dim() < 1:
            raise ValueError("Input tensor must have at least one dimension for the batch size.")
        
        # Initialize a list to store the mixed tensors
        mixed_tensors = []
        
        # Loop through the multiplier to create multiple shifted and mixed tensors
        for i in range(1, multiplier + 1):
            # Shift the tensor down along the batch dimension by i
            shifted_tensor = torch.roll(input_tensor, shifts=i, dims=0)
            
            # Calculate the average of the input tensor and the shifted tensor
            mixed_tensor = (input_tensor + shifted_tensor) / 2
            
            # Append the mixed tensor to the list
            mixed_tensors.append(mixed_tensor)
        
        # Concatenate all mixed tensors along the batch dimension
        output_tensor = torch.cat(mixed_tensors, dim=0)
        
        return output_tensor
    loss_fn = nn.MSELoss()
    #loss_fn_cosine = nn.CosineSimilarity(dim=1)
     
    augment = False
    pMin = 0
    pMax = 0
    pStep = (pMax-pMin)/(num_epochs)
    p = pMin
    ec = 0
    def count_files(directory):
        return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

    mha_output_save_folder = f"./saves/{args.model_name}/{args.task}/mha/outputs/encoder_{args.encoder_idx}"
    batch_count = count_files(mha_output_save_folder)
    
    print("Batch count : ",batch_count)
    for epoch in range(0,num_epochs):

        print_memory_usage()        
        total_loss = 0
        noisy_loss = 0

        normal_loss = 0
        augment_loss = 0
        for bIdx in range(batch_count):
            #Load input and output
            try:
                x_inputs = dropout_tensor( torch.load(f"{input_save_folder}/a_batch_{bIdx}.pt"), p ).to(device)
                a_inputs = torch.load(f"{input_save_folder}/b_batch_{bIdx}.pt").to(device)
                
            except Exception as e:
                print(f"[e{epoch}b{bIdx}]Unable to dropout inputs?",e)
                ec +=1
                continue
            try:
                normal_outputs = torch.load(f"{output_save_folder}/o_batch_{bIdx}.pt").to(device)

            except Exception as e:
                ec += 1
                continue
            aug_outputs = None
            aug_x_inputs = None
            aug_a_inputs = None
            if (augment):
                aug_x_inputs = augment_tensor(x_inputs,multiplier=1).to(device)
                aug_a_inputs = augment_tensor(a_inputs,multiplier=1).to(device)
                aug_outputs = original_sa(aug_x_inputs,aug_x_inputs,aug_x_inputs,mask=aug_a_inputs)[0]
           
            predicted_normal_outputs = attention_layer(x_inputs,x_inputs,x_inputs,mask=a_inputs)[0]
            predicted_aug_outputs = attention_layer(aug_x_inputs,aug_x_inputs,aug_x_inputs,mask=aug_a_inputs)[0] if augment else None
            # Compute loss
            loss_normal = loss_fn(predicted_normal_outputs, normal_outputs)
            loss_aug = loss_fn(predicted_aug_outputs,aug_outputs) if augment else 0
            loss = loss_normal if (not augment) else loss_normal + loss_aug
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            total_loss += loss.item() # target ~0.001
            normal_loss += loss_normal.item()
            augment_loss += loss_aug.item() if augment else 0

        print(f"[MHA]Epoch {epoch+1}, Loss: {total_loss} = {normal_loss} (normal) + {augment_loss} (augmented)  \r")
        p+=pStep
    print("EC ",ec)
    save_dir = f"./saves/{args.model_name}/{args.job_name}/model"
    create_directory_if_not_exists(save_dir) 
    torch.save(attention_layer.state_dict(),f"{save_dir}/mha_enc{encoder_idx}_epoch{num_epochs}.pth")

    print("Training complete.")    
    
if __name__ == "__main__":
    main()
