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
from low_rank_modules.distilbert import FFNLowRank 
from low_rank_modules.modeling_roberta import RobertaForSequenceClassification, RobertaIntermediateLowRank, RobertaOutputLowRank, RobertaFFNLowRank
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
parser.add_argument("--multiplier",type=int)
args = parser.parse_args()


def truncated_svd_for_init(
    original_weight: torch.Tensor,
    reduce_weight: torch.Tensor,
    expand_weight: torch.Tensor,
):
    """
    Factor 'original_weight' using truncated SVD up to the maximum feasible rank,
    then copy as much as fits into 'reduce_weight' and 'expand_weight' without
    dimension mismatch.

    Args:
        original_weight: Tensor of shape (old_out_dim, old_in_dim)
        reduce_weight:   Tensor of shape (new_rank,    new_in_dim)
        expand_weight:   Tensor of shape (new_out_dim, new_rank)

    After SVD:
        W ~ U_r * sqrt(S_r) @ sqrt(S_r) * Vh_r
        => expand = U_r * sqrt(S_r),  reduce = sqrt(S_r) @ Vh_r
    """
    # Original shapes
    old_out_dim, old_in_dim = original_weight.shape

    # New shapes
    new_out_dim, new_rank_expand = expand_weight.shape
    new_rank_reduce, new_in_dim  = reduce_weight.shape

    # We want rank <= min(old_out_dim, old_in_dim, new_rank_expand, new_rank_reduce)
    # because expand must be (old_out_dim, rank), reduce must be (rank, old_in_dim).
    # We'll also be limited by the actual S size from SVD.
    rank_upper_bound = min(old_out_dim, old_in_dim, new_rank_expand, new_rank_reduce)

    # Perform SVD
    #   U:  (old_out_dim,  min(old_out_dim, old_in_dim))
    #   S:  (min(old_out_dim, old_in_dim),)
    #   Vh: (min(old_out_dim, old_in_dim), old_in_dim)
    U, S, Vh = torch.linalg.svd(original_weight, full_matrices=False)

    # Some old_out_dim × old_in_dim might be smaller or bigger than we think,
    # so clamp rank_upper_bound by S.size(0) as well.
    rank = min(rank_upper_bound, S.size(0))

    # Truncate to that rank
    U_r  = U[:, :rank]           # (old_out_dim, rank)
    S_r  = S[:rank]              # (rank,)
    Vh_r = Vh[:rank, :]          # (rank, old_in_dim)

    # sqrt(S) (rank, rank)
    sqrtS = torch.sqrt(torch.diag(S_r))  # => (rank, rank)

    # Now the factorization is:
    #   expand_matrix = U_r * sqrt(S_r)       => shape (old_out_dim, rank)
    #   reduce_matrix = sqrt(S_r) * Vh_r      => shape (rank, old_in_dim)
    expand_matrix = U_r @ sqrtS     # (old_out_dim, rank)
    reduce_matrix = sqrtS @ Vh_r    # (rank, old_in_dim)

    # We'll copy partial slices into expand_weight and reduce_weight, in case
    # new_out_dim < old_out_dim or new_in_dim < old_in_dim, etc.
    out_dim_to_copy   = min(new_out_dim, old_out_dim)
    rank_to_copy_exp  = min(new_rank_expand, rank)  # for expand
    rank_to_copy_red  = min(new_rank_reduce, rank)  # for reduce
    in_dim_to_copy    = min(new_in_dim, old_in_dim)

    # Fill expand_weight[:out_dim_to_copy, :rank_to_copy_exp]
    expand_weight[:out_dim_to_copy, :rank_to_copy_exp] = \
        expand_matrix[:out_dim_to_copy, :rank_to_copy_exp]

    # Fill reduce_weight[:rank_to_copy_red, :in_dim_to_copy]
    reduce_weight[:rank_to_copy_red, :in_dim_to_copy] = \
        reduce_matrix[:rank_to_copy_red, :in_dim_to_copy]
def copy_ffn_weights_svd(
    old_ffn: nn.Module,     # RobertaFFN with {intermediate, output}
    new_ffn: nn.Module      # RobertaFFNLowRank with {intermediate, output}
):
    """
    SVD-based initialization from a full-rank RobertaFFN ('old_ffn') to a
    low-rank RobertaFFNLowRank ('new_ffn'), robust to dimension mismatches.

    Steps:
      1) Factor old_ffn.intermediate.dense (shape = (intermediate_size, hidden_size))
         => new_ffn.intermediate.{dense_reduce, dense_expand}
      2) Factor old_ffn.output.dense       (shape = (hidden_size, intermediate_size))
         => new_ffn.output.{dense_reduce, dense_expand}
      3) Copy biases and LayerNorm if shapes allow partial copying.
    """
    with torch.no_grad():
        # -------------------------------------------
        # 1) Intermediate
        # -------------------------------------------
        # Original:  old_ffn.intermediate.dense.weight: (old_int_size, old_hidden_size)
        old_intermediate_w = old_ffn.intermediate.dense.weight
        old_intermediate_b = old_ffn.intermediate.dense.bias  # (old_int_size,)

        # New: new_ffn.intermediate.dense_reduce.weight: (new_int_size//comp, hidden_size(?))
        #      new_ffn.intermediate.dense_expand.weight: (new_int_size,       new_int_size//comp)
        new_reduce_w = new_ffn.intermediate.dense_reduce.weight
        new_expand_w = new_ffn.intermediate.dense_expand.weight
        # Factor + copy
        truncated_svd_for_init(old_intermediate_w, new_reduce_w, new_expand_w)

        # Copy partial bias
        old_b_size = old_intermediate_b.shape[0]
        new_b_size = new_ffn.intermediate.dense_expand.bias.shape[0]
        copy_b_size = min(old_b_size, new_b_size)
        new_ffn.intermediate.dense_expand.bias[:copy_b_size] = old_intermediate_b[:copy_b_size]

        # -------------------------------------------
        # 2) Output
        # -------------------------------------------
        # old_ffn.output.dense.weight: (old_hidden_size, old_int_size)
        # old_ffn.output.dense.bias:   (old_hidden_size,)
        old_output_w = old_ffn.output.dense.weight
        old_output_b = old_ffn.output.dense.bias

        # new_ffn.output.dense_reduce.weight: (new_hidden_size//comp, new_int_size)
        # new_ffn.output.dense_expand.weight: (new_hidden_size,       new_hidden_size//comp)
        new_reduce_w2 = new_ffn.output.dense_reduce.weight
        new_expand_w2 = new_ffn.output.dense_expand.weight
        # Factor + copy
        truncated_svd_for_init(old_output_w, new_reduce_w2, new_expand_w2)

        # Copy partial bias
        old_b2_size = old_output_b.shape[0]
        new_b2_size = new_ffn.output.dense_expand.bias.shape[0]
        copy_b2_size = min(old_b2_size, new_b2_size)
        new_ffn.output.dense_expand.bias[:copy_b2_size] = old_output_b[:copy_b2_size]

        # -------------------------------------------
        # 3) Copy LayerNorm if shapes allow
        # -------------------------------------------
        old_ln = old_ffn.output.LayerNorm
        new_ln = new_ffn.output.LayerNorm

        # old_ln.weight, old_ln.bias => shape (old_hidden_size,)
        # new_ln.weight, new_ln.bias => shape (new_hidden_size,)
        ln_w_size = min(old_ln.weight.shape[0], new_ln.weight.shape[0])
        ln_b_size = min(old_ln.bias.shape[0],   new_ln.bias.shape[0])

        new_ln.weight[:ln_w_size] = old_ln.weight[:ln_w_size]
        new_ln.bias[:ln_b_size]   = old_ln.bias[:ln_b_size]
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
def main():
    
    set_seed(42)
    save_dir = "./downloads"

    # Check if data is saved for cluster\
    model_name_short = args.model_name.split("/")[-1]
    config_path = os.path.join(save_dir, f"{args.task}_{model_name_short}_config")
    tokenizer_path = os.path.join(save_dir, f"{args.task}_{model_name_short}_tokenizer")
    model_path = os.path.join(save_dir, f"{args.task}_{model_name_short}_model") 

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    # Skip

    num_epochs = int(args.epochs)
    encoder_idx = int(args.encoder_idx)

    op_save_dir = f"./saves/{args.model_name}/{args.job_name}/model"
    create_directory_if_not_exists(op_save_dir)
    
    if os.path.exists(f"{op_save_dir}/ffn_enc{encoder_idx}_epoch{num_epochs}.pth"):
        print(f"FFN exists, SKIP")
        exit()


    if not os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=False,
        )
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if not os.path.exists(config_path):
        config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=args.num_labels,
            finetuning_task=args.task,
            trust_remote_code=False,
        )
        
        config.save_pretrained(config_path)
    else:
        print("LOAD CONFIG FROM SAVE")
        config = AutoConfig.from_pretrained(config_path)
    
    # Load or save model
    if not os.path.exists(model_path):

        if ("roberta" in args.model_name.lower() ):

            model = RobertaForSequenceClassification.from_pretrained(
                args.model_name,
                config=config,
                trust_remote_code=False,
                ignore_mismatched_sizes=False,
            ) 

            model.config.use_cache = False
        else:

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
        model.save_pretrained(model_path,safe_serialization=False)

    else:
        if ("roberta" in args.model_name ):
            model = RobertaForSequenceClassification.from_pretrained(model_path)
    
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)



    print(f"Encoder idx = {encoder_idx}")
    compression = int(args.compression)
    print(f"Compression = {compression}")
    
    if ("roberta" in args.model_name.lower()):
        original_ffn_layer = model.roberta.encoder.layer[encoder_idx].ffn

        intermediate_low_rank = RobertaIntermediateLowRank(config,compression=compression)
        output_low_rank       = RobertaOutputLowRank(config,compression=compression)
        new_ffn_layer         = RobertaFFNLowRank( intermediate_low_rank, output_low_rank )
        new_ffn_layer.intermediate = intermediate_low_rank
        new_ffn_layer.output  = output_low_rank
        
         
        copy_ffn_weights_svd(
            old_ffn=original_ffn_layer,
            new_ffn=new_ffn_layer#,
         #   compression=args.compression
        )

    else:
        original_ffn_layer = model.distilbert.transformer.layer[encoder_idx].ffn
        new_ffn_layer = FFNLowRank(config,compression=compression) # New bert layer to be train
    
    #Load saved inputs
    input_save_folder = f"./saves/{args.model_name}/{args.task}/ffn/inputs/encoder_{encoder_idx}/"
    output_save_folder = f"./saves/{args.model_name}/{args.task}/ffn/outputs/encoder_{encoder_idx}/"

    optimizer = optim.Adam(new_ffn_layer.parameters(), lr=1e-4)

    device = torch.device("cuda:1")
    original_ffn_layer = original_ffn_layer.to(device)
    new_ffn_layer = new_ffn_layer.to(device)
    new_ffn_layer.train()
    original_ffn_layer.eval()

    def add_noise_and_check(input_tensor, threshold=5.0):
        noisy_tensor = input_tensor.clone()
        distance = 0.0
    
        while distance < threshold:
            noise = torch.randn_like(input_tensor)  # Adjust the noise level if necessary
            noisy_tensor = input_tensor + 0.1 * noise
            distance = torch.norm(input_tensor - noisy_tensor).item()
    
        return noisy_tensor
    

    
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
            raise ValueError("Percentage must be between 0 and 1.")
        
        # Create a mask of the same shape as the input_tensor
        # Each element of the mask is 0 with the probability of 'percentage', otherwise 1
        assert input_tensor != None
        mask = torch.bernoulli((1 - percentage) * torch.ones_like(input_tensor)).to(input_tensor.device)
        assert mask != None

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


    #loss_fn = nn.MSELoss() 
    loss_fn = nn.MSELoss()
    #loss_fn_cosine = nn.CosineSimilarity(dim=1)
    multiplier = args.multiplier
    
    augment = multiplier != 0
    pMin = 0
    pMax = 0 

    pStep = (pMax-pMin)/num_epochs
    p = pMin

    def count_files(directory):
        return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

    mha_output_save_folder = f"./saves/{args.model_name}/{args.task}/mha/outputs/encoder_{args.encoder_idx}"
    batch_count = count_files(mha_output_save_folder)
    for epoch in range(num_epochs):

        total_loss = 0
        noisy_loss = 0
        normal_loss = 0
        augment_loss = 0

        for bIdx in range(batch_count):
            
            #Load input and output
            try:
                h_inputs = dropout_tensor( torch.load(f"{input_save_folder}/h_batch_{bIdx}.pt") , p ).to(device)
                
                #dropped_h_inputs = dropout_tensor(h_inputs,0.05).to(device)
                normal_outputs = torch.load(f"{output_save_folder}/o_batch_{bIdx}.pt").to(device)
                
            except:
                print("Unable load/dropout input/output")
                continue
            
            assert h_inputs != None

            aug_outputs = None
            aug_h_inputs = None

            if (augment):
                

                aug_h_inputs = augment_tensor(h_inputs,multiplier).to(device)

                aug_outputs = original_ffn_layer(aug_h_inputs).to(device)

            predicted_normal_outputs = new_ffn_layer(h_inputs)
            predicted_aug_outputs = new_ffn_layer(aug_h_inputs) if augment else None 
            #predicted_dropped_outputs = new_ffn_layer(h_inputs)


            loss_normal = loss_fn(predicted_normal_outputs, normal_outputs)
            loss_aug = loss_fn(predicted_aug_outputs,aug_outputs) if augment else 0
            #loss_dropped = loss_fn(predicted_dropped_outputs,normal_outputs)

            loss = loss_normal if (not augment) else loss_normal + loss_aug
            
            optimizer.zero_grad()

            # Backward pass
            loss.backward()
    
            # Update parameters
            optimizer.step()

            total_loss += loss.item() # target ~0.001
            normal_loss += loss_normal.item()
            augment_loss += loss_aug.item() if augment else 0

        print(f"[FFN] Epoch {epoch+1}, Loss: {total_loss} = {normal_loss} (normal) + {augment_loss} (augmented)  \r")
        p += pStep

    torch.save(new_ffn_layer.state_dict(),f"{op_save_dir}/ffn_enc{encoder_idx}_epoch{num_epochs}.pth")

    print("Training complete.")    
    
if __name__ == "__main__":
    main()
