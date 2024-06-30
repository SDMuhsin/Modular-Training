#!/bin/bash

# Define the array of TASK_NAMEs
TASK_NAMES=( "cb" )  # Modify this array with your actual task names

MODEL_TYPE="distilbert/distilbert-base-uncased"

# Export MODEL_TYPE to make it available to parallel sub-shells
export MODEL_TYPE

# Use GNU Parallel to run tasks directly
parallel -j 1 -u 'python3 loss_landscape_run.py \
  --model_name_or_path $MODEL_TYPE \
  --task_name {} \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 218 \
  --output_dir ./tmp/{} \
  --random_seed 42 \
  --seed 42' ::: "${TASK_NAMES[@]}" > nores.txt

