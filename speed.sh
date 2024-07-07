#!/bin/bash

# Define the array of model names
MODELS=( "squeezebert/squeezebert-uncased" "microsoft/deberta-v3-xsmall" ) #("albert-base-v2" "t5-small" "bert-base-uncased" "huawei-noah/TinyBERT_General_6L_768D" "google/mobilebert-uncased" "distilbert-base-uncased")

# Define the array of TASK_NAMEs
TASK_NAMES=("cb")  # Modify this array with your actual task names

# Loop over each model in the MODELS array
for MODEL in "${MODELS[@]}"; do
    echo "Running tasks for model: $MODEL"

    # Export MODEL to make it available to parallel sub-shells
    export MODEL

    # Use GNU Parallel to run tasks directly
    parallel -j 1 -u 'python3 run_speed.py \
      --model_name_or_path $MODEL \
      --task_name {} \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 18 \
      --output_dir ./tmp/$MODEL/{} \
      --moddistilbert n \
      --seed 42' ::: "${TASK_NAMES[@]}"
done
