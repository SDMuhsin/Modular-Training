#!/bin/bash

# Define the array of TASK_NAMES
TASK_NAMES=( "copa" "wsc" "wic" "cb" "boolq" "stsb" "rte" "mrpc" )

# Define the array of MODEL_TYPES
MODEL_TYPES=( "microsoft/deberta-v3-xsmall" )  #("bert-base-uncased" "huawei-noah/TinyBERT_General_6L_768D" "google/mobilebert-uncased" "distilbert-base-uncased")

# Define the array of SEEDS
SEEDS=( 44 45 )

# Loop over each model
for MODEL_TYPE in "${MODEL_TYPES[@]}"
do
  # Export MODEL_TYPE to make it available to parallel sub-shells
  export MODEL_TYPE

  # Loop over each seed
  for SEED in "${SEEDS[@]}"
  do
    # Use GNU Parallel to run tasks directly for each combination of model and seed
    parallel -j 5 "python3 run_superglue_baselines.py \
      --model_name_or_path ${MODEL_TYPE} \
      --task_name {} \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 18 \
      --output_dir ./tmp/{}_{}_{} \
      --random_seed ${SEED} \
      --seed ${SEED}" ::: "${TASK_NAMES[@]}"
  done
done
