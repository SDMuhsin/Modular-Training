#!/bin/bash

# Define the array of TASK_NAMES
TASK_NAMES=( "cb" ) #( "boolq" "cb" "wic" "wsc" ) #( "copa" "wsc" "wic" "cb" "boolq" "cola" "stsb" "rte" "mrpc" )

# Define the array of MODEL_TYPES
MODEL_TYPES=("unsloth/Llama-3.2-1B")  #("distilbert/distilbert-base-uncased") 

# Define the array of SEEDS
SEEDS=(41)

# Loop over each model
for MODEL_TYPE in "${MODEL_TYPES[@]}"
do
  # Export MODEL_TYPE to make it available to parallel sub-shells
  export MODEL_TYPE

  # Loop over each seed
  for SEED in "${SEEDS[@]}"
  do
    # Use GNU Parallel to run tasks directly for each combination of model and seed
    parallel -j 1 "python3 run_superglue_moded.py \
      --model_name_or_path ${MODEL_TYPE} \
      --task_name {} \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 18 \
      --output_dir ./tmp/{}_{}_{} \
      --random_seed ${SEED} \
      --job_name 'fn-{}m200aug3x'\
      --seed ${SEED}" ::: "${TASK_NAMES[@]}"
  done
done

