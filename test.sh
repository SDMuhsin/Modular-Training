#!/bin/bash

export TASK_NAME=stsb
which python3
python3 test.py \
  --model_name_or_path google/mobilebert-uncased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
