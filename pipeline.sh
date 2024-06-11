#!/bin/sh

TASK="cb"
EXP_DIR="./exp"
MODEL_TYPE="distilbert/distilbert-base-uncased"


python jiant/jiant/proj/main/export_model.py \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_base_path ${EXP_DIR}/models/${MODEL_TYPE}

python jiant/jiant/scripts/download_data/runscript.py \
    download \
    --tasks ${TASK} \
    --output_path ${EXP_DIR}/tasks/

python jiant/jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${EXP_DIR}/tasks/configs/${TASK}_config.json \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_dir ${EXP_DIR}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val \
    --max_seq_length 256 \
    --smart_truncate


python3 run_glue_no_trainer.py \
  --model_name_or_path $MODEL_TYPE \
  --task_name $TASK_NAME \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 18 \
  --output_dir /tmp/$TASK \
  --seed 42

exit 1

python jiant/jiant/proj/main/scripts/configurator.py \
    SingleTaskConfigurator \
    ${EXP_DIR}/runconfigs/${MODEL_TYPE}/${TASK}.json \
    --task_name ${TASK} \
    --task_config_base_path ${EXP_DIR}/tasks/configs \
    --task_cache_base_path ${EXP_DIR}/cache/${MODEL_TYPE} \
    --epochs 3 \
    --train_batch_size 16 \
    --eval_batch_multiplier 2 \
    --do_train --do_val
python jiant/jiant/proj/main/runscript.py \
    run \
    --ZZsrc ${EXP_DIR}/models/${MODEL_TYPE}/config.json \
    --jiant_task_container_config_path ${EXP_DIR}/runconfigs/${MODEL_TYPE}/${TASK}.json \
    --model_load_mode from_transformers \
    --learning_rate 1e-5 \
    --do_train --do_val \
    --do_save --force_overwrite \
    --output_dir ${EXP_DIR}/runs/${MODEL_TYPE}/${TASK}
