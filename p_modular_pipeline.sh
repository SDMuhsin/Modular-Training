#!/bin/sh

job_name=$1
epochs=$2
do_capture=$3
do_modular_train_sa=$4
do_modular_train_bl=$5
do_run_glue=$6



encoder_compression=$7
do_plug_mods=$8
aug_n=$9
task=${10}

model_name="distilbert/distilbert-base-uncased" 
#./dp_sa_pipeline.sh test 1 n y y y 2 y

random_seed=42
encoder_idx=0

rm ./saves/"${task}_augmented_dataset.pkl"

export model_name task epochs job_name aug_n do_capture random_seed
export CUDA_VISIBLE_DEVICES=1  # Set this once if it's constant, or handle dynamically within the parallel command if it varies.

# Check if data capture should be performed
unset CUDA_VISIBLE_DEVICES
if [ "$do_capture" = "y" ]; then
    seq 0 5 | parallel -j 6 --env model_name,task,epochs,job_name,aug_n,CUDA_VISIBLE_DEVICES 'echo "Generating data for encoder {}" && \
    python3 capture_data.py \
      --model_name_or_path $model_name \
      --task_name $task \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs $epochs \
      --output_dir /tmp/$task/ \
      --overwrite_output_dir \
      --job_name $job_name \
      --encoder_idx {} \
      --aug_n $aug_n \
      --random_seed $random_seed \
      && echo "Data generated successfully for encoder {}"'
fi

export job_name epochs task encoder_compression random_seed

if [ "$do_modular_train_sa" = "y" ] && [ "$do_modular_train_bl" = "y" ]; then
    echo "\n \n \n Beginning modular training of SA and BL modules"
    echo "Encoder compression: $encoder_compression"

    # Use seq to generate encoder indices and pass each index to parallel
    seq 0 5 | parallel -j 5 'echo "Processing encoder index {}"; \
        unset CUDA_VISIBLE_DEVICES; \
        export encoder_idx={}; \
        parallel -j 2 -u ::: \
            "echo \"Modular training for SA module, encoder $encoder_idx\"; python3 mha_modular.py --encoder_idx=$encoder_idx --model_name=distilbert/distilbert-base-uncased --job_name=$job_name --num_labels=2 --epochs=$epochs --task=$task --threshold_scale=0 --compression=$encoder_compression --random_seed=$random_seed" \
            "echo \"Modular training for BL module, encoder $encoder_idx\"; python3 ffn_modular.py --encoder_idx=$encoder_idx --model_name=distilbert/distilbert-base-uncased --job_name=$job_name --num_labels=2 --epochs=$epochs --task=$task --threshold_scale=0 --compression=$encoder_compression --random_seed=$random_seed"'
fi

if [ "$do_run_glue" = "y" ];
then

	modularity="N"
	if [ "$do_plug_mods" = "y" ];
	then
		modularity="MF"
	fi

	echo "\n \n \n Run glue for job $job_name"
	echo "Modularity : $modularity"

	python3 run_superglue.py \
	  --model_name_or_path $model_name \
	  --task_name $task \
	  --per_device_train_batch_size 32 \
	  --learning_rate 2e-5 \
	  --num_train_epochs 18 \
	  --output_dir /tmp/$task/ \
	  --job_name $job_name \
	  --last_mod_trained_for $epochs \
	  --encoder_modularity $modularity\
	  --encoder_compression $encoder_compression \
	  --random_seed $random_seed
fi


