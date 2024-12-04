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

model_name="distilbert/distilbert-base-uncased" #"unsloth/Llama-3.2-1B" 
#./dp_sa_pipeline.sh test 1 n y y y 2 y


encoder_idx=0
random_seed=42
rm ./saves/"${task}_augmented_dataset.pkl"
while [ $encoder_idx -le 1 ]
do
	rm  ./saves/$model_name/$task/* -r
	echo "Generating data for encoder $encoder_idx"
	if [ "$do_capture" = "y" ];
	then
		export CUDA_VISIBLE_DEVICES=1
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
		  --encoder_idx $encoder_idx \
		  --aug_n $aug_n \
		  --random_seed $random_seed 
	fi
	echo "Data generated succesfully"
	

	unset CUDA_VISIBLE_DEVICES


	if [ "$do_modular_train_sa" = "y" ] && [ "$do_modular_train_bl" = "y" ];
	then
	    echo "\n \n \n Beginning modular training of SA and BL modules"
	    echo "Encoder compression: $encoder_compression"

	    export job_name epochs task noise_threshold_scale encoder_compression model_name

	    # Execute one instance of dis_mha_modular.py and one instance of dis_ffn_modular.py in parallel for the same encoder index
	    parallel -j 1 -u ::: \
		"echo 'Modular training for SA module, encoder $encoder_idx'; python3 mha_modular.py --encoder_idx=$encoder_idx --model_name=$model_name --job_name=$job_name --num_labels=2 --epochs=$epochs --task=$task --threshold_scale=0 --compression=$encoder_compression --random_seed=$random_seed"\
		"echo 'Modular training for BL module, encoder $encoder_idx'; python3 ffn_modular.py --encoder_idx=$encoder_idx --model_name=$model_name --job_name=$job_name --num_labels=2 --epochs=$epochs --task=$task --threshold_scale=0 --compression=$encoder_compression --random_seed=$random_seed"
	fi	
	encoder_idx=$((encoder_idx + 1))
done


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


