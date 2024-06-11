import os

model_name_or_path = "distilbert/distilbert-base-uncased"
task_name = "cb"
encoder_idx = 0

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

mha_output_save_folder = f"./saves/{model_name_or_path}/{task_name}/mha/outputs/encoder_{encoder_idx}"
mha_input_save_folder = f"./saves/{model_name_or_path}/{task_name}/mha/inputs/encoder_{encoder_idx}"


ffn_input_save_folder = f"./saves/{model_name_or_path}/{task_name}/ffn/inputs/encoder_{encoder_idx}"
ffn_output_save_folder = f"./saves/{model_name_or_path}/{task_name}/ffn/outputs/encoder_{encoder_idx}"

print(f"Number of files in {mha_output_save_folder}: {count_files(mha_output_save_folder)}")
print(f"Number of files in {mha_input_save_folder}: {count_files(mha_input_save_folder)//2}")

print(f"Number of files in {ffn_input_save_folder}: {count_files(ffn_input_save_folder)}")
print(f"Number of files in {ffn_output_save_folder}: {count_files(ffn_output_save_folder)}")

