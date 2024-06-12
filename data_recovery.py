import os
import torch
model_name_or_path = "distilbert/distilbert-base-uncased"
task_name = "cb"
encoder_idx = 0

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

mha_output_save_folder = f"./saves/{model_name_or_path}/{task_name}/mha/outputs/encoder_{encoder_idx}"
mha_input_save_folder = f"./saves/{model_name_or_path}/{task_name}/mha/inputs/encoder_{encoder_idx}"


ffn_input_save_folder = f"./saves/{model_name_or_path}/{task_name}/ffn/inputs/encoder_{encoder_idx}"
ffn_output_save_folder = f"./saves/{model_name_or_path}/{task_name}/ffn/outputs/encoder_{encoder_idx}"

# Function calls extracted as variables
mha_output_file_count = count_files(mha_output_save_folder)
mha_input_file_count = count_files(mha_input_save_folder) // 2
ffn_input_file_count = count_files(ffn_input_save_folder)
ffn_output_file_count = count_files(ffn_output_save_folder)

# Using the variables in the print statements
print(f"Number of files in {mha_output_save_folder}: {mha_output_file_count}")
print(f"Number of files in {mha_input_save_folder}: {mha_input_file_count}")
print(f"Number of files in {ffn_input_save_folder}: {ffn_input_file_count}")
print(f"Number of files in {ffn_output_save_folder}: {ffn_output_file_count}")

for encoder_idx in range(6):
    mha_output_save_folder = f"./saves/{model_name_or_path}/{task_name}/mha/outputs/encoder_{encoder_idx}"
    mha_input_save_folder = f"./saves/{model_name_or_path}/{task_name}/mha/inputs/encoder_{encoder_idx}"
    for bIdx in range(mha_input_file_count):
        print(f"Encoder {encoder_idx}, batchIdx = {bIdx}",end="\r")
        # Load all inputs a and b
        a_inputs = torch.load(f"{mha_input_save_folder}/b_batch_{bIdx}.pt")
        x_inputs = torch.load(f"{mha_input_save_folder}/a_batch_{bIdx}.pt")
        normal_outputs = torch.load(f"{mha_output_save_folder}/o_batch_{bIdx}.pt")
        if(x_inputs == None or a_inputs == None or normal_outputs == None):
            print("None Problem with input i = ",bIdx,encoder_idx)
