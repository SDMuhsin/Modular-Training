import os
import torch
model_name_or_path = "distilbert/distilbert-base-uncased"
task_name = "cb"
encoder_idx = 5
encoder_idx_end = encoder_idx+1

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

a = set()
x = set()
o = set()
for encoder_idx in range(encoder_idx , encoder_idx_end):
    mha_output_save_folder = f"./saves/{model_name_or_path}/{task_name}/mha/outputs/encoder_{encoder_idx}"
    mha_input_save_folder = f"./saves/{model_name_or_path}/{task_name}/mha/inputs/encoder_{encoder_idx}"
    for bIdx in range(mha_input_file_count):
        print(f"Encoder {encoder_idx}, batchIdx = {bIdx}",end="\r")
        # Load all inputs a and b
        a_inputs = torch.load(f"{mha_input_save_folder}/b_batch_{bIdx}.pt")
        a.add(a_inputs.shape[0])
    
        x_inputs = torch.load(f"{mha_input_save_folder}/a_batch_{bIdx}.pt")
        x.add(x_inputs.shape[0])

        normal_outputs = torch.load(f"{mha_output_save_folder}/o_batch_{bIdx}.pt")
        if(x_inputs == None or a_inputs == None or normal_outputs == None):
            print("None Problem with input i = ",bIdx,encoder_idx)

print()
print(a)
print(x)
print(o)


for encoder_idx in range(encoder_idx , encoder_idx_end):

    ffn_input_save_folder = f"./saves/{model_name_or_path}/{task_name}/ffn/inputs/encoder_{encoder_idx}"
    ffn_output_save_folder = f"./saves/{model_name_or_path}/{task_name}/ffn/outputs/encoder_{encoder_idx}"
    
    for bIdx in range(mha_input_file_count):
        print(f"Encoder {encoder_idx}, batchIdx = {bIdx}",end="\r")
        # Load all inputs a and b
        h_inputs = torch.load(f"{ffn_input_save_folder}/h_batch_{bIdx}.pt")
        normal_outputs = torch.load(f"{ffn_output_save_folder}/o_batch_{bIdx}.pt")

        if(h_inputs == None or normal_outputs == None):
            print("None Problem with input i = ",bIdx,encoder_idx)


