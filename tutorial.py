from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader

EXP_DIR = "./exp"

# Download the Data
downloader.download_data(["mrpc"], f"{EXP_DIR}/tasks")

# Set up the arguments for the Simple API
args = run.RunConfiguration(
   run_name="simple",
   exp_dir=EXP_DIR,
   data_dir=f"{EXP_DIR}/tasks",
   hf_pretrained_model_name_or_path="google-bert/bert-base-uncased",
   tasks="mrpc",
   train_batch_size=32,
   num_train_epochs=3
)

# Run!
run.run_simple(args)
