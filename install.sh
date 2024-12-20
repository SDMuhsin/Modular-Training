#!/bin/bash


#python3 -m venv torch_sayed
# Activate venv



module load python/3.11.5
module load arrow/16.1.0
source ../torch_sayed/bin/activate

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install nltk nlpaug numpy pandas

#scp -r ./downloads sdmuhsin@beluga.alliancecan.ca://home/sdmuhsin/scratch/sdmuhsin/transformers/modular-training-super-glue
#scp -r ../transformers sdmuhsin@beluga.alliancecan.ca://home/sdmuhsin/scratch/sdmuhsin/transformers/
#scp -r ./glove-embeddings sdmuhsin@beluga.alliancecan.ca://home/sdmuhsin/scratch/sdmuhsin/transformers/modular-training-super-glue
#scp -r ./saves sdmuhsin@beluga.alliancecan.ca://home/sdmuhsin/scratch/sdmuhsin/transformers/modular-training-super-glue
#scp -r ~/nltk_data sdmuhsin@beluga.alliancecan.ca:/home/sdmuhsin/scratch/sdmuhsin/transformers/modular-training-super-glue/nltk_data
python3 -m pip install urllib3
python3 -m pip install -e ../transformers/
python3 -m pip install scipy==1.10.1 dill
python3 -m pip install accelerate -U
python3 -m pip install scikit-learn
python3 -m pip install gensim
python3 -m pip install datasets evaluate
python3 -m pip install urllib3
#in run_sbatch.sh add export NLTK_DATA=..../nltk_data
