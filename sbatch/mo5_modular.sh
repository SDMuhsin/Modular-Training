#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=2           # 
#SBATCH --mem=16000M
#SBATCH --time=7-00:00
#SBATCH --chdir=/scratch/sdmuhsin/Modular-Training
#SBATCH --output=mo5_modular-%N-%j.out

module load python/3.10
module load arrow/16.1.0
source ./env/bin/activate

echo "Environment loaded"
which python3

export PYTHONPATH="$PYTHONPATH:$(pwd)"

./scripts/mo5_modular.sh

echo "Run complete"
