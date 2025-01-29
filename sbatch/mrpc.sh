#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=2           # 
#SBATCH --mem=8000M
#SBATCH --time=0-01:00
#SBATCH --chdir=/scratch/sdmuhsin/Modular-Training
#SBATCH --output=mrpc-%N-%j.out

module load python/3.10
module load arrow/16.1.0
source ./env/bin/activate

echo "Environment loaded"
which python3

export PYTHONPATH="$PYTHONPATH:$(pwd)"

./scripts/modular_pipeline.sh modroberta-mprc-m200Aug3xMult1x 200 n n n y 2 y 3 mrpc 1

echo "Run complete"
