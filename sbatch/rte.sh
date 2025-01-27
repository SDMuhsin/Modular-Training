#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=2           # 
#SBATCH --mem=32000M
#SBATCH --time=4-00:00
#SBATCH --chdir=/scratch/sdmuhsin/Modular-Training
#SBATCH --output=rte-%N-%j.out

module load python/3.10
module load arrow/16.1.0
source ./env/bin/activate

echo "Environment loaded"
which python3

export PYTHONPATH="$PYTHONPATH:$(pwd)"

./scripts/modular_pipeline.sh modroberta-rte-m200Aug3xMult1x 200 y y y y 2 y 3 rte 1


echo "Run complete"
