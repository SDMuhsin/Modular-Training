# Modular Training for Transformer Compression

This repository houses the source code for the paper "Modular Training for Transformer Compression". Modular training is an approach to transformer compression that trains low rank submodules in isolation before integrating them into a full model. This model achieves 31\% compression and 2.5x inference speedup (on CPU) over its baseline DistilBERT while retaining 99% of it's task performance.


# Setup

All libraries required for running this project are specified in the file install.sh. The script can be run directly if a virtual environment is presented in the parent directory and the code is being run in a slurm cluster. Note that the transformers library is installed from source (version 4.42.0.dev0) but can be installed directly as well.


# Reproducing results.

The project pipeline consists of several stages.

1. `capture_data.py` passes a dataset through the baseline model and captures intermediate activations, storing them on disk.
2. `mha_modular.py` and `ffn_modular.py` uses the generated data to train low rank versions of MHA and FFN blocks
3. `run_superglue.py` integrates the trained submodules, fine tunes, and evaluates them on the specific dataset

A full end-to-end run can be triggered by using the script `./modular_pipeline.sh`. This only evaluates the model on the dataset for one seed. The script `./rs\_moded.sh` can be used to evaluate the model on 5 seeds and the python script `consolidate.py`can be used to tabulate the median of 5 results.

# Results

```
+---------------------------------+---------+-------------+--------+-------+-------+-------+-------------+-------------+-----------------+
| Model                           |   boolq | cb          |   copa |   rte |   wic |   wsc | stsb        | mrpc        |   Average Score |
+=================================+=========+=============+========+=======+=======+=======+=============+=============+=================+
| bert-base-uncased               |   74.46 | 73.21/51.09 |     63 | 70.04 | 66.61 | 63.46 | 89.42/88.91 | 85.29/89.76 |           73.19 |
+---------------------------------+---------+-------------+--------+-------+-------+-------+-------------+-------------+-----------------+
| distilbert-base-uncased         |   73.18 | 80.36/66.39 |     57 | 63.54 | 65.2  | 64.42 | 86.81/86.61 | 85.78/89.97 |           72.04 |
+---------------------------------+---------+-------------+--------+-------+-------+-------+-------------+-------------+-----------------+
| moddistilbert-base-uncased      |   72.75 | 80.36/66.31 |     58 | 63.9  | 63.48 | 63.46 | 86.05/85.73 | 86.52/90.79 |           71.81 |
+---------------------------------+---------+-------------+--------+-------+-------+-------+-------------+-------------+-----------------+
| t5-small                        |   67.25 | 64.29/44.82 |     49 | 58.84 | 63.79 | 64.42 | 84.49/84.56 | 83.82/88.74 |           66.99 |
+---------------------------------+---------+-------------+--------+-------+-------+-------+-------------+-------------+-----------------+
| squeezebert/squeezebert-uncased |   75.35 | 67.86/47.23 |     57 | 70.04 | 65.2  | 64.42 | 88.73/88.34 | 86.27/90.14 |           71.86 |
+---------------------------------+---------+-------------+--------+-------+-------+-------+-------------+-------------+-----------------+


```

