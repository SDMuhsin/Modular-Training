#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import dill as pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import psutil

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import BertForSequenceClassification, AutoConfig

from low_rank_modules.modeling_llama import LlamaForSequenceClassification
import nlpaug.augmenter.sentence as nas
# Will error if the minimal version of Transformers is not installed. Remove at your own risks. [I am upgrading from 4.39.3.]
check_min_version("4.41.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
'''
    Stuff I am doing to make it pull BERT from a local copy:
    export PYTHONPATH="./transformers:$PYTHONPATH"

'''
task_to_keys = {
    "cola": ("sentence", None,None),
    "mnli": ("premise", "hypothesis",None),
    "mrpc": ("sentence1", "sentence2",None),
    "qnli": ("question", "sentence",None),
    "qqp": ("question1", "question2",None),
    "rte": ("sentence1", "sentence2",None),
    "sst2": ("sentence", None, None),
    "stsb": ("sentence1", "sentence2",None),
    "wnli": ("sentence1", "sentence2",None),
    "cb"  : ("premise","hypothesis",None),
    "boolq": ("question","passage",None),
    "copa" : ("premise","choice1","choice2"),
    "multirc" : ("paragraph","question","answer"),
    "wic" : ("sentence1","sentence2",None),
    "wsc" : ("span1_text","span2_text",None)
}



logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    
    encoder_idx:int = field(default =-1)
    #encoder_idx: int = field(default = 0, metadata={"help":"Which encoder index do you want to capture inputs/outputs from?"})
    
    job_name : str = field(default="FORGOTJOBNAME")
    random_seed : int = field(default=42)
    post_ft_capture : str = field(default="n")

    aug_n : int = field(default = 1)
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

from transformers import BertModel
import transformers
import torch
from torch.utils.data import DataLoader
import time
import unicodedata

# Print the location of the BertModel class definition
print("BertModel loaded from:", BertModel.__module__)
print("Transformers module loaded from:", transformers.__file__)

'''

linear_projection: 1.06% of total time
prepare_query_key_value: 0.10% of total time
handle_cross_attention_and_past_key: 0.00% of total time
attention_score_calculation: 33.12% of total time
normalize_and_mask_attention_scores: 19.71% of total time
finalize_attention_output: 46.01% of total time
Average runtime for original module: 0.048446 seconds
Average runtime for shared module: 0.033481 seconds

'''

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024)} MB")  # RSS memory in MB


def assert_model_weights_equal(model1, model2, exclude_class_name="BertSharedSelfAttention"):
    mismatched_layers = []
    for (name1, module1), (name2, module2) in zip(model1.named_modules(), model2.named_modules()):
        # Skip comparison for self-attention layers or any other specified layers
        if module1.__class__.__name__ == exclude_class_name or module2.__class__.__name__ == exclude_class_name:
            print(f"Skipping {name1} as it is an instance of {exclude_class_name}")
            continue
        
        # Check if modules are functionally equivalent
        if module1.__class__.__name__ != module2.__class__.__name__:
            print(f"Warning: Type mismatch but checking weights due to functional equivalency {module1.__class__.__name__} and {module2.__class__.__name__} at {name1}")
            continue  # Skip mismatched types that aren't explicitly excluded
        
        # Compare the weights
        if hasattr(module1, 'weight') and hasattr(module2, 'weight') and module1.weight is not None and module2.weight is not None:
            if module1.weight.shape != module2.weight.shape:
                print(f"Shape mismatch in {name1}, not comparing weights")
                continue
            if not torch.allclose(module1.weight, module2.weight, atol=1e-6):
                mismatched_layers.append(name1)
                print(f"Weights do not match in layer {name1}")
        
        # Compare the biases
        if hasattr(module1, 'bias') and hasattr(module2, 'bias') and module1.bias is not None and module2.bias is not None:
            if module1.bias.shape != module2.bias.shape:
                print(f"Shape mismatch in {name1}, not comparing biases")
                continue
            if not torch.allclose(module1.bias, module2.bias, atol=1e-6):
                mismatched_layers.append(name1)
                print(f"Biases do not match in layer {name1}")
    
    if mismatched_layers:
        raise AssertionError(f"Weight mismatches found in layers: {mismatched_layers}")
    else:
        print("All comparable weights and biases are equal.")

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def print_learnable_weights_info(model):
    # Calculate the total number of learnable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Iterate through all parameters that are learnable
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Calculate the percentage contribution of each parameter
            percentage = 100.0 * param.numel() / total_params
            print(f"{name}: {percentage:.2f}% of total learnable parameters")

def print_first_encoder_weights_info(model):
    # Access the first encoder layer
    first_encoder = model.distilbert.transformer.layer[0]

    # Calculate the total number of learnable parameters in the first encoder
    total_params = sum(p.numel() for p in first_encoder.parameters() if p.requires_grad)

    # Iterate through all parameters of the first encoder that are learnable
    for name, param in first_encoder.named_parameters():
        if param.requires_grad:
            # Calculate the percentage contribution of each parameter within the first encoder
            percentage = 100.0 * param.numel() / total_params
            print(f"{name}: {percentage:.2f}% of total learnable parameters in the first encoder")

import nlpaug.augmenter.word as naw
import nltk
from nltk.corpus import stopwords
import random

# Ensure you have the stopwords dataset downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def augment_sentence(sentence, glove_model_path='./glove-embeddings/glove.6B.100d.txt', aug_n=1, replace_percentage=0.2):
    """Applies augmentation to a single sentence using GloVe embeddings, replacing about 40% of non-stop words."""
    # Initialize the GloVe augmenter
    aug = naw.WordEmbsAug(
        model_type='glove',
        model_path=glove_model_path,
        action='substitute',
    )

    words = sentence.split()
    
    # Identify non-stop word indices
    non_stop_word_indices = [i for i, word in enumerate(words) if word.lower() not in stop_words]
    
    # Determine the number of words to replace
    num_to_replace = max(1, int(len(non_stop_word_indices) * replace_percentage))

    augmented_sentences = set()  # Use a set to avoid duplicate sentences
    max_attempts = 20
    while len(augmented_sentences) < aug_n and max_attempts > 0 and num_to_replace <= len(non_stop_word_indices):
        random_indices = random.sample(non_stop_word_indices, num_to_replace)
        new_words = words.copy()

        for idx in random_indices:
            new_words[idx] = aug.augment([words[idx]])[0]  # Augment the single word directly

        augmented_sentence = ' '.join(new_words)
        augmented_sentences.add(augmented_sentence)  # Add to set, automatically handling uniqueness
        max_attempts -= 1
    return list(augmented_sentences)

def augment_superglue_sentence(sentence, exclude_words, glove_model_path='./glove-embeddings/glove.6B.100d.txt', aug_n=1, replace_percentage=0.4):
    """Applies augmentation to a single sentence for SuperGLUE datasets using GloVe embeddings, excluding specified words."""
    aug = naw.WordEmbsAug(
        model_type='glove',
        model_path=glove_model_path,
        action='substitute',
    )

    words = sentence.split()

    # Identify non-stop word indices excluding the specified words
    non_stop_word_indices = [i for i, w in enumerate(words) if w.lower() not in stop_words and w not in exclude_words]

    num_to_replace = max(1, int(len(non_stop_word_indices) * replace_percentage))

    augmented_sentences = set()
    max_attempts = 20
    while len(augmented_sentences) < aug_n and max_attempts > 0 and num_to_replace <= len(non_stop_word_indices):
        random_indices = random.sample(non_stop_word_indices, num_to_replace)
        new_words = words.copy()

        for idx in random_indices:
            new_words[idx] = aug.augment([words[idx]])[0]

        augmented_sentence = ' '.join(new_words)
        augmented_sentences.add(augmented_sentence)
        max_attempts -= 1

    return list(augmented_sentences)


def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

save_dir = "./downloads"

def main():
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    training_args.dataloader_num_workers = 0
    random.seed(data_args.random_seed)
    np.random.seed(data_args.random_seed)
    torch.manual_seed(data_args.random_seed)
    torch.cuda.manual_seed_all(data_args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(data_args.random_seed)
    
    # Check if data is saved for cluster
    model_name_short = model_args.model_name_or_path.split("/")[-1]
    config_path = os.path.join(save_dir, f"{data_args.task_name}_{model_name_short}_config")
    tokenizer_path = os.path.join(save_dir, f"{data_args.task_name}_{model_name_short}_tokenizer")
    model_path = os.path.join(save_dir, f"{data_args.task_name}_{model_name_short}_model")
    metric_path = os.path.join(save_dir, f"{data_args.task_name}_{model_name_short}_metric.pkl")


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(data_args.random_seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)


    if data_args.task_name is not None:
        dataset_path = os.path.join(save_dir, f"{data_args.task_name}")

        if not os.path.exists(dataset_path):
            # Downloading and loading a dataset from the hub.
            if data_args.task_name in ["rte","stsb","mrpc","cola"]:
                raw_datasets = load_dataset("glue", data_args.task_name)
            else:
                raw_datasets = load_dataset("aps/super_glue", data_args.task_name)

            # Save the dataset to the specified directory
            raw_datasets.save_to_disk(dataset_path)
            print("Saved dataset to disk")
        else:
            # Load the dataset from the specified directory
            raw_datasets = datasets.load_from_disk(dataset_path)
            print("Loaded dataset from disk")

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    augment = True
    dataset_path = f"./saves/{data_args.task_name}_augmented_dataset.pkl"  # You can change the extension and serialization method

    # Check if the dataset has been augmented and saved already
    if os.path.exists(dataset_path):
        augment = False
        with open(dataset_path, 'rb') as f:
            augmented_data = pickle.load(f)
        print("Loaded augmented data from file.")
    
    #raw_datasets['train'] = raw_datasets['train'].select(range(0,2))

    '''
    print("------ BEFORE AUGMENT ---------")
    for data in raw_datasets['train']:

        for k in data.keys():
            if k !=None:
                print("____")
                print("\t", k)
                print("\t", data[k])
                print("_____")
    '''

    # Only augment if the dataset doesn't exist
    do_augment = int(data_args.aug_n) != 0
    aug_n = int(data_args.aug_n)
    if augment and do_augment:
        train_dataset = raw_datasets['train']
        keys = task_to_keys[data_args.task_name]
        
        # Initialize augmented data dictionary with dynamic key creation based on keys provided in task_to_keys
        augmented_data = {key: [] for key in train_dataset.column_names}
        
        other_keys = [k for k in train_dataset.column_names if k not in keys ]

        #augmented_data.update({'label': [], 'idx': []})
        
        for i in range(len(train_dataset)):
            print(f"Data augmentation: {i}/{len(train_dataset)}", end="\r")
            if i % 100 == 0:
                print_memory_usage()
            
            entry = train_dataset[i]
            entries = [entry[key] for key in keys if key is not None]
             
            # Append original data
            for key, value in zip(keys, entries):
                if value is not None:
                    augmented_data[key].append(value)
            
            for k in other_keys:
                augmented_data[k].append(entry[k])
            #augmented_data['label'].append(entry['label'])
            #augmented_data['idx'].append(entry['idx'])
            
            # Handle augmentation
            augmented_entries = []
            for value in entries:
                if value is not None:
                    
                    if(data_args.task_name in ["wic","wsc"]):
                        
                        exclude_words = [train_dataset[i]["word"]] if data_args.task_name == "wic" else [ train_dataset[i]["span1_text"], train_dataset[i]["span2_text"] ]
                        augmented_entries.append( augment_superglue_sentence(value,aug_n=aug_n,exclude_words=exclude_words) ) 
                    else:
                        augmented_entries.append(augment_sentence(value,aug_n=aug_n))

                else:
                    augmented_entries.append(None)
            
            # Determine minimum size for augmented samples
            aug_samples_count = min(len(aug) for aug in augmented_entries if aug is not None)
            
            # Append augmented data
            for j in range(aug_samples_count):
                for key, aug in zip(keys, augmented_entries):
                    if aug is not None:
                        augmented_data[key].append(aug[j])
            
            other_keys = [k for k in train_dataset.column_names if k not in keys ]
            for k in other_keys:
                augmented_data[k].extend( [ entry[k] ] * aug_samples_count )
            #augmented_data['label'].extend([entry['label']] * aug_samples_count)
            #augmented_data['idx'].extend([entry['idx']] * aug_samples_count)

        # Save the augmented data
        with open(dataset_path, 'wb') as f:
            pickle.dump(augmented_data, f)
        print("Augmented data saved to file.")


    if(do_augment):
        augmented_dataset = datasets.Dataset.from_dict(augmented_data)
        raw_datasets['train'] = augmented_dataset

    def update_idx(example, idx):
        example["idx"] = idx
        return example

    # Assuming raw_datasets['train'] is a Dataset object
    raw_datasets['train'] = raw_datasets['train'].map(update_idx, with_indices=True)
    
    ''' 
    print("------ AFTER AUGMENT ---------")
    for data in raw_datasets['train']:

        for k in task_to_keys[data_args.task_name]:
            if k !=None:
                print("____")
                print("\t",k)
                print("\t",data[k])
                print("_____")
    exit() '''

    if not os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            #cache_dir=args.cache_dir,
            #use_fast=args.use_fast_tokenizer,
            #revision=args.model_revision,
            #token=args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if ("Llama" in model_args.model_name_or_path ):
        tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists(config_path):
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            #revision=args.model_revision,
            #token=args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        
        if ("Llama" in model_args.model_name_or_path ):
            config.pad_token_id = tokenizer.pad_token_id    
            config.use_cache = False
        
        config.save_pretrained(config_path)
    else:
        print("LOAD FROM SAVE")
        config = AutoConfig.from_pretrained(config_path)
    

    if ("Llama" in model_args.model_name_or_path ):
        config.pad_token_id = tokenizer.pad_token_id    
        config.use_cache = False 

    # Load or save model
    if not os.path.exists(model_path):

        if ("Llama" in model_args.model_name_or_path ):

            model = LlamaForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                #revision=args.model_revision,
                #token=args.token,
                trust_remote_code=model_args.trust_remote_code,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            ) 

            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.use_cache = False
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        model.save_pretrained(model_path)


    else:
        if ("Llama" in model_args.model_name_or_path ):
            model = LlamaForSequenceClassification.from_pretrained(model_path)
    
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    fine_tuned = data_args.post_ft_capture == 'y' 
    if fine_tuned:
        print("Using already fine tuned Model")
        model.load_state_dict(torch.load(f"./saves/models/baseline/{model_args.model_name_or_path}/{data_args.task_name}/baseline_model.pth"))    

    # model = my_model

    if data_args.task_name is not None:
        sentence1_key, sentence2_key,sentence3_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    
    def preprocess_wic_function(examples):
        processed_texts = []

        for s1, s2, word, start1, end1, start2, end2 in zip(examples['sentence1'], examples['sentence2'], examples['word'], examples['start1'], examples['end1'], examples['start2'], examples['end2']):
            # Mark the word in sentence1
            s1 = s1[:start1] + "[unused0]" + s1[start1:end1] + "[unused1]" + s1[end1:]
            # Mark the word in sentence2
            s2 = s2[:start2] + "[unused0]" + s2[start2:end2] + "[unused1]" + s2[end2:]
            # Concatenate the two sentences
            processed_texts.append(s1 + " " + s2)
        
        result = tokenizer(processed_texts, padding='max_length', max_length=128, truncation=True)

        if "label" in examples:
            # Handling unexpected labels by defaulting to a specific label, e.g., 0
            label_to_id = {0: 0, 1: 1, -1: 0}  # Adding -1 mapping to handle the KeyError
            result["labels"] = [label_to_id.get(l, 0) for l in examples["label"]]  # Default to 0 if label not found

        return result
    
    def preprocess_wsc_function(examples):
        # Extract the required fields from the examples
        texts = examples['text']
        span1_indices = examples['span1_index']
        span2_indices = examples['span2_index']
        span1_texts = examples['span1_text']
        span2_texts = examples['span2_text']

        # Prepare modified texts by marking the spans
        marked_texts = []
        for text, s1_idx, s2_idx, s1_text, s2_text in zip(texts, span1_indices, span2_indices, span1_texts, span2_texts):
            if s1_idx < s2_idx:
                marked_text = (text[:s1_idx] + '[SPAN1]' + s1_text + '[/SPAN1]' +
                               text[s1_idx + len(s1_text):s2_idx] + '[SPAN2]' + s2_text + '[/SPAN2]' +
                               text[s2_idx + len(s2_text):])
            else:
                marked_text = (text[:s2_idx] + '[SPAN2]' + s2_text + '[/SPAN2]' +
                               text[s2_idx + len(s2_text):s1_idx] + '[SPAN1]' + s1_text + '[/SPAN1]' +
                               text[s1_idx + len(s1_text):])
            
            marked_texts.append(marked_text)

        # Tokenize the marked texts
        result = tokenizer(marked_texts, padding='max_length', max_length=128, truncation=True)

        # Handle labels if present
        if "label" in examples:
            # Standardizing label format
            # Assuming labels might be in boolean format or as strings 'true'/'false'
            label_to_id = {0: 0, 1: 1,-1:0}  # Removing unnecessary -1 mapping
            result["labels"] = [label_to_id[int(l)] for l in examples['label']]

        return result


    with training_args.main_process_first(desc="dataset map pre-processing"):

        if(data_args.task_name == "wic"):
            raw_datasets = raw_datasets.map(
                preprocess_wic_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
        elif(data_args.task_name == 'wsc'):

            raw_datasets = raw_datasets.map(
                preprocess_wsc_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
        else:
            
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Load or save metric
    # Get the metric function
    if data_args.task_name not in ["rte","mrpc","stsb","cola"]:
        metric = evaluate.load("./downloads/evaluate/metrics/super_glue/super_glue.py", data_args.task_name)
    else:
        metric = evaluate.load("./downloads/evaluate/metrics/glue/glue.py", data_args.task_name)    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    def create_directory_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
   
    class AttentionHook:
        def __init__(self, encoder_idx):
            self.encoder_idx = encoder_idx
            self.batch_idx = 0

        def __call__(self, module, inputs, outputs):
            output_save_folder = f"./saves/{model_args.model_name_or_path}/{data_args.task_name}/mha/outputs/encoder_{self.encoder_idx}"
            create_directory_if_not_exists(output_save_folder)
            o = outputs[0]
            torch.save(o, f"{output_save_folder}/o_batch_{self.batch_idx}.pt")
            self.batch_idx += 1

    class AttentionHookV2:
        def __init__(self, encoder_idx):
            self.encoder_idx = encoder_idx
            self.batch_idx = 0

        def __call__(self, module, inputs, outputs):
            base_save_folder = f"./saves/{model_args.model_name_or_path}/{data_args.task_name}/mha/encoder_{self.encoder_idx}"
            
            # Save inputs
            input_save_folder = f"{base_save_folder}/inputs"
            create_directory_if_not_exists(input_save_folder)
            
            a = inputs[0]
            b = inputs[1]
            torch.save(a, f"{input_save_folder}/a_batch_{self.batch_idx}.pt")
            torch.save(b, f"{input_save_folder}/b_batch_{self.batch_idx}.pt")

            # Save outputs
            output_save_folder = f"{base_save_folder}/outputs"
            create_directory_if_not_exists(output_save_folder)
            
            o = outputs[0]
            torch.save(o, f"{output_save_folder}/o_batch_{self.batch_idx}.pt")

            self.batch_idx += 1

    class FfnHook:
        def __init__(self, encoder_idx):
            self.encoder_idx = encoder_idx
            self.batch_idx = 0

        def __call__(self, module, inputs, outputs):
            input_save_folder = f"./saves/{model_args.model_name_or_path}/{data_args.task_name}/ffn/inputs/encoder_{self.encoder_idx}"
            create_directory_if_not_exists(input_save_folder)
            output_save_folder = f"./saves/{model_args.model_name_or_path}/{data_args.task_name}/ffn/outputs/encoder_{self.encoder_idx}"
            create_directory_if_not_exists(output_save_folder)

            o = outputs
            torch.save(o, f"{output_save_folder}/o_batch_{self.batch_idx}.pt")

            h = inputs[0]
            torch.save(h, f"{input_save_folder}/h_batch_{self.batch_idx}.pt")

            self.batch_idx += 1

    class LayerHook:
        def __init__(self, encoder_idx):
            self.inputs = []
            self.outputs = []
            self.encoder_idx = encoder_idx
            self.batch_idx = 0

        def __call__(self, module, inputs, outputs):


            input_save_folder = f"./saves/{model_args.model_name_or_path}/{data_args.task_name}/mha/inputs/encoder_{self.encoder_idx}"
            create_directory_if_not_exists(input_save_folder)

            a = inputs[0]
            b = inputs[1]

            torch.save(a, f"{input_save_folder}/a_batch_{self.batch_idx}.pt")
            torch.save(b, f"{input_save_folder}/b_batch_{self.batch_idx}.pt")
            self.batch_idx += 1

    # Check if the model is Llama and adjust the hook registration accordingly
    if "Llama" in model_args.model_name_or_path:
        # Assuming model.layers is a list of layers in Llama
        hooks = [AttentionHook(i) for i in range(len(model.model.layers))]
        layer_hooks = [LayerHook(i) for i in range(len(model.model.layers))]
        ffn_hooks = [FfnHook(i) for i in range(len(model.model.layers))]

        def pre_forward_hook(module, args):
            #hidden_states, attention_mask, *rest = args
            print(f"Attention mask in pre-forward hook: {len(args)}")
            return args

        for i, hook in enumerate(hooks):
            if i == data_args.encoder_idx:
                model.model.layers[i].self_attn.register_forward_hook(hooks[i])
                model.model.layers[i].mlp.register_forward_hook(ffn_hooks[i])
                model.model.layers[i].register_forward_hook(layer_hooks[i])

    else:
        # Original DistilBERT hook registration
        modelbert = model.distilbert
        hooks = [AttentionHook(i) for i in range(len(modelbert.transformer.layer))]
        layer_hooks = [LayerHook(i) for i in range(len(modelbert.transformer.layer))]
        ffn_hooks = [FfnHook(i) for i in range(len(modelbert.transformer.layer))]


        for i, hook in enumerate(hooks):
            if i == data_args.encoder_idx:
                modelbert.transformer.layer[i].attention.register_forward_hook(hooks[i])
                modelbert.transformer.layer[i].ffn.register_forward_hook(ffn_hooks[i])
                modelbert.transformer.layer[i].register_forward_hook(layer_hooks[i])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Evaluation
    logger.info("*** Evaluate ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    eval_datasets = [train_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        valid_mm_dataset = raw_datasets["validation_mismatched"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
            valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
        eval_datasets.append(valid_mm_dataset)
        combined = {}

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        trainer.log_metrics("eval", metrics)
        #trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

   

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
