
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import matplotlib.pyplot as plt
import copy
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import numpy as np
import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from evaluate import load
from low_rank_modules.distilbert import FFNLowRank,MultiHeadSelfAttentionLowRank 
import loss_landscapes
import loss_landscapes.metrics
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.42.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42
    )
    

    parser.add_argument(
        "--modular_job_name",
        type=str,
        default='NONE'
    )
    parser.add_argument(
        "--compress",
        type=str,
        default='n'
    )
    parser.add_argument(
        "--modular",
        type=str,
        default='n'
    )
    
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

save_dir = "./downloads"
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")
def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    config_path = os.path.join(save_dir, f"{args.task_name}_config")
    tokenizer_path = os.path.join(save_dir, f"{args.task_name}_tokenizer")
    model_path = os.path.join(save_dir, f"{args.task_name}_model")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        logger.info(f"******* Setting seed : {args.seed} ********")
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    ''' CHANGE THESE LINES '''
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if (args.task_name in ["rte","stsb","mrpc","cola"]):
            raw_datasets = load_dataset("glue", args.task_name)

        else:
            raw_datasets = load_dataset("aps/super_glue", args.task_name)

    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
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
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    print(raw_datasets['train']) 
    print(f"Labels : ",num_labels)
    print(set(raw_datasets['train']['label']))
    print(set(raw_datasets['validation']['label']))
    
    print(set(raw_datasets['test']['label']))
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if not os.path.exists(config_path):
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
        )
        config.save_pretrained(config_path)
    else:
        print("LOAD FROM SAVE")
        config = AutoConfig.from_pretrained(config_path)

    # Load or save tokenizer
    if not os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            use_fast=args.use_fast_tokenizer,
            revision=args.model_revision,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load or save model
    if not os.path.exists(model_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
        model.save_pretrained(model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    if(args.compress == 'y'):
        my_model = copy.deepcopy(model)
        module_trained_for = 200
        for i in range(6):
            
            module_path = f"./saves/{args.model_name_or_path}/{args.modular_job_name}/model/mha_enc{i}_epoch{module_trained_for}.pth"
            mha = MultiHeadSelfAttentionLowRank(config,compression=2)
            
            if(args.modular == 'y'):
                mha.load_state_dict(torch.load(module_path))
            
            my_model.distilbert.transformer.layer[i].attention = mha
            
            
            module_path = f"./saves/{args.model_name_or_path}/{args.modular_job_name}/model/ffn_enc{i}_epoch{module_trained_for}.pth"
            ffn = FFNLowRank(config,compression=2)
            
            if( args.modular == 'y'):
                ffn.load_state_dict(torch.load(module_path))

            my_model.distilbert.transformer.layer[i].ffn = ffn
    
        model = my_model 

    

    non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
    print(f"Non label column names",non_label_column_names)

    loss_save_folder = f"./saves/models/loss_landscape/{args.model_name_or_path}/{args.task_name}"
        

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key, sentence3_key = task_to_keys[args.task_name]
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

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False
    #print(model.config.label2id)
    #print(model.config.id2label)


    def preprocess_function(examples):
        # Tokenize the texts
        if(sentence3_key is None):
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
        else:
            texts = (
                (examples[sentence1_key],examples[sentence2_key],examples[sentence3_key])
            )

        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
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

    with accelerator.main_process_first():

        if(args.task_name == "wic"):
            
            processed_datasets = raw_datasets.map(
                preprocess_wic_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
        elif(args.task_name == 'wsc'):

            processed_datasets = raw_datasets.map(
                preprocess_wsc_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
        else:
            
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    def metric_acc_and_f1(eval_pred):
        
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        
        accuracy_metric = load("accuracy")
        f1_metric = load("f1")

        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels)
        return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}
    
    if args.task_name is not None:
            
        if(args.task_name in ["rte","stsb","mrpc","cola"]):
            metric = load("glue",args.task_name)
        else:
            metric = load("super_glue",args.task_name)
    
    # Extract a single batch to use for visualization
    batch = next(iter(train_dataloader))
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    labels = batch['labels'].to(model.device)  # Make sure 'labels' is the correct key for your dataset

    # You might need other inputs depending on the model architecture, such as 'token_type_ids'
    if 'token_type_ids' in batch:
        token_type_ids = batch['token_type_ids'].to(model.device)
    else:
        token_type_ids = None

    # Now structure the inputs in a way expected by the model
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    if token_type_ids is not None:
        inputs['token_type_ids'] = token_type_ids

    # Define the loss function directly
    criterion = torch.nn.CrossEntropyLoss()
    model200 = copy.deepcopy(model)
    model217 = copy.deepcopy(model)

    # Load state

    sufix = ''
    sufix += '_compressed' if args.compress == 'y' else ''
    sufix += '_modular' if args.modular == 'y' else ''

    model200.load_state_dict( torch.load(f"{loss_save_folder}/e0_compressed_modular.pth") )
    model217.load_state_dict( torch.load(f"{loss_save_folder}/best_model_compressed_modular.pth") )

    def interpolate_model(model1, model2, alpha):
        model_interp = copy.deepcopy(model1)
        for name, param1 in model1.named_parameters():
            param2 = model2.state_dict()[name]
            # Update the interpolated model's parameters
            model_interp.state_dict()[name].copy_(param1 * (1 - alpha) + param2 * alpha)
        return model_interp
    alphas = torch.linspace(0, 1, steps=20)
    losses = []

    # Assuming you have an eval_dataloader set up and metric loaded
    for alpha in alphas:
        model_interp = interpolate_model(model200, model217, alpha.item())
        model_interp.eval()
        total_loss = 0
        count_batches = 0

        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model_interp(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                count_batches += 1

        avg_loss = total_loss / count_batches
        losses.append(avg_loss)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(alphas.numpy(), losses, marker='o')
    plt.title('Loss Landscape Interpolation between Model200 and Model217')
    plt.xlabel('Interpolation Coefficient (Alpha)')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()
    plt.savefig("loss_compressed_modular.png")
    exit()

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    global_results = {}

    ''' Change num train steps depending on use case '''
    if(args.modular == 'y'):
        args.num_train_epochs = 18
    
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            
            metric.add_batch(
                 predictions=predictions,
                 references=references
            )  
        eval_metric = metric.compute()        
        logger.info(f"[{args.task_name}][EVAL] epoch {epoch}: {eval_metric}")
        global_results[str(epoch)] = eval_metric
        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()
    
    
    #baseline_model_dir = f"./saves/models/baseline/{args.model_name_or_path}/{args.task_name}"
    #create_directory_if_not_exists(baseline_model_dir)
    #torch.save(model.state_dict(),f"{baseline_model_dir}/baseline_model.pth")
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "218_results_{args.model_name_or_path}_{args.task_name}.json"), "w") as f:
            json.dump(global_results, f)
    
    

if __name__ == "__main__":
    main()
