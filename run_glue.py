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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import torch

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import ruamel.yaml as yaml
from transformers import BertTokenizer, RobertaTokenizer
from models.model_classification import XFMForClassification

import utils
from utils.checkpointer import Checkpointer

import transformers
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)


logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--config", type=str, default=None, help="A yaml file containing the hyperparameters.", required=True
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="A checkpoint."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output_dir is not None and not args.output_dir.startswith('hdfs'):
        os.makedirs(args.output_dir, exist_ok=True)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # Sanity checks
    if config['task_name'] is None and config['train_file'] is None and config['validation_file'] is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if config['train_file'] is not None:
            extension = config['train_file'].split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if config['validation_file'] is not None:
            extension = config['validation_file'].split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args, config


def main():
    args, config = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if config['task_name'] is not None:
        # Downloading and loading a dataset from the hub.
        if config.get('glue_datasets', ''):
            raw_datasets=datasets.load_from_disk(os.path.join(config.get('glue_datasets', ''), config['task_name']))
        else:
            raw_datasets = load_dataset("glue", config['task_name'])
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if config['train_file'] is not None:
            data_files["train"] = config['train_file']
        if config.validation_file is not None:
            data_files["validation"] = config['validation_file']
        extension = (config['train_file'] if config['train_file'] is not None else config['valid_file']).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if config['task_name'] is not None:
        is_regression = config['task_name'] == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if config['use_huggingface_models']:
        model_config = AutoConfig.from_pretrained(config['model_name_or_path'], num_labels=num_labels, finetuning_task=config['task_name'])
        tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], use_fast=not config['use_slow_tokenizer'])
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name_or_path'],
            from_tf=bool(".ckpt" in config['model_name_or_path']),
            config=model_config,
        )
    else:
        print('use our models for GLUE test. e.l. XFM')
        if ('bert-base-uncased' in config['text_encoder']) or ('bert-large-uncased' in config['text_encoder']):
            tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
        elif ('roberta-base' in config['text_encoder']) or ('roberta-large' in config['text_encoder']):
            tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
        else:
            pass
        print(f"Creating model XFM for classification", flush=True)
        PretrainModel = XFMForClassification
        config['num_labels'] = num_labels
        model = PretrainModel(config=config)
        if os.path.exists(args.checkpoint):
            model.load_pretrained(args.checkpoint, config)

    # Preprocessing the datasets
    if config['task_name'] is not None:
        sentence1_key, sentence2_key = task_to_keys[config['task_name']]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models in huggingface model zoo have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if config['use_huggingface_models'] and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        print(label_to_id)
        if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and config['task_name'] is not None
            and not is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                logger.info(
                    f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                    "Using it!"
                )
                label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
            else:
                logger.warn(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif config['task_name'] is None:
            label_to_id = {v: i for i, v in enumerate(label_list)}
        
    padding = "max_length" if config['pad_to_max_length'] else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=config['max_length'], truncation=True)
        
        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if config['task_name'] == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if config['pad_to_max_length']:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config['per_device_train_batch_size']
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config['per_device_eval_batch_size'])

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['gradient_accumulation_steps'])
    if config['max_train_steps'] is None:
        config['max_train_steps'] = config['num_train_epochs'] * num_update_steps_per_epoch
    else:
        config['num_train_epochs'] = math.ceil(config['max_train_steps'] / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=config['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=config['num_warmup_steps'],
        num_training_steps=config['max_train_steps'],
    )

    # Get the metric function
    if config['task_name'] is not None:
        metric = load_metric("glue", config['task_name'])

    # Train!
    total_batch_size = config['per_device_train_batch_size'] * accelerator.num_processes * config['gradient_accumulation_steps']

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {config['per_device_train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {config['max_train_steps']}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(config['max_train_steps']), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    checkpointer = Checkpointer(args.output_dir)
    
    for epoch in range(config['num_train_epochs']):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if config['use_huggingface_models']:
                outputs = model(**batch)
                loss = outputs.loss
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                loss = model(image=None, text_ids=input_ids, text_atts=attention_mask, targets=labels)
            loss = loss / config['gradient_accumulation_steps']
            accelerator.backward(loss)
            if step % config['gradient_accumulation_steps'] == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= config['max_train_steps']:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            if config['use_huggingface_models']:
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = model(image=None, text_ids=input_ids, text_atts=attention_mask, targets=labels, train=False)
                predictions = outputs.argmax(dim=-1) if not is_regression else outputs.squeeze()
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")
        if (epoch+1) % config['ckpt_frequent'] == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_obj = {'model': unwrapped_model, 'config': config}
            checkpointer.save_checkpoint(model_state=save_obj,
                                            epoch=epoch,
                                            training_states=optimizer.state_dict())

    if config['task_name'] == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=config['per_device_eval_batch_size']
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            if config['use_huggingface_models']:
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
            else:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = model(image=None, text_ids=input_ids, text_atts=attention_mask, targets=labels, train=False)
            predictions = outputs.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")


if __name__ == "__main__":
    main()