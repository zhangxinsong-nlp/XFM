# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import argparse
import os
import sys

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import math

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from models.model_classification import XVLMForClassification

import utils
from dataset import create_dataset, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer

from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hcopy
from accelerators import ACCELERATOR_MAP


def reinit_scheduler_properties_mysched(optimizer: Optimizer, scheduler, cfg) -> None:
    """
    with ApexDDP, do re-init to avoid lr_scheduler warning.
    issue: https://github.com/pytorch/pytorch/issues/27595
    issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/841
    """
    args = cfg

    if scheduler.optimizer == optimizer:
        # from transformers import get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        scheduler.__init__(optimizer, lr_lambda, last_epoch=-1)


def run_text_iter(model, tokenizer, batch, optimizer, accelerator, metric_logger, device):
    text, targets = batch

    optimizer.zero_grad()

    targets = targets.to(device)
    text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)
    with autocast(accelerator.cfg.AUTO_CAST):
        loss = model(None, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=True)
    accelerator.backward_step(loss, optimizer)
    accelerator.optimizer_step(optimizer, model)

    # accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    # if accelerator_clip_grad_norm > 0:
    #     accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    # optimizer.step()

    metric_logger.update(text_loss=loss.item())


def run_image_iter(model, tokenizer, batch, optimizer, accelerator, metric_logger, device):
    optimizer.zero_grad()

    image, text, targets = batch
    image, targets = image.to(device, non_blocking=True), targets.to(device)
    text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)
    with autocast(accelerator.cfg.AUTO_CAST):
        loss = model(image, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=True)
    accelerator.backward_step(loss, optimizer)
    accelerator.optimizer_step(optimizer, model)

    # accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    # if accelerator_clip_grad_norm > 0:
    #     accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    # optimizer.step()

    metric_logger.update(loss=loss.item())


def train(model, image_loader, text_loader, test_loader_dict, optimizer, epoch_info, device, scheduler, config, accelerator, checkpointer, tokenizer):
    model.train()
    start_epoch, _ = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_large', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train step: [{}]'.format(start_epoch)
    assert start_epoch == 0
    print_freq = 50

    world_size = utils.get_world_size()
    step_per_epoch = math.ceil(config['train_dataset_size']/(config['batch_size']*world_size))
    assert step_per_epoch > 1
    global_step = 0  # start from 0

    best_acc = 0
    best_step = 0

    if text_loader is not None:
        text_iter = iter(text_loader)
        metric_logger.add_meter('text_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    else:
        text_iter = None

    for i, batch in enumerate(metric_logger.log_every(image_loader, print_freq, header, step_per_epoch, epoch_info)):
        if (text_iter is not None) and ((global_step == 0) or (random.random() < config['texts']['iter_perc'])):
            run_text_iter(model, tokenizer, next(text_iter), optimizer, accelerator, metric_logger, device)

        run_image_iter(model, tokenizer, batch, optimizer, accelerator, metric_logger, device)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_large=optimizer.param_groups[2]["lr"])
        scheduler.step()

        current_epoch = global_step // step_per_epoch

        if (global_step + 1) % step_per_epoch == 0:
            if utils.is_main_process():
                train_stats = {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch}
                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if (current_epoch+1) % config['ckpt_frequent'] == 0:
                    model_without_ddp = model
                    if hasattr(model, 'module'):
                        model_without_ddp = model.module
                    save_obj = {'model': model_without_ddp.state_dict(), 'config': config}
                    checkpointer.save_checkpoint(model_state=save_obj,
                                                 epoch=current_epoch,
                                                 training_states=optimizer.state_dict())
            dist.barrier()

        if (global_step+1) % config['ckpt_frequent_step'] == 0:
            if utils.is_main_process():
                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module
                save_obj = {'model': model_without_ddp.state_dict(), 'config': config}
                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=current_epoch, step=global_step,
                                             training_states=optimizer.state_dict())
            dist.barrier()

        if (global_step+1) % config['test_frequent_step'] == 0:
            test_stats = evaluate(model, test_loader_dict, tokenizer, device, config)
            model.train()

            if utils.is_main_process():
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': current_epoch, 'step': global_step}

                print(log_stats, flush=True)

                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if float(test_stats['acc']) > best_acc:
                    best_acc = float(test_stats['acc'])
                    best_step = global_step

                print("### Best Step: {:}, Best Accuracy: {:.3f}".format(best_step, best_acc), flush=True)

            dist.barrier()

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader_dict, tokenizer, device, config):
    model.eval()

    header = 'Evaluation:'
    print_freq = 50
    rtn = {}
    acc = 0.

    for t, data_loader in data_loader_dict.items():
        num_test_steps = math.ceil(config['test_dataset_size'][t] / (config['batch_size_test'] * utils.get_world_size()))
        metric_logger = utils.MetricLogger(delimiter="  ")
        for image, text, targets in metric_logger.log_every_test(data_loader, print_freq, header, dataset_len=num_test_steps):
            image = image.to(device, non_blocking=True)
            targets = targets.to(device)
            text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

            prediction = model(image, text_inputs.input_ids, text_inputs.attention_mask, targets=targets, train=False)

            _, pred_class = prediction.max(1)
            accuracy = (targets == pred_class).sum() / targets.size(0)

            metric_logger.meters['acc'].update(accuracy.item(), n=image.size(0))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats: type: ", t, 'result: ', metric_logger.global_avg())
        acc += metric_logger.meters['acc'].global_avg
        rtn.update({t+'_'+k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()})
    rtn['acc'] = acc / len(data_loader_dict)
    
    return rtn


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    config['train_image_file'] = ','.join(config['train_image_file'])
    config['train_text_file'] = ','.join(config['train_text_file'])
    config['batch_size'] = config['batch_size_train']

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    if args.bs > 0:
        config['batch_size'] = args.bs // utils.get_world_size()

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset", flush=True)
    image_dataset, test_dataset_dict, text_dataset = create_dataset('classify', config)
    config['num_labels'] = image_dataset.num_labels

    if utils.is_main_process():
        print(f"### train_image_file: {config['train_image_file']}", flush=True)
        print(f"### train_text_file: {config['train_text_file']}", flush=True)
        print(f"### test_file: {config['test_file']}", flush=True)
        print(f"### batch size, {config['batch_size']} x {int(os.environ.get('WORLD_SIZE', 1))}")

    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=config['batch_size'],
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=None)

    if text_dataset is not None:
        text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=config['batch_size'],
                                                   num_workers=4,
                                                   pin_memory=True,
                                                   drop_last=False,
                                                   collate_fn=None)
    else:
        text_loader = None

    test_loader_dict = {}
    for k, v in test_dataset_dict.items():
        test_loader_dict[k] = torch.utils.data.DataLoader(v, batch_size=config['batch_size_test'],
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=None)

    print("Creating model", flush=True)

    use_xbrain = config.get('use_xbrain', True)
    if config.get('use_text_classifier', False):
        from models.model_classification import TextClassifier
        model = TextClassifier(config=config)
        if use_xbrain:
            model.load_pretrained(args.checkpoint, config)
    elif use_xbrain:
        from models.model_classification import XFMForClassification
        model = XFMForClassification(config=config)
        model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    else:
        model = XVLMForClassification(config=config)
        model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)

    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    tokenizer = build_tokenizer(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")
        test_stats = evaluate(model, test_loader_dict, tokenizer, device, config)
        if utils.is_main_process():
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
            print(log_stats, flush=True)
        dist.barrier()

    else:
        print("Start training")
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)

        arg_sche = utils.AttrDict(config['schedular'])
        world_size = utils.get_world_size()
        arg_sche['step_per_epoch'] = math.ceil(config['train_dataset_size'] / (config['batch_size'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        arg_acc = utils.AttrDict(config['accelerator'])
        accelerator = ACCELERATOR_MAP[arg_acc['ACCELERATOR']](arg_acc, logger=None)

        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, world_size, rank)
        reinit_scheduler_properties_mysched(optimizer, lr_scheduler, arg_sche)

        checkpointer = Checkpointer(args.output_dir)

        start_epoch = 0
        max_epoch = config['schedular']['epochs']
        epoch_info = (start_epoch, max_epoch)

        print("Start training", flush=True)
        train(model, image_loader, text_loader, test_loader_dict, optimizer, epoch_info, device, lr_scheduler, config,
              accelerator, checkpointer, tokenizer)

    if utils.is_main_process():
        os.system("cat log.txt")
        hcopy('log.txt', args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), flush=True)

    print('### Time {}'.format(total_time_str))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    hmkdir(args.output_dir)

    yaml.dump(config, open('config.yaml', 'w'))
    hcopy('config.yaml', args.output_dir)

    main(args, config)