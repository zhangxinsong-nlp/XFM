# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import argparse
import os
import sys
import warnings

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import math

import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.cuda.amp import autocast

from models.model_pretrain import XFM

import utils
from dataset import create_dataset
from scheduler import create_scheduler
from optim import create_optimizer

from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hcopy, hexists
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


def run_image_iter(model, image_batch, optimizer, accelerator, metric_logger, device, data_source, ret_mim_loss=True, ret_match_loss=True, ret_mlm_loss=True, ret_itc_loss=True, do_optm=False):
    image, batch = image_batch[0].to(device, non_blocking=True), [t.to(device) if t is not None else None for t in image_batch[1:]]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = batch

    # optimizer.zero_grad()
    with autocast(accelerator.cfg.AUTO_CAST):
        loss = model(image, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, 
                    masked_ids=masked_ids, ret_match_loss=ret_match_loss, ret_mim_loss=ret_mim_loss,
                    ret_mlm_loss=ret_mlm_loss, ret_itc_loss=ret_itc_loss, data_source=data_source)

    loss_in_total = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm'] + loss['loss_mim']
    accelerator.backward_step(loss_in_total, optimizer)
    
    if do_optm:
        accelerator.optimizer_step(optimizer, model)
        optimizer.zero_grad()

    if data_source == 'image':
        metric_logger.update(loss_itc=loss['loss_itc'].item())
        metric_logger.update(loss_itm=loss['loss_itm'].item())
        metric_logger.update(loss_mlm=loss['loss_mlm'].item())
        metric_logger.update(loss_mim=loss['loss_mim'].item())
    elif data_source == 'web':
        metric_logger.update(loss_witc=loss['loss_itc'].item())
        metric_logger.update(loss_witm=loss['loss_itm'].item())
        metric_logger.update(loss_wmlm=loss['loss_mlm'].item())
        metric_logger.update(loss_wmim=loss['loss_mim'].item())
    elif data_source == 'imagenet':
        metric_logger.update(loss_imim=loss['loss_mim'].item())
    elif data_source == 'aux':
        metric_logger.update(loss_amlm=loss['loss_mlm'].item())


def run_region_iter(model, region_batch, optimizer, accelerator, metric_logger, device, data_source, ret_mim_loss=True, ret_match_loss=True, ret_mlm_loss=True, ret_itc_loss=True):
    image, region_batch = region_batch[0].to(device, non_blocking=True), [
        t.to(device) if t is not None else None for t in region_batch[1:]]

    idx_to_group_img, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, \
    image_atts, target_bbox, is_image = region_batch

    if config['calc_image_bbox_loss']:
        is_image = None

    # optimizer.zero_grad()
    with autocast(accelerator.cfg.AUTO_CAST):
        loss = model(image, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                    image_atts=image_atts, idx_to_group_img=idx_to_group_img, target_bbox=target_bbox, is_image=is_image,
                    ret_mim_loss=ret_mim_loss, ret_bbox_loss=config['ret_bbox_loss'], ret_match_loss=ret_match_loss, 
                    ret_mlm_loss=ret_mlm_loss, ret_bbox_giou=config['ret_bbox_giou'], ret_itc_loss=ret_itc_loss, data_source=data_source)

    loss_in_total = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm'] + loss['loss_bbox'] + loss['loss_giou']
    accelerator.backward_step(loss_in_total, optimizer)
    # accelerator.optimizer_step(optimizer, model)
    # optimizer.zero_grad()

    metric_logger.update(loss_ritc=loss['loss_itc'].item())
    metric_logger.update(loss_ritm=loss['loss_itm'].item())
    metric_logger.update(loss_rmlm=loss['loss_mlm'].item())
    metric_logger.update(loss_rbbox=loss['loss_bbox'].item())
    metric_logger.update(loss_rgiou=loss['loss_giou'].item())


def run_text_iter(model, batch, optimizer, accelerator, metric_logger, device):
    batch = [t.to(device) if t is not None else None for t in batch]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, = batch

    optimizer.zero_grad()
    with autocast(accelerator.cfg.AUTO_CAST):
        loss = model(None, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids)
        loss_in_total = loss['loss_mlm']
    accelerator.backward_step(loss_in_total, optimizer)
    accelerator.optimizer_step(optimizer, model)
    optimizer.zero_grad()
    # accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    # if accelerator_clip_grad_norm > 0:
    #     accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
    # optimizer.step()

    metric_logger.update(loss_tmlm=loss['loss_mlm'].item())

def train(model, image_loader, data_loaders, optimizer, epoch_info, device, scheduler, config, accelerator, checkpointer):
    model.train()
    image_loader_aux, image_loader_web, image_loader_imagenet, region_loader, text_loader = data_loaders
    start_epoch, _ = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_large', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train step: [{}]'.format(start_epoch)
    # assert start_epoch == 0
    print_freq = 50

    world_size = utils.get_world_size()
    step_per_epoch = math.ceil(config['train_dataset_size']/(config['batch_size']*world_size))
    assert step_per_epoch > 1
    current_step = start_epoch * step_per_epoch
    global_step = current_step + 1
    # global_step = 0  # start from 0

    stop_calc_itm = config.get('stop_calc_itm', float('inf'))  # steps
    print(f"### Stop Calculate Matching Loss After {stop_calc_itm} Steps", flush=True)

    stop_calc_mlm = config.get('stop_calc_mlm', float('inf'))  # steps
    print(f"### Stop Calculate MLM Loss After {stop_calc_mlm} Steps", flush=True)

    stop_calc_itc = config.get('stop_calc_itc', float('inf'))  # steps
    print(f"### Stop Calculate ITC Loss After {stop_calc_itc} Steps", flush=True)

    stop_calc_mim = config.get('stop_calc_mim', float('inf'))  # steps
    print(f"### Stop Calculate MIM Loss After {stop_calc_mim} Steps", flush=True)

    stop_calc_mm = config.get('stop_calc_mm', float('inf'))  # steps
    print(f"### Stop Calculate MM Loss After {stop_calc_mm} Steps", flush=True)

    if stop_calc_mm != 0:
        metric_logger.add_meter('loss_mim', utils.SmoothedValue(window_size=50, fmt='{value:.4f}')) 
        metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}')) 

    if region_loader is not None:
        region_iter = iter(region_loader)
        metric_logger.add_meter('loss_ritc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_ritm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_rmlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_rbbox', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_rgiou', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    else:
        region_iter = None

    if image_loader_web is not None:
        image_iter_web = iter(image_loader_web)
        metric_logger.add_meter('loss_witc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_wmlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_wmim', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_witm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    else:
        image_iter_web = None

    if image_loader_imagenet is not None:
        image_iter_imagenet = iter(image_loader_imagenet)
        metric_logger.add_meter('loss_imim', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    else:
        image_iter_imagenet = None

    if text_loader is not None:
        text_iter = iter(text_loader)
        metric_logger.add_meter('loss_tmlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    else:
        text_iter = None

    if image_loader_aux is not None:
        image_iter_aux = iter(image_loader_aux)  # large-scale image-text pairs, more general
        metric_logger.add_meter('loss_amlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    else:
        image_iter_aux = None  

    for i, batch in enumerate(metric_logger.log_every(image_loader, print_freq, header, step_per_epoch, epoch_info)):
        
        if text_iter is not None:
            run_text_iter(model, next(text_iter), optimizer, accelerator, metric_logger, device)

        if region_iter is not None:
            run_region_iter(model, next(region_iter), optimizer, accelerator, metric_logger, device, data_source='region', ret_mim_loss=global_step < stop_calc_mim,
                            ret_match_loss=global_step < stop_calc_itm, ret_mlm_loss=global_step < stop_calc_mlm, ret_itc_loss=global_step < stop_calc_itc)

        if image_iter_web is not None:
            do_optm = False if global_step < stop_calc_mm else True
            run_image_iter(model, next(image_iter_web), optimizer, accelerator, metric_logger, device, data_source='web', ret_mim_loss=global_step < stop_calc_mim,
                ret_match_loss=global_step < stop_calc_itm, ret_mlm_loss=global_step < stop_calc_mlm, ret_itc_loss=global_step < stop_calc_itc)

        if image_iter_aux is not None:
            run_image_iter(model, next(image_iter_aux), optimizer, accelerator, metric_logger, device, data_source='aux', ret_mim_loss=global_step < stop_calc_mim,
                ret_match_loss=False, ret_mlm_loss=global_step < stop_calc_mlm, ret_itc_loss=False)
        
        if image_iter_imagenet is not None:
            do_optm = False if (global_step < stop_calc_mm and image_iter_web is not None) else True
            run_image_iter(model, next(image_iter_imagenet), optimizer, accelerator, metric_logger, device, data_source='imagenet', ret_mim_loss=global_step < stop_calc_mim,
                ret_match_loss=global_step < stop_calc_itm, ret_mlm_loss=global_step < stop_calc_mlm, ret_itc_loss=global_step < stop_calc_itc, do_optm=do_optm)         

        if global_step < stop_calc_mm:
            run_image_iter(model, batch, optimizer, accelerator, metric_logger, device, data_source='image', ret_mim_loss=global_step < stop_calc_mim,
                            ret_match_loss=global_step < stop_calc_itm, ret_mlm_loss=global_step < stop_calc_mlm, ret_itc_loss=global_step < stop_calc_itc, do_optm=True)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_large=optimizer.param_groups[2]["lr"])
        scheduler.step()

        current_epoch = global_step // step_per_epoch
        if (global_step+1) % step_per_epoch == 0:
            if utils.is_main_process():
                train_stats = {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch,
                             }

                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if (current_epoch+1) % config['ckpt_frequent'] == 0:
                    model_without_ddp = model
                    if hasattr(model, 'module'):
                        model_without_ddp = model.module

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(),
                        'config': config,
                        'epoch': current_epoch,
                    }
                    checkpointer.save_checkpoint(model_state=save_obj,
                                                 epoch=current_epoch,
                                                 training_states=optimizer.state_dict())

            dist.barrier()

        if (global_step+1) % config['ckpt_frequent_step'] == 0:
            if utils.is_main_process():
                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': scheduler.state_dict(),
                    'config': config,
                    # 'epoch': current_epoch,
                }

                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=current_epoch, step=global_step,
                                             training_states=optimizer.state_dict())

            dist.barrier()

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    warnings.filterwarnings("ignore")

    assert 'xlm' not in config['text_encoder'], "For multilingual pre-training, use Pretrain_multilingual.py instead"

    config['train_file'] = ','.join(config['train_file'])
    config['train_file_regions'] = ','.join(config['train_file_regions'])
    if 'train_file_text' in config:
        config['train_file_text'] = ','.join(config['train_file_text'])

    if 'train_file_aux' in config:
        config['train_file_aux'] = ','.join(config['train_file_aux'])

    if 'train_file_web' in config:
        config['train_file_web'] = ','.join(config['train_file_web'])

    if 'train_file_imagenet' in config:
        config['train_file_imagenet'] = ','.join(config['train_file_imagenet'])

    config['batch_size'] = config['images']['batch_size']

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset", flush=True)
    image_dataset, region_dataset, text_dataset, image_dataset_aux, image_dataset_web, image_dataset_imagenet = create_dataset('pretrain', config)

    if utils.is_main_process():
        assert hexists(os.path.dirname(args.output_dir))
        hmkdir(args.output_dir)

        print(f"### images: {config['train_file']}", flush=True)
        print(f"### regions: {config['train_file_regions']}", flush=True)
        print(f"### images_aux: {config['train_file_aux'] if 'train_file_aux' in config else ''}", flush=True)
        print(f"### images_web: {config['train_file_web'] if 'train_file_web' in config else ''}", flush=True)
        print(f"### images_imagenet: {config['train_file_imagenet'] if 'train_file_imagenet' in config else ''}", flush=True)
        print(f"### texts: {config['train_file_text'] if 'train_file_text' in config else ''}", flush=True)
        print(f"### batch size, {config['batch_size']} x {int(os.environ.get('WORLD_SIZE', 1))}")

    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=config['images']['batch_size'],
                                               num_workers=config['images']['num_workers'],
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=image_dataset.collate_fn)

    if image_dataset_aux is not None:  # for small-scale high-quality images
        image_loader_aux = torch.utils.data.DataLoader(image_dataset_aux,
                                                       batch_size=config['images_aux']['batch_size'],
                                                       num_workers=config['images_aux']['num_workers'],
                                                       pin_memory=True,
                                                       drop_last=False,
                                                       collate_fn=image_dataset_aux.collate_fn)
    else:
        image_loader_aux = None

    if image_dataset_web is not None:  # for small-scale high-quality images
        image_loader_web = torch.utils.data.DataLoader(image_dataset_web,
                                                       batch_size=config['images_web']['batch_size'],
                                                       num_workers=config['images_web']['num_workers'],
                                                       pin_memory=True,
                                                       drop_last=False,
                                                       collate_fn=image_dataset_web.collate_fn)
    else:
        image_loader_web = None

    if image_dataset_imagenet is not None:  # for small-scale high-quality images
        image_loader_imagenet = torch.utils.data.DataLoader(image_dataset_imagenet,
                                                       batch_size=config['images_imagenet']['batch_size'],
                                                       num_workers=config['images_imagenet']['num_workers'],
                                                       pin_memory=True,
                                                       drop_last=False,
                                                       collate_fn=image_dataset_imagenet.collate_fn)
    else:
        image_loader_imagenet = None

    if region_dataset is not None:
        region_loader = torch.utils.data.DataLoader(region_dataset, batch_size=config['regions']['max_images'],
                                                    # batch_size = max_images * max_regions
                                                    num_workers=config['regions']['num_workers'],
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    collate_fn=region_dataset.collate_fn)
    else:
        region_loader = None

    if text_dataset is not None:
        text_loader = torch.utils.data.DataLoader(text_dataset, batch_size=config['texts']['batch_size'],
                                                  num_workers=config['texts']['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  collate_fn=text_dataset.collate_fn)
    else:
        text_loader = None

    print(f"Creating model XFM", flush=True)
    PretrainModel = XFM

    if config.get('train_from_scratch', False):
        model = PretrainModel(config=config, load_text_params=True, load_vision_params=False)
    else:
        model = PretrainModel(config=config)

    # print(model)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'fusion_encoder.roberta.embeddings' not in name), flush=True)
    for name in model.state_dict():
        print(name)

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    arg_sche['step_per_epoch'] = math.ceil(config['train_dataset_size'] / (config['batch_size'] * world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    arg_acc = utils.AttrDict(config['accelerator'])
    accelerator = ACCELERATOR_MAP[arg_acc['ACCELERATOR']](arg_acc, logger=None)

    start_epoch = 0
    if config.get('resume', False):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']+1 
        
    if os.path.exists(args.checkpoint):
        model.load_pretrained(args.checkpoint, config, is_domain_pretrain=True)

    model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, world_size, rank)

    checkpointer = Checkpointer(args.output_dir)

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    max_epoch = config['schedular']['epochs']
    epoch_info = (start_epoch, max_epoch)

    data_loaders = (image_loader_aux, image_loader_web, image_loader_imagenet, region_loader, text_loader)
    print("Start training", flush=True)
    train(model, image_loader, data_loaders, optimizer, epoch_info, device, lr_scheduler, config,
          accelerator, checkpointer)
    dist.barrier()

    if utils.is_main_process():
        os.system("cat log.txt")
        hcopy('log.txt', args.output_dir)

        yaml.dump(config, open('config.yaml', 'w'))
        hcopy('config.yaml', args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), flush=True)

    print('### Time {}'.format(total_time_str))

    
if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='output/pretrain')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # hmkdir(args.output_dir)

    main(args, config)