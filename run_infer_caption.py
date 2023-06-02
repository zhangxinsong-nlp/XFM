# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.
# By Xinsong Zhang
# Based on X-VLM code base
# https://github.com/zengyan-97/X-VLM


import argparse
import os
import math
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_generation import XFMForCaptioning


import utils
from utils.hdfs_io import hmkdir, hexists

from dataset.utils import collect_result
from dataset import create_dataset, create_sampler, create_loader


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 50
    
    result = []

    for image, image_names in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device, non_blocking=True)

        captions = model_without_ddp.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                  min_length=config['min_length'])

        for caption, img_id in zip(captions, image_names):
            assert isinstance(img_id, str)
            result.append({"image_id": img_id.strip(), "caption": caption})

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if world_size > 8:
        assert hexists(args.output_hdfs) and args.output_hdfs.startswith('hdfs'), "for collect_result among nodes"

    if args.bs > 0:
        config['batch_size_test'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']

    print(f"Generating captions for {args.img_rdir}")
    config['img_rdir'] = args.img_rdir
    dataset = create_dataset('infer_caption', config)

    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {len(dataset)}, batch size, {config['batch_size_test']} x {world_size}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([dataset], [False], num_tasks, global_rank)
    else:
        samplers = [None]

    test_loader = create_loader([dataset], samplers,
                                batch_size=[config['batch_size_test']],
                                num_workers=[4], is_trains=[False],
                                collate_fns=[None])[0]

    print("Creating model")
    model = XFMForCaptioning(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate, load_capt_pretrain=args.load_capt_pretrain)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)

    print("Start evaluating")
    test_result = evaluation(model, test_loader, device, config)
    test_result_file = collect_result(test_result, 'xvlm_caption', local_wdir=args.result_dir,
                                      hdfs_wdir=args.output_hdfs,
                                      write_to_hdfs=world_size > 8, save_result=True, remove_duplicate='image_id')

    dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")

    parser.add_argument('--img_rdir', type=str, required=True, help="images to generate captions")

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--load_capt_pretrain', action='store_true')
    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--evaluate', action='store_true')

    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    hmkdir(args.output_dir)
    hmkdir(args.result_dir)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)

    main(args, config)