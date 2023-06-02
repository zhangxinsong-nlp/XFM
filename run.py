# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.
# By Xinsong Zhang
# Based on X-VLM code base
# https://github.com/zengyan-97/X-VLM

import os
import sys
import argparse

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy

############ Set it correctly for distributed training across nodes
NNODES = 1  # e.g. 1/2/3/4
NPROC_PER_NODE = 1  # e.g. 1 gpu1

MASTER_ADDR = '127.0.0.1' # set master address
MASTER_PORT = 12345
NODE_RANK = 0  # e.g. 0/1/2
############

print("NNODES, ", NNODES)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NODE_RANK, ", NODE_RANK)


def get_nnodes(args):  # when using only part of nodes
    if args.dist == 'all':
        return NNODES

    elif args.dist == '2':
        assert NNODES >= 2
        return 2

    else:
        return 1


def get_dist_launch(args):  # some examples
    if args.dist == 'all':  # use all nodes
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes={:} --node_rank={:} --master_addr={:} --master_port={:}".format(
            NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, args.master_port)

    elif args.dist == '2':
        assert int(os.getenv("ARNOLD_WORKER_NUM")) >= 2
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes=2 --node_rank={:} --master_addr={:} --master_port={:}".format(
            NPROC_PER_NODE, NODE_RANK, MASTER_ADDR, args.master_port)

    elif args.dist == '1':
        return "python3 -m torch.distributed.launch --nproc_per_node={:} --master_port={:} " \
               "--nnodes=1 ".format(NPROC_PER_NODE, args.master_port)

    elif args.dist == 'f4':
        return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.launch --nproc_per_node=4 " \
               "--nnodes=1 "

    elif args.dist == 'l4':
        return "CUDA_VISIBLE_DEVICES=4,5,6,7 WORLD_SIZE=4 python3 -m torch.distributed.launch --master_port=12345 --nproc_per_node=4 " \
               "--nnodes=1 "

    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.launch --nproc_per_node=1 " \
               "--nnodes=1 --master_port={:}".format(num, args.master_port)

    else:
        raise ValueError


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]
        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local


def run_pretrain(args):
    dist_launch = get_dist_launch(args)

    use_env = 'Pretrain.py'

    print(f"### Start pre-training {use_env}", flush=True)
    os.system(f"{dist_launch} --use_env {use_env} --seed {args.seed} "
              f"--epoch {args.epoch} --config {args.config} --output_dir {args.output_dir} "
              f"{f'--checkpoint {args.checkpoint}' if args.checkpoint else ''}")


def run_pretrain_refcoco_bbox(args):
    print("### Start refcoco bbox domain pre-training", flush=True)

    dist_launch = get_dist_launch(args)

    if len(args.load_ckpt_from):
        print(f"### Loading domain pre-trained results from: {args.load_ckpt_from}")
        domain_ckpt = get_from_hdfs(args.load_ckpt_from)

    else:  # domain pre-train
        if not os.path.exists(args.config): args.config = f'./configs/{args.model}/Grounding_bbox_pretrain_O1.yaml'

        os.system(f"{dist_launch} "
                  f"--use_env Grounding_bbox_pretrain.py --seed {args.seed} --config {args.config} "
                  f"--output_dir {args.output_dir} --checkpoint {args.checkpoint}")

        domain_ckpt = get_from_hdfs(f"{args.output_dir}/model_state_epoch_latest.th")

    return domain_ckpt


def run_pretrain_captioning(args):
    print("### Start captioning domain pre-training", flush=True)

    dist_launch = get_dist_launch(args)

    if len(args.load_ckpt_from):
        print(f"### Loading domain pre-trained results from: {args.load_ckpt_from}")
        domain_ckpt = get_from_hdfs(args.load_ckpt_from)

    else:  # domain pre-train
        if not os.path.exists(args.config): args.config = f'configs/{args.model}/Captioning_pretrain_O1.yaml'

        os.system(f"{dist_launch} --use_env Captioning_pretrain.py --seed {args.seed} --config {args.config} "
                  f"--output_dir {args.output_dir} --checkpoint {args.checkpoint}")

        domain_ckpt = get_from_hdfs(f"{args.output_dir}/model_state_epoch_latest.th")

    return domain_ckpt


def run_nlvr2(args, load_nlvr_pretrain=False):
    dist_launch = get_dist_launch(args)
    if not os.path.exists(args.config): args.config = f'./configs/{args.model}/NLVR.yaml'

    print("### Training NLVR2", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env NLVR.py --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} {'--load_nlvr_pretrain' if load_nlvr_pretrain else ''} "
              f"{'--evaluate' if args.evaluate else ''}")


def run_itr_flickr(args, use_itc_only=False):
    dist_launch = get_dist_launch(args)
    if not os.path.exists(args.config): args.config = f"configs/{args.model}/Retrieval_flickr.yaml"

    print("### Training Retrieval Flickr", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env {'Retrieval_itc.py' if use_itc_only else 'Retrieval.py'} --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_itr_coco(args, use_itc_only=False):
    dist_launch = get_dist_launch(args)
    if not os.path.exists(args.config): args.config = f"configs/{args.model}/Retrieval_coco.yaml"

    print("### Training Retrieval COCO", flush=True)
    os.system(f"{dist_launch} "
              f"--use_env {'Retrieval_itc.py' if use_itc_only else 'Retrieval.py'} --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --epoch {args.epoch} "
              f"--checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_vqa(args, load_vqa_pretrain=False):
    dist_launch = get_dist_launch(args)
    print("### Training VQA", flush=True)
    if not os.path.exists(args.config): args.config = f'./configs/{args.model}/VQA.yaml'

    if not os.path.exists(os.path.join(args.output_dir, 'result')):
        os.mkdir(os.path.join(args.output_dir, 'result'))

    os.system(f"{dist_launch} "
              f"--use_env VQA.py --config {args.config} {'--load_vqa_pretrain' if load_vqa_pretrain else ''}"
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_refcoco(args, use_bbox=False, block_num=-1, load_bbox_pretrain=False, epochs=-1):
    dist_launch = get_dist_launch(args)

    if use_bbox:
        print("### Training RefCOCO with bbox", flush=True)
        if not os.path.exists(args.config): args.config = f"configs/{args.model}/Grounding_bbox.yaml"

        os.system(f"{dist_launch} "
                  f"--use_env Grounding_bbox.py --config {args.config} "
                  f"--output_dir {args.output_dir} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} "
                  f"--bs {args.bs} --seed {args.seed} {'--load_bbox_pretrain' if load_bbox_pretrain else ''} --checkpoint {args.checkpoint} "
                  f"{'--evaluate' if args.evaluate else ''}")

    else:
        print("### Training RefCOCO", flush=True)
        if not os.path.exists(args.config): args.config = f"configs/{args.model}/Grounding.yaml"

        os.system(f"{dist_launch} "
                  f"--use_env Grounding.py --config {args.config} "
                  f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} "
                  f"--gradcam_mode itm --block_num {block_num} --epochs {epochs} --checkpoint {args.checkpoint} "
                  f"{'--evaluate' if args.evaluate else ''}")


def run_coco_captioning(args, load_capt_pretrain=False, scst=False):
    dist_launch = get_dist_launch(args)
    print("### Training COCO Captioning", flush=True)

    if not os.path.exists(args.config):
        args.config = f'./configs/{args.model}/Captioning.yaml'

    if scst:
        load_capt_pretrain = True 

    os.system(f"{dist_launch} "
              f"--use_env {'Captioning_scst.py' if scst else 'Captioning.py'} --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --epoch {args.epoch} --checkpoint {args.checkpoint} "
              f"{'--scst' if scst else ''}  {'--load_capt_pretrain' if load_capt_pretrain else ''} {'--evaluate' if args.evaluate else ''}")


def run_infer_caption(args):
    dist_launch = get_dist_launch(args)
    print("### Inferring captions", flush=True)

    args.config = f'./configs/{args.model}/Captioning.yaml'
    args.bs = 2048  # batch_size = bs // world_size

    os.system(f"{dist_launch} "
              f"--use_env run_infer_caption.py --img_rdir images/tmp --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} --evaluate")


def run_classify(args):
    dist_launch = get_dist_launch(args)
    print("### Training classifier", flush=True)

    if not os.path.exists(args.config):
        args.config = f'./configs/{args.model}/Classify.yaml'

    os.system(f"{dist_launch} "
              f"--use_env Classify.py --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint}")



def run_glue(args):
    dist_launch = get_dist_launch(args)
    print("### Training glue", flush=True)

    if not os.path.exists(args.config):
        args.config = f'./configs/{args.model}/glue_mrpc.yaml'

    os.system(f"{dist_launch} "
              f"--use_env run_glue.py --config {args.config} "
              f"--output_dir {args.output_dir} --seed {args.seed} --checkpoint {args.checkpoint}")


def run_imagenet(args):
    dist_launch = get_dist_launch(args)

    print("### Training imagenet1k classification", flush=True)
    sys.stdout.flush()

    if not os.path.exists(args.config):
            args.config = f'./configs/{args.model}/imagenet1k.yaml'
    
    os.system(f"{dist_launch} "
              f"--use_env Imagenet.py --config {args.config} "
              f"--output_dir {args.output_dir} --seed {args.seed} "
              f"{f'--checkpoint {args.checkpoint}' if args.checkpoint else ''}")

def run(args):
    if args.task == 'pretrain_DIY':
        if not os.path.exists(args.config):
            args.config = 'configs/xfm-pt/Pretrain_XBrain_base_4m.yaml'
        run_pretrain(args)

    elif args.task == 'infer_caption':
        run_infer_caption(args)

    # Academic Tasks ->
    elif args.task == 'itr_coco':
        run_itr_coco(args)

    elif args.task == 'itr_flickr':
        run_itr_flickr(args)

    elif args.task == 'vqa':
        run_vqa(args)

    elif args.task == 'nlvr':
        run_nlvr2(args)

    elif args.task == 'refcoco_bbox':
        domain_ckpt = run_pretrain_refcoco_bbox(args)
        # domain_ckpt = get_from_hdfs(f"{args.output_dir}/model_state_epoch_latest.th")
        # run fine-tune, reset args
        args.checkpoint = domain_ckpt
        if hexists(args.output_dir): args.output_dir = os.path.join(args.output_dir, 'refcoco_ft')
        args.config = f"configs/{args.model}/Grounding_bbox.yaml"
        run_refcoco(args, use_bbox=True, load_bbox_pretrain=True)

    elif args.task == 'glue':
        run_glue(args)

    elif args.task == 'imagenet':
        run_imagenet(args)

    elif args.task == 'coco_captioning':
        domain_ckpt = run_pretrain_captioning(args)

        # run fine-tune, reset args
        args.checkpoint = domain_ckpt
        if hexists(args.output_dir): args.output_dir = os.path.join(args.output_dir, 'coco_capt_ft')
        args.config = f'./configs/{args.model}/Captioning.yaml'
        run_coco_captioning(args, load_capt_pretrain=True)

    elif args.task == 'classify':
        run_classify(args)

    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--dist', type=str, required=True, help="see func get_dist_launch for details")

    parser.add_argument('--config', default='', type=str, help="if not given, use default")
    parser.add_argument('--model', default='xfm-ft', type=str, help="to set default fine-tuning configs")

    parser.add_argument('--epoch', default=-1, type=int, help="for pre-training (debug) only")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                           "this option only works for fine-tuning scripts.")

    parser.add_argument('--checkpoint', default='', type=str, help="for domain pretraining or fine-tuning")
    parser.add_argument('--load_ckpt_from', default='', type=str, help="load domain pre-trained params")

    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, required=True, help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--output_hdfs', type=str, default='', help="HDFS path required by VQA and Refcoco, "
                                                                    "to collect eval results among nodes")

    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--master_port', default=12345, type=int)

    args = parser.parse_args()

    if MASTER_ADDR == 'SET_IT':
        print("### warning: the settings for distributed training is not filled (ignore this if you only use one node)")

    if '/SET/PATH/TO/hadoop/bin/hdfs' in HADOOP_BIN:
        print("### warning: you have not set the path to hadoop_bin (ignore this if you don't use HDFS)")

    assert hexists(os.path.dirname(args.output_dir))
    hmkdir(args.output_dir)

    if len(args.output_hdfs):
        assert hexists(os.path.dirname(args.output_hdfs))
        hmkdir(args.output_hdfs)

    if len(args.config):
        assert hexists(args.config)

        if args.config.startswith('hdfs://'):
            args.config = get_from_hdfs(args.config)

    if args.checkpoint.startswith('hdfs://'):
        args.checkpoint = get_from_hdfs(args.checkpoint)

    run(args)

