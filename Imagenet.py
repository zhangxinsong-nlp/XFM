# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path
import PIL
import math

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets

from models.model_classification import XFMForClassification
from models.vit import interpolate_pos_embed
from optim import LARS
from torchvision import transforms

from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from timm.utils import accuracy as timm_accuracy

import utils


DATASETS = {
    "celeba": datasets.CelebA,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "emnist": datasets.EMNIST,
    "fakedata": datasets.FakeData,
    "fashionmnist": datasets.FashionMNIST,
    "flickr8k": datasets.Flickr8k,
    "flickr30k": datasets.Flickr30k,
    "inaturalist": datasets.INaturalist,
    "kmnist": datasets.KMNIST,
    "lfwpeople": datasets.LFWPeople,
    "lsun": datasets.LSUN,
    "mnist": datasets.MNIST,
    "omniglot": datasets.Omniglot,
    "places365": datasets.Places365,
    "qmnist": datasets.QMNIST,
    "semeion": datasets.SEMEION,
    "sbu": datasets.SBU,
    "stl10": datasets.STL10,
    "svhn": datasets.SVHN,
    "usps": datasets.USPS,
    #----below are only supported by torch1.11 + torchvision0.12
    "sun397": datasets.SUN397,
    "country211": datasets.Country211,
    "dtd": datasets.DTD,
    "caltech101": datasets.Caltech101,
    "caltech256": datasets.Caltech256,
    "stanfordcars": datasets.StanfordCars,
    "renderedsst2": datasets.RenderedSST2,
    "pcam": datasets.PCAM,
    "oxfordiiitpet": datasets.OxfordIIITPet,
    "flowers102": datasets.Flowers102,
    "food101": datasets.Food101,
    "gtsrb": datasets.GTSRB,
    "fer2013": datasets.FER2013,
    "fgvcaircraft": datasets.FGVCAircraft,
    "eurosat": datasets.EuroSAT,
    "kitti": datasets.Kitti,
}

dataset2nlabels = {
    'imagenet': 1000,
    'food101': 101,
    'cifar10': 10,
    'cifar100': 100,
    'stanfordcars': 196,
    'fgvcaircraft': 102,
    'dtd': 47,
    'oxfordiiitpet': 37,
    'flowers102': 103, # flowers 1 - 102
    'mnist': 10,
    'stl10': 10,
    'sun397': 397,
    'caltech101': 101,
    'caltech256': 256,
    'gtsrb': 43, # data unavailable
    # 'kitti': unclear structure
    'country211': 211,
    'fer2013': 7,
    'pcam': 2,
    'kitti': 9,
    'renderedsst2': 2,
}


class FER2013(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    #def __init__(self, split='Training', transform=None):
    def __init__(self, split, transform=None):    
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('/opt/tiger/xvlm/images/fer2013/data.h5', 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((28709, 48, 48))
            self.train_data = np.array(self.train_data)
           

        elif self.split == 'PublicTest':
            self.PublicTest_data = self.data['PublicTest_pixel']
            self.PublicTest_labels = self.data['PublicTest_label']
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = self.PublicTest_data.reshape((3589, 48, 48))
            self.PublicTest_data = np.array(self.PublicTest_data)

        else :
            self.PrivateTest_data = self.data['PrivateTest_pixel']
            self.PrivateTest_labels = self.data['PrivateTest_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3589, 48, 48))
            self.PrivateTest_data = np.array(self.PrivateTest_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]   #  np.newaxis的作用是增加一个维度
        img = np.concatenate((img, img, img), axis=2)  #完成多个数组的拼接
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else :
            return len(self.PrivateTest_data)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        # acc1, acc5 = timm_accuracy(output, target, topk=topk)
        # print("acc1:", acc1)
        # return [acc1, acc5]


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_epochs = config['schedular']['warmup_epochs']
    epochs = config['schedular']['epochs']
    peak_lr = config['schedular']['lr']
    min_lr = config['schedular']['min_lr']
    if epoch < warmup_epochs:
        lr = peak_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (peak_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def build_transform(is_train, config):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config['image_res'],
            is_training=True,
            color_jitter=config['color_jitter'],
            auto_augment=config['aa'],
            interpolation='bicubic',
            re_prob=config['reprob'],
            re_mode=config['remode'],
            re_count=config['recount'],
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    input_size = config['image_res']
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def gen_loader(config, is_training=True):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    num_readers = config['num_readers']

    data_path = config['dataset']
    sampler = None
    if is_training:
        batch_size = config['batch_size_train']
        traindir = os.path.join(data_path, 'train')
        train_transform = build_transform(True, config)
        dataset = datasets.ImageFolder(traindir, transform=train_transform)
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=num_readers, pin_memory=True, sampler=sampler)
    else:
        batch_size = config['batch_size_test']
        valdir = os.path.join(data_path, 'val')
        # data_path = 'hdfs://haruna/home/byte_ailab_litg/user/zhangxinsong/pretrain/vlm/data/imagenet/val'
        val_transform = build_transform(False, config)
        dataset = datasets.ImageFolder(valdir, transform=val_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_readers, pin_memory=True)


    return data_loader, sampler


def gen_loader_others(config, is_training=True):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    num_readers = config['num_readers']

    data_path = config['dataset']
    sampler = None
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])

    train_dataset, validate_dataset = None, None
    if config['task_name'] in ['sun397', 'caltech101', 'caltech256']:
        transform = transforms.Compose([
                transforms.RandomResizedCrop(config['image_res'], interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if config['task_name'] == 'caltech101':
            transform = transforms.Compose([
                    transforms.RandomResizedCrop(config['image_res'], interpolation=3),
                    transforms.Grayscale(3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])     

        dataset = DATASETS[config['task_name']](data_path, download=True, transform=transform)  
        train_size = int(len(dataset) * 0.8)
        validate_size = len(dataset) - train_size
        train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size])

    if is_training:
        batch_size = config['batch_size_train']
        traindir = os.path.join(data_path, 'train')
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(config['image_res'], interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if config['task_name'] in  ['mnist', 'fer2013']:
            train_transform = transforms.Compose([
                    transforms.Resize(config['image_res'], interpolation=3),
                    transforms.Grayscale(3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        if config['task_name'] in ['mnist', 'cifar10', 'cifar100', 'kitti']:
            dataset = DATASETS[config['task_name']](
                traindir, train=True, download=True, transform=train_transform)
        elif config['task_name'] in ['dtd', 'fgvcaircraft', 'food101', 'stanfordcars', 'gtsrb', 'pcam']:
            dataset = DATASETS[config['task_name']](
                traindir, split='train', download=True, transform=train_transform)
        elif config['task_name'] in ['oxfordiiitpet']:
            dataset = DATASETS[config['task_name']](
                traindir, split='trainval', download=True, transform=train_transform)
        elif config['task_name'] in ['sun397', 'caltech101', 'caltech256']:
            dataset = train_dataset 
        elif config['task_name'] == 'fer2013':
            dataset = FER2013(split = 'Training', transform=train_transform)
        else:
            dataset = DATASETS[config['task_name']](
                traindir, split='train', download=True, transform=train_transform) 
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=num_readers, pin_memory=True, sampler=sampler)
    else:
        batch_size = config['batch_size_test']
        valdir = os.path.join(data_path, 'val')
        test_transform = transforms.Compose([
            transforms.Resize(288, interpolation=3),
            transforms.CenterCrop(config['image_res']),
            transforms.ToTensor(),
            normalize,
        ])
        if config['task_name'] in  ['mnist', 'fer2013']:
            test_transform = transforms.Compose([
                transforms.Resize(config['image_res'], interpolation=3),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                normalize,
            ])
        if config['task_name'] in ['mnist', 'cifar10', 'cifar100', 'kitti']:
            dataset = DATASETS[config['task_name']](
                valdir, train=False, download=True, transform=test_transform)
        elif config['task_name'] in ['dtd', 'fgvcaircraft', 'food101', 'stanfordcars', 'gtsrb', 'pcam']:
            dataset = DATASETS[config['task_name']](
                valdir, split='test', download=True, transform=test_transform)   
        elif config['task_name'] in ['oxfordiiitpet']:
            dataset = DATASETS[config['task_name']](
                valdir, split='test', download=True, transform=test_transform) 
        elif config['task_name'] in ['sun397', 'caltech101', 'caltech256']:
            dataset = validate_dataset
        elif config['task_name'] == 'fer2013':
            dataset = FER2013(split = 'PrivateTest', transform=test_transform)
        else:
            dataset = DATASETS[config['task_name']](
                valdir, split='test', download=True, transform=test_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_readers, pin_memory=True)

    return data_loader, sampler


def train(model, train_loader, optimizer, criterion, epoch, mixup_fn, device, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.8f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    lr_log = AverageMeter('lr', ':.8f')
    progress = ProgressMeter(
        len(train_loader),
        [lr_log, batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # train
    model.train()  

    end = time.time()
    for i, (images, target) in enumerate(train_loader): # images: [4096, 3, 224, 224], target: [4096]
        # measure data loading time
        data_time.update(time.time() - end)

        # FROM MAE: we use a per iteration (instead of per epoch) lr scheduler
        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)

        images = images.cuda(device, non_blocking=True)

        if config['task_name'] == 'kitti':
            target = target['type']
            target = target.cuda(device, non_blocking=True)
        else:
            target = target.cuda(device, non_blocking=True)

        if mixup_fn is not None:
            images, target = mixup_fn(images, target)

        # compute output
        output = model(images, None, None, None, False) # output: [4096, 1000]
        loss = criterion(output, target)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))
        lr_log.update(optimizer.param_groups[0]["lr"], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)  


@torch.no_grad()
def evaluate(model, val_loader, device):
    # test
    eval_criterion = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader): # images [512, 3, 256, 256] target [512]
        images = images.cuda(device, non_blocking=True)
        target = target.cuda(device, non_blocking=True)
            
        # compute output
        output = model(images, None, None, None, False) 
        loss = eval_criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg
    
    
def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    rank = utils.get_rank()
    config['num_labels'] = dataset2nlabels[config['task_name']]

    #### Model #### 
    print(f"Creating model XFM for classification", flush=True)
    PretrainModel = XFMForClassification
    model = PretrainModel(config=config)
    if os.path.exists(args.checkpoint):
        model.load_pretrained(args.checkpoint, config)

    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module  

    print("config['optimizer']", config['optimizer'])
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if config['optimizer']['opt'] == 'lars':
        optimizer = LARS(parameters, lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['opt'] == 'adamW':
        optimizer = torch.optim.AdamW(parameters, lr=config['optimizer']['lr'])
    else:
        optimizer = torch.optim.SGD(parameters, config['optimizer']['lr'],
                                    momentum=config['optimizer']['momentum'],
                                    weight_decay=config['optimizer']['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    print("optimizer = ", optimizer)

    max_epoch = config['schedular']['epochs']
    # warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    
    print("Start training")
    start_time = time.time()
    if config.get('task_name', 'imagenet') == 'imagenet':
        train_loader, train_sampler = gen_loader(config, is_training=True)
        val_loader, _ = gen_loader(config, is_training=False)
    else:
        train_loader, train_sampler = gen_loader_others(config, is_training=True)
        val_loader, _ = gen_loader_others(config, is_training=False)

    mixup_fn = None
    is_lp = config.get('is_lp', False)
    mixup_active = (config['mixup'] > 0 or config['cutmix'] > 0. or config['cutmix_minmax'] is not None) and not is_lp
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=config['mixup'], cutmix_alpha=config['cutmix'], cutmix_minmax=config['cutmix_minmax'],
            prob=config['mixup_prob'], switch_prob=config['mixup_switch_prob'], mode=config['mixup_mode'],
            label_smoothing=config['smoothing'], num_classes=config['num_labels'])

    # # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy().cuda(args.gpu)
    elif config['smoothing'] > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config['smoothing']).cuda(args.gpu)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    print("criterion = %s" % str(criterion))

    best_acc1 = 0
    for epoch in range(0, max_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(model, train_loader, optimizer, criterion, epoch, mixup_fn, device, config)  
        acc1 = evaluate(model, val_loader, device)
        # test_stats = evaluate(model, test_loader, tokenizer, device, config)

        if utils.is_main_process():  
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                print("best_acc1 = ", best_acc1)
        
        dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/imagenet_zeroshot.yaml')
    parser.add_argument('--output_dir', default='output/imagenet')  
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)