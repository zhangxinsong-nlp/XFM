# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

from accelerators.accelerator import Accelerator
from accelerators.ddp_accelerator import DDPAccelerator
from accelerators.apex_ddp_accelerator import ApexDDPAccelerator
from accelerators.torch_ddp_accelerator import TorchAMPDDPAccelerator

ACCELERATOR_MAP = {'ApexDDP': ApexDDPAccelerator,
                   'DDP': DDPAccelerator,
                   'TorchAMPDDP': TorchAMPDDPAccelerator,
                   }