# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import torch
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler

Net = torch.nn.Module
from accelerators import DDPAccelerator

# from fex.engine.accelerators.max_scaler import MaxClipGradScaler


class TorchAMPDDPAccelerator(DDPAccelerator):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.scaler = GradScaler()
        self.accelerator_clip_grad_norm = self.cfg.CLIP_GRAD_NORM
        # self.scaler = MaxClipGradScaler() # PartialFC 可能需要用到这个，未经过测试，先留一行注释以备以后遇到问题作为提醒

    def backward_step(self, loss: torch.Tensor, optimizer: Optimizer):
        """
        backward step
        """
        self.scaler.scale(loss).backward()

    def backward_step_with_grad(self, features: torch.Tensor, grad: torch.Tensor, optimizer: Optimizer):
        """
        backward step
        """
        features.backward(self.scaler.scale(grad))

    def optimizer_step(self, optimizer: Optimizer, model: Net) -> float:
        """
        optimizer step,
        1. Gradient clipping (if has)
        """
        # Unscales the gradients of optimizer's assigned params in-place

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        total_norm = 0
        if self.accelerator_clip_grad_norm > 0:
            self.scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.accelerator_clip_grad_norm)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        self.scaler.step(optimizer)

        # Updates the scale for next iteration.
        self.scaler.update()

        optimizer.zero_grad()

        return float(total_norm)

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict):
        return self.scaler.load_state_dict(state_dict)

    def get_metrics(self):
        return {'loss_scale_0': self.scaler.get_scale()}
