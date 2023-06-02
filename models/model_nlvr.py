
# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import os
import torch
from torch import nn
import torch.nn.functional as F

from models import XFMBase, build_mlp


class XFMForNLVR(XFMBase):

    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=None)

        # self.share_cross_attention(self.text_encoder.encoder)
        self.cls_head = build_mlp(input_dim=self.text_width*2, output_dim=2)
        if 'load_domain_pretrained' not in config or not config['load_domain_pretrained']:
            self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def forward(self, image, text_ids, text_atts, targets, train=True):
        image_embeds, image_atts = self.get_vision_embeds(image)
        encoder_embeds = self.get_text_embeds(text_ids, text_atts)
        image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))

        output_cls_image1 = self.get_cross_embeds(image0_embeds, image_atts[:image0_embeds.size(0)],
                                           text_embeds=encoder_embeds, text_atts=text_atts, is_pretrain=False)[:, 0, :]

        output_cls_image2 = self.get_cross_embeds(image1_embeds, image_atts[image0_embeds.size(0):],
                                           text_embeds=encoder_embeds, text_atts=text_atts, is_pretrain=False)[:, 0, :]

        output_cls = torch.cat((output_cls_image1, output_cls_image2), dim=-1)

        assert output_cls.shape[-1] == self.text_width*2

        prediction = self.cls_head(output_cls)

        return F.cross_entropy(prediction, targets) if train else prediction