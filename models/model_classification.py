
# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss

from models import XFMBase, build_mlp
# from transformers import OFATokenizer, OFAModel


class XFMForClassification(XFMBase):
    def __init__(self, config, init_scale=0.001):
        super().__init__(config, load_vision_params=True, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, 
                         use_bbox_loss=False)

        feature_dim = self.vision_width if config.get('task_name') == 'imagenet' else self.text_width
        self.is_lp = config.get('is_lp', False)
        self.use_ofa = config.get('use_ofa', False)
        # if self.use_ofa:
        #     self.tokenizer = OFATokenizer.from_pretrained('/opt/tiger/xvlm/tmp/ofa_huggingface/OFA-base')
        #     self.model_ofa = OFAModel.from_pretrained('/opt/tiger/xvlm/tmp/ofa_huggingface/OFA-base', use_cache=False)

        task_name = config.get('task_name', 'glue')
        if task_name == 'imagenet' or self.is_lp:
            self.cls_head = self.build_mlp(input_dim=feature_dim*2, output_dim=config['num_labels'])
        else:
            self.cls_head = build_mlp(input_dim=feature_dim, output_dim=config['num_labels'])

        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.LayerNorm(input_dim*2),
            nn.GELU(),
            nn.Linear(input_dim*2, input_dim * 4),
            nn.LayerNorm(input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim*4, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim*2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, image, text_ids, text_atts, targets, train=True):

        if image is None:
            output_cls = self.text_encoder(text_ids, attention_mask=text_atts,
                                           encoder_hidden_states=None,
                                           encoder_attention_mask=None,
                                           return_dict=True).last_hidden_state[:, 0, :]
        elif text_ids is None:
            # image_embeds, _ = self.get_vision_embeds(image)
            if self.is_lp:
                with torch.no_grad():
                    if self.use_ofa:
                        b = image.shape[0]
                        txt = " what does the image describe?"
                        inputs = self.tokenizer([txt], return_tensors="pt").input_ids
                        inputs = inputs.repeat(b, 1).to(image.device)
                        # patch_img = patch_resize_transform(img).unsqueeze(0)
                        image_embeds = self.model_ofa(inputs, patch_images=image, decoder_input_ids=inputs).encoder_last_hidden_state
                    else:
                        image_embeds, _ = self.get_vision_embeds(image) 
            else:
                image_embeds, _ = self.get_vision_embeds(image)
            output_cls = image_embeds[:, 0, :]
            output_mean = torch.mean(image_embeds[:, 1:, :], dim=1)
            output_cls = torch.cat([output_cls, output_mean], dim=-1)
        
        else:
            image_embeds, image_atts = self.get_vision_embeds(image)
            encoder_embeds = self.get_text_embeds(text_ids, text_atts)
            output_cls = self.get_cross_embeds(image_embeds, image_atts,
                                               text_embeds=encoder_embeds, text_atts=text_atts,
                                               is_pretrain=False)[:, 0, :]

        prediction = self.cls_head(output_cls)
        if prediction.shape[-1] == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(prediction.view(-1), targets.view(-1))
            return loss if train else prediction

        return F.cross_entropy(prediction, targets) if train else prediction


class TextClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        from transformers import BertModel, RobertaModel
        if 'roberta-base' in config['text_encoder']:
            self.text_encoder, msg = RobertaModel.from_pretrained(config['text_encoder'], add_pooling_layer=False, output_loading_info=True)
        else:
            self.text_encoder, msg = BertModel.from_pretrained(config['text_encoder'], add_pooling_layer=False, output_loading_info=True)

        print("Loading Text encoder: ", msg, flush=True)

        self.text_width = self.text_encoder.config.hidden_size
        self.cls_head = build_mlp(input_dim=self.text_width, output_dim=config['num_labels'])
        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def load_pretrained(self, ckpt_rpath, config, is_domain_pretrain=False):
        if is_domain_pretrain:
            checkpoint = torch.load(ckpt_rpath, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        else:
            print("### Loading pretrained text encoder", flush=True)
            checkpoint = torch.load(ckpt_rpath, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
            for key in list(state_dict.keys()):
                if key.startswith('text_encoder.'):

                    name_to_replace = 'bert.'

                    # TODO: 我在试 xroberta.py
                    if 'roberta' in config['text_encoder']:
                        name_to_replace = 'roberta.'

                    if name_to_replace in key:
                        encoder_key = key.replace(name_to_replace, '')
                        state_dict[encoder_key] = state_dict[key]
                        del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, targets, train=True):

        output_cls = self.text_encoder(text_ids, attention_mask=text_atts,
                                       encoder_hidden_states=None,
                                       encoder_attention_mask=None,
                                       return_dict=True).last_hidden_state[:, 0, :]

        prediction = self.cls_head(output_cls)

        return F.cross_entropy(prediction, targets) if train else prediction

