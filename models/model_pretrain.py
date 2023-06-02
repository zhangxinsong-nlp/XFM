# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.


import torch
import torch.nn as nn
from models import XFMBase


class XFM(XFMBase):
    def __init__(self, config, load_vision_params=True, load_text_params=True):
        super().__init__(config, load_vision_params=load_vision_params, load_text_params=load_text_params,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True, use_bbox_loss=True, config_text=None)
        
        wregion = config.get('wregion', 1.0)
        wweb = config.get('wweb', 1.0)
        wimagenet = config.get('wimagenet', 1.0)
        wimage = config.get('wimage', 1.0)
        waux = config.get('waux', 1.0)
        self.weights_map = {'region':wregion,'web':wweb,'imagenet':wimagenet,'image':wimage,'aux':waux}
        self.do_image_mask = config.get('do_image_mask', True)
        self.use_mm_mim_loss = config.get('use_mm_mim_loss', True)
        self.min_temp = config.get('min_temp', 0.001)
        self.max_temp = config.get('max_temp', 0.5)


    def forward_multimodal(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                text_ids_2=None, text_atts_2=None, text_ids_masked_2=None, masked_pos_2=None, masked_ids_2=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None, ret_mim_loss=False,
                ret_bbox_loss=False, ret_match_loss=True, ret_mlm_loss=True, ret_bbox_giou=False, ret_itc_loss=True, data_source=None):

        if self.learnable_temp:
            with torch.no_grad():
                self.temp.clamp_(self.min_temp, self.max_temp)    
        
        if ret_bbox_loss:
            image_embeds, image_atts, image_embeds_fullatts = \
                self.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img)
        else:
            image_embeds, image_atts = self.get_vision_embeds(image)

        if data_source != 'imagenet':
            text_embeds = self.get_text_embeds(text_ids, text_atts)  # 12L text encoder, i.e. cross encoder w/o image input
            image_feat, text_feat = self.get_features(image_embeds, text_embeds)
        
        loss_itc = torch.tensor(0.0)
        if ret_itc_loss and data_source != 'imagenet':
            loss_itc = self.get_contrastive_loss(image_feat, text_feat)
            if data_source in self.weights_map.keys():
                loss_itc *= self.weights_map[data_source]
            
        loss_itm = torch.tensor(0.0)
        if ret_match_loss and data_source != 'imagenet':
            loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_ids, text_atts, text_feat, text_embeds=text_embeds)  # 12L cross
            if data_source in self.weights_map.keys():
                loss_itm *= self.weights_map[data_source]

        loss_mlm = torch.tensor(0.0)
        if ret_mlm_loss and data_source != 'imagenet':
            loss_mlm = self.get_fuse_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)  # 12L cross 
            if data_source in self.weights_map.keys():
                loss_mlm *= self.weights_map[data_source]

        loss_mim = torch.tensor(0.0)
        if ret_mim_loss and not ret_bbox_loss:
            image_embeds_masked, image_atts, ids_mask = self.get_vision_embeds(image, do_mask=self.do_image_mask)
            if self.use_vision_tokenizer:
                if (data_source == 'imagenet' or self.use_mm_mim_loss):
                    loss_mim = self.get_mim_loss(image_embeds_masked, image, ids_mask)
            else:
                if (data_source == 'imagenet' or self.use_mm_mim_loss):
                    loss_mim = self.get_mim_loss(image_embeds_masked, image_embeds, ids_mask)
            if data_source in self.weights_map.keys():
                loss_mim *= self.weights_map[data_source]

        loss = {'loss_itc': loss_itc, 'loss_itm': loss_itm, 'loss_mlm': loss_mlm, 'loss_mim': loss_mim}

        if ret_bbox_giou:
            output_coord = self.predict_bbox(image_embeds_fullatts, text_ids, text_atts, text_embeds)
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)

            loss['loss_bbox'] = loss_bbox
            loss['loss_giou'] = loss_giou
        else:
            loss['loss_bbox'] = torch.tensor(0.0)
            loss['loss_giou'] = torch.tensor(0.0) 

        return loss

    def forward_text(self, text_ids=None, text_atts=None,
                     text_ids_masked=None, masked_pos=None, masked_ids=None):

        loss = self.get_mlm_loss(text_ids_masked, text_atts, None, None, masked_pos, masked_ids)

        return {'loss_mlm': loss}

    def forward(self, image=None, text_ids=None, text_atts=None,
                text_ids_masked=None, masked_pos=None, masked_ids=None,
                text_ids_2=None, text_atts_2=None, text_ids_masked_2=None, masked_pos_2=None, masked_ids_2=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None, ret_mim_loss=False,
                ret_bbox_loss=False, ret_match_loss=True, ret_mlm_loss=True, ret_bbox_giou=False, ret_itc_loss=True, data_source=None):

        if image is None:  # text
            loss = self.forward_text(text_ids, text_atts, text_ids_masked, masked_pos, masked_ids)

        else:
            loss = self.forward_multimodal(image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids,
                                           text_ids_2, text_atts_2, text_ids_masked_2, masked_pos_2, masked_ids_2,
                                           image_atts, idx_to_group_img, target_bbox, is_image, ret_mim_loss, ret_bbox_loss,
                                           ret_match_loss=ret_match_loss, ret_mlm_loss=ret_mlm_loss, ret_bbox_giou=ret_bbox_giou,
                                           ret_itc_loss=ret_itc_loss, data_source=data_source)

        return loss