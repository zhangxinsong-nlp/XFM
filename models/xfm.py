# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.
# By Xinsong Zhang
# Based on BEiT, X-VLM code base
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/zengyan-97/X-VLM

from logging import raiseExceptions
import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from functools import partial

from models import box_ops
from utils import read_json
from models.model_vqkd import vqkd_encoder_base_decoder_3x768x12_clip


def load_params_change_prefix(state_dict: dict, prefix: str, new_prefix: str):
    if prefix == new_prefix:
        return state_dict

    state_dict_new = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k.replace(prefix, new_prefix)

        state_dict_new[k] = v

    return state_dict_new


def load_roberta_lm_head(state_dict):
    def _replace(old_key: str, new_key: str):
        if new_key != old_key:
            state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]

    _replace('lm_head.bias', 'cls.predictions.bias')
    _replace('lm_head.dense.weight', 'cls.predictions.transform.dense.weight')
    _replace('lm_head.dense.bias', 'cls.predictions.transform.dense.bias')
    _replace('lm_head.layer_norm.weight', 'cls.predictions.transform.LayerNorm.weight')
    _replace('lm_head.layer_norm.bias', 'cls.predictions.transform.LayerNorm.bias')
    _replace('lm_head.decoder.weight', 'cls.predictions.decoder.weight')


def rename_tf_layernorm(state_dict):
    for k in list(state_dict.keys()):
        if 'LayerNorm.' in k:
            new_k = k.strip().replace('LayerNorm.beta', 'LayerNorm.bias')
            new_k = new_k.strip().replace('LayerNorm.gamma', 'LayerNorm.weight')
            state_dict[new_k] = state_dict[k]
            if new_k != k:
                del state_dict[k]


def load_params_choose_layers(prefix: str, state_dict: dict, mapper: dict):
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            new_k = None
            for i in mapper.keys():
                if k.startswith(f'{prefix}.{i}.'):
                    new_k = k.replace(f'{prefix}.{i}.', f'{prefix}.{mapper[i]}.')
                    break

            if new_k:
                state_dict[new_k] = state_dict[k]

            del state_dict[k]

    return state_dict


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


def get_visual_tokenizer(tokenizer_model, tokenizer_weight, image_res, codebook_size, codebook_dim):
    print(f"Creating visual tokenizer: {tokenizer_model}")
    if tokenizer_model == 'vqkd_encoder_base_decoder_3x768x12_clip':
        model = vqkd_encoder_base_decoder_3x768x12_clip(pretrained=True, pretrained_weight=tokenizer_weight, 
                                                        as_tokenzer=True, img_size=image_res, 
                                                        n_code=codebook_size, code_dim=codebook_dim).eval()
    else:
        raise ValueError
    return model


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )


def build_vision_encoder(config, load_params=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2

    if config.get('use_clip_vit', False):  # good performance, but only base model available
        from models.clip_vit import CLIPVisionTransformer, interpolate_pos_embed

        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        vision_encoder = CLIPVisionTransformer(image_size=config['image_res'], patch_size=vision_config['patch_size'],
                                               hidden_size=vision_config['vision_width'],
                                               hidden_act=vision_config['hidden_act'],
                                               num_attention_heads=vision_config['num_attention_heads'],
                                               attention_dropout=vision_config['attention_dropout'],
                                               intermediate_size=vision_config['intermediate_size'],
                                               num_hidden_layers=vision_config['num_hidden_layers'],
                                               local_attn_depth=vision_config['local_attn_depth'])

        if load_params:
            # download from https://huggingface.co/openai/clip-vit-base-patch16/tree/main
            state_dict_orig = torch.load(vision_config['ckpt'], map_location="cpu")
            state_dict = {}
            for k, v in state_dict_orig.items():
                if k.startswith('vision_model.'):
                    k = k[13:]
                    if k.startswith('embeddings.'):
                        k = k[11:]
                        k = k.replace('patch_embedding.weight', 'patch_embed.weight')
                        k = k.replace('position_embedding.weight', 'pos_embed.weight')

                    if k != 'position_ids':
                        state_dict[k] = v

            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed.weight'].unsqueeze(dim=0),
                                                       num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

            assert vision_config['num_hidden_layers'] in [6, 12], "param initialization not implemented"
            if vision_config['num_hidden_layers'] == 6:
                mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                load_params_choose_layers('encoder.layers', state_dict, mapper)


    elif config.get('use_swin', False):
        from models.swin_transformer import SwinTransformer, interpolate_relative_pos_embed

        vision_config = read_json(config['vision_config'])
        assert config['image_res'] == vision_config['image_res']
        assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']

        vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                         patch_size=4,
                                         in_chans=3,
                                         embed_dim=vision_config['embed_dim'],
                                         depths=vision_config['depths'],
                                         num_heads=vision_config['num_heads'],
                                         window_size=vision_config['window_size'],
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         drop_rate=0.0,
                                         drop_path_rate=0.1,
                                         ape=False,
                                         patch_norm=True,
                                         use_checkpoint=False)

        if load_params:
            # download from https://github.com/microsoft/Swin-Transformer
            state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

            for k in list(state_dict.keys()):
                if 'relative_position_bias_table' in k:
                    dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                    state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                elif ('relative_position_index' in k) or ('attn_mask' in k):
                    del state_dict[k]

    elif config.get('use_beit_v2', False):

        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']
        num_masking_patches = config.get('num_masking_patches', 75)
        min_num_patches = config.get('min_num_patches', 16)

        if 'base' in config['vision_config']:
            from models.beit2 import beit_base_patch16 as beit_model
        elif 'large' in config['vision_config']:
            from models.beit2 import beit_large_patch16 as beit_model
        else:
            raise ValueError

        vision_encoder = beit_model(img_size=config['image_res'],
                                    drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
                                    use_mean_pooling=True,
                                    init_scale=0.001,
                                    use_rel_pos_bias=True, use_abs_pos_emb=False,
                                    init_values=0.1, qkv_bias=True, local_attn_depth=config['local_attn_depth'],
                                    num_masking_patches=num_masking_patches, min_num_patches=min_num_patches,)

        if load_params:
            from models.beit2 import load_pretrained_beit2
            load_pretrained_beit2(vision_encoder, vision_config['ckpt'])  
    
    else:  # deit, worse than clip-vit/swin...
        raise ValueError
        # assert config['patch_size'] == 16
        # vision_width = 768

        # vision_encoder = VisionTransformer(
        #     img_size=config['image_res'], patch_size=config['patch_size'], embed_dim=768, depth=12, num_heads=12,
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        #     local_attn_depth=4)

        # if load_params:
        #     # download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
        #     state_dict = torch.load("data/deit_base_patch16_224-b5f2ef4d.pth", map_location="cpu")["model"]
        #     pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], num_patches=num_patches, num_extra_tokens=1)
        #     state_dict['pos_embed'] = pos_embed_reshaped

    if load_params and (not config.get('use_beit_v2', False)):
        print("### Load ViT: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)

    return vision_encoder, vision_width


def build_text_encoder(config, vision_width, load_text_params=False, use_mlm_loss=False, config_text=None):

    if 'roberta' in config['text_encoder']:
        from models.xroberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
    elif 'bert' in config['text_encoder']:
        from models.xbert import BertConfig, BertForMaskedLM, BertModel
    else:
        raise ValueError


    init_params = []  # train from scratch with larger lr

    if config_text is None:

        if 'roberta' in config['text_encoder']:
            config_text = RobertaConfig.from_json_file(os.path.join(config['text_encoder'], 'config.json'))
        else:
            config_text = BertConfig.from_json_file(os.path.join(config['text_encoder'], 'config.json'))

        # set configs
        config_text.num_hidden_layers = config.get('text_num_hidden_layers', 12)
        if config_text.num_hidden_layers == 0:
            print("Attention!!! text_num_hidden_layers is setting to zero!!!")
        config_text.fusion_layer = config.get('text_fusion_start_at', config_text.num_hidden_layers // 2)

    else:
        assert isinstance(config_text, BertConfig)

    config_text.encoder_width = vision_width

    if use_mlm_loss:
        # assert load_text_params is True  # for domain pre-training
        if ('accelerator' in config.keys()) and (config['accelerator']['FP16_OPT_LEVEL'] != 'O0'):
            config_text.fp16 = True  # will use some operations to avoid gradient overflow

        if 'roberta' in config['text_encoder']:
            text_encoder = RobertaForMaskedLM(config=config_text)
        else:
            text_encoder = BertForMaskedLM(config=config_text)

        if load_text_params:
            print("### Initializing text encoder from ", os.path.join(config['text_encoder'], 'pytorch_model.bin'))
            state_dict = torch.load(os.path.join(config['text_encoder'], 'pytorch_model.bin'))

            if 'roberta' in config['text_encoder']:  # 基于 xroberta.py 的导入方式。现在我们采纳的。
                if ('xlm-roberta-large' in config['text_encoder']) and (config_text.num_hidden_layers == 12):
                    # I use 12 layers from in total 24 layers

                    # for Cross-View Language Modeling (https://arxiv.org/abs/2206.00621)
                    # i need 1024d hidden states to encode both image-caption pairs and parallel text pairs

                    mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8, 19: 9, 21: 10, 23: 11}
                    load_params_choose_layers('roberta.encoder.layer', state_dict, mapper)

                elif ('roberta-large' in config['text_encoder']) and (config_text.num_hidden_layers == 12):
                    mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8, 19: 9, 21: 10, 23: 11}
                    load_params_choose_layers('roberta.encoder.layer', state_dict, mapper)

            else:
                if 'bert-base-uncased-6l' in config['text_encoder']:
                    assert (config['text_num_hidden_layers'] == 12) and (config['text_fusion_start_at'] == 6), "otherwise, not implemneted"
                    mapper = {0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11}  # bert-base-6L, 两次初始化前6层和后6层
                    load_params_choose_layers('bert.encoder.layer', state_dict, mapper, do_expand=True)

                elif 'bert-base-uncased-18l' in config['text_encoder']:
                    assert (config['text_num_hidden_layers'] == 18) and (
                                config['text_fusion_start_at'] == 12), "otherwise, not implemneted"

                elif 'bert-base-uncased' in config['text_encoder']:
                    rename_tf_layernorm(state_dict)
                    if (config_text.num_hidden_layers == 6) and ('text_fusion_start_at' not in config):  # 在训练自己的bert
                        mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                        load_params_choose_layers('bert.encoder.layer', state_dict, mapper)

                    elif config_text.num_hidden_layers == 18:
                        assert config['text_fusion_start_at'] == 12
                        mapper = {6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 17}
                        load_params_choose_layers('bert.encoder.layer', state_dict, mapper, do_expand=True)

                    elif (config_text.num_hidden_layers == 18) and ('text_fusion_start_at' not in config):  # 在训练自己的bert
                        mapper = {6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 17}
                        load_params_choose_layers('bert.encoder.layer', state_dict, mapper, do_expand=True)

                elif 'bert-large-uncased-18l' in config['text_encoder']:
                    assert (config['text_num_hidden_layers'] == 18) and (
                                config['text_fusion_start_at'] == 12), "otherwise, not implemneted"

                elif 'bert-large-uncased-12l' in config['text_encoder']:
                    if config['text_num_hidden_layers'] == 18:
                        assert config['text_fusion_start_at'] == 12
                        mapper = {6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 17}
                        load_params_choose_layers('bert.encoder.layer', state_dict, mapper, do_expand=True)

                    elif config['text_num_hidden_layers'] == 24:
                        assert config['text_fusion_start_at'] == 12
                        mapper = {i: 12+i for i in range(12)}
                        load_params_choose_layers('bert.encoder.layer', state_dict, mapper, do_expand=True)

                    else:
                        raise NotImplementedError

                elif 'bert-large-uncased' in config['text_encoder']:  # large 写这里啦，先随便点。
                    rename_tf_layernorm(state_dict)

                    if config_text.num_hidden_layers == 12:
                        mapper = {layer: i for i, layer in enumerate(list(range(1, 24+1, 2)))}
                        load_params_choose_layers('bert.encoder.layer', state_dict, mapper)

                    elif config_text.num_hidden_layers == 18:
                        mapper = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10, 17: 11,
                                  18: 12, 19: 13, 20: 14, 21: 15, 22: 16, 23: 17}
                        load_params_choose_layers('bert.encoder.layer', state_dict, mapper)
                    elif config_text.num_hidden_layers == 24:
                        pass  # large 参数全部导入就是

                    elif config_text.num_hidden_layers == 30:
                        # 独立的6层cross, 只是这个branch里面没分开。使用后6层
                        mapper = {18: 24, 19: 25, 20: 26, 21: 27, 22: 28, 23: 29}
                        load_params_choose_layers('bert.encoder.layer', state_dict, mapper, do_expand=True)
                    else:
                        raise NotImplementedError

                elif 'chinese-roberta-wwm-ext' in config['text_encoder']:
                    if config_text.num_hidden_layers == 6:
                        mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                        load_params_choose_layers('bert.encoder.layer', state_dict, mapper)

                else:
                    raise NotImplementedError

                # if config_text.num_hidden_layers == 6:
                #     mapper = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}
                #     load_params_choose_layers('bert.encoder.layer', state_dict, mapper)

            msg = text_encoder.load_state_dict(state_dict, strict=False)
            print("missing_keys: ", msg.missing_keys, flush=True)
            print("unexpected_keys: ", msg.unexpected_keys, flush=True)
            init_params += [f'text_encoder.{k}' for k in msg.missing_keys]

    else:  # for fine-tuning, not load_text_params by default
        assert load_text_params is False

        if 'roberta' in config['text_encoder']:
            text_encoder = RobertaModel(config=config_text, add_pooling_layer=False)
        else:
            text_encoder = BertModel(config=config_text, add_pooling_layer=False)

    return text_encoder, init_params


def load_pretrained(model, ckpt_rpath, config, is_eval=False, load_text=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

    if is_eval:
        return state_dict

    num_patches = (config['image_res'] // config['patch_size']) ** 2

    print("### Loading pretrained vision encoder", flush=True)
    if config.get('use_clip_vit', False):
        from models.clip_vit import interpolate_pos_embed
        del state_dict['vision_encoder.position_ids']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed.weight'].unsqueeze(dim=0),
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    elif config.get('use_swin', False):
        from models.swin_transformer import interpolate_relative_pos_embed
        window_size = read_json(config['vision_config'])['window_size']

        for k in list(state_dict.keys()):
            if 'relative_position_bias_table' in k:
                dst_num_pos = (2 * window_size - 1) ** 2
                state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
            elif ('relative_position_index' in k) or ('attn_mask' in k):
                del state_dict[k]

    elif config.get('use_beit_v2', False):
        from models.beit2 import interpolate_pos_embed

        vision_state_dict = {}
        for k in list(state_dict.keys()):
            if k.startswith('vision_encoder.'):
                vision_state_dict[k[15:]] = state_dict[k]
                del state_dict[k]

        vision_state_dict = interpolate_pos_embed(model.vision_encoder, vision_state_dict)
        for k in vision_state_dict.keys():
            state_dict['vision_encoder.' + k] = vision_state_dict[k]

    else:
        raise ValueError

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
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

    return state_dict


class XFMBase(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=False,
                 use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                 config_text=None):
        super().__init__()
        self.init_params = []  # train from scratch with larger lr
        self.vision_encoder, vision_width = build_vision_encoder(config, load_params=load_vision_params)
        self.text_encoder, init_params = build_text_encoder(config, vision_width=vision_width, load_text_params=load_text_params,
                                                            use_mlm_loss=use_mlm_loss,
                                                            config_text=config_text)  # text & cross-modal
        self.init_params.extend(init_params)
        self.num_text_layers = self.text_encoder.config.fusion_layer
        self.num_cross_layers = self.text_encoder.config.num_hidden_layers - self.num_text_layers

        self.vision_width = vision_width
        self.text_width = self.text_encoder.config.hidden_size  # i.e. cross_width
        self.use_vision_tokenizer = config.get('use_vision_tokenizer', False)
        if config.get('use_vision_tokenizer', False):
            tokenizer_model = config['tokenizer_model']
            image_res = config['image_res']
            tokenizer_weight = config['tokenizer_weight']
            codebook_size = config['codebook_size']
            codebook_dim = config['codebook_dim']
            self.vqkd = get_visual_tokenizer(tokenizer_model, tokenizer_weight, image_res, codebook_size, codebook_dim)
            self.lm_head = nn.Linear(self.vision_width, codebook_size)
            self.loss_ce = nn.CrossEntropyLoss()

        if use_contrastive_loss:
            self.embed_dim = config['embed_dim']
            self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
            self.text_proj = nn.Linear(self.text_width, self.embed_dim)
            self.init_params.extend(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
            self.init_params.extend(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])

            self.learnable_temp = config.get('learnable_temp', True)
            if not self.learnable_temp:
                self.temp = config.get('temp', 0.07)
            else:
                self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            self.init_params.extend(['temp'])

        if use_matching_loss:
            self.itm_head = build_mlp(input_dim=self.text_width, output_dim=2)
            self.init_params.extend(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])

        if use_bbox_loss:
            self.bbox_head = build_mlp(input_dim=self.text_width, output_dim=4)
            self.init_params.extend(['bbox_head.' + n for n, _ in self.bbox_head.named_parameters()])

        # check
        named_parameters = set([n for n, _ in self.named_parameters()])
        for n in set(self.init_params):
            if n not in named_parameters:
                print(f"warning: {n} not in named_parameters")
                self.init_params.remove(n)

        from models.xroberta import RobertaForMaskedLM, RobertaConfig
        config_fusion = RobertaConfig.from_json_file(os.path.join(config['text_encoder'], 'config.json'))
        config_fusion.num_hidden_layers = config['fusion_num_hidden_layers']
        config_fusion.fusion_layer = config['fusion_fusion_start_at']
        config_fusion.encoder_width = self.vision_width
        self.fusion_layers = config_fusion.num_hidden_layers
        self.text_layers = config['text_num_hidden_layers']
        self.fusion_encoder = RobertaForMaskedLM(config=config_fusion)
        self.detach_text_forMLM = config.get('detach_text_forMLM', True)
        self.mim_cls_only = config.get('mim_cls_only', False)
        if self.vision_width != self.text_width:
            self.fusion_proj = nn.Linear(self.text_width, self.vision_width)
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        

    def load_pretrained(self, ckpt_rpath, config, is_eval=False, is_domain_pretrain=False):
        if is_domain_pretrain:
            checkpoint = torch.load(ckpt_rpath, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
            for key in list(state_dict.keys()):
                if 'visual_encoder' in key:
                    encoder_key = key.replace('visual_encoder','vision_encoder')         
                    state_dict[encoder_key] = state_dict[key] 
                    del state_dict[key]
        else:
            state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=is_eval, load_text=True)

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)


    def get_vision_embeds(self, image, image_atts=None, idx_to_group_img=None, do_mask=False):
        """
        vision_embeds: cls + patch embeds
        """
        if idx_to_group_img is None:
            if do_mask:
                image_embeds, id_masked = self.vision_encoder(image, do_mask=do_mask)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                return image_embeds, image_atts, id_masked  # full attention
            else:
                image_embeds = self.vision_encoder(image)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                return image_embeds, image_atts  # full attention

        else:  # image < bsz
            if image_atts is None:
                image_embeds_fullatts = self.vision_encoder(image)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))  # expend to bsz

                image_atts = torch.ones(image_embeds_fullatts.size()[:-1], dtype=torch.long).to(image.device)

                return image_embeds_fullatts, image_atts

            else:
                assert image_atts.size(0) == idx_to_group_img.size(0)  # bsz
                image_embeds, image_embeds_fullatts = \
                    self.vision_encoder(image, idx_to_group_img=idx_to_group_img, image_atts=image_atts)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))

                return image_embeds, image_atts, image_embeds_fullatts


    def get_text_embeds(self, text_ids, text_atts):
        """
        12-L text encoder, i.e. cross-encoder w/o image input
        """
        assert text_atts is not None
        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        return encoder(text_ids,
                       attention_mask=text_atts,
                       encoder_hidden_states=None,
                       encoder_attention_mask=None,
                       return_dict=True,
                       ).last_hidden_state


    def get_features(self, image_embeds=None, text_embeds=None):
        if image_embeds is None:
            return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        elif text_embeds is None:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        else:
            return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1), \
                   F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)


    def get_mim_loss(self, image_embeds_masked, targets, mask_tokens):
        if self.use_vision_tokenizer:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    input_ids = self.vqkd.get_codebook_indices(targets)            
            return self.loss_ce(self.lm_head(image_embeds_masked[:,1:,:][mask_tokens]), input_ids[mask_tokens])

        targets = targets.detach()
        if self.mim_cls_only:
            return self.loss_fn(image_embeds_masked[:,1:,:][mask_tokens], targets[:,1:,:][mask_tokens])
        else:
            return self.loss_fn(image_embeds_masked[:,1:,:][mask_tokens], targets[:,1:,:][mask_tokens]) + self.loss_fn(image_embeds_masked[:,0,:], targets[:,0,:])


    def get_fuse_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):

        bsz, _ = text_ids_masked.shape
        text_ids_masked = text_ids_masked[:bsz,:]
        text_atts = text_atts[:bsz,:]
        image_embeds = image_embeds[:bsz,:, :]
        image_atts = image_atts[:bsz,:]
        masked_pos = masked_pos[:bsz,:]
        masked_ids = masked_ids[:bsz,:]
        encoder_embeds = self.get_text_embeds(text_ids_masked, text_atts)
        if self.detach_text_forMLM:
            encoder_embeds = encoder_embeds.detach()
        return self.fusion_encoder(encoder_embeds=encoder_embeds,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True,
                                 labels=masked_ids,
                                 masked_pos=masked_pos).loss


    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None, is_pretrain=True):
        """
        12-L cross encoder
        text_ids as input is required.
        """
        encoder = self.fusion_encoder.bert if hasattr(self.fusion_encoder, 'bert') else self.fusion_encoder
        
        if text_embeds is None:
            return encoder(text_ids,
                        attention_mask=text_atts,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                        ).last_hidden_state
        else:
            encoder_embeds = text_embeds.detach() if is_pretrain else text_embeds
            return encoder(encoder_embeds=encoder_embeds,
                        attention_mask=text_atts,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                        ).last_hidden_state            


    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        """
        Args:
            image_feat, text_feat: normalized

        Returns: contrastive loss

        """
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_hard_negatives(self, image_feat, text_feat, idx=None):
        bs = image_feat.size(0)
        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_neg_idx = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_neg_idx.append(neg_idx)

        text_neg_idx = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_neg_idx.append(neg_idx)

        return image_neg_idx, text_neg_idx


    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_ids, text_atts, text_feat, idx=None,
                          return_cross_embeds=False, text_embeds=None, is_pretrain=True):
        """
        Matching Loss with hard negatives
        """
        assert text_ids.dim() == 2, "X-Brain uses text_ids for matching."

        image_neg_idx, text_neg_idx = self.get_hard_negatives(image_feat, text_feat, idx=idx)

        bs = image_feat.size(0)
        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = image_neg_idx[b]
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_ids_neg = []
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = text_neg_idx[b]
            text_ids_neg.append(text_ids[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
            text_embeds_neg.append(text_embeds[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)

        text_ids_all = torch.cat([text_ids, text_ids_neg], dim=0)
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        cross_pos = self.get_cross_embeds(image_embeds, image_atts, text_ids=text_ids, text_atts=text_atts, 
                                            is_pretrain=is_pretrain, text_embeds=text_embeds)[:, 0,
                    :]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all, text_ids=text_ids_all,
                                          text_atts=text_atts_all, text_embeds=text_embeds_all,
                                          is_pretrain=is_pretrain)[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)

        if return_cross_embeds:
            return F.cross_entropy(output, itm_labels), cross_pos

        return F.cross_entropy(output, itm_labels)


    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):
        return self.text_encoder(text_ids_masked,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True,
                                 labels=masked_ids,
                                 masked_pos=masked_pos).loss


    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes


    def predict_bbox(self, image_embeds, text_ids, text_atts, text_embeds, is_pretrain=True):
        """
        Returns:
            output_coord: bsz, 4
        """
        assert image_embeds.size(0) == text_ids.size(0) == text_atts.size(0)

        output_cls = self.get_cross_embeds(image_embeds, torch.ones(image_embeds.shape[:2]).to(image_embeds.device),
                                           text_ids=text_ids, text_atts=text_atts, text_embeds=text_embeds, is_pretrain=is_pretrain)[:, 0, :]

        output_coord = self.bbox_head(output_cls).sigmoid()
        return output_coord