# -*- coding: utf-8 -*-
# Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks (https://arxiv.org/abs/2301.05065)
# Github: https://github.com/zhangxinsong-nlp/XFM
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.
# By Xinsong Zhang
# Based on X-VLM code base
# https://github.com/zengyan-97/X-VLM

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import build_tokenizer
from models.xbert import BertLMHeadModel
from models.xroberta import RobertaForCausalLM
from models import XFMBase, load_pretrained


class XFMForVQA(XFMBase):
    """
    Generative Model, but model the task as ranking at inference
    Use XVLMForGeneration for purely generative tasks.
    """
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                         config_text=None)

        assert isinstance(config['pad_token_id'], int)
        self.pad_token_id = config['pad_token_id']
        config_enc = self.text_encoder.config

        if 'roberta' in config['text_encoder']:
            from models.xroberta import RobertaConfig
            config_dec = RobertaConfig.from_json_file(os.path.join(config['text_encoder'], 'config.json'))
        else:
            config_dec = copy.deepcopy(config_enc)

        config_dec.encoder_width = config_enc.hidden_size
        config_dec.fusion_layer = config['decoder_fusion_start_at']  # start index
        config_dec.num_hidden_layers = config['num_dec_layers']
        self.cross_encoder_width = config_enc.encoder_width  # i.e. vision_width
        self.dec_encoder_width = config_enc.hidden_size

        if 'roberta' in config['text_encoder']:
            from models.xroberta import RobertaForCausalLM
            self.text_decoder = RobertaForCausalLM(config=config_dec)
        else:
            self.text_decoder = BertLMHeadModel(config=config_dec)

        if self.dec_encoder_width != self.cross_encoder_width:
            self.init_params = ['text_decoder.' + n for n, _ in self.text_decoder.named_parameters()
                                if ('crossattention.self.key' in n) or ('crossattention.self.value' in n)]
        else:
            self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(self, ckpt_rpath, config, load_text=False)

            print("### Loading pretrained text encoder", flush=True)
            for key in list(state_dict.keys()):

                name_to_replace = 'bert.'

                # TODO: 我在试 xroberta.py
                if 'roberta' in config['text_encoder']:
                    name_to_replace = 'roberta.'

                if name_to_replace in key and 'text_encoder' in key:
                    encoder_key = key.replace(name_to_replace, '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

                    # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                if 'fusion_encoder.' in key:
                    decoder_key = key.replace('fusion_encoder', 'text_decoder')
                    state_dict[decoder_key] = state_dict[key]
                    # del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, quesiton, answer=None, k=None, weights=None, train=True):
        image_embeds, image_atts = self.get_vision_embeds(image)
        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        if train:
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.pad_token_id, -100)
            text_embeds = encoder(quesiton.input_ids,
                                    attention_mask=quesiton.attention_mask,
                                    encoder_hidden_states=None,
                                    encoder_attention_mask=None,
                                    return_dict=True,
                                    ).last_hidden_state
            question_output = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, 
                                                    text_atts=quesiton.attention_mask, is_pretrain=False)
            
            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output[b]] * n
                question_atts += [quesiton.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)
            
            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_states,
                                              encoder_attention_mask=question_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )
            loss = weights * answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss

        else:
            text_embeds = encoder(quesiton.input_ids,
                                    attention_mask=quesiton.attention_mask,
                                    encoder_hidden_states=None,
                                    encoder_attention_mask=None,
                                    return_dict=True,
                                    ).last_hidden_state
            question_output = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, 
                                                    text_atts=quesiton.attention_mask, is_pretrain=False)
            question_atts = torch.ones(question_output.size()[:-1], dtype=torch.long).to(question_output.device)
            topk_ids, topk_probs = self.rank_answer(question_output, question_atts,
                                                    answer.input_ids, answer.attention_mask, k)
            return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token
        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

class XFMForCaptioningDomainPretrain(XFMBase):
    """
    domain pre-train for image captioning
    Attention!!! not debug yet
    """
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False, config_text=None)

        self.pad_token_id = config['pad_token_id']
        config_enc = self.text_encoder.config

        self.text_encoder = None
        self.text_decoder = BertLMHeadModel(config=config_enc)

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(self, ckpt_rpath, config, load_text=False)

            print("### Loading pretrained text encoder", flush=True)
            for key in list(state_dict.keys()):
                assert isinstance(key, str)
                if key.startswith('text_encoder.'):
                    decoder_key = key.replace('text_encoder.', 'text_decoder.')
                    state_dict[decoder_key] = state_dict[key]
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts):
        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        decoder_targets = text_ids.masked_fill(text_ids == self.pad_token_id, -100)

        loss = self.text_decoder(text_ids,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 labels=decoder_targets,
                                 return_dict=True,
                                 ).loss

        return loss


class XFMForCaptioning(XFMBase):
    """
    generation based on images
    Attention!!! not debug yet
    """
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False, config_text=None)

        self.tokenizer = build_tokenizer(config['text_encoder'])
        self.tokenizer.add_special_tokens({'bos_token': self.tokenizer.cls_token, 'eos_token': self.tokenizer.sep_token})

        self.prompt = config['prompt']
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1
        self.max_tokens = config['max_tokens']

        config_enc = self.text_encoder.config

        self.text_encoder = None
        self.text_decoder = RobertaForCausalLM(config=config_enc, label_smoothing=config['label_smoothing'])

    def load_pretrained(self, ckpt_rpath, config, load_capt_pretrain=False, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(ckpt_rpath, config, load_text=False)
            print("### Loading pretrained text encoder", flush=True)
            print("load_capt_pretrain, ", load_capt_pretrain)
            if not load_capt_pretrain:
                print("### Loading pretrained text encoder", flush=True)
                for key in list(state_dict.keys()):
                    assert isinstance(key, str)
                    if key.startswith('text_encoder.'):
                        decoder_key = key.replace('text_encoder.', 'text_decoder.')
                        state_dict[decoder_key] = state_dict[key]
                        del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, caption):
        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=self.max_tokens, return_tensors="pt").to(
            image.device)

        # text.input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        loss = self.text_decoder(text.input_ids,
                                 attention_mask=text.attention_mask,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 labels=decoder_targets,
                                 return_dict=True,
                                 ).loss

        return loss

    def generate(self, image, sample=False, num_beams=1, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0, num_return_sequences=1, greedy=False):

        prompt = [self.prompt] * image.size(0)

        image_embeds = self.vision_encoder(image)

        if num_beams > 1:
            assert (sample is False) and (num_return_sequences == 1)
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        if num_return_sequences > 1:
            assert (sample is True) and (num_beams == 1)
            image_embeds = image_embeds.repeat_interleave(num_return_sequences, dim=0)
            prompt = [self.prompt] * image_embeds.size(0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        # input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        def _get_captions(caption_ids):
            captions = []
            for output in caption_ids:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[len(self.prompt):])
            return captions

        if greedy:
            # greedy generation from OSCAR
            assert (num_beams == 1) and (num_return_sequences == 1)
            outputs, logprobs = self.text_decoder._generate_no_beam_search(input_ids=input_ids, cur_len=input_ids.shape[1], max_length=max_length,
                                          do_sample=False, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          pad_token_id=self.tokenizer.pad_token_id, eos_token_ids=[self.tokenizer.sep_token_id],
                                          batch_size=image_embeds.size(0), **model_kwargs)

            return _get_captions(outputs)

        elif sample:
            # sampling from OSCAR
            outputs, logprobs = self.text_decoder._generate_no_beam_search(input_ids=input_ids, cur_len=input_ids.shape[1], max_length=max_length,
                                          do_sample=True, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          pad_token_id=self.tokenizer.pad_token_id, eos_token_ids=[self.tokenizer.sep_token_id],
                                          batch_size=image_embeds.size(0), **model_kwargs)

            # outputs: (bs x num_return_sequences, max_length)
            # logprobs: (bs x num_return_sequences,)

            return _get_captions(outputs), logprobs

        else:
            # beam search from huggingface
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

            return _get_captions(outputs)


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))