# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from dotmap import DotMap
from torch import nn, Tensor

import main.globals as g
from main.detr.models.misc_nets import MLP
from utils.file_handler import FileHandler

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos, src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask, pos=pos, src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first attention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TRXREFINE01(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(TRXREFINE01, self).__init__()

        # stem net
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.transformer.dim_model, nhead=cfg.transformer.n_head,
            dim_feedforward=cfg.transformer.dim_feedforward,
            activation='relu')
        self.global_encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                                 num_layers=cfg.transformer.encoder_layers)

        self.backbone = self.load_backbone(backbone=cfg.backbone)
        w, h = cfg.image_size
        self._make_position_embedding(w, h, cfg.transformer.dim_model, cfg.transformer.pos_embedding)
        self.final_layer = MLP(input_dim=4096, hidden_dim=1024, output_dim=2, num_layers=3)
        self.mm = MLP(input_dim=4096, hidden_dim=1024, output_dim=68, num_layers=3)

    @staticmethod
    def load_backbone(backbone, **kwargs):
        if backbone == g.HRNET:
            from models.HRNET.HRNET import get_face_alignment_net
            hrnet_config = DotMap(FileHandler.load_yaml(g.HRNET_ARGS))
            hrnet = get_face_alignment_net(config=hrnet_config.hrnet_args)
        return hrnet

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 4
                self.pe_w = w // 4
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(self._make_sine_position_embedding(d_model), requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000, scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = F.interpolate(pos.flatten(2), 4096).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def forward(self, x):
        # input is bsx256,256,3
        x = self.backbone.forward(x)  # bsx68x64x64

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.global_encoder(x.cuda(), pos=self.pos_embedding.cuda())
        enc_out = self.mm(x.permute(1, 2, 0))

        x = self.final_layer(x.permute(1, 2, 0)) * 255
        return x, enc_out


def freeze_training_on_block_(model):
    for param in model.parameters():
        param.requires_grad = False


def get_pose_net_trxrefine01(cfg, **kwargs):
    model = TRXREFINE01(cfg, **kwargs)
    freeze_training_on_block_(model.backbone)
    return model
