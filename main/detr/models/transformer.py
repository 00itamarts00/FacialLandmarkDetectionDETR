# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from main.detr.models.misc_nets import MLP


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, num_encoder_layers=6,  num_encoder_blocks=1,
                 num_decoder_layers=6, num_decoder_blocks=1, normalize_before=False,
                 return_intermediate=False, decoder_blocks_weight_sharing=True,
                 dropout=0.1, activation="relu"):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        encoder_block = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder = encoder_block

        self.decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                     dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)

        # decoder_block = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                    return_intermediate=return_intermediate)
        #
        # pre_decoder = PreDecoderBlock(input_dim=[68, 2], hidden_dim=d_model, output_dim=40)
        # post_decoder = PostDecoderBlock(hidden_dim=d_model, sfactor=256)
        # TODO: apply query_pos for the decoder. Init it s.t. it will be the position of all the average faces
        self.decoder = MultipleDecoder(decoder_layer=self.decoder_layer, hidden_dim=d_model,
                                       num_blocks=num_decoder_blocks, num_decoder_layers=num_decoder_layers,
                                       weight_sharing=decoder_blocks_weight_sharing,
                                       return_intermediate=return_intermediate)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.permute(0, 2, 1, 3), memory.permute(1, 2, 0).view(bs, c, h, w)


# TODO change name with MultipleDecoder
class MultipleDecoder(nn.Module):
    def __init__(self, decoder_layer, hidden_dim, num_decoder_layers, num_blocks, weight_sharing, return_intermediate,
                 sfactor=256):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.num_decoder_layers = num_decoder_layers
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.weight_sharing = weight_sharing
        self.return_intermediate = return_intermediate
        self.sfactor = sfactor
        self.nn_list = nn.ModuleList()
        self.pre_decoder = self.gen_pre_decoder()  # input: bs x n_lndmk x 2 / output : n_lndmk x bs x hidden_dim
        self.decoder_block = self.gen_decoder_block()  # input: n_lndmk x bs x hidden_dim, memory, memory_key_padding_mask, pos, query_pos
        self.post_decoder = self.gen_post_decoder()     # input: n_lndmk x bs x hidden_dim / output : bs x n_lndmk x 2
        self.multi_decoder_blocks = self.init_multi_decoder_net()

    def gen_pre_decoder(self):
        return PreDecoderBlock(input_dim=[68, 2], hidden_dim=self.hidden_dim)

    def gen_post_decoder(self):
        return PostDecoderBlock(hidden_dim=self.hidden_dim, sfactor=self.sfactor)

    def gen_decoder_block(self):
        decoder_norm = nn.LayerNorm(self.hidden_dim)
        return TransformerDecoder(self.decoder_layer, self.num_decoder_layers, decoder_norm,
                                  return_intermediate=self.return_intermediate)

    def init_multi_decoder_net(self):
        # input n_lndmk x bs x hidden_dim, (memory, memory_key_padding_mask, pos, query_pos)
        net = {}
        for stage_num in range(self.num_blocks):
            net[stage_num] = {}
            if stage_num == 0:
                net[stage_num]['pre_decoder'] = None
            else:
                self.nn_list.append(self.pre_decoder if self.weight_sharing else self.gen_pre_decoder())
                net[stage_num]['pre_decoder'] = self.pre_decoder if self.weight_sharing else self.gen_pre_decoder()
            self.nn_list.append(self.decoder_block if self.weight_sharing else self.gen_decoder_block())
            net[stage_num]['decoder_block'] = self.decoder_block if self.weight_sharing else self.gen_decoder_block()
            self.nn_list.append(self.post_decoder if self.weight_sharing else self.gen_post_decoder())
            net[stage_num]['post_decoder'] = self.post_decoder if self.weight_sharing else self.gen_post_decoder()
        return net

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt
        intermediate = []

        for i, stage in enumerate(self.nn_list):
            if stage._get_name() == 'TransformerDecoder':
                output = stage(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
                output = output[-1] if output.ndim == 4 else output
            else:
                output = stage(output)
                if stage._get_name() == 'PostDecoderBlock' and self.return_intermediate:
                    intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class PreDecoderBlock(nn.Module):
    # turns output of decoder/encoder block to num_landmarks x 2
    def __init__(self, input_dim, hidden_dim, num_layer=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pre_dec = MLP(2, hidden_dim // 2, hidden_dim, num_layers=num_layer).cuda()

    def forward(self, x):
        return self.pre_dec(x)


class PostDecoderBlock(nn.Module):
    # turns output of decoder/encoder block to num_landmarks x 2
    def __init__(self, hidden_dim, sfactor):
        super().__init__()
        self.sfactor = sfactor
        self.coord_embed = MLP(hidden_dim, hidden_dim // 2, 2, 3)

    def forward(self, hs):
        return self.coord_embed(hs).sigmoid() * self.sfactor * 1.2 + 0.5


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, multi_enc_loss=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            output = self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        else:
            output = self.forward_post(src, src_mask, src_key_padding_mask, pos)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_encoder_blocks=args.encoder_blocks,
        num_decoder_layers=args.dec_layers,
        num_decoder_blocks=args.decoder_blocks,
        normalize_before=args.pre_norm,
        return_intermediate=args.return_intermediate,
        decoder_blocks_weight_sharing=args.decoder_blocks_weight_sharing,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
