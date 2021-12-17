#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020/11/17 21:37
@Author  : Tomas S. Fang
@Email   : fangsen1996@gmail.com
@File    : Layers.py
@Software: PyCharm
"""


from abc import ABC

import torch.nn as nn

from transformer.SubLayers import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer


class EncoderLayer(nn.Module, ABC):
    """
    Encoder layer in composed with self-attention layer and position-wise
    feed forward network.
    """
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, use_bias=None):

        # self attention
        residual1 = src
        src = self.layer_norm(src)
        _src, _ = self.self_attention(src, src, src, src_mask, use_bias=use_bias)

        # residual connection
        src = residual1 + self.dropout(_src)

        # position-wise feedforward
        residual2 = src
        src = self.layer_norm(src)
        _src = self.positionwise_feedforward(src)

        # residual connection
        src = residual2 + self.dropout(_src)

        return src


class DecoderLayer(nn.Module, ABC):
    """
    Decoder layer is composed with self-attention network, encoder-decoder network, and
    position-wise feed forward network. We use pre-layer-normalization to replace post-
    layer-normalization.
    """
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask=None, src_mask=None):

        # self attention
        residual1 = trg
        trg = self.layer_norm(trg)
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # residual connection
        trg = residual1 + self.dropout(_trg)

        # encoder-decoder attention
        residual2 = trg
        trg = self.layer_norm(trg)
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # residual connection
        trg = residual2 + self.dropout(_trg)

        # position-wise feedforward
        residual3 = trg
        trg = self.layer_norm(trg)
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = residual3 + self.dropout(_trg)

        return trg, attention


# test
# if __name__ == '__main__':
#     inputs = torch.randn((2, 2, 16))
#     enc = EncoderLayer(16, 8, 32, 0.1, device=None)
#     output, _ = enc(inputs)
#     print('enc；', output.size())
#     inputs_dec = torch.randn((2, 1, 16))
#     dec = DecoderLayer(16, 8, 32, 0.1, device=None)
#     dec_output, _ = dec(inputs_dec, output)
#     print('dec:', dec_output.size())

