#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/1 21:55
@Author  : Tomas Anderson Fang
@Email   : fsaunknow@gmail.com
@File    : Models.py
@Software: PyCharm
"""


from abc import ABC

import torch
import torch.nn as nn
from transformer.Layers import EncoderLayer, DecoderLayer
from utils import make_src_mask, make_trg_mask, PositionalEncoding, device


class Encoder(nn.Module, ABC):
    """
    Encoder is composed with multiple encoder layers.
    """
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=500):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_enc = PositionalEncoding(hid_dim, n_position=max_length)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(hid_dim)

    def forward(self, src, src_mask, use_bias=None):

        src = self.dropout(self.pos_enc(self.tok_embedding(src)))

        for index, layer in enumerate(self.layers):
            if index == 0:
                src = layer(src, src_mask, use_bias=use_bias)
            else:
                src = layer(src, src_mask)

        src = self.layer_norm(src)

        return src


class Decoder(nn.Module, ABC):
    """
    Decoder is composed with multiple encoder layer.
    """
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=200):
        super().__init__()

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_enc = PositionalEncoding(hid_dim, n_position=max_length)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(hid_dim)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.pos_enc(self.tok_embedding(trg))
        embed_x = trg.detach()
        trg = self.dropout(trg)

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        trg = self.layer_norm(trg)

        output = self.fc_out(trg)

        return output, attention


class Transformer(nn.Module, ABC):
    """
    A sequence to sequence model with hybrid attention network.
    """

    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 share_weight=False,
                 use_bias=None):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.use_bias = use_bias

        if share_weight:
            self.encoder.tok_embedding.weight = self.decoder.tok_embedding.weight

    def forward(self, src_seq, trg_seq):

        use_bias = self.use_bias
        src_mask = make_src_mask(src_seq, self.src_pad_idx)
        trg_mask = make_trg_mask(trg_seq, self.trg_pad_idx)

        enc_output = self.encoder(src_seq, src_mask, use_bias)
        dec_output, attn = self.decoder(trg_seq, enc_output, trg_mask, src_mask)

        return dec_output, attn


# test
if __name__ == '__main__':
    src = torch.randint(0, 10, (2, 2)).to(device)
    trg = torch.randint(0, 10, (2, 2)).to(device)
    encoder = Encoder(7855, 512, 6, 8, 1024, 0.1)
    decoder = Decoder(7654, 512, 6, 8, 1024, 0.1)
    model = Transformer(encoder, decoder, 1, 1).to(device)
    output, _ = model(src, trg)
    print(output.size())





