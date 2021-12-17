#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tomas S. Fang
@contact: fangsen1996@gmail.com
@software: PyCharm
@file: utils.py
@time: 2020/11/18 17:50
"""


from abc import ABC

import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_src_mask(src, src_pad_idx):

    # src_mask = [batch size, 1, 1, src len]
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    return src_mask


def make_trg_mask(trg, trg_pad_idx):

    # trg_pad_mask = [batch size, 1, trg len, 1]
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(3)

    trg_len = trg.shape[1]

    # trg_sub_mask = [trg len, trg len]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()

    # trg_mask = [batch size, 1, trg len, trg len]
    trg_mask = trg_pad_mask & trg_sub_mask

    return trg_mask


class PositionalEncoding(nn.Module, ABC):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
