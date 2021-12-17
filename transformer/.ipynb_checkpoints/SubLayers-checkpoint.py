#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020/11/17 20:53
@Author  : Tomas S. Fang
@Email   : fangsen1996@gmail.com
@File    : SubLayers.py
@Software: PyCharm
"""


from abc import ABC

import torch
import torch.nn as nn

from transformer.Module import ScaledDotProductAttention


class MultiHeadAttentionLayer(nn.Module, ABC):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        # For self-attention.
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        # For local attention
        self.fc_p = nn.Linear(hid_dim, hid_dim)
        self.fc_up = nn.Linear(self.head_dim, 1)
        self.fc_ud = nn.Linear(self.head_dim, 1)
        self.fc_us = nn.Linear(hid_dim, self.n_heads)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.slf_attn = ScaledDotProductAttention(self.head_dim ** 0.5)

        self.dropout = nn.Dropout(dropout)
        # self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None, use_bias=None):
        batch_size = query.shape[0]
        len_q = query.shape[1]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        # q = [batch size, query len, hid dim]
        # k = [batch size, key len, hid dim]
        # v = [batch size, value len, hid dim]
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        # Calculate gaussian bias.
        # p = [batch_size, n_heads, len_q, head_dim]
        # pos = [1, 1, 1, len_q], cen_pos = [batch_size, n_heads, len_q, 1]
        # win_size=[batch_size, n_heads, len_q, 1], bias = [batch_size, n_heads, len_q, len_q]
        pos = torch.arange(len_q).view(1, -1).unsqueeze(0).unsqueeze(1).cuda()
        p = self.fc_p(q).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        p_i = self.fc_up(self.tanh(p))
        z_i = self.fc_ud(self.tanh(p))
        cen_pos = len_q * self.sigmoid(p_i)
        win_size = len_q * self.sigmoid(z_i)
        bias = (pos - cen_pos) ** 2 / (1/2 * (win_size ** 2))

        # Gated factor.
        alpha = self.sigmoid(self.fc_us(query))

        # q = [batch size, n heads, query len, head dim]
        # k = [batch size, n heads, key len, head dim]
        # v = [batch size, n heads, value len, head dim]
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        alpha = alpha.view(batch_size, -1, self.n_heads, 1).permute(0, 2, 1, 3)

        if use_bias is None:
            output, attn = self.slf_attn(q, k, v, mask=mask)
        else:
            output, attn = self.slf_attn(q, k, v, mask=mask, bias=bias, alpha=alpha)

        # x = [batch size, seq len, n_heads, head_dim]
        x = output.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, seq len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, seq len, hid dim]
        x = self.fc_o(x)

        return x, attn


class PositionwiseFeedforwardLayer(nn.Module, ABC):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, pf dim]
        # in paper, dropout is not used in position wise feed forward layer
        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, hid dim]
        x = self.fc_2(x)

        return x


















