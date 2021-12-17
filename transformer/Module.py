#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020/11/17 20:56
@Author  : Tomas S. Fang
@Email   : fangsen1996@gmail.com
@File    : Module.py
@Software: PyCharm
"""


from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module, ABC):
    """
    Hybrid attention composed with self-attention and local attention.
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, bias=None, alpha=None):

        #  Scaled Dot-Product
        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.temperature

        # We only add local attention to the encoder.
        if bias is not None:
            attn = (1 - alpha) * attn + alpha * bias

        # Replace padding mask with a large negative number.
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        # Gain the context vector.
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
