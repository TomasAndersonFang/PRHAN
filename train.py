#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/3 13:25
@Author  : Tomas Anderson Fang
@Email   : fsaunknow@gmail.com
@File    : train.py
@Software: PyCharm
"""


import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformer.Models import Transformer, Encoder, Decoder

from datasets import src_pad_idx, trg_pad_idx
from datasets import src_vocab_size, trg_vocab_size
from datasets import train_iter, valid_iter

from utils import device


def cal_performance(pred, real, trg_pad_idx, smoothing=False):
    """Apply label smoothing if needed."""

    loss = cal_loss(pred, real, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    real = real.contiguous().view(-1)
    non_pad_mask = real.ne(trg_pad_idx)
    n_correct = pred.eq(real).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, real, trg_pad_idx, smoothing=False):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    # [batch_size x s_len]
    real = real.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        # one_hot = [batch_size, s_len, trg_vocab_size]
        one_hot = torch.zeros_like(pred).scatter(1, real.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = real.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, real, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def train_epoch(model, train_data, optimizer, smoothing):
    """Epoch operation in training parse."""

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Training)   '
    for i, batch in enumerate(tqdm(train_data, mininterval=2, desc=desc, leave=False)):
        # prepare data
        src_seq = batch.source[:, :400]
        trg_seq = batch.target[:, :-1][:, :50]
        trg_real = batch.target[:, 1:][:, :50]

        # forward
        optimizer.zero_grad()
        pred, _ = model(src_seq, trg_seq)

        # pred = [batch size, trg len, output dim]
        pred = pred.view(-1, pred.size(-1))
        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, trg_real, trg_pad_idx, smoothing=smoothing)
        loss.backward()
        optimizer.step()
        #optimizer.step()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, valid_data):
    """
    Epoch operation in evaluation parse.
    """

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(valid_data, mininterval=2, desc=desc, leave=False):
            # prepare data
            src_seq = batch.source[:, :400]
            trg_seq = batch.target[:, :-1][:, :50]
            trg_real = batch.target[:, 1:][:, :50]

            # forward
            pred, _ = model(src_seq, trg_seq)
            pred = pred.view(-1, pred.size(-1))

            loss, n_correct, n_word = cal_performance(
                pred, trg_real, trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


epochs = 200
lr=0.0005

def train(model, train_data, valid_data, lr, label_smoothing, save_model=False):
    """
    Start training.
    """

    def print_performances(header, loss, accu, start_time):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", ppl=math.exp(min(loss, 100)),
            accu=100 * accu, elapse=(time.time() - start_time) / 60))

    valid_losses = []
    # best_valid_loss = float('inf')
    acc = 0.
    loss = 1e10
    for epoch_i in range(epochs):
        print('[ Epoch', epoch_i, ']')
        optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.998), eps=1e-9, lr=lr)
        start = time.time()
        train_loss, train_accu = train_epoch(
            model, train_data, optimizer, smoothing=label_smoothing)
        print_performances('Training', train_loss, train_accu, start)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, valid_data)
        print_performances('Validation', valid_loss, valid_accu, start)

        valid_losses += [valid_loss]

        if save_model:
            if valid_loss < loss:
                loss = valid_loss
                torch.save(model.state_dict(), './model_save/test.pt')
        if (epoch_i + 1) % 5 == 0:
            lr = lr * 0.97


enc = Encoder(src_vocab_size, 128, 2, 8, 512, 0.1)
dec = Decoder(trg_vocab_size, 128, 2, 8, 512, 0.1)
transformer = Transformer(enc, dec, src_pad_idx, trg_pad_idx, share_weight=True, use_bias=True).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(transformer):,} trainable parameters')


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


transformer.apply(initialize_weights)

#optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.998), eps=1e-9, lr=lr)

train(transformer, train_iter, valid_iter, lr, label_smoothing=True, save_model=True)
