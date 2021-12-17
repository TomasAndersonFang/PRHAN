#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/3 13:28
@Author  : Tomas Anderson Fang
@Email   : fsaunknow@gmail.com
@File    : datasets.py
@Software: PyCharm
"""


import torchtext
from torchtext.data import BucketIterator, Field, TabularDataset

import spacy
import torch

# set tokens
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_token = '<sos>'
eos_token = '<eos>'
spacy_en = spacy.load('en')


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


print('Start creating dataset!')

TEXT = Field(tokenize=tokenize_en,
             init_token=init_token,
             eos_token=eos_token,
             lower=True,
             batch_first=True)

#data_fields = [("id", TEXT),
#               ("target", TEXT),
#               ("source", TEXT)]

data_fields = [("source", TEXT),
               ("target", TEXT)]

train_data, valid_data, test_data = TabularDataset.splits(
        path='./dataset/',
        train='train_proj.csv',
        validation='valid_proj.csv',
        test='test_proj.csv',
        format='csv',
        skip_header=True,
        fields=data_fields)
#         filter_pred=lambda x: len(vars(x)['source']) <= 410 and len(vars(x)['target']) <= 160)

train_iter, valid_iter = BucketIterator.splits(
        (train_data, valid_data),
        batch_sizes=(128, 128),
        sort_key=lambda x: len(x.source),
        sort_within_batch=False,
        device=device,
        shuffle=True)

TEXT.build_vocab(train_data)

src_pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
trg_pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
src_vocab_size = len(TEXT.vocab)
trg_vocab_size = len(TEXT.vocab)

print('Finish creating dataset!')
