#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/17 16:53
@Author  : Tomas Anderson Fang
@Email   : fsaunknow@gmail.com
@File    : translator.py
@Software: PyCharm
"""


from tqdm import tqdm
import spacy
import torch

from transformer.Models import Transformer
from transformer.Models import Encoder, Decoder
from utils import make_src_mask, make_trg_mask

from datasets import src_vocab_size, trg_vocab_size
from datasets import src_pad_idx, trg_pad_idx
from datasets import test_data, train_data
from datasets import TEXT
from Summarization import Summary

from utils import device
import time


def translate_sentence(sentence, TEXT, model, device, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [TEXT.init_token] + tokens[:398] + [TEXT.eos_token]

    src_indexes = [TEXT.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    pred = model.summarization(src_tensor)

    trg_tokens = [TEXT.vocab.itos[i] for i in pred]

    return trg_tokens[1:]


def calculate_bleu(data, TEXT, model, device):
    trgs = []
    pred_trgs = []
    rouge = Rouge()

    for index, datum in enumerate(tqdm(data, mininterval=2, desc='--Translate--', leave=False)):
        src = vars(datum)['source']
        trg = vars(datum)['target']

        pred_trg = translate_sentence(src, TEXT, model, device)

        pred_trg = pred_trg[:-1]
        
        with open('pred.txt', 'a') as f:
            f.write(' '.join(pred_trg) + '\n')
            
        with open('gold.txt', 'a') as f:
            f.write(' '.join(trg) + '\n')

        pred_trgs.append(' '.join(pred_trg))
        trgs.append(' '.join(trg))

    return trgs


enc = Encoder(src_vocab_size, 128, 2, 8, 512, 0.1)
dec = Decoder(trg_vocab_size, 128, 2, 8, 512, 0.1)
transformer = Transformer(enc, dec, src_pad_idx, trg_pad_idx).to(device)
transformer.load_state_dict(torch.load('./model_save/test.pt'))

pad_idx = TEXT.vocab.stoi['<pad>']
sos_idx = TEXT.vocab.stoi['<sos>']
eos_idx = TEXT.vocab.stoi['<eos>']

summary = Summary(transformer, 4, 50, pad_idx, sos_idx, eos_idx, device)
bleu = calculate_bleu(test_data, TEXT, summary, device)

