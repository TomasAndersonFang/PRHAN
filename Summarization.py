#!/usr/bin/env python
# encoding: utf-8
"""
@author: Tomas S. Fang
@contact: fangsen1996@gmail.com
@software: PyCharm
@file: Summarization.py
@time: 2020/11/17 16:59
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from utils import make_src_mask, make_trg_mask


class Summary(nn.Module, ABC):
    """
    Load the trained model and summary articles via beam search fashion.
    """

    def __init__(self, model, beam_size, max_seq_len, pad_idx, sos_idx, eos_idx, device):
        super(Summary, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[self.sos_idx]]))
        self.init_seq = self.init_seq.to(device)
        self.register_buffer('blank_seq', torch.full((beam_size, max_seq_len), 0, dtype=torch.long))
        self.blank_seq[:, 0] = self.sos_idx
        self.blank_seq = self.blank_seq.to(device)
        self.register_buffer('len_map', torch.arange(1, max_seq_len+1, dtype=torch.long).unsqueeze(0))
        self.len_map = self.len_map.to(device)

    def inference(self, trg_seq, enc_output, src_mask):
        trg_mask = make_trg_mask(trg_seq, self.pad_idx)
        dec_output, _ = self.model.decoder(trg_seq, enc_output, trg_mask, src_mask)

        return F.softmax(dec_output, dim=-1)

    def initialization(self, src_seq, src_mask):
        beam_size = self.beam_size

        # shape = [1, src_seq_len, d_model]
        enc_output = self.model.encoder(src_seq, src_mask)
        # shape = [1, trg_seq_len, vocab_size]
        # print(enc_output.size())
        dec_output = self.inference(self.init_seq, enc_output, src_mask)

        # get the top-k best result, k = beam_size
        # shape = [1, trg_seq_len, k]
        best_k_prob, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        # shape = [k]
        scores = torch.log(best_k_prob).view(beam_size)

        # shape = [k, 50]
        pred_seq = self.blank_seq.clone().detach()
        pred_seq[:, 1] = best_k_idx[0]
        # shape = [k, src_seq_lem, d_model]
        enc_output = enc_output.repeat(beam_size, 1, 1)

        return enc_output, pred_seq, scores

    def get_best_scores(self, pred_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # get top-k results, thus we will get k^2 candidates in total.
        # shape = [k, 1, k]
        best_k2_prob, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # calculate scores. shape = [k, k]
        scores = torch.log(best_k2_prob).view(beam_size, -1) + scores.view(beam_size, 1)

        # get top-k scores from k^2 scores
        # shape = [k]
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # get the top-k positions of the top-k candidates
        best_k_r_idx, best_k_c_idx = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idx, best_k_c_idx]

        pred_seq[:, :step] = pred_seq[best_k_r_idx, :step]
        pred_seq[:, step] = best_k_idx

        return pred_seq, scores

    def summarization(self, article):
        # shape = [1, seq_lem]
        assert article.size(0) == 1, 'the batch size of article must equal to 1'

        pad_idx, trg_eos_idx = self.pad_idx, self.eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            pad_mask = make_src_mask(article, pad_idx)
            enc_output, pred_seq, scores = self.initialization(src_seq=article, src_mask=pad_mask)

            ans_idx = 0
            for step in range(2, max_seq_len):
                # shape = [k, step, vocab_size]
                dec_output = self.inference(pred_seq[:, : step], enc_output, pad_mask)
                pred_seq, scores = self.get_best_scores(pred_seq, dec_output, scores, step)

                # Check if all path finished inference.
                # locate the eos token
                # shape = [k, 50]
                eos_pos = pred_seq == trg_eos_idx
                # replace the eos with its position for the length penalty
                # shape = [k], it contains the length for each sentence in every row in the pred_seq.
                seq_lens, _ = self.len_map.masked_fill(~eos_pos, max_seq_len).min(1)

                if (eos_pos.sum(1) > 0).sum(0).item() == beam_size:
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break

        return pred_seq[ans_idx][: seq_lens[ans_idx]].tolist()

