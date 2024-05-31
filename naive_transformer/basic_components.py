# -*- coding:utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt

import seaborn

seaborn.set_context(context="talk")

# padding mask 
def create_padding_mask(seq_q: torch.Tensor, seq_k: torch.Tensor) -> torch.Tensor:
    batch_size, len_q = seq_q.size() # batch_size x len_q
    batch_size, len_k = seq_k.size() # batch_size x len_k
    pad_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k

    # true mean masked
    return pad_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


# look ahead mask
def create_look_ahead_mask(seq: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = seq.size() # batch_size x seq_len
    look_ahead_mask = torch.ones(seq_len, seq_len).triu(1) # seq_len x seq_len

    # 1 mean masked
    return look_ahead_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len) # batch_size x seq_len x seq_len


# scaled dot product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Q: batch_size x n_head x len_q x d_k
        # K: batch_size x n_head x len_k x d_k
        # V: batch_size x n_head x len_k x d_k
        # mask: batch_size x n_head x len_q x len_k

        # K.transpose(-1, -2): batch_size x n_head x d_k x len_k

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k) # batch_size x n_head x len_q x len_k

        if mask is not None:
            scores.masked_fill_(mask, -1e9) # replace masked value to -1e9, batch_size x n_head x len_q x len_k, to ignore the masked value

        attention = torch.nn.functional.softmax(scores, dim=-1) # batch_size x n_head x len_q x len_k         

        context = torch.matmul(attention, V) # batch_size x n_head x len_q x d_k

        return context, attention

# multi-head attention


# feed forward

# layer normalization

# positional encoding

# embedding

# transformer encoder

# transformer decoder

# transformer
