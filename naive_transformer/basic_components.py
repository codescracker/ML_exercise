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

    # true means masked
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
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, n_head: int):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model        
        self.n_head = n_head
        assert self.d_model % self.n_head == 0

        self.d_k = d_model // self.n_head
        self.d_v = d_model // self.n_head

        self.W_Q = nn.Linear(self.d_model, self.d_model)
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)

        self.attention = ScaledDotProductAttention(d_k)
        self.linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Q: batch_size x len_q x d_model
        # K: batch_size x len_k x d_model
        # V: batch_size x len_k x d_model
        # mask: batch_size x len_q x len_k

        batch_size = Q.size(0)

        # linear projection
        Q = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2) # batch_size x n_head x len_q x d_k
        K = self.W_K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2) # batch_size x n_head x len_k x d_k
        V = self.W_V(V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2) # batch_size x n_head x len_k x d_v

        if mask is not None:        
            mask = mask.unsqueeze(1).expand(batch_size, self.n_head, mask.size(1), mask.size(2)) # batch_size x n_head x len_q x len_k

        context, attention = self.attention(Q, K, V, mask) # batch_size x n_head x len_q x d_v

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # batch_size x len_q x d_model

        output = self.linear(context) # batch_size x len_q x d_model

        return output, attention


# layer normalization
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size x len_q x d_model
        mean = x.mean(-1, keepdim=True) # batch_size x len_q x 1
        std = x.std(-1, keepdim=True) # batch_size x len_q x 1

        x_normalized = (x - mean) / (std + self.eps) # batch_size x len_q x d_model

        return self.a_2 * x_normalized + self.b_2 # batch_size x len_q x d_model

# feed forward
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size x len_q x d_model

        x = self.dropout(torch.nn.functional.relu(self.linear1(x))) # batch_size x len_q x d_ff
        x = self.linear2(x) # batch_size x len_q x d_model

        return x

# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model) # max_len x d_model
        position = torch.arange(0, max_len).unsqueeze(1).float() # max_len x 1

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # d_model / 2

        pe[:, 0::2] = torch.sin(position * div_term) # max_len x d_model / 2
        pe[:, 1::2] = torch.cos(position * div_term) # max_len x d_model / 2

        pe = pe.unsqueeze(0) # 1 x max_len x d_model

        self.register_buffer('pe', pe) # register pe as buffer to save the state of the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size x len_q x d_model

        x = x + self.pe[:, :x.size(1)] # batch_size x len_q x d_model

        return self.dropout(x)

# embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size x len_q

        return self.embedding(x) * math.sqrt(self.d_model) # batch_size x len_q x d_model


# add and norm
class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        # x: batch_size x len_q x d_model

        return x + self.dropout(sublayer(self.norm(x))) # batch_size x len_q x d_model

# encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_k: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, d_k, n_head)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: batch_size x len_q x d_model
        # mask: batch_size x len_q x len_k

        x = self.add_norm1(x, lambda x: self.multi_head_attention(x, x, x, mask)) # batch_size x len_q x d_model
        x = self.add_norm2(x, self.feed_forward) # batch_size x len_q x d_model

        return x

# decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_k: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, d_k, n_head)
        self.multi_head_attention = MultiHeadAttention(d_model, d_k, n_head)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, mask: torch.Tensor, look_ahead_mask: torch.Tensor) -> torch.Tensor:
        # x: batch_size x len_q x d_model
        # enc_output: batch_size x len_k x d_model
        # mask: batch_size x len_q x len_k
        # look_ahead_mask: batch_size x len_q x len_q

        x = self.add_norm1(x, lambda x: self.masked_multi_head_attention(x, x, x, look_ahead_mask)) # batch_size x len_q x d_model
        x = self.add_norm2(x, lambda x: self.multi_head_attention(x, enc_output, enc_output, mask)) # batch_size x len_q x d_model
        x = self.add_norm3(x, self.feed_forward) # batch_size x len_q x d_model

        return x


# encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_k: int, n_head: int, d_ff: int, n_layer: int, max_len:int,  dropout: float = 0.1):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, n_head, d_ff, dropout) for _ in range(n_layer)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: batch_size x len_q
        # mask: batch_size x len_q x len_k

        x = self.positional_encoding(self.embedding(x)) # batch_size x len_q x d_model

        for layer in self.layers:
            x = layer(x, mask) # batch_size x len_q x d_model

        return x


# decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_k: int, n_head: int, d_ff: int, n_layer: int, max_len: int, dropout: float = 0.1):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_k, n_head, d_ff, dropout) for _ in range(n_layer)])

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, mask: torch.Tensor, look_ahead_mask: torch.Tensor) -> torch.Tensor:
        # x: batch_size x len_q
        # enc_output: batch_size x len_k x d_model
        # mask: batch_size x len_q x len_k
        # look_ahead_mask: batch_size x len_q x len_q

        x = self.positional_encoding(self.embedding(x)) # batch_size x len_q x d_model

        for layer in self.layers:
            x = layer(x, enc_output, mask, look_ahead_mask) # batch_size x len_q x d_model

        return x

# transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                d_model: int, d_k: int, n_head: int, d_ff: int, 
                n_encoder_layer: int, n_decoder_layer: int, 
                input_max_len:int, output_max_len:int,
                dropout: float = 0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_k, n_head, d_ff, n_encoder_layer, input_max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, d_k, n_head, d_ff, n_decoder_layer, output_max_len, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src: batch_size x len_q
        # tgt: batch_size x len_k

        src_mask = create_padding_mask(src, src) # batch_size x len_q x len_k
        look_ahead_mask = create_look_ahead_mask(tgt) # batch_size x len_q x len_q

        enc_output = self.encoder(src, src_mask) # batch_size x len_q x d_model
        dec_output = self.decoder(tgt, enc_output, src_mask, look_ahead_mask) # batch_size x len_q x d_model

        output = self.linear(dec_output) # batch_size x len_q x tgt_vocab_size

        return output

    def generate(self, src: torch.Tensor, max_len: int, start_symbol: int) -> torch.Tensor:
        # src: batch_size x len_q

        src_mask = create_padding_mask(src, src) # batch_size x len_q x len_k
        enc_output = self.encoder(src, src_mask) # batch_size x len_q x d_model
        
        # initialize decoder input, the first input is start_symbol
        dec_input = torch.zeros(src.size(0), 1).fill_(start_symbol).type_as(src.data) # batch_size x 1
        dec_output = dec_input
        
        for _ in range(max_len - 1):
            look_ahead_mask = create_look_ahead_mask(dec_output)
            dec_output = self.decoder(dec_output, enc_output, src_mask, look_ahead_mask)
            dec_output = self.linear(dec_output)
            dec_output = torch.cat([dec_input, dec_output[:, -1].argmax(dim=-1, keepdim=True)], dim=-1)
            dec_input = dec_output
            
        return dec_output
        