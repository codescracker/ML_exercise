# -*- coding:utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn

from naive_transformer.basic_components import FeedForward,MultiHeadAttention, AddNorm, create_look_ahead_mask


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x) -> torch.Tensor:
        return self.embedding(x)
        
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, block_size: int):
        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(block_size, d_model)
        
    def forward(self, x) -> torch.Tensor:
        return self.pe(x)


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float):
        super(Block, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.add_norm_1(x, self.attn(x, x, x, mask))
        x = self.add_norm_2(x, self.ff(x))
        return x
    

class GPT2(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_ff: int, n_heads: int, n_layers: int, block_size: int, dropout: float):
        super(GPT2, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model, block_size)
        self.blocks = nn.ModuleList([Block(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
        self.block_size = block_size
        
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # id : (batch_size, seq_len)
        
        device = idx.device
        look_ahead_mask = create_look_ahead_mask(idx.size(1)).to(device) # (batch_size, seq_len, seq_len)
        
        _b, _t = idx.size()
        tok_emb = self.token_embedding(idx).to(device) # (batch_size, seq_len, d_model)
        pos_emb = self.position_embedding(torch.arange(_t).unsqueeze(0).repeat(_b, 1)).to(device) # (batch_size, seq_len, d_model)
        x = tok_emb + pos_emb # (batch_size, seq_len, d_model)
                
        for block in self.blocks:
            x = block(x, look_ahead_mask)
        return self.fc(x) # (batch_size, seq_len, vocab_size)
    
    @torch.no_grad()
    def generate(self, max_new_tokens:int, idx: torch.Tensor, temperature=1.0, top_k=None):
        self.eval()
        
        # idx : (batch_size, seq_len)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:] # (batch_size, seq_len)
            logits = self.forward(idx_cond) # (batch_size, seq_len, vocab_size)
            
            logits = logits[:, -1, :] / temperature # (batch_size, vocab_size) only the last token
            if top_k is not None:
                v, i = logits.topk(top_k)
                logits[logits < v[:, -1]] = -float('Inf')
            
            probs = logits.softmax(-1) # (batch_size, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1) # (batch_size, 1) sample from the distribution
            idx = torch.cat([idx, next_token], dim=-1) # (batch_size, seq_len+1)
            
        return idx