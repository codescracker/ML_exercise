# -*- coding:utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# replace layernorm with RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, epsilon: float=1e-8) -> None:
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(d_model)) # (d_model, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, seq_len, d_model)
        
        rms = torch.sqrt((x ** 2).mean(-1, keepdim=True) + self.epsilon) # (batch_size, seq_len, 1)
        
        x_normed = (x / rms) * self.gamma # (batch_size, seq_len, d_model)
        
        return x_normed
    
# replace absolute positional embedding with rotational positional embedding
class RotatePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int) -> None:
        self.dim = d_model
        self.seq_len = seq_len
        assert d_model % 2 == 0, 'd_model must be even.'
        
        # calculate the rotation frequencies for positional embeddings
        theta_numerator = torch.arange(0, d_model, 2.0) / self.dim # (d_model // 2, )
        theta = 1.0 / torch.pow(10000, theta_numerator/self.dim) # (d_model // 2, )
        
        # Generate frequency values for positional embeddings
        m = torch.arange(seq_len)
        freqs = torch.outer(m, theta).float()

        # Convert frequency values to complex numbers (polar form)
        self.freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_complex", self.freqs_complex)
        
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # x : (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.dim, 'd_model must be same as the model dimension.'
        
        # Reshape the input into a complex tensor for rotational operations
        # (B, SeqLen, H, Head_Dim) -> (B, SeqLen, H, Head_Dim // 2)
        x_complex = torch.view_as_complex(x.float().reshape(x.shape[:-1], seq_len, -1, 2))
        
        # Extract rotational frequencies for the given sequence length and start position
        # (SeqLen, Head_Dim // 2) -> (1, SeqLen, 1, Head_Dim // 2)
        freq_complex = self.freqs_complex[start_pos:start_pos+seq_len].unsqueeze(0).unsqueeze(2)
        
        # Apply rotational transformation to the input using frequency values
        # (B, SeqLen, H, Head_Dim // 2) * (1, SeqLen, 1, Head_Dim // 2) -> (B, SeqLen, H, Head_Dim // 2)
        x_rotated = x_complex * freq_complex
        
        # Convert the rotated complex tensor back to real-valued tensor
        # (B, SeqLen, H, Head_Dim // 2) -> (B, SeqLen, H , Head_Dim // 2, 2)
        x_out = torch.view_as_real(x_rotated)
        
        # Reshape to match the original input shape
        # (B, SeqLen, H , Head_Dim // 2, 2) -> (B, SeqLen, H, Head_Dim)
        x_out = x_out.reshape(batch_size, seq_len, d_model)
        
        return x_out
    
# FeedForward Layer for LLAMA2 model with swish activation
class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim_multiplier: int = None) -> None:
        super(FeedForward, self).__init__()
        
        d_ff = 4 * d_model
        d_ff = int(2 * d_model / 3)
        
        if ffn_dim_multiplier is not None:
            d_ff = int(ffn_dim_multiplier * d_model)
            
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.linear3 = nn.Linear(d_model, d_ff, bias=False)
        
        self.swish = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, seq_len, d_model)
        
        swish = self.swish(self.linear1(x)) # (batch_size, seq_len, d_ff)
        
        x_v = self.linear3(x) # (batch_size, seq_len, d_ff)
        
        x = swish * x_v # (batch_size, seq_len, d_ff)
        
        x = self.linear2(x) # (batch_size, seq_len, d_model)
        
        return x
        

# Attention Layer for LLAMA2 model with group query attention
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_n_heads: int, max_batch_size: int, max_seq_len: int) -> None:
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        self.n_kv_heads = kv_n_heads if kv_n_heads is not None else n_heads # number of heads for key and value
        
        self.n_q_heads = n_heads # number of heads for query
        self.n_rep = self.n_q_heads // self.n_kv_heads # number of query heads per key-value head
        
        self.head_dim = d_model // self.n_q_heads
        
        # linear transformation for queries, keys, and values, and output
        self.q_linear = nn.Linear(d_model, self.n_q_heads * self.head_dim, bias=False)
        self.k_linear = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_linear = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        
        self.output_linear = nn.Linear(self.n_q_heads * self.head_dim, d_model, bias=False)
        
        # initialize key and value caches with zeros
        self.cache_k = torch.zeros((self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim))
        
        # rotary position embeddings
        self.rope = RotatePositionalEmbedding(self.head_dim, self.max_seq_len)
        
    @staticmethod
    def repeat_heads(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
        # Repeat the heads of K and V to match the number of heads in Q
        
        batch_size, seq_len, n_heads, head_dim = x.shape
        
        if n_repeat == 1:
            return x
        else:
            x = x.unsqueeze(-2).expand(batch_size, seq_len, n_heads, n_repeat, head_dim)
            x = x.reshape(batch_size, seq_len, n_heads * n_repeat, head_dim)
            
            return x
        
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # x : (batch_size, 1, d_model)
        
        batch_size, seq_len, d_model = x.shape
        
        # linear transformation for queries, keys, and values
        q = self.q_linear(x) # (batch_size, 1, n_q_heads * head_dim)
        k = self.k_linear(x) # (batch_size, 1, n_kv_heads * head_dim)
        v = self.v_linear(x) # (batch_size, 1, n_kv_heads * head_dim)
        
        # split the heads
        q = q.view(batch_size, seq_len, self.n_q_heads, self.head_dim) # (batch_size, 1, n_q_heads, head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim) # (batch_size, 1, n_kv_heads, head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim) # (batch_size, 1, n_kv_heads, head_dim)
        
        q = self.rope(q, start_pos) # (batch_size, 1, n_q_heads, head_dim)
        k = self.rope(k, start_pos)  # (batch_size, 1, n_kv_heads, head_dim)
        
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = k
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = v
        
        keys = self.cache_k[:batch_size, :start_pos + seq_len] # (batch_size, seq_len, n_kv_heads, head_dim)
        values = self.cache_v[:batch_size, :start_pos + seq_len] # (batch_size, seq_len, n_kv_heads, head_dim)
        
        # repeat the keys and values to match the number of heads in queries
        keys = self.repeat_heads(keys, self.n_rep) # (batch_size, seq_len, n_q_heads, head_dim)
        values = self.repeat_heads(values, self.n_rep) # (batch_size, seq_len, n_q_heads, head_dim)
        
        q = q.transpose(1, 2) # (batch_size, n_q_heads, 1, head_dim)
        keys = keys.transpose(1, 2) # (batch_size, n_q_heads, seq_len, head_dim)
        values = values.transpose(1, 2) # (batch_size, n_q_heads, seq_len, head_dim)
        
        scores = torch.matmul(q, keys.transpose(-2, -1)) / math.sqrt(self.head_dim) # (batch_size, n_q_heads, 1, seq_len)
        scores = F.softmax(scores, dim=-1) # (batch_size, n_q_heads, 1, seq_len)
        
        context = torch.matmul(scores, values) # (batch_size, n_q_heads, 1, head_dim) 
        
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.n_q_heads * self.head_dim) # (batch_size, 1, n_q_heads * head_dim)
        
        output = self.output_linear(context) # (batch_size, 1, d_model)-> (batch_size, 1, d_model)
        
        return output
    

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_n_heads: int, ffn_dim_multiplier: int, max_batch_size: int, max_seq_len: int) -> None:
        super(Block, self).__init__()
        
        self.attn = SelfAttention(d_model, n_heads, kv_n_heads, max_batch_size, max_seq_len)
        self.ffn = FeedForward(d_model, ffn_dim_multiplier)
        self.norm1 = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # x : (batch_size, seq_len, d_model)
        
        h = x + self.attn(self.norm1(x), start_pos) # (batch_size, seq_len, d_model)
        out = h + self.ffn(self.ffn_norm(h)) # (batch_size, seq_len, d_model)
        
        return out
    
    
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, 
                n_heads: int, kv_n_heads: int, ffn_dim_multiplier: int, max_batch_size: int, max_seq_len: int) -> None:
        super(Transformer, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, kv_n_heads, ffn_dim_multiplier, max_batch_size, max_seq_len) for _ in range(n_layers)])
        
        self.norm = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # x : (batch_size, seq_len) seq_len must be 1.
        batch_size, seq_len = x.shape
        assert seq_len == 1, 'seq_len must be 1.'
                
        x = self.token_embedding(x) # (batch_size, 1, d_model)
        
        for block in self.blocks:
            x = block(x, start_pos) # (batch_size, 1, d_model)
        
        x = self.norm(x) # (batch_size, 1, d_model)
        
        x = self.fc(x) # (batch_size, 1, vocab_size)
        
        return x
    


