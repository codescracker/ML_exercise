# -*- coding:utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from llama2.components import FeedForward, RMSNorm, SelfAttention

# implement mix of expert layer with feedforward layer
class MixOfExpertFeedForward(nn.Module):
    def __init__(self, d_model: int, num_experts: int, num_expert_per_tok:int ,ffn_dim_multiplier: int):
        super(MixOfExpertFeedForward, self).__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_tok = num_expert_per_tok
        self.expert_ffn_dim_multiplier = ffn_dim_multiplier
        
        self.experts = nn.ModuleList([FeedForward(d_model, ffn_dim_multiplier) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        
        x = x.view(-1, d_model) # [batch_size*seq_len, d_model]
        router = self.gate(x) # [batch_size*seq_len, num_experts]
        probs, indices = torch.topk(router, self.num_experts_per_tok) # [batch_size*seq_len, num_experts_per_tok], [batch_size*seq_len, num_experts_per_tok]
        probs = F.softmax(probs, dim=-1) # [batch_size*seq_len, num_experts_per_tok]
        
        masks = indices.unsqueeze(-1) == torch.arange(self.num_experts) # [batch_size*seq_len, num_experts_per_tok, 1], [num_experts,] -> [batch_size*seq_len, num_experts_per_tok, num_experts]
        masks = masks.permute(2, 0, 1) # [num_experts, batch_size*seq_len, num_experts_per_tok]
        
        y = torch.zeros_like(x) # [batch_size*seq_len, d_model]
        
        for i, expert in enumerate(self.experts):
            mask = masks[i] # [batch_size*seq_len, num_experts_per_tok]

            token_idx, expert_idx = torch.where(mask) 
            
            token_emb = x[token_idx] # [d_model]
            token_emb = expert(token_emb) # [d_model]
            prob = probs[token_idx, expert_idx, None] # [1] 
            
            token_emb_expert = prob * token_emb # [d_model]
            
            y[token_idx]+= token_emb_expert
            
        return y.view(batch_size, seq_len, d_model) # [batch_size, seq_len, d_model]
            

class MixOfExpertBlock(nn.Module):
    def __init__(self, d_model: int, num_experts: int, num_expert_per_tok:int ,ffn_dim_multiplier: int,
                n_heads: int, kv_n_heads: int, max_batch_size: int, max_seq_len: int):
        super(MixOfExpertBlock, self).__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_tok = num_expert_per_tok
        self.expert_ffn_dim_multiplier = ffn_dim_multiplier
        
        self.moe_ff = MixOfExpertFeedForward(d_model, num_experts, num_expert_per_tok, ffn_dim_multiplier)
        self.norm1 = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, kv_n_heads, max_batch_size, max_seq_len)
        
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # x : (batch_size, seq_len, d_model)
        
        h = x + self.attn(self.norm1(x), start_pos) # (batch_size, seq_len, d_model)
        out = h + self.moe_ff(self.ffn_norm(h)) # (batch_size, seq_len, d_model)
        
        return out
    
class MixOfExpertTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, kv_n_heads: int, 
                 ffn_dim_multiplier: int, max_batch_size: int, max_seq_len: int, num_experts: int, num_experts_per_tok: int):
        super(MixOfExpertTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_ffn_dim_multiplier = ffn_dim_multiplier
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MixOfExpertBlock(d_model, num_experts, num_experts_per_tok, ffn_dim_multiplier, n_heads, kv_n_heads, max_batch_size, max_seq_len) for _ in range(n_layers)])
        
        self.norm = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # x : (batch_size, seq_len)
        
        batch_size, seq_len = x.shape
        assert seq_len == 1, 'seq_len must be 1.'
                
        x = self.token_embedding(x) # (batch_size, 1, d_model)
        
        for block in self.blocks:
            x = block(x, start_pos) # (batch_size, 1, d_model)
        
        x = self.norm(x) # (batch_size, 1, d_model)
        
        x = self.fc(x) # (batch_size, 1, vocab_size)
        
        return x
    
