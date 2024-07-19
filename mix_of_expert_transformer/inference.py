# -*- coding:utf-8 -*-

from typing import List, Tuple, Dict, Optional, Union
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from dataclasses import dataclass

from mix_of_expert_transformer.components import MixOfExpertTransformer


class LlamaMoE:
    """
    Llama2 is a transformer-based model that can be used for text generation tasks.
    """
    def __init__(self, model: MixOfExpertTransformer, tokenizer: SentencePieceProcessor, 
                config: Dict) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    @staticmethod
    def build(checkpoint_dir: str, load_model: bool, 
              tokenizer_path: str,
              d_model: int, max_seq_len: int, device: str, num_layers: int,
                num_heads: int, kv_num_heads: int, ffn_dim_multiplier: int,
                max_batch_size: int,
                num_experts: int, num_experts_per_tok: int) -> MixOfExpertTransformer:
        
        prev_time = time.time()
        
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        
        vocab_size = tokenizer.vocab_size()
        
        model = MixOfExpertTransformer(vocab_size=vocab_size, d_model=d_model, n_layers=num_layers, 
                        n_heads=num_heads,kv_n_heads=kv_num_heads, 
                        ffn_dim_multiplier=ffn_dim_multiplier, max_batch_size=max_batch_size, max_seq_len=max_seq_len, 
                        num_experts=num_experts, num_experts_per_tok=num_experts_per_tok
                    ).to(device)
        
        config = dict()
        config["d_model"] = d_model
        config["max_seq_len"] = max_seq_len
        config["device"] = device
        config["num_layers"] = num_layers
        config["num_heads"] = num_heads
        config["kv_num_heads"] = kv_num_heads
        config["ffn_dim_multiplier"] = ffn_dim_multiplier
        config["max_batch_size"] = max_batch_size
        config["vocab_size"] = vocab_size
        config["num_experts"] = num_experts
        config["num_experts_per_tok"] = num_experts_per_tok
        
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint from {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu")
            print("Loaded Checkpoint in {:.2f}s".format(time.time() - prev_time))
            prev_time = time.time()
        
            model.load_state_dict(checkpoint, strict=True)
            print("Loaded Model in {:.2f}s".format(time.time() - prev_time))
        
        return LlamaMoE(model, tokenizer, config)
    
    
    @staticmethod
    def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor: # remove the least probable tokens until the cumulative probability exceeds top_p
        
        # probs: [batch_size, vocab_size]
        probs_sort, probs_indices = torch.sort(probs, dim=-1, descending=True) # probs_sort: [batch_size, vocab_size], probs_indices: [batch_size, vocab_size]
        cumulative_probs = torch.cumsum(probs_sort, dim=-1) # [batch_size, vocab_size]
        
        mask = cumulative_probs - probs_sort > top_p # [batch_size, vocab_size]
        
        probs_sort[mask] = 0 # [batch_size, vocab_size]
        probs_sort /= probs_sort.sum(dim=-1, keepdim=True) # [batch_size, vocab_size]
        
        sampled_indices = torch.multinomial(probs_sort, 1) # [batch_size, 1]
        next_token = probs_indices.gather(-1, sampled_indices) # [batch_size, 1]
        
        return next_token
        
    def generate(self, prompts: List[str], max_length: int, top_p: float, temperature: float):
        
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompts)
        
        max_prompt_len = max([len(prompt) for prompt in prompt_tokens])
        
        total_len = min(max_prompt_len + max_length, self.config["max_seq_len"])
        
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.config["device"]) # [batch_size, total_len]

        for idx, prompt in enumerate(prompt_tokens):
            tokens[idx, :len(prompt)] = torch.tensor(prompt, device=self.config["device"])
            
        eos_reached = torch.Tensor([False] * batch_size).to(self.config["device"]) # [batch_size]
        prompt_tokens_mask = (tokens != pad_id) # [batch_size, total_len]
        
        for cur_pos in range(1, total_len):
            with torch.no_grad():
                logits = self.model(tokens[:, :cur_pos], prompt_tokens_mask[:, :cur_pos])
                
                if temperature >0:
                    logits = logits[:, -1] / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = self._sample_top_p(probs, top_p)
                else:                   
                    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                    
                next_token = next_token.reshape(-1) # [batch_size]
                
                cur_mask = prompt_tokens_mask[:, cur_pos] # [batch_size]
                cur_token = tokens[:, cur_pos] # [batch_size]
                
                next_token = torch.where(cur_mask, cur_token, next_token) # [batch_size]
                
                tokens[:, cur_pos] = next_token 
                
                eos_reached |= (next_token == self.tokenizer.eos_id()) # [batch_size]
                
                if eos_reached.all():
                    break
        
        out_tokens = []
        out_text = []
        for idx, token in enumerate(tokens.tolist()):
            
            if self.tokenizer.eos_id() in token:
                token = token[:token.index(self.tokenizer.eos_id())]
            
            out_tokens.append(token)
            out_text.append(self.tokenizer.decode(token))
            
        return out_tokens, out_text