# -*- coding:utf-8 -*-

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from naive_transformer.basic_components import Transformer


# 生成句子
def greedy_decode(model:Transformer, src: torch.Tensor, max_len: int, start_symbol: int) -> torch.Tensor:

    ouptut = model.generate(src, max_len, start_symbol)

    return ouptut


if __name__ == '__main__':
    src_len = 5 # 输入序列enc_input的最长序列长度，其实就是最长的那句话的token数
    tgt_len = 6 # 输出序列dec_input/dec_output的最长序列长度

    src_vocab = {'P':0, 'ich':1,'mochte':2,'ein':3,'bier':4,'cola':5}
    src_vocab_size = len(src_vocab) # 6
    # 目标词典（包含特殊符）
    tgt_vocab = {'P':0,'i':1,'want':2,'a':3,'beer':4,'coke':5,'S':6,'E':7,'.':8}
    # 反向映射词典，idx ——> word
    idx2word = {v:k for k,v in tgt_vocab.items()}
    tgt_vocab_size = len(tgt_vocab) # 9

    d_model = 512 # 用来表示一个词的向量长度
    d_ff = 2048 # FFN的隐藏层神经元个数
    d_k = d_v = 64 # 分头后的q、k、v词向量长度，依照原文我们都设为64 # 原文：queries and kes of dimention d_k,and values of dimension d_v .所以q和k的长度都用d_k来表示
    n_layers = 6 # Encoder Layer 和 Decoder Layer的个数
    n_heads = 8 # 多头注意力中head的个数，原文：we employ h = 8 parallel attention layers, or heads

    transformer = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, 
                                d_model=d_model, d_k=d_k, n_head=n_heads, d_ff=d_ff, n_encoder_layer=n_layers, n_decoder_layer=n_layers,
                                input_max_len=src_len, output_max_len=tgt_len,
                                dropout=0.1)

    # 加载模型
    transformer.load_state_dict(torch.load('./transformer.pth'))

    # 推断
    transformer.eval()
    
    src = Variable(torch.LongTensor([[1,2,3,4,0]]))
    print(greedy_decode(transformer, src, max_len=6, start_symbol=6)) 