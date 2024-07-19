# -*- coding:utf-8 -*-

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from naive_transformer.basic_components import Transformer


# 构建模型输入的Tensor
def make_data(sentence):
    enc_inputs, dec_inputs, dec_outputs = [],[],[]
    for i in range(len(sentence)):
        enc_input = [src_vocab[word] for word in sentence[i][0].split()]
        dec_input = [tgt_vocab[word] for word in sentence[i][1].split()]
        dec_output = [tgt_vocab[word] for word in sentence[i][2].split()]
        
        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)
        
    # LongTensor是专用于存储整型的，Tensor则可以存浮点、整数、bool等多种类型
    return torch.LongTensor(enc_inputs),torch.LongTensor(dec_inputs),torch.LongTensor(dec_outputs)


# 使用Dataset加载数据
class MyDataSet(Data.Dataset):
    def __init__(self,enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet,self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        
    def __len__(self):
        # 我们前面的enc_inputs.shape = [2,5],所以这个返回的是2
        return self.enc_inputs.shape[0] 
    
    # 根据idx返回的是一组 enc_input, dec_input, dec_output
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
 
 
if __name__ == '__main__':
    # S: 起始标记
    # E: 结束标记
    # P：意为padding，将当前序列补齐至最长序列长度的占位符
    sentence = [
        # enc_input   dec_input    dec_output
        ['ich mochte ein bier P','S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P','S i want a coke .', 'i want a coke . E'],
    ]

    # 词典，padding用0来表示
    # 源词典
    src_vocab = {'P':0, 'ich':1,'mochte':2,'ein':3,'bier':4,'cola':5}
    src_vocab_size = len(src_vocab) # 6
    # 目标词典（包含特殊符）
    tgt_vocab = {'P':0,'i':1,'want':2,'a':3,'beer':4,'coke':5,'S':6,'E':7,'.':8}
    # 反向映射词典，idx ——> word
    idx2word = {v:k for k,v in tgt_vocab.items()}
    tgt_vocab_size = len(tgt_vocab) # 9

    src_len = 5 # 输入序列enc_input的最长序列长度，其实就是最长的那句话的token数
    tgt_len = 6 # 输出序列dec_input/dec_output的最长序列长度

    enc_inputs, dec_inputs, dec_outputs = make_data(sentence)
 
    print(' enc_inputs: \n', enc_inputs)  # enc_inputs: [2,5]
    print(' dec_inputs: \n', dec_inputs)  # dec_inputs: [2,6]
    print(' dec_outputs: \n', dec_outputs) # dec_outputs: [2,6]

    # 构建DataLoader
    loader = Data.DataLoader(dataset=MyDataSet(enc_inputs,dec_inputs, dec_outputs),batch_size=2,shuffle=True)

    for enc_inputs, dec_inputs, dec_outputs in loader:
        print('enc_inputs:',enc_inputs)
        print('dec_inputs:',dec_inputs)
        print('dec_outputs:',dec_outputs)
 
    # 用来表示一个词的向量长度
    d_model = 512
    # FFN的隐藏层神经元个数
    d_ff = 2048
    # 分头后的q、k、v词向量长度，依照原文我们都设为64
    # 原文：queries and kes of dimention d_k,and values of dimension d_v .所以q和k的长度都用d_k来表示
    d_k = d_v = 64
    # Encoder Layer 和 Decoder Layer的个数
    n_layers = 6
    # 多头注意力中head的个数，原文：we employ h = 8 parallel attention layers, or heads
    n_heads = 8
    
    transformer = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, 
                              d_model=d_model, d_k=d_k, n_head=n_heads, d_ff=d_ff, n_encoder_layer=n_layers, n_decoder_layer=n_layers,
                              input_max_len=src_len, output_max_len=tgt_len,
                              dropout=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
    
    transformer.train()
    for epoch in range(20):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            # 清空梯度
            optimizer.zero_grad()
            # enc_inputs: [batch_size, src_len]
            # dec_inputs: [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]
            
            # 前向传播
            outputs = transformer(enc_inputs, dec_inputs) # [batch_size, tgt_len, tgt_vocab_size]
            # 计算损失
            loss = criterion(outputs.view(-1,outputs.size(-1)), dec_outputs.view(-1))
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            
    # 导出模型
    torch.save(transformer.state_dict(), './transformer.pth')
            