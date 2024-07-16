# -*- coding:utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn

from naive_transformer.basic_components import EncoderLayer, create_padding_mask

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self.get_positional_encoding(max_len, d_model)
        
    def get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class BertEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super(BertEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model, max_len)
        self.segment_embedding = nn.Embedding(3, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, segment_label):
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(x)
        segment_embedding = self.segment_embedding(segment_label)
        embedding =  token_embedding + position_embedding + segment_embedding
        
        return self.dropout(embedding)
        
    
class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, vocab_size, max_len, d_ff, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_ff = d_ff
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.d_k = hidden // attn_heads
        
        self.embedding = BertEmbedding(vocab_size=self.vocab_size, d_model=self.hidden, max_len=self.max_len, dropout=dropout)
        self.encoder_blocks = nn.ModuleList(
            [EncoderLayer(d_model=self.hidden, d_k=self.d_k, n_head=self.attn_heads, d_ff=self.d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        
    def forward(self, x, segment_info):
        # attention masking for padded token
        attention_mask = create_padding_mask(x,x)
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)
        
        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, attention_mask)
        
        return x
    
    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        return checkpoint["epoch"], checkpoint["loss"]


class NextSentencePrediction(torch.nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # use only the first token which is the [CLS]
        return self.softmax(self.linear(x[:, 0]))
    
class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: BERT model output size [batch_size, max_len, hidden]
        :param vocab_size: total vocab size 
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
    
class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, next_sentence: NextSentencePrediction, mask_lm: MaskedLanguageModel):
        super(BERTLM, self).__init__()
        self.bert = bert
        self.next_sentence = next_sentence
        self.mask_lm = mask_lm
        
    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)
    
    
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr