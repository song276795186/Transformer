import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from Transformer.Encoder import clones, LayerNorm, SublayerConnection


def subsequent_mask(size):
    """
    Mask out subsequent positions
    :param size:
    :return:

    example:
    import numpy as np
    x = (1,4,4)
    np.triu(np.ones(x), k=1)
    array([[[0., 1., 1., 1.],
            [0., 0., 1., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.]]])
    """
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  #scaling factor d_k
    #the fuction of the scaling is to prevent the softmax function has gradient vanish problem
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    """
    key.transpose(-2,-1) 即将最后两个维度进行换轴， 例如将一个(4,3,2)的tensor转化为一个(4,2,3)的tensor
    """
    if mask is None:
        scores = scores.masked_fill(mask == 0, -1e9)
        """
        masked_fill 是什么作用：
            
        """
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads
        :param h: number of heads
        :param d_model: model_size
        :param dropout: dropout ratio
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implement Figure 2
        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """
        if mask is None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h * d_k
        query, key, value = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
                 for l, x in zip(self.linears, (query, key, value))]
        """
        self.linears :  d_model x d_model
        query,key,value : nbatches x d_q x d_model  
        l(x)    :    nbatches x d_q x d_model
        l(x).view(...)  :  nbacthes x d_q x h x d_k
        l(x).view().transpose(1,2)  :  nbacthes x h x d_q x d_k
        """

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        """
        x  :  nbacthes x h x d_q x d_k
        """

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

        """
        x  :  nbatches x   d_q  x (h x d_k)
        即    nbatches x   d_q  x  d_model
        """


class PositionWiseFeedForward(nn.Module):
    """
    In addition to attention sub-layers,
    each of the layers in our encoder and decoder
    contains a fully connected feed-forward network,
    which is applied to each position separately and identically.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    #Implement the positional encoding function
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #Compute the positional encoding once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, max_len).unsqueeze(1) -
                             (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)