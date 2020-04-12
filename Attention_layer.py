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
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
