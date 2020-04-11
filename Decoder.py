import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from Transformer.Encoder import clones, LayerNorm, SublayerConnection


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward
    这是一个decoder层,decoder解码器应该由6个decoder组成
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        #这里sublayer 0,1,2都是一样的
        #self-attn层之后sublayer一下
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        #src-attn层之后再sublayer一下
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        #feed_forward之后再sublayer
        return self.sublayer[2](x, self.feed_forward)
