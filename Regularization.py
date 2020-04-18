import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from Transformer.Encoder import clones, LayerNorm, SublayerConnection

"""
During training, we employed label smoothing of value ϵls=0.1 (cite). 
This hurts perplexity,
as the model learns to be more unsure,
but improves accuracy and BLEU score.
"""

class LabelSmoothing(nn.Module):
    pass