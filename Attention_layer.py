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