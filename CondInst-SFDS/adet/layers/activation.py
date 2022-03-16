import torch
import torch.nn.functional as F
from torch import nn


class Mish(nn.Module):
    # @gxl
    # Mish activation
    # Non-offical code. The new version of PyTorch includes this function.
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def get_activation(name = 'ReLU', inplace = False):
    
    if name == 'ReLU':
        activation = nn.ReLU(inplace = inplace)
    elif name == 'Mish':
        activation = Mish(inplace = inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return activation

def mish(x):
    return x * torch.tanh(F.softplus(x))