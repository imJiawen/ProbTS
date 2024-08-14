# ---------------------------------------------------------------------------------
# Portions of this file are derived from RevIN
# - Source: https://github.com/ts-kim/RevIN
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
from probts.utils import repeat
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, num_samples=None, dim=0):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x, num_samples=num_samples, dim=dim)
        elif mode == 'denorm':
            x = self._denormalize(x, num_samples=num_samples, dim=dim)
        elif mode =='norm_only':
            x = self._normalize(x, num_samples=num_samples, dim=dim)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x, num_samples=None, dim=0):
        if num_samples is None:
            mean = self.mean
            std = self.stdev
        else:
            mean = repeat(self.mean, num_samples, dim=dim)
            std = repeat(self.stdev, num_samples, dim=dim)
            
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - mean
        x = x / std
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, num_samples=None, dim=0):
        if num_samples is None:
            mean = self.mean
            std = self.stdev
        else:
            mean = repeat(self.mean, num_samples, dim=dim)
            std = repeat(self.stdev, num_samples, dim=dim)
            
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        # print("x,shape")
        x = x * std
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + mean
        return x
