import numbers
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor

class ScaleNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5, bias: bool = False) -> None:
        super(ScaleNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        scalenorm = self.weight * input / (torch.norm(input, p=2, dim=-1, keepdim=True) + self.eps)

        if self.bias is not None:
            scalenorm = scalenorm + self.bias

        return scalenorm
