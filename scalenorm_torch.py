import math
import numbers
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor

class ScaleNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        eps: float = 1e-5, 
        bias: bool = False
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        C = int(torch.tensor(self.normalized_shape).prod().item())
        self.register_buffer("sqrt_C", torch.tensor(math.sqrt(C)), persistent=False)
        self.eps = eps
        self.gamma = nn.Parameter(torch.empty(1))
        self.softplus = nn.Softplus()
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        s = self.sqrt_C.item()
        init_val = s if s > 20.0 else math.log(math.expm1(s))
        init.constant_(self.gamma, init_val)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        norm = torch.linalg.vector_norm(input, ord=2, dim=dims, keepdim=True)
        gamma = self.softplus(self.gamma)
        scalenorm = gamma * input / (norm + self.eps)
        
        if self.bias is not None:
            scalenorm = scalenorm + self.bias

        return scalenorm