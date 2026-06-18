import math
import numbers
from typing import Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor, Size

import triton
import triton.language as tl


def maybe_contiguous_lastdim(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def triton_autotune_configs():
    max_threads_per_block = 1024
    warp_size = getattr(
        torch.cuda.get_device_properties(torch.cuda.current_device()),
        "warp_size", 32,
    )
    return [
        triton.Config({}, num_warps=w)
        for w in [1, 2, 4, 8, 16, 32]
        if w * warp_size <= max_threads_per_block
    ]


@triton.autotune(configs=triton_autotune_configs(), key=["N", "HAS_BIAS"])
@triton.jit
def _scale_norm_fwd_kernel(
    X,       # input
    Y,       # output
    B,       # bias (optional)
    Norm,    # per-row L2 norm, float32
    g,       # softplus(gamma), float scalar
    stride_x_row,
    stride_y_row,
    N,
    eps,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    tl.store(Norm + row, norm)
    denom = norm + eps
    y = g * x / denom
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = y + b
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(configs=triton_autotune_configs(), key=["N", "HAS_BIAS"])
@triton.jit
def _scale_norm_bwd_kernel(
    X,
    DY,
    DX,
    DB,      # partial bias grad: (num_blocks, N)
    DG,      # partial g grad: (num_blocks,)
    Norm,
    g,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    M,
    N,
    eps,
    rows_per_program,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dg = 0.0
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        norm = tl.load(Norm + row)
        denom = norm + eps
        c = tl.sum(dy * x, axis=0)
        dx = (g / denom) * (dy - x * c / (norm * denom))
        tl.store(DX + cols, dx, mask=mask)
        dg += c / denom
        if HAS_BIAS:
            db += dy
        X += stride_x_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DG + row_block_id, dg)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


class ScaleNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, bias, eps):
        x = maybe_contiguous_lastdim(x)
        M, N = x.shape
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_N:
            raise RuntimeError("ScaleNorm doesn't support feature dim >= 64KB.")
        out = torch.empty_like(x)
        norm = torch.empty((M,), dtype=torch.float32, device=x.device)
        bias_arg = bias.contiguous() if bias is not None else None
        g_val = g.item()
        with torch.cuda.device(x.device.index):
            _scale_norm_fwd_kernel[(M,)](
                x, out, bias_arg, norm, g_val,
                x.stride(0), out.stride(0),
                N, eps,
                HAS_BIAS=bias is not None,
                BLOCK_N=BLOCK_N,
            )
        ctx.save_for_backward(x, norm, g)
        ctx.eps = eps
        ctx.g_val = g_val
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return out

    @staticmethod
    def backward(ctx, dy):
        x, norm, g = ctx.saved_tensors
        M, N = x.shape
        dy = maybe_contiguous_lastdim(dy)
        dx = torch.empty_like(x)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count * 8
        sm_count = max(1, min(sm_count, M))
        rows_per_program = math.ceil(M / sm_count)
        num_blocks = math.ceil(M / rows_per_program)
        grid = (num_blocks,)

        _db = (
            torch.empty((num_blocks, N), dtype=torch.float32, device=x.device)
            if ctx.has_bias else None
        )
        _dg = torch.empty((num_blocks,), dtype=torch.float32, device=x.device)

        with torch.cuda.device(x.device.index):
            _scale_norm_bwd_kernel[grid](
                x, dy, dx, _db, _dg, norm, ctx.g_val,
                x.stride(0), dy.stride(0), dx.stride(0),
                M, N, ctx.eps, rows_per_program,
                HAS_BIAS=ctx.has_bias,
                BLOCK_N=BLOCK_N,
            )

        dg = _dg.sum().reshape(g.shape).to(g.dtype)
        db = _db.sum(0).to(ctx.bias_dtype) if ctx.has_bias else None
        return dx, dg, db, None


class FusedScaleNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        C = int(torch.tensor(self.normalized_shape).prod().item())
        self.C = C
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
        g = self.softplus(self.gamma)
        x_2d = input.reshape(-1, self.C)
        bias_flat = self.bias.reshape(-1) if self.bias is not None else None
        out = ScaleNormFn.apply(x_2d, g, bias_flat, self.eps)
        return out.reshape(input.shape)
