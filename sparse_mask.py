# %%
import math

import torch

@torch.jit.script
def fuzzy_gate_impl(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    return 1 / (1 + (x / a) ** (4 * b))
    
@torch.jit.script
def d_fuzzy_gate(grad, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    p = (x / a) ** (4 * b)
    return -4 * b / (x * (1 / p + 2 + p)) * grad

class FuzzyGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x, a, b)
        return fuzzy_gate_impl(x, a, b)
    @staticmethod
    def backward(ctx, grad):
        x, a, b = ctx.saved_tensors
        return d_fuzzy_gate(grad, x, a, b), None, None


def fuzzy_gate(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    return FuzzyGate.apply(x, a, b)

def compute_sparsity(x: torch.Tensor):
    zeros = (x == 0).float().sum()
    n_elem = math.prod(list(x.shape))
    return float(zeros / n_elem)


import triton
import triton.language as tl

@triton.jit
def fuzzy_gate(x, a, b):
    return 1 / (1 + tl.exp(tl.log(tl.abs(x / a)) * (4 * b)))

@triton.jit
def d_fuzzy_gate(x, a, b):
    power = tl.exp(tl.log(tl.abs(x / a)) * (4 * b))
    return tl.where(x == 0, 0, -4 * b / (x * (1 / power + 2 + power)))

@triton.jit
def softmax_kernel(
    input, output, gate, a_gate, b_gate, gate_sum_ptr, n_cols, n_rows,
    BLOCK_SIZE: tl.constexpr):

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    head_idx = row_idx // n_rows

    input_ptrs = input + row_idx * n_cols + col_offsets
    input_row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    gate_ptrs = gate + row_idx * n_cols + col_offsets
    gate_row = tl.load(gate_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    a_val = tl.load(a_gate + head_idx)
    b_val = tl.load(b_gate + head_idx)

    sub_max = input_row - tl.max(input_row, axis=0)
    num = tl.exp(sub_max)
    gates = fuzzy_gate(gate_row, a_val, b_val)
    num *= gates
    denom = tl.sum(num, axis=0)

    out = num / denom
    output_row_start_ptr = output + row_idx * n_cols
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, out, mask=col_offsets < n_cols)

    sum_gates = tl.sum(gates, 0) / (n_cols * n_rows)
    tl.store(gate_sum_ptr + row_idx, sum_gates)


@triton.jit
def d_gated_softmax_kernel(
    grad, gate_grad, prev_output, gate, d_input, d_gate, a_gate, b_gate, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    head_idx = row_idx // n_rows

    row_start = row_idx * n_cols

    input_ptrs = prev_output + row_start + col_offsets
    input_row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0)

    gate_ptrs = gate + row_start + col_offsets
    gate_row = tl.load(gate_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    grad_ptrs = grad + row_start + col_offsets
    grad_row = tl.load(grad_ptrs, mask=col_offsets < n_cols, other=0)

    a_val = tl.load(a_gate + head_idx)
    b_val = tl.load(b_gate + head_idx)

    s = tl.sum(grad_row * input_row, 0)
    dy_dx = input_row * (grad_row - s)
    tl.store(d_input + row_start + col_offsets, dy_dx, mask=col_offsets < n_cols)

    pow = tl.exp(tl.log(tl.abs(gate_row / a_val)) * (4 * b_val))
    gates = 1 / (1 + pow)
    dx_dg = tl.where(gate_row == 0, 0, -4 * b_val / (gate_row * (1 / pow + 2 + pow)))
    
    ds_dg = tl.load(gate_grad + head_idx) / (n_cols * n_rows)
    dy_dg = (dy_dx / gates + ds_dg) * dx_dg
    tl.store(d_gate + row_start + col_offsets, dy_dg, mask=col_offsets < n_cols)


def gated_softmax(x, gate, a_gate, b_gate):
    b, h, n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x)
    gate_sum = torch.empty([b, h, n_rows], dtype=x.dtype, device=x.device)
    softmax_kernel[(b * h * n_rows,)](
        x, y, gate, a_gate, b_gate, gate_sum, n_cols, (n_rows * b * h),
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, gate_sum.sum(-1)

def d_gated_softmax(grad, gate_grad, x, gate, a_gate, b_gate):
    b, h, n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    dy_dx = torch.empty_like(x)
    dy_dg = torch.empty_like(x)
    d_gated_softmax_kernel[(b * h * n_rows,)](
        grad, gate_grad, x, gate, dy_dx, dy_dg, a_gate, b_gate, (n_rows * b * h), n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dy_dx, dy_dg

def forward_naive(a, g, a1, b1):
    xmax = a.max(dim=-1)[0]
    x = (a - xmax.unsqueeze(-1)).exp()
    gate = 1 / (1 + (g / a1) ** (4 * b1))
    x = x * gate
    x = x / (x.sum(-1, keepdim=True))

    gsum = gate.sum(-1).sum(-1) / (g.nelement())
    return x, gsum

def backward_naive(grad, gate_grad, x, g, a1, b1):
    da = x * (grad - (x * grad).sum(-1, keepdim=True))

    p = (g / a1) ** (4 * b1)
    gate = 1 / (1 + p)
    d_gate = -4*b1 / (g * (1 / p + 2 + p))
    d_gate = torch.where(x == 0, torch.zeros_like(d_gate), d_gate)
    dg = (da / gate + (gate_grad / g.nelement())[:, :, None, None]) * d_gate
    return da, dg


class GatedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, g, a1, b1):
        x, g_sum = gated_softmax(a, g, a1, b1)
        ctx.save_for_backward(x, g, a1, b1)
        return x, g_sum
    
    @staticmethod
    def backward(ctx, grad, gate_grad):
        x, g, a1, b1 = ctx.saved_tensors
        #da, dg = d_gated_softmax(grad, a, g, a1, b1)
        da, dg = d_gated_softmax(grad, gate_grad, x, g, a1, b1)
        return da, dg, None, None, None

"""a_gate = torch.tensor([1], dtype=torch.float32, device='cuda')[:, None, None, None].repeat(4, 8, 1, 1)
b_gate = torch.tensor([10], dtype=torch.float32, device='cuda')[:, None, None, None].repeat(4, 8, 1, 1)

x = torch.rand(4, 8, 768, 256, device='cuda')

y, s = gated_softmax(x, x, a_gate, b_gate)

print(torch.isnan(y).any(), torch.isnan(s).any())"""
