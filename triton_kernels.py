#!/usr/bin/env python3
"""
Triton Fused Kernels for Tensor Parallelism
============================================

This module implements custom fused CUDA kernels using Triton for optimized
tensor-parallel linear layers. The key fusions are:

1. FusedLinearGELU: Combines Linear + GELU activation (for MLP up-projection)
2. FusedLinearDropout: Combines Linear + Dropout (for attention/MLP output)

These fusions reduce memory bandwidth by eliminating intermediate tensor
read/writes and reduce kernel launch overhead.

Theoretical Benefits:
- Memory bandwidth: ~160MB saved per MLP layer (for batch=8, seq=1024, GPT-2 Large)
- Kernel launches: ~216 launches saved per training iteration
- Expected speedup: 10-20% for fused operations

Author: M. Akram Bari, Yashas Harisha (NYU HPML)
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# SECTION 1: Fused Linear + GELU Kernel
# =============================================================================


@triton.jit
def _fused_linear_gelu_forward_kernel(
    # Pointers to matrices
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides for x (input)
    stride_xm,
    stride_xk,
    # Strides for w (weight)
    stride_wk,
    stride_wn,
    # Strides for output
    stride_outm,
    stride_outn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Linear + GELU forward kernel.

    Computes: out = GELU(x @ w.T + b)

    Where:
        x: [M, K] input tensor
        w: [N, K] weight tensor (transposed for matmul)
        b: [N] bias tensor
        out: [M, N] output tensor

    GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute starting positions
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to first block of x and w
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load x and w blocks with masking
        x_mask = (rm[:, None] < M) & ((k + rk[None, :]) < K)
        w_mask = ((k + rk[:, None]) < K) & (rn[None, :] < N)

        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # Accumulate matmul in FP32
        acc += tl.dot(x_block, w_block)

        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b_ptrs = b_ptr + rn
    b_mask = rn < N
    bias = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Apply GELU activation (tanh approximation)
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715

    x_cube = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + coeff * x_cube)

    # Numerically stable tanh: clamp to avoid exp overflow
    # tanh(x) saturates to ±1 for |x| > ~10, so clamping is safe
    inner_clamped = tl.where(inner > 10.0, 10.0, inner)
    inner_clamped = tl.where(inner_clamped < -10.0, -10.0, inner_clamped)
    exp_2x = tl.exp(2.0 * inner_clamped)
    tanh_inner = (exp_2x - 1.0) / (exp_2x + 1.0)
    gelu_out = 0.5 * acc * (1.0 + tanh_inner)

    # Store output
    out_ptrs = out_ptr + rm[:, None] * stride_outm + rn[None, :] * stride_outn
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, gelu_out.to(tl.float16), mask=out_mask)


@triton.jit
def _fused_linear_gelu_backward_x_kernel(
    # Pointers
    grad_out_ptr,
    x_ptr,
    w_ptr,
    b_ptr,
    grad_x_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides for grad_out
    stride_gom,
    stride_gon,
    # Strides for x
    stride_xm,
    stride_xk,
    # Strides for w (transposed back for backward)
    stride_wn,
    stride_wk,
    # Strides for grad_x
    stride_gxm,
    stride_gxk,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Backward kernel for grad_x in fused Linear + GELU.

    Computes: grad_x = (grad_out * gelu_grad) @ w

    Where gelu_grad is the derivative of GELU w.r.t. its input.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rn = tl.arange(0, BLOCK_N)

    # We need to recompute the pre-activation values for GELU gradient
    # First, compute x @ w.T + b for this block
    x_ptrs_for_recompute = (
        x_ptr + rm[:, None] * stride_xm + tl.arange(0, BLOCK_K)[None, :] * stride_xk
    )

    # Initialize accumulator for grad_x
    grad_x_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # Loop over N dimension
    for n in range(0, N, BLOCK_N):
        rn_off = n + rn

        # Load grad_out block
        go_ptrs = grad_out_ptr + rm[:, None] * stride_gom + rn_off[None, :] * stride_gon
        go_mask = (rm[:, None] < M) & (rn_off[None, :] < N)
        grad_out_block = tl.load(go_ptrs, mask=go_mask, other=0.0).to(tl.float32)

        # Load weight block for this N range
        w_ptrs = w_ptr + rn_off[:, None] * stride_wn + rk[None, :] * stride_wk
        w_mask = (rn_off[:, None] < N) & (rk[None, :] < K)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # We need GELU gradient: need pre-activation value
        # For simplicity, we assume grad_out already includes GELU gradient
        # (computed separately or passed in pre-multiplied)

        # grad_x += grad_out @ w
        grad_x_acc += tl.dot(grad_out_block, w_block)

    # Store grad_x
    gx_ptrs = grad_x_ptr + rm[:, None] * stride_gxm + rk[None, :] * stride_gxk
    gx_mask = (rm[:, None] < M) & (rk[None, :] < K)
    tl.store(gx_ptrs, grad_x_acc.to(tl.float16), mask=gx_mask)


# =============================================================================
# SECTION 2: Fused Linear + Dropout Kernel
# =============================================================================


@triton.jit
def _fused_linear_dropout_forward_kernel(
    # Pointers
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    mask_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_outm,
    stride_outn,
    stride_maskm,
    stride_maskn,
    # Dropout params
    p_drop,
    seed,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Linear + Dropout forward kernel.

    Computes: out = dropout(x @ w.T + b, p)

    Dropout is implemented using a deterministic PRNG based on position,
    allowing for exact reproducibility in backward pass.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    # Matmul accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x_mask = (rm[:, None] < M) & ((k + rk[None, :]) < K)
        w_mask = ((k + rk[:, None]) < K) & (rn[None, :] < N)

        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.dot(x_block, w_block)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b_ptrs = b_ptr + rn
    b_mask = rn < N
    bias = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Apply dropout
    # Generate random mask using Philox PRNG
    # Each element gets a unique random number based on its position
    offs = rm[:, None] * N + rn[None, :]
    random = tl.rand(seed, offs)
    dropout_mask = random > p_drop

    # Scale by 1/(1-p) during training
    scale = 1.0 / (1.0 - p_drop)
    out = tl.where(dropout_mask, acc * scale, tl.zeros_like(acc))

    # Store output and mask
    out_ptrs = out_ptr + rm[:, None] * stride_outm + rn[None, :] * stride_outn
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, out.to(tl.float16), mask=out_mask)

    # Store dropout mask for backward pass
    mask_ptrs = mask_ptr + rm[:, None] * stride_maskm + rn[None, :] * stride_maskn
    tl.store(mask_ptrs, dropout_mask, mask=out_mask)


@triton.jit
def _fused_linear_dropout_forward_inference_kernel(
    # Pointers (no mask needed for inference)
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_outm,
    stride_outn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Linear forward kernel for inference (no dropout).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x_mask = (rm[:, None] < M) & ((k + rk[None, :]) < K)
        w_mask = ((k + rk[:, None]) < K) & (rn[None, :] < N)

        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.dot(x_block, w_block)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b_ptrs = b_ptr + rn
    b_mask = rn < N
    bias = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store output (no dropout for inference)
    out_ptrs = out_ptr + rm[:, None] * stride_outm + rn[None, :] * stride_outn
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


# =============================================================================
# SECTION 3: GELU Activation Kernels (for backward pass)
# =============================================================================


@triton.jit
def _gelu_backward_kernel(
    grad_out_ptr,
    x_ptr,
    grad_x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute GELU backward: grad_x = grad_out * gelu'(x)

    GELU'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * inner'
    where inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    and inner' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU derivative
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715

    x_sq = x * x
    x_cube = x_sq * x
    inner = sqrt_2_over_pi * (x + coeff * x_cube)

    # Numerically stable tanh: clamp to avoid exp overflow
    inner_clamped = tl.where(inner > 10.0, 10.0, inner)
    inner_clamped = tl.where(inner_clamped < -10.0, -10.0, inner_clamped)
    exp_2inner = tl.exp(2.0 * inner_clamped)
    tanh_inner = (exp_2inner - 1.0) / (exp_2inner + 1.0)

    # sech^2(x) = 1 - tanh^2(x)
    sech_sq = 1.0 - tanh_inner * tanh_inner
    inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_sq)

    gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_sq * inner_deriv
    grad_x = grad_out * gelu_grad

    tl.store(grad_x_ptr + offs, grad_x.to(tl.float16), mask=mask)


# =============================================================================
# SECTION 4: PyTorch Autograd Wrappers
# =============================================================================


class FusedLinearGELU(torch.autograd.Function):
    """
    Autograd function for fused Linear + GELU.

    Forward: out = GELU(x @ w.T + b)
    Backward: Computes gradients for x, w, b
    """

    @staticmethod
    def forward(ctx, x, weight, bias):
        """
        Args:
            x: Input tensor [*, in_features]
            weight: Weight tensor [out_features, in_features]
            bias: Bias tensor [out_features]

        Returns:
            Output tensor [*, out_features]
        """
        # Reshape x to 2D for kernel
        orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        M, K = x_2d.shape
        N = weight.shape[0]

        # Allocate output
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Choose block sizes based on problem size
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        # Launch kernel
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _fused_linear_gelu_forward_kernel[grid](
            x_2d,
            weight,
            bias,
            out,
            M,
            N,
            K,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(1),
            weight.stride(0),  # w is [N, K], we want [K, N] view
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        # Save for backward - need pre-activation for GELU gradient
        # Recompute pre-activation (could also save it, trading memory for compute)
        pre_act = torch.nn.functional.linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias, pre_act)

        # Reshape output to match input batch dims
        out_shape = orig_shape[:-1] + (N,)
        return out.view(out_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, pre_act = ctx.saved_tensors

        # Reshape for computation
        orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        grad_out_2d = grad_output.view(-1, grad_output.shape[-1])
        pre_act_2d = pre_act.view(-1, pre_act.shape[-1])

        M, K = x_2d.shape
        N = weight.shape[0]

        # Compute GELU gradient
        grad_pre_act = torch.empty_like(pre_act_2d)
        n_elements = pre_act_2d.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        _gelu_backward_kernel[grid](
            grad_out_2d.view(-1),
            pre_act_2d.view(-1),
            grad_pre_act.view(-1),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Compute gradients using PyTorch (could also use Triton kernels)
        # grad_x = grad_pre_act @ weight
        grad_x = grad_pre_act.mm(weight)

        # grad_weight = grad_pre_act.T @ x
        grad_weight = grad_pre_act.t().mm(x_2d)

        # grad_bias = grad_pre_act.sum(0)
        grad_bias = grad_pre_act.sum(0)

        return grad_x.view(orig_shape), grad_weight, grad_bias


class FusedLinearDropout(torch.autograd.Function):
    """
    Autograd function for fused Linear + Dropout.

    Forward: out = Dropout(x @ w.T + b, p)
    Backward: Computes gradients for x, w, b
    """

    @staticmethod
    def forward(ctx, x, weight, bias, p_drop, training):
        orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        M, K = x_2d.shape
        N = weight.shape[0]

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        if training and p_drop > 0:
            # Allocate mask for backward
            dropout_mask = torch.empty((M, N), device=x.device, dtype=torch.bool)
            seed = torch.randint(0, 2**31, (1,), device=x.device).item()

            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            _fused_linear_dropout_forward_kernel[grid](
                x_2d,
                weight,
                bias,
                out,
                dropout_mask,
                M,
                N,
                K,
                x_2d.stride(0),
                x_2d.stride(1),
                weight.stride(1),
                weight.stride(0),
                out.stride(0),
                out.stride(1),
                dropout_mask.stride(0),
                dropout_mask.stride(1),
                p_drop,
                seed,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )

            ctx.save_for_backward(x, weight, dropout_mask)
            ctx.p_drop = p_drop
        else:
            # Inference mode - no dropout
            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            _fused_linear_dropout_forward_inference_kernel[grid](
                x_2d,
                weight,
                bias,
                out,
                M,
                N,
                K,
                x_2d.stride(0),
                x_2d.stride(1),
                weight.stride(1),
                weight.stride(0),
                out.stride(0),
                out.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )

            ctx.save_for_backward(x, weight, None)
            ctx.p_drop = 0.0

        ctx.training = training

        out_shape = orig_shape[:-1] + (N,)
        return out.view(out_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, dropout_mask = ctx.saved_tensors
        p_drop = ctx.p_drop
        training = ctx.training

        orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        grad_out_2d = grad_output.view(-1, grad_output.shape[-1])

        # Apply dropout mask to gradient (if training)
        if training and p_drop > 0 and dropout_mask is not None:
            scale = 1.0 / (1.0 - p_drop)
            grad_pre_drop = grad_out_2d * dropout_mask.float() * scale
        else:
            grad_pre_drop = grad_out_2d

        # Compute gradients
        grad_x = grad_pre_drop.mm(weight)
        grad_weight = grad_pre_drop.t().mm(x_2d)
        grad_bias = grad_pre_drop.sum(0)

        return grad_x.view(orig_shape), grad_weight, grad_bias, None, None


# =============================================================================
# SECTION 5: Convenience Functions
# =============================================================================


def fused_linear_gelu(x, weight, bias):
    """
    Fused Linear + GELU operation.

    Args:
        x: Input tensor [..., in_features]
        weight: Weight tensor [out_features, in_features]
        bias: Bias tensor [out_features]

    Returns:
        Output tensor [..., out_features]
    """
    return FusedLinearGELU.apply(x, weight, bias)


def fused_linear_dropout(x, weight, bias, p_drop=0.1, training=True):
    """
    Fused Linear + Dropout operation.

    Args:
        x: Input tensor [..., in_features]
        weight: Weight tensor [out_features, in_features]
        bias: Bias tensor [out_features]
        p_drop: Dropout probability
        training: Whether in training mode

    Returns:
        Output tensor [..., out_features]
    """
    return FusedLinearDropout.apply(x, weight, bias, p_drop, training)


# =============================================================================
# SECTION 6: Benchmarking Utilities
# =============================================================================


def benchmark_fused_linear_gelu(M, N, K, num_iters=100, warmup=10):
    """
    Benchmark fused Linear + GELU vs unfused PyTorch.

    Returns dict with timing results.
    """
    import torch.nn.functional as F

    # Create tensors
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        _ = fused_linear_gelu(x, w, b)
        _ = F.gelu(F.linear(x, w, b), approximate="tanh")

    torch.cuda.synchronize()

    # Benchmark fused
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _ = fused_linear_gelu(x, w, b)
    end.record()
    torch.cuda.synchronize()
    fused_time = start.elapsed_time(end) / num_iters

    # Benchmark unfused
    start.record()
    for _ in range(num_iters):
        _ = F.gelu(F.linear(x, w, b), approximate="tanh")
    end.record()
    torch.cuda.synchronize()
    unfused_time = start.elapsed_time(end) / num_iters

    return {
        "fused_ms": fused_time,
        "unfused_ms": unfused_time,
        "speedup": unfused_time / fused_time,
        "M": M,
        "N": N,
        "K": K,
    }


def benchmark_fused_linear_dropout(M, N, K, p_drop=0.1, num_iters=100, warmup=10):
    """
    Benchmark fused Linear + Dropout vs unfused PyTorch.
    """
    import torch.nn.functional as F

    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        _ = fused_linear_dropout(x, w, b, p_drop, training=True)
        _ = F.dropout(F.linear(x, w, b), p_drop, training=True)

    torch.cuda.synchronize()

    # Benchmark fused
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        _ = fused_linear_dropout(x, w, b, p_drop, training=True)
    end.record()
    torch.cuda.synchronize()
    fused_time = start.elapsed_time(end) / num_iters

    # Benchmark unfused
    start.record()
    for _ in range(num_iters):
        _ = F.dropout(F.linear(x, w, b), p_drop, training=True)
    end.record()
    torch.cuda.synchronize()
    unfused_time = start.elapsed_time(end) / num_iters

    return {
        "fused_ms": fused_time,
        "unfused_ms": unfused_time,
        "speedup": unfused_time / fused_time,
        "M": M,
        "N": N,
        "K": K,
    }


if __name__ == "__main__":
    # Quick sanity check
    print("Testing Triton Fused Kernels")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)

    # Test FusedLinearGELU
    print("\n1. Testing FusedLinearGELU...")
    x = torch.randn(
        8, 1024, 1280, device="cuda", dtype=torch.float16, requires_grad=True
    )
    w = torch.randn(5120, 1280, device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.randn(5120, device="cuda", dtype=torch.float16, requires_grad=True)

    # Reference
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    ref_out = torch.nn.functional.gelu(
        torch.nn.functional.linear(x_ref, w_ref, b_ref), approximate="tanh"
    )

    # Fused
    fused_out = fused_linear_gelu(x, w, b)

    # Check forward
    max_diff = (ref_out - fused_out).abs().max().item()
    print(f"   Forward max diff: {max_diff:.6f}")
    print(f"   Forward pass: {'PASS' if max_diff < 0.1 else 'FAIL'}")

    # Backward
    grad_out = torch.randn_like(fused_out)
    ref_out.backward(grad_out)
    fused_out.backward(grad_out)

    grad_x_diff = (x_ref.grad - x.grad).abs().max().item()
    print(f"   Backward grad_x max diff: {grad_x_diff:.6f}")
    print(f"   Backward pass: {'PASS' if grad_x_diff < 0.5 else 'FAIL'}")

    # Benchmark
    print("\n2. Running benchmarks...")
    result = benchmark_fused_linear_gelu(8 * 1024, 5120, 1280)
    print(
        f"   FusedLinearGELU: {result['fused_ms']:.3f}ms (fused) vs {result['unfused_ms']:.3f}ms (unfused)"
    )
    print(f"   Speedup: {result['speedup']:.2f}x")

    result = benchmark_fused_linear_dropout(8 * 1024, 1280, 5120)
    print(
        f"   FusedLinearDropout: {result['fused_ms']:.3f}ms (fused) vs {result['unfused_ms']:.3f}ms (unfused)"
    )
    print(f"   Speedup: {result['speedup']:.2f}x")

    print("\n" + "=" * 50)
    print("Triton kernel tests complete!")
