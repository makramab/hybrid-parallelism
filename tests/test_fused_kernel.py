#!/usr/bin/env python3
"""
Test script for fused bias + GELU CUDA kernel.

Tests:
1. Correctness: Compare fused kernel output vs PyTorch reference
2. Gradient correctness: Compare gradients vs PyTorch autograd
3. Performance: Benchmark fused vs unfused operations

Usage:
    python tests/test_fused_kernel.py
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_correctness():
    """Test that fused kernel produces correct results."""
    print("=" * 60)
    print("TEST: Fused Bias+GELU Correctness")
    print("=" * 60)

    try:
        from fused_kernels.fused_bias_gelu import fused_bias_gelu
    except Exception as e:
        print(f"Could not import fused_bias_gelu: {e}")
        print("Make sure you're running on a CUDA-enabled machine.")
        return False

    # Test configurations
    test_cases = [
        {"batch": 2, "seq": 128, "hidden": 256, "dtype": torch.float32},
        {"batch": 4, "seq": 512, "hidden": 1024, "dtype": torch.float32},
        {
            "batch": 8,
            "seq": 1024,
            "hidden": 5120,
            "dtype": torch.float32,
        },  # GPT-2 Large MLP
        {"batch": 2, "seq": 128, "hidden": 256, "dtype": torch.float16},
        {"batch": 4, "seq": 512, "hidden": 1024, "dtype": torch.float16},
    ]

    all_passed = True

    for i, tc in enumerate(test_cases):
        B, T, H = tc["batch"], tc["seq"], tc["hidden"]
        dtype = tc["dtype"]

        # Create test inputs
        x = torch.randn(B, T, H, dtype=dtype, device="cuda")
        bias = torch.randn(H, dtype=dtype, device="cuda")

        # Reference: PyTorch implementation
        ref = F.gelu(x + bias, approximate="tanh")

        # Fused kernel
        fused = fused_bias_gelu(x, bias)

        # Compare
        if dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-4, 1e-4

        is_close = torch.allclose(ref, fused, rtol=rtol, atol=atol)
        max_diff = (ref - fused).abs().max().item()

        status = "PASS" if is_close else "FAIL"
        print(f"  [{status}] Case {i + 1}: B={B}, T={T}, H={H}, dtype={dtype}")
        print(f"         Max diff: {max_diff:.2e}")

        if not is_close:
            all_passed = False

    return all_passed


def test_gradients():
    """Test that gradients are computed correctly."""
    print("\n" + "=" * 60)
    print("TEST: Gradient Correctness")
    print("=" * 60)

    try:
        from fused_kernels.fused_bias_gelu import fused_bias_gelu
    except Exception as e:
        print(f"Could not import fused_bias_gelu: {e}")
        return False

    test_cases = [
        {"batch": 2, "seq": 64, "hidden": 128},
        {"batch": 4, "seq": 256, "hidden": 512},
    ]

    all_passed = True

    for i, tc in enumerate(test_cases):
        B, T, H = tc["batch"], tc["seq"], tc["hidden"]

        # Create inputs with gradients
        x_ref = torch.randn(
            B, T, H, dtype=torch.float32, device="cuda", requires_grad=True
        )
        bias_ref = torch.randn(
            H, dtype=torch.float32, device="cuda", requires_grad=True
        )

        # Clone for fused kernel
        x_fused = x_ref.detach().clone().requires_grad_(True)
        bias_fused = bias_ref.detach().clone().requires_grad_(True)

        # Forward pass
        out_ref = F.gelu(x_ref + bias_ref, approximate="tanh")
        out_fused = fused_bias_gelu(x_fused, bias_fused)

        # Create gradient
        grad_out = torch.randn_like(out_ref)

        # Backward pass
        out_ref.backward(grad_out)
        out_fused.backward(grad_out)

        # Compare gradients
        rtol, atol = 1e-4, 1e-4

        grad_x_close = torch.allclose(x_ref.grad, x_fused.grad, rtol=rtol, atol=atol)
        grad_bias_close = torch.allclose(
            bias_ref.grad, bias_fused.grad, rtol=rtol, atol=atol
        )

        max_diff_x = (x_ref.grad - x_fused.grad).abs().max().item()
        max_diff_bias = (bias_ref.grad - bias_fused.grad).abs().max().item()

        status = "PASS" if (grad_x_close and grad_bias_close) else "FAIL"
        print(f"  [{status}] Case {i + 1}: B={B}, T={T}, H={H}")
        print(f"         grad_x max diff: {max_diff_x:.2e}")
        print(f"         grad_bias max diff: {max_diff_bias:.2e}")

        if not (grad_x_close and grad_bias_close):
            all_passed = False

    return all_passed


def test_performance():
    """Benchmark fused vs unfused operations."""
    print("\n" + "=" * 60)
    print("TEST: Performance Benchmark")
    print("=" * 60)

    try:
        from fused_kernels.fused_bias_gelu import fused_bias_gelu
    except Exception as e:
        print(f"Could not import fused_bias_gelu: {e}")
        return False

    # GPT-2 Large dimensions
    B, T, H = 8, 1024, 5120  # intermediate_size for GPT-2 Large

    x = torch.randn(B, T, H, dtype=torch.float16, device="cuda")
    bias = torch.randn(H, dtype=torch.float16, device="cuda")

    # Warmup
    for _ in range(10):
        _ = F.gelu(x + bias, approximate="tanh")
        _ = fused_bias_gelu(x, bias)
    torch.cuda.synchronize()

    # Benchmark unfused (PyTorch)
    n_iters = 100
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = F.gelu(x + bias, approximate="tanh")
    torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark fused
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = fused_bias_gelu(x, bias)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / n_iters * 1000

    speedup = unfused_time / fused_time
    print(f"  Configuration: B={B}, T={T}, H={H}, dtype=float16")
    print(f"  PyTorch (x + bias + gelu): {unfused_time:.3f} ms")
    print(f"  Fused kernel:             {fused_time:.3f} ms")
    print(f"  Speedup:                  {speedup:.2f}x")

    # Memory comparison
    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        y = F.gelu(x + bias, approximate="tanh")
    unfused_mem = torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        y = fused_bias_gelu(x, bias)
    fused_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n  Memory (PyTorch):    {unfused_mem:.1f} MB")
    print(f"  Memory (Fused):      {fused_mem:.1f} MB")
    print(
        f"  Memory saved:        {unfused_mem - fused_mem:.1f} MB ({(unfused_mem - fused_mem) / unfused_mem * 100:.1f}%)"
    )

    return True


def test_with_linear():
    """Test fused kernel in realistic Linear + GELU scenario."""
    print("\n" + "=" * 60)
    print("TEST: Linear + Fused GELU Integration")
    print("=" * 60)

    try:
        from fused_kernels.fused_bias_gelu import fused_bias_gelu
    except Exception as e:
        print(f"Could not import fused_bias_gelu: {e}")
        return False

    # Simulate GPT-2 Large MLP up-projection
    B, T = 4, 512
    hidden_size = 1280
    intermediate_size = 5120

    # Create linear layer
    linear = torch.nn.Linear(hidden_size, intermediate_size, bias=False).cuda().half()
    bias = torch.nn.Parameter(
        torch.randn(intermediate_size, device="cuda", dtype=torch.float16)
    )

    x = torch.randn(B, T, hidden_size, dtype=torch.float16, device="cuda")

    # Reference: Linear with bias built-in + GELU
    linear_with_bias = (
        torch.nn.Linear(hidden_size, intermediate_size, bias=True).cuda().half()
    )
    linear_with_bias.weight.data.copy_(linear.weight.data)
    linear_with_bias.bias.data.copy_(bias.data)

    out_ref = F.gelu(linear_with_bias(x), approximate="tanh")

    # Fused: Linear without bias + fused bias+GELU
    out_fused = fused_bias_gelu(linear(x), bias)

    # Compare
    is_close = torch.allclose(out_ref, out_fused, rtol=1e-2, atol=1e-2)
    max_diff = (out_ref - out_fused).abs().max().item()

    status = "PASS" if is_close else "FAIL"
    print(f"  [{status}] Linear + Fused GELU")
    print(f"         Max diff: {max_diff:.2e}")

    return is_close


def main():
    print("\n" + "=" * 60)
    print("FUSED BIAS+GELU CUDA KERNEL TESTS")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. These tests require a GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    results = []

    # Run tests
    results.append(("Correctness", test_correctness()))
    results.append(("Gradients", test_gradients()))
    results.append(("Performance", test_performance()))
    results.append(("Linear Integration", test_with_linear()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
