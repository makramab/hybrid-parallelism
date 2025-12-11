#!/usr/bin/env python3
"""
Correctness Tests for Triton Fused Kernels
==========================================

This module provides comprehensive tests to verify that the Triton fused kernels
produce numerically correct results compared to PyTorch reference implementations.

Tests include:
1. Forward pass numerical correctness (FusedLinearGELU, FusedLinearDropout)
2. Backward pass gradient correctness
3. Different input sizes and shapes
4. Edge cases (small batches, large batches, etc.)

Run with: python tests/test_triton_kernels.py

Author: M. Akram Bari, Yashas Harisha (NYU HPML)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math

# Import Triton kernels
from triton_kernels import (
    fused_linear_gelu,
    fused_linear_dropout,
    FusedLinearGELU,
    FusedLinearDropout,
)


def check_close(ref, test, name, rtol=1e-2, atol=1e-2):
    """Check if two tensors are close within tolerance."""
    max_diff = (ref - test).abs().max().item()
    mean_diff = (ref - test).abs().mean().item()
    is_close = torch.allclose(ref, test, rtol=rtol, atol=atol)

    status = "PASS" if is_close else "FAIL"
    print(f"  {name}: {status}")
    print(f"    Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    print(f"    Tolerance: rtol={rtol}, atol={atol}")

    return is_close


class TestFusedLinearGELU:
    """Tests for FusedLinearGELU kernel."""

    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16

    def test_forward_correctness(self, M, K, N):
        """Test forward pass numerical correctness."""
        print(f"\n  Testing forward [M={M}, K={K}, N={N}]...")

        x = torch.randn(M, K, device=self.device, dtype=self.dtype)
        w = torch.randn(N, K, device=self.device, dtype=self.dtype)
        b = torch.randn(N, device=self.device, dtype=self.dtype)

        # Reference (PyTorch)
        ref = F.gelu(F.linear(x, w, b), approximate="tanh")

        # Fused (Triton)
        fused = fused_linear_gelu(x, w, b)

        return check_close(ref, fused, "Forward", rtol=0.05, atol=0.05)

    def test_forward_3d_input(self, B, T, K, N):
        """Test forward pass with 3D input [batch, seq, features]."""
        print(f"\n  Testing forward 3D [B={B}, T={T}, K={K}, N={N}]...")

        x = torch.randn(B, T, K, device=self.device, dtype=self.dtype)
        w = torch.randn(N, K, device=self.device, dtype=self.dtype)
        b = torch.randn(N, device=self.device, dtype=self.dtype)

        # Reference
        ref = F.gelu(F.linear(x, w, b), approximate="tanh")

        # Fused
        fused = fused_linear_gelu(x, w, b)

        return check_close(ref, fused, "Forward 3D", rtol=0.05, atol=0.05)

    def test_backward_correctness(self, M, K, N):
        """Test backward pass gradient correctness."""
        print(f"\n  Testing backward [M={M}, K={K}, N={N}]...")

        # Reference inputs
        x_ref = torch.randn(
            M, K, device=self.device, dtype=self.dtype, requires_grad=True
        )
        w_ref = torch.randn(
            N, K, device=self.device, dtype=self.dtype, requires_grad=True
        )
        b_ref = torch.randn(N, device=self.device, dtype=self.dtype, requires_grad=True)

        # Fused inputs (cloned)
        x_fused = x_ref.detach().clone().requires_grad_(True)
        w_fused = w_ref.detach().clone().requires_grad_(True)
        b_fused = b_ref.detach().clone().requires_grad_(True)

        # Forward
        ref_out = F.gelu(F.linear(x_ref, w_ref, b_ref), approximate="tanh")
        fused_out = fused_linear_gelu(x_fused, w_fused, b_fused)

        # Backward
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)
        fused_out.backward(grad_out)

        # Check gradients
        results = []
        results.append(
            check_close(x_ref.grad, x_fused.grad, "grad_x", rtol=0.1, atol=0.1)
        )
        results.append(
            check_close(w_ref.grad, w_fused.grad, "grad_w", rtol=0.1, atol=0.1)
        )
        results.append(
            check_close(b_ref.grad, b_fused.grad, "grad_b", rtol=0.1, atol=0.1)
        )

        return all(results)

    def run_all(self):
        """Run all FusedLinearGELU tests."""
        print("\n" + "=" * 60)
        print("FusedLinearGELU Tests")
        print("=" * 60)

        results = []

        # Test various sizes
        test_cases = [
            (128, 256, 512),  # Small
            (1024, 1280, 5120),  # GPT-2 Large MLP size
            (8192, 1280, 5120),  # Batch * seq for GPT-2
            (256, 256, 256),  # Square
        ]

        for M, K, N in test_cases:
            results.append(self.test_forward_correctness(M, K, N))

        # Test 3D inputs
        results.append(self.test_forward_3d_input(8, 1024, 1280, 5120))
        results.append(self.test_forward_3d_input(4, 512, 256, 1024))

        # Test backward
        results.append(self.test_backward_correctness(1024, 256, 512))
        results.append(self.test_backward_correctness(4096, 1280, 5120))

        passed = sum(results)
        total = len(results)
        print(f"\nFusedLinearGELU: {passed}/{total} tests passed")

        return passed == total


class TestFusedLinearDropout:
    """Tests for FusedLinearDropout kernel."""

    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16

    def test_forward_inference(self, M, K, N):
        """Test forward pass in inference mode (no dropout)."""
        print(f"\n  Testing forward inference [M={M}, K={K}, N={N}]...")

        x = torch.randn(M, K, device=self.device, dtype=self.dtype)
        w = torch.randn(N, K, device=self.device, dtype=self.dtype)
        b = torch.randn(N, device=self.device, dtype=self.dtype)

        # Reference (no dropout in inference)
        ref = F.linear(x, w, b)

        # Fused (training=False)
        fused = fused_linear_dropout(x, w, b, p_drop=0.1, training=False)

        # Use larger tolerances for large matrices due to FP16 accumulation differences
        rtol = 0.05 if M * K > 1000000 else 0.02
        atol = 0.1 if M * K > 1000000 else 0.02
        return check_close(ref, fused, "Forward inference", rtol=rtol, atol=atol)

    def test_forward_training_statistics(self, M, K, N, p_drop=0.1):
        """Test that dropout has correct statistics in training mode."""
        print(f"\n  Testing dropout statistics [M={M}, K={K}, N={N}, p={p_drop}]...")

        x = torch.randn(M, K, device=self.device, dtype=self.dtype)
        w = torch.randn(N, K, device=self.device, dtype=self.dtype)
        b = torch.randn(N, device=self.device, dtype=self.dtype)

        # Run multiple times and check statistics
        outputs = []
        for _ in range(10):
            out = fused_linear_dropout(x, w, b, p_drop=p_drop, training=True)
            outputs.append(out)

        # Stack outputs
        stacked = torch.stack(outputs)

        # Check that outputs vary (dropout is stochastic)
        variance = stacked.var(dim=0).mean().item()
        print(f"    Output variance across runs: {variance:.6f}")

        # Check that approximately p_drop fraction of values are zero
        # (After scaling, zeros should still be zero)
        # Actually, with scaling, non-zero values are scaled by 1/(1-p), so mean should be preserved

        # Check mean is approximately preserved
        ref_linear = F.linear(x, w, b)
        ref_mean = ref_linear.float().mean().item()
        fused_mean = stacked.float().mean().item()

        # Use absolute difference for near-zero means, relative otherwise
        abs_diff = abs(ref_mean - fused_mean)
        if abs(ref_mean) < 0.1:
            # For near-zero means, use absolute difference
            mean_diff = abs_diff
            threshold = 0.1  # Allow 0.1 absolute difference
            print(f"    Reference mean: {ref_mean:.4f}, Fused mean: {fused_mean:.4f}")
            print(
                f"    Absolute mean difference: {mean_diff:.4f} (threshold: {threshold})"
            )
            return mean_diff < threshold and variance > 0.001
        else:
            # For larger means, use relative difference
            mean_diff = abs_diff / abs(ref_mean)
            threshold = 0.3
            print(f"    Reference mean: {ref_mean:.4f}, Fused mean: {fused_mean:.4f}")
            print(
                f"    Relative mean difference: {mean_diff:.4f} (threshold: {threshold})"
            )
            return mean_diff < threshold and variance > 0.001

    def test_backward_correctness(self, M, K, N):
        """Test backward pass gradient correctness (inference mode for exact comparison)."""
        print(f"\n  Testing backward [M={M}, K={K}, N={N}]...")

        # In inference mode (no dropout), gradients should match exactly
        x_ref = torch.randn(
            M, K, device=self.device, dtype=self.dtype, requires_grad=True
        )
        w_ref = torch.randn(
            N, K, device=self.device, dtype=self.dtype, requires_grad=True
        )
        b_ref = torch.randn(N, device=self.device, dtype=self.dtype, requires_grad=True)

        x_fused = x_ref.detach().clone().requires_grad_(True)
        w_fused = w_ref.detach().clone().requires_grad_(True)
        b_fused = b_ref.detach().clone().requires_grad_(True)

        # Forward (inference mode)
        ref_out = F.linear(x_ref, w_ref, b_ref)
        fused_out = fused_linear_dropout(
            x_fused, w_fused, b_fused, p_drop=0.0, training=False
        )

        # Backward
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)
        fused_out.backward(grad_out)

        # Check gradients
        results = []
        results.append(
            check_close(x_ref.grad, x_fused.grad, "grad_x", rtol=0.05, atol=0.05)
        )
        results.append(
            check_close(w_ref.grad, w_fused.grad, "grad_w", rtol=0.05, atol=0.05)
        )
        results.append(
            check_close(b_ref.grad, b_fused.grad, "grad_b", rtol=0.05, atol=0.05)
        )

        return all(results)

    def run_all(self):
        """Run all FusedLinearDropout tests."""
        print("\n" + "=" * 60)
        print("FusedLinearDropout Tests")
        print("=" * 60)

        results = []

        # Test inference mode
        test_cases = [
            (128, 256, 512),
            (1024, 1280, 1280),
            (8192, 5120, 1280),
        ]

        for M, K, N in test_cases:
            results.append(self.test_forward_inference(M, K, N))

        # Test training statistics
        results.append(
            self.test_forward_training_statistics(4096, 1280, 1280, p_drop=0.1)
        )
        results.append(
            self.test_forward_training_statistics(4096, 1280, 1280, p_drop=0.2)
        )

        # Test backward
        results.append(self.test_backward_correctness(1024, 256, 512))
        results.append(self.test_backward_correctness(4096, 1280, 1280))

        passed = sum(results)
        total = len(results)
        print(f"\nFusedLinearDropout: {passed}/{total} tests passed")

        return passed == total


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16

    def test_large_values(self):
        """Test with large input values."""
        print("\n  Testing large values...")

        # Use moderately large values (not extreme) to test stability
        # Values * 3 is enough to test edge cases without causing FP16 overflow
        x = torch.randn(1024, 256, device=self.device, dtype=self.dtype) * 3
        w = torch.randn(512, 256, device=self.device, dtype=self.dtype)
        b = torch.randn(512, device=self.device, dtype=self.dtype)

        fused = fused_linear_gelu(x, w, b)

        has_nan = torch.isnan(fused).any().item()
        has_inf = torch.isinf(fused).any().item()

        print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
        return not has_nan and not has_inf

    def test_small_values(self):
        """Test with small input values."""
        print("\n  Testing small values...")

        x = torch.randn(1024, 256, device=self.device, dtype=self.dtype) * 0.01
        w = torch.randn(512, 256, device=self.device, dtype=self.dtype) * 0.01
        b = torch.randn(512, device=self.device, dtype=self.dtype) * 0.01

        fused = fused_linear_gelu(x, w, b)

        has_nan = torch.isnan(fused).any().item()
        has_inf = torch.isinf(fused).any().item()

        print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
        return not has_nan and not has_inf

    def test_zero_input(self):
        """Test with zero input."""
        print("\n  Testing zero input...")

        x = torch.zeros(1024, 256, device=self.device, dtype=self.dtype)
        w = torch.randn(512, 256, device=self.device, dtype=self.dtype)
        b = torch.randn(512, device=self.device, dtype=self.dtype)

        fused = fused_linear_gelu(x, w, b)

        # GELU(b) should be the output since x is zero
        ref = F.gelu(b.unsqueeze(0).expand(1024, -1), approximate="tanh")

        return check_close(ref, fused, "Zero input", rtol=0.1, atol=0.1)

    def run_all(self):
        """Run all numerical stability tests."""
        print("\n" + "=" * 60)
        print("Numerical Stability Tests")
        print("=" * 60)

        results = []
        results.append(self.test_large_values())
        results.append(self.test_small_values())
        results.append(self.test_zero_input())

        passed = sum(results)
        total = len(results)
        print(f"\nNumerical Stability: {passed}/{total} tests passed")

        return passed == total


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Triton Fused Kernels - Correctness Tests")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return False

    results = []

    # Run test suites
    results.append(TestFusedLinearGELU().run_all())
    results.append(TestFusedLinearDropout().run_all())
    results.append(TestNumericalStability().run_all())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(results)
    if all_passed:
        print("All test suites PASSED!")
    else:
        print("Some test suites FAILED!")
        for i, (name, passed) in enumerate(
            zip(
                ["FusedLinearGELU", "FusedLinearDropout", "NumericalStability"], results
            )
        ):
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
