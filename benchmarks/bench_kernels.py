#!/usr/bin/env python3
"""
Performance Benchmarks for Triton Fused Kernels
================================================

This module provides comprehensive benchmarks comparing Triton fused kernels
against PyTorch reference implementations.

Metrics measured:
1. Execution time (forward and backward)
2. Memory bandwidth utilization
3. Peak memory usage
4. Speedup over baseline

Run with: python benchmarks/bench_kernels.py

Author: M. Akram Bari, Yashas Harisha (NYU HPML)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
from datetime import datetime
import argparse


def benchmark_kernel(fn, num_iters=100, warmup=20):
    """
    Benchmark a kernel function.

    Args:
        fn: Function to benchmark (should take no arguments)
        num_iters: Number of iterations for timing
        warmup: Number of warmup iterations

    Returns:
        dict with timing results
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_iters
    return elapsed_ms


def benchmark_fused_linear_gelu(M, N, K, num_iters=100):
    """
    Benchmark FusedLinearGELU vs PyTorch.

    Args:
        M: Batch size (or batch * seq_len)
        N: Output features
        K: Input features
        num_iters: Number of iterations

    Returns:
        dict with benchmark results
    """
    from triton_kernels import fused_linear_gelu

    # Create tensors
    x = torch.randn(M, K, device='cuda', dtype=torch.float16, requires_grad=True)
    w = torch.randn(N, K, device='cuda', dtype=torch.float16, requires_grad=True)
    b = torch.randn(N, device='cuda', dtype=torch.float16, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    # Forward benchmarks
    def fused_forward():
        return fused_linear_gelu(x, w, b)

    def pytorch_forward():
        return F.gelu(F.linear(x_ref, w_ref, b_ref), approximate='tanh')

    fused_fwd_ms = benchmark_kernel(fused_forward, num_iters)
    pytorch_fwd_ms = benchmark_kernel(pytorch_forward, num_iters)

    # Backward benchmarks
    grad_out = torch.randn(M, N, device='cuda', dtype=torch.float16)

    def fused_backward():
        x.grad = None
        w.grad = None
        b.grad = None
        out = fused_linear_gelu(x, w, b)
        out.backward(grad_out)

    def pytorch_backward():
        x_ref.grad = None
        w_ref.grad = None
        b_ref.grad = None
        out = F.gelu(F.linear(x_ref, w_ref, b_ref), approximate='tanh')
        out.backward(grad_out)

    fused_bwd_ms = benchmark_kernel(fused_backward, num_iters // 2)
    pytorch_bwd_ms = benchmark_kernel(pytorch_backward, num_iters // 2)

    # Calculate memory bandwidth
    # Unfused: Read X, W, Y; Write Y, result = 2*(M*K + N*K + M*N + M*N) bytes (FP16)
    # Fused: Read X, W; Write result = 2*(M*K + N*K + M*N) bytes (FP16)
    unfused_bytes = 2 * (M * K + N * K + 2 * M * N + M * N)  # Extra M*N for intermediate
    fused_bytes = 2 * (M * K + N * K + M * N)

    unfused_bw = (unfused_bytes / 1e9) / (pytorch_fwd_ms / 1000)  # GB/s
    fused_bw = (fused_bytes / 1e9) / (fused_fwd_ms / 1000)  # GB/s

    return {
        "operation": "FusedLinearGELU",
        "shape": {"M": M, "N": N, "K": K},
        "forward": {
            "fused_ms": fused_fwd_ms,
            "pytorch_ms": pytorch_fwd_ms,
            "speedup": pytorch_fwd_ms / fused_fwd_ms,
        },
        "backward": {
            "fused_ms": fused_bwd_ms,
            "pytorch_ms": pytorch_bwd_ms,
            "speedup": pytorch_bwd_ms / fused_bwd_ms,
        },
        "total": {
            "fused_ms": fused_fwd_ms + fused_bwd_ms,
            "pytorch_ms": pytorch_fwd_ms + pytorch_bwd_ms,
            "speedup": (pytorch_fwd_ms + pytorch_bwd_ms) / (fused_fwd_ms + fused_bwd_ms),
        },
        "memory_bandwidth_gb_s": {
            "fused": fused_bw,
            "pytorch": unfused_bw,
        },
        "bytes_saved": unfused_bytes - fused_bytes,
    }


def benchmark_fused_linear_dropout(M, N, K, p_drop=0.1, num_iters=100):
    """
    Benchmark FusedLinearDropout vs PyTorch.
    """
    from triton_kernels import fused_linear_dropout

    x = torch.randn(M, K, device='cuda', dtype=torch.float16, requires_grad=True)
    w = torch.randn(N, K, device='cuda', dtype=torch.float16, requires_grad=True)
    b = torch.randn(N, device='cuda', dtype=torch.float16, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    # Forward benchmarks
    def fused_forward():
        return fused_linear_dropout(x, w, b, p_drop, training=True)

    def pytorch_forward():
        return F.dropout(F.linear(x_ref, w_ref, b_ref), p_drop, training=True)

    fused_fwd_ms = benchmark_kernel(fused_forward, num_iters)
    pytorch_fwd_ms = benchmark_kernel(pytorch_forward, num_iters)

    # Backward benchmarks
    grad_out = torch.randn(M, N, device='cuda', dtype=torch.float16)

    def fused_backward():
        x.grad = None
        w.grad = None
        b.grad = None
        out = fused_linear_dropout(x, w, b, p_drop, training=True)
        out.backward(grad_out)

    def pytorch_backward():
        x_ref.grad = None
        w_ref.grad = None
        b_ref.grad = None
        out = F.dropout(F.linear(x_ref, w_ref, b_ref), p_drop, training=True)
        out.backward(grad_out)

    fused_bwd_ms = benchmark_kernel(fused_backward, num_iters // 2)
    pytorch_bwd_ms = benchmark_kernel(pytorch_backward, num_iters // 2)

    return {
        "operation": "FusedLinearDropout",
        "shape": {"M": M, "N": N, "K": K},
        "dropout_p": p_drop,
        "forward": {
            "fused_ms": fused_fwd_ms,
            "pytorch_ms": pytorch_fwd_ms,
            "speedup": pytorch_fwd_ms / fused_fwd_ms,
        },
        "backward": {
            "fused_ms": fused_bwd_ms,
            "pytorch_ms": pytorch_bwd_ms,
            "speedup": pytorch_bwd_ms / fused_bwd_ms,
        },
        "total": {
            "fused_ms": fused_fwd_ms + fused_bwd_ms,
            "pytorch_ms": pytorch_fwd_ms + pytorch_bwd_ms,
            "speedup": (pytorch_fwd_ms + pytorch_bwd_ms) / (fused_fwd_ms + fused_bwd_ms),
        },
    }


def benchmark_memory_usage(M, N, K):
    """
    Benchmark peak memory usage.
    """
    from triton_kernels import fused_linear_gelu

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # PyTorch unfused
    x = torch.randn(M, K, device='cuda', dtype=torch.float16, requires_grad=True)
    w = torch.randn(N, K, device='cuda', dtype=torch.float16, requires_grad=True)
    b = torch.randn(N, device='cuda', dtype=torch.float16, requires_grad=True)

    out = F.gelu(F.linear(x, w, b), approximate='tanh')
    out.backward(torch.randn_like(out))

    pytorch_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB

    # Clear
    del x, w, b, out
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Triton fused
    x = torch.randn(M, K, device='cuda', dtype=torch.float16, requires_grad=True)
    w = torch.randn(N, K, device='cuda', dtype=torch.float16, requires_grad=True)
    b = torch.randn(N, device='cuda', dtype=torch.float16, requires_grad=True)

    out = fused_linear_gelu(x, w, b)
    out.backward(torch.randn_like(out))

    fused_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB

    return {
        "shape": {"M": M, "N": N, "K": K},
        "pytorch_peak_mb": pytorch_peak,
        "fused_peak_mb": fused_peak,
        "memory_saved_mb": pytorch_peak - fused_peak,
        "memory_saved_percent": (pytorch_peak - fused_peak) / pytorch_peak * 100,
    }


def run_gpt2_benchmarks():
    """Run benchmarks for GPT-2 Large sizes."""
    print("\n" + "=" * 70)
    print("GPT-2 Large Benchmark (batch=8, seq=1024, hidden=1280, intermediate=5120)")
    print("=" * 70)

    # GPT-2 Large dimensions
    batch = 8
    seq = 1024
    hidden = 1280
    intermediate = 5120

    M = batch * seq  # 8192

    results = []

    # MLP up-projection: [batch*seq, hidden] -> [batch*seq, intermediate] with GELU
    print("\n--- MLP Up-Projection (Linear + GELU) ---")
    result = benchmark_fused_linear_gelu(M, intermediate, hidden, num_iters=50)
    results.append(result)
    print(f"Forward:  Fused={result['forward']['fused_ms']:.3f}ms, PyTorch={result['forward']['pytorch_ms']:.3f}ms, Speedup={result['forward']['speedup']:.2f}x")
    print(f"Backward: Fused={result['backward']['fused_ms']:.3f}ms, PyTorch={result['backward']['pytorch_ms']:.3f}ms, Speedup={result['backward']['speedup']:.2f}x")
    print(f"Total:    Speedup={result['total']['speedup']:.2f}x")
    print(f"Memory BW: Fused={result['memory_bandwidth_gb_s']['fused']:.1f} GB/s, PyTorch={result['memory_bandwidth_gb_s']['pytorch']:.1f} GB/s")

    # MLP down-projection: [batch*seq, intermediate] -> [batch*seq, hidden] with Dropout
    print("\n--- MLP Down-Projection (Linear + Dropout) ---")
    result = benchmark_fused_linear_dropout(M, hidden, intermediate, p_drop=0.1, num_iters=50)
    results.append(result)
    print(f"Forward:  Fused={result['forward']['fused_ms']:.3f}ms, PyTorch={result['forward']['pytorch_ms']:.3f}ms, Speedup={result['forward']['speedup']:.2f}x")
    print(f"Backward: Fused={result['backward']['fused_ms']:.3f}ms, PyTorch={result['backward']['pytorch_ms']:.3f}ms, Speedup={result['backward']['speedup']:.2f}x")
    print(f"Total:    Speedup={result['total']['speedup']:.2f}x")

    # Attention output projection: [batch*seq, hidden] -> [batch*seq, hidden] with Dropout
    print("\n--- Attention Output Projection (Linear + Dropout) ---")
    result = benchmark_fused_linear_dropout(M, hidden, hidden, p_drop=0.1, num_iters=50)
    results.append(result)
    print(f"Forward:  Fused={result['forward']['fused_ms']:.3f}ms, PyTorch={result['forward']['pytorch_ms']:.3f}ms, Speedup={result['forward']['speedup']:.2f}x")
    print(f"Backward: Fused={result['backward']['fused_ms']:.3f}ms, PyTorch={result['backward']['pytorch_ms']:.3f}ms, Speedup={result['backward']['speedup']:.2f}x")
    print(f"Total:    Speedup={result['total']['speedup']:.2f}x")

    # Memory benchmark
    print("\n--- Memory Usage ---")
    mem_result = benchmark_memory_usage(M, intermediate, hidden)
    print(f"PyTorch Peak: {mem_result['pytorch_peak_mb']:.1f} MB")
    print(f"Fused Peak:   {mem_result['fused_peak_mb']:.1f} MB")
    print(f"Memory Saved: {mem_result['memory_saved_mb']:.1f} MB ({mem_result['memory_saved_percent']:.1f}%)")

    return results, mem_result


def run_scaling_benchmarks():
    """Run benchmarks across different batch sizes."""
    print("\n" + "=" * 70)
    print("Scaling Benchmark (varying batch size)")
    print("=" * 70)

    hidden = 1280
    intermediate = 5120

    batch_sizes = [1, 2, 4, 8, 16, 32]
    seq_len = 1024

    results = []

    print(f"\n{'Batch':<8} {'M':<10} {'Fused (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 60)

    for batch in batch_sizes:
        M = batch * seq_len
        result = benchmark_fused_linear_gelu(M, intermediate, hidden, num_iters=30)
        results.append(result)
        print(f"{batch:<8} {M:<10} {result['forward']['fused_ms']:<12.3f} {result['forward']['pytorch_ms']:<14.3f} {result['forward']['speedup']:<10.2f}x")

    return results


def run_full_model_estimate():
    """Estimate full model speedup based on kernel benchmarks."""
    print("\n" + "=" * 70)
    print("Full Model Speedup Estimate")
    print("=" * 70)

    # GPT-2 Large: 36 layers
    # Each layer has:
    # - MLP up-projection with GELU
    # - MLP down-projection (no dropout in our implementation for simplicity)
    # - Attention output projection (no dropout for simplicity)

    batch = 8
    seq = 1024
    hidden = 1280
    intermediate = 5120
    num_layers = 36

    M = batch * seq

    # Benchmark individual operations
    mlp_up = benchmark_fused_linear_gelu(M, intermediate, hidden, num_iters=30)
    mlp_down = benchmark_fused_linear_dropout(M, hidden, intermediate, p_drop=0.0, num_iters=30)

    # Per-layer savings
    mlp_up_savings = mlp_up['total']['pytorch_ms'] - mlp_up['total']['fused_ms']
    mlp_down_savings = mlp_down['total']['pytorch_ms'] - mlp_down['total']['fused_ms']

    total_savings_per_layer = mlp_up_savings + mlp_down_savings
    total_savings = total_savings_per_layer * num_layers

    # Estimate full iteration time (rough)
    # Full iteration includes: embeddings, attention (QKV, softmax), MLP, etc.
    # MLP is roughly 60-70% of compute in transformers
    # Let's estimate MLP time

    mlp_pytorch_time = (mlp_up['total']['pytorch_ms'] + mlp_down['total']['pytorch_ms']) * num_layers
    mlp_fused_time = (mlp_up['total']['fused_ms'] + mlp_down['total']['fused_ms']) * num_layers

    print(f"\nPer-Layer MLP Time:")
    print(f"  PyTorch: {mlp_up['total']['pytorch_ms'] + mlp_down['total']['pytorch_ms']:.3f} ms")
    print(f"  Fused:   {mlp_up['total']['fused_ms'] + mlp_down['total']['fused_ms']:.3f} ms")
    print(f"  Savings: {total_savings_per_layer:.3f} ms/layer")

    print(f"\nFull Model MLP Time ({num_layers} layers):")
    print(f"  PyTorch: {mlp_pytorch_time:.1f} ms")
    print(f"  Fused:   {mlp_fused_time:.1f} ms")
    print(f"  Total Savings: {total_savings:.1f} ms")

    print(f"\nEstimated Impact:")
    print(f"  If MLP is ~60% of iteration, and iteration takes ~500ms:")
    print(f"  Estimated iteration speedup: {total_savings/500*100:.1f}% faster")


def save_results(results, filename="benchmark_results.json"):
    """Save benchmark results to JSON file."""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"triton_benchmark_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton Fused Kernels")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark only")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("Triton Fused Kernels - Performance Benchmarks")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return

    # Print GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    all_results = {}

    # Run benchmarks
    gpt2_results, mem_result = run_gpt2_benchmarks()
    all_results["gpt2"] = gpt2_results
    all_results["memory"] = mem_result

    if not args.quick:
        scaling_results = run_scaling_benchmarks()
        all_results["scaling"] = scaling_results

        run_full_model_estimate()

    # Save results
    if args.save:
        save_results(all_results)

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
