# Hybrid Parallelism for Efficient Multi-GPU Training

This project investigates hybrid parallelism techniques for improving the performance and efficiency of large-scale language model training on multi-GPU systems.

## Overview

We train **GPT-2 Large (774M parameters)** on **WikiText-103** using 4 NVIDIA A100 GPUs, comparing different parallelism strategies.

## Results

### Training Configurations Comparison

| Configuration | Batch/GPU | Throughput (tok/s) | MFU | Memory | Training Time |
|--------------|-----------|-------------------|-----|--------|---------------|
| **Baseline DDP** | 8 | 75,130 | 33.2% | 21.37 GB | 75.7s |
| **ZeRO-3** | 8 | 44,948 | 19.8% | 8.33 GB | 109.5s |
| **ZeRO-3 + TP** | 8 | 43,093 | 19.0% | 6.36 GB | 116.0s |
| **ZeRO-3 + TP + CUDA Fused** | 8 | 42,750 | 18.9% | 6.36 GB | 117.9s |
| **ZeRO-3 + TP** | 32 | 121,000 | **52.6%** | 17.81 GB | 46.6s |
| **ZeRO-3 + TP + CUDA Fused** | 32 | 118,717 | 51.7% | 17.81 GB | 47.6s |

### Microbenchmark Results (GELU Kernel)

| Configuration | PyTorch (ms) | CUDA Fused (ms) | Speedup |
|--------------|--------------|-----------------|---------|
| Small [4, 512, 1024] | 0.0228 | 0.0286 | 0.80x |
| Medium [8, 512, 2048] | 0.0813 | 0.0687 | 1.18x |
| GPT-2 Large [8, 1024, 5120] | 0.3594 | 0.3212 | 1.12x |

### MLP Operation Breakdown

| Operation | Time (ms) | % of MLP |
|-----------|-----------|----------|
| Linear (up-projection) | 0.5992 | 48.4% |
| GELU | 0.1319 | 10.7% |
| Linear (down-projection) | 0.4457 | 36.0% |
| Dropout | 0.0492 | 4.0% |
| **Full MLP** | **1.2378** | **100%** |

## Key Findings

### 1. Memory Reduction with ZeRO-3 + Tensor Parallelism

- **70% memory reduction**: From 21.37 GB (Baseline) to 6.36 GB (ZeRO-3 + TP)
- ZeRO-3 alone reduces memory to 8.33 GB (61% reduction)
- Adding Tensor Parallelism further reduces to 6.36 GB

### 2. Throughput vs Memory Trade-off (Batch Size 8)

- At small batch sizes, memory-efficient configurations have lower throughput
- Baseline DDP: 75,130 tok/s (33.2% MFU)
- ZeRO-3 + TP: 43,093 tok/s (19.0% MFU)
- This is due to communication overhead from parameter sharding and tensor parallel all-reduce operations

### 3. Leveraging Memory Savings with Larger Batch Sizes

- **Key insight**: Memory savings enable larger batch sizes, which improves MFU
- With batch size 32, ZeRO-3 + TP achieves:
  - **121,000 tok/s** throughput (vs 75,130 tok/s baseline)
  - **52.6% MFU** (vs 33.2% baseline)
  - **1.6x faster training** (46.6s vs 75.7s)

### 4. Custom CUDA Fused Kernel Analysis

- The fused bias+GELU kernel shows **1.12x speedup** for the GELU operation itself on large tensors
- However, **no end-to-end improvement** observed because:
  - GELU accounts for only **10.7%** of MLP compute time
  - Linear layers (matmul) dominate at **84.4%** of MLP time
  - cuBLAS is already highly optimized for matrix multiplications
- **Conclusion**: Kernel fusion is effective for the target operation, but the operation is not the bottleneck

### 5. Model Convergence

All configurations converge to similar loss/accuracy, proving correctness:

- Baseline DDP: Loss 6.97, Accuracy 12.5%
- ZeRO-3: Loss 6.81, Accuracy 13.2%
- ZeRO-3 + TP: Loss 7.17, Accuracy 10.6%

(Lower accuracy at batch 32 is expected due to fewer gradient updates with same number of samples)

## Conclusion

1. **ZeRO-3 + Tensor Parallelism** is the recommended configuration for memory-constrained scenarios, achieving 70% memory reduction
2. **Larger batch sizes** should be used to leverage memory savings - this transforms the memory-throughput trade-off into a net positive
3. **Custom CUDA kernels** for elementwise operations provide limited benefit when compute is dominated by matrix multiplications
4. **Future optimization** should target communication overhead (e.g., computation-communication overlap) rather than elementwise kernel fusion

## Project Structure

```
hpml/
├── train.py                    # Main training script with all modes
├── ds_config_zero3.json        # DeepSpeed ZeRO-3 configuration
├── fused_kernels/              # Custom CUDA fused kernels
│   ├── __init__.py
│   ├── bias_gelu_cuda.cu       # CUDA kernel implementation
│   ├── bias_gelu.cpp           # C++ bindings
│   └── fused_bias_gelu.py      # Python wrapper with autograd
├── benchmarks/
│   └── microbenchmark.py       # Detailed operation profiling
├── tests/
│   ├── test_fused_kernel.py    # CUDA kernel correctness tests
│   └── test_triton_kernels.py  # Triton kernel tests
├── results/                    # Training results (JSON files)
└── README.md                   # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support)
- DeepSpeed
- Transformers (HuggingFace)
- 4× NVIDIA A100 GPUs (40GB each)
- CUDA 12.x (for custom kernels)

## Quick Start (NYU HPC)

### Step 1: Enter Singularity Container

```bash
/scratch/work/public/singularity/run-cuda-12.2.2.bash
```

### Step 2: Navigate to Project

```bash
cd /path/to/hpml
```

### Step 3: Install Dependencies

```bash
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install --user transformers datasets tqdm accelerate deepspeed
```

### Step 4: Run Tests

```bash
# Test data loading
python3 train.py --test-data

# Test model creation (single GPU)
python3 train.py --test-model

# Test DDP training (4 GPUs)
torchrun --nproc_per_node=4 train.py --test-ddp

# Test CUDA fused kernel correctness
python3 tests/test_fused_kernel.py
```

### Step 5: Run Microbenchmark

```bash
# Basic microbenchmark
python3 benchmarks/microbenchmark.py

# With PyTorch profiler for kernel-level details
python3 benchmarks/microbenchmark.py --detailed

# Export to CSV
python3 benchmarks/microbenchmark.py --export-csv results/microbench.csv
```

## Training Commands

### 1. Baseline DDP (4 GPUs)

```bash
torchrun --nproc_per_node=4 train.py --mode baseline --max-samples 5000
```

### 2. ZeRO-3 (4 GPUs)

```bash
deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 5000
```

### 3. ZeRO-3 + Tensor Parallelism (DP=2, TP=2)

```bash
# Default batch size 8 (memory efficient)
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 5000

# Larger batch size 32 (higher MFU)
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --batch-size 32 --max-samples 5000
```

### 4. ZeRO-3 + TP + Custom CUDA Fused Kernel

```bash
# Default batch size 8
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --max-samples 5000

# Larger batch size 32
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --batch-size 32 --max-samples 5000
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Training mode: `single`, `baseline`, `hybrid` | `baseline` |
| `--tp-size` | Tensor parallel size (1=off, 2=split across 2 GPUs) | `1` |
| `--batch-size` | Batch size per GPU | `8` |
| `--max-samples` | Limit training samples for quick tests | None (full) |
| `--use-fused-kernel` | Use custom CUDA fused bias+GELU kernel | False |
| `--test` | Use small model/data for testing | False |

## Architecture

### Tensor Parallelism Implementation

The project implements custom tensor parallel layers:

- **ColumnParallelLinear**: Splits output dimension across GPUs (used for QKV projections, MLP up-projection)
- **RowParallelLinear**: Splits input dimension across GPUs with All-Reduce (used for attention output, MLP down-projection)

```
For 4 GPUs with TP=2, DP=2:
- TP groups: [0,1], [2,3] - GPUs that share tensor parallel work
- DP groups: [0,2], [1,3] - GPUs that share data parallel work
```

### Custom CUDA Fused Kernels

The `fused_kernels/` directory contains custom CUDA kernels for fusing operations:

**Fused Bias + GELU Kernel** (`bias_gelu_cuda.cu`):

- Computes `y = GELU(x + bias)` in a single kernel pass
- Reduces memory traffic by avoiding intermediate tensor materialization
- Uses tanh approximation: `GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
- Supports FP16 and FP32
- Includes backward pass for autograd compatibility

### Memory Optimization Stack

1. **FP16 Mixed Precision**: Reduces memory by 50%
2. **Gradient Checkpointing**: Trades compute for memory in transformer blocks
3. **Flash Attention**: O(n) memory instead of O(n²) for attention
4. **ZeRO-3**: Shards parameters, gradients, and optimizer states
5. **Tensor Parallelism**: Splits model layers across GPUs

## Metrics

The training script tracks:

- **Loss**: Cross-entropy loss for next-token prediction
- **Accuracy**: Top-1 token prediction accuracy (%)
- **Throughput**: Tokens/second and samples/second
- **MFU (Model FLOPs Utilization)**: Achieved FLOPs / Theoretical peak FLOPs
- **Peak Memory**: GPU memory usage

## Troubleshooting

### CUDA Kernel Compilation Issues

If the fused kernel fails to compile:

```bash
# Clear cached compiled kernels
rm -rf ~/.cache/torch_extensions/

# Ensure CUDA toolkit is available
nvcc --version

# Check PyTorch CUDA version matches system CUDA
python3 -c "import torch; print(torch.version.cuda)"
```

### DeepSpeed Issues

```bash
# Check DeepSpeed installation
ds_report

# Verify distributed setup
python3 -c "import torch.distributed as dist; print('OK')"
```

### Memory Issues

If running out of memory:

1. Reduce `--batch-size`
2. Enable gradient checkpointing (already enabled by default)
3. Use ZeRO-3 + TP mode for lower memory usage

## References

1. S. Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," SC20, 2020.

2. M. Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," arXiv:1909.08053, 2019.

3. Microsoft, "DeepSpeed: Accelerating Deep Learning Training," GitHub repository, 2024.

## Authors

- M. Akram Bari (<ma9091@nyu.edu>)
- Yashas Harisha (<yh5569@nyu.edu>)

New York University - High Performance Machine Learning (HPML)
