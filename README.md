# Hybrid Parallelism for Efficient Multi-GPU Training

This project investigates hybrid parallelism techniques for improving the performance and efficiency of large-scale language model training on multi-GPU systems.

## Overview

We train **GPT-2 Large (774M parameters)** on **WikiText-103** using 4 NVIDIA A100 GPUs, comparing different parallelism strategies:

| Configuration | Description | Memory | MFU |
|---------------|-------------|--------|-----|
| **Single GPU** | Baseline for scaling efficiency | ~21 GB | ~45% |
| **Baseline DDP** | PyTorch Distributed Data Parallel | ~21 GB | ~43% |
| **Hybrid ZeRO-3** | DeepSpeed ZeRO-3 optimizer sharding | ~20 GB | ~26% |
| **Hybrid DP×TP** | ZeRO-3 + Tensor Parallelism (DP=2, TP=2) | **~6 GB** | ~52%* |
| **Hybrid DP×TP + Fused** | With custom CUDA fused kernels | **~6 GB** | ~52%* |

*MFU varies with batch size - larger batches leverage memory savings for higher MFU.

## Key Results

- **70% Memory Reduction**: Hybrid DP×TP reduces memory from 21GB to 6GB per GPU
- **Batch Size Scaling**: With lower memory, batch size can be increased 4× for better MFU
- **52% MFU Achieved**: With batch size 32, Hybrid DP×TP reaches 52% MFU (vs 19% with batch 8)
- **Custom CUDA Kernels**: Fused bias+GELU kernel achieves 1.12-1.18× speedup for the operation

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

### 1. Single GPU (Scaling Baseline)

```bash
python3 train.py --mode single --max-samples 5000
```

### 2. Baseline DDP (4 GPUs)

```bash
torchrun --nproc_per_node=4 train.py --mode baseline --max-samples 5000
```

### 3. Hybrid ZeRO-3 (4 GPUs)

```bash
deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 5000
```

### 4. Hybrid DP×TP (DP=2, TP=2)

```bash
# Default batch size (memory efficient)
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 5000

# Larger batch size (higher MFU)
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --batch-size 32 --max-samples 5000
```

### 5. Hybrid DP×TP + Custom CUDA Fused Kernel

```bash
# Default batch size
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --max-samples 5000

# Larger batch size
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

## Full Comparison (For Report)

Run all configurations sequentially:

```bash
# 1. Single GPU baseline
python3 train.py --mode single --max-samples 5000

# 2. Baseline DDP (4 GPUs)
torchrun --nproc_per_node=4 train.py --mode baseline --max-samples 5000

# 3. Hybrid ZeRO-3 (4 GPUs)
deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 5000

# 4. Hybrid DP×TP (DP=2, TP=2)
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 5000

# 5. Hybrid DP×TP + CUDA fused kernel
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --max-samples 5000

# 6. Hybrid DP×TP with larger batch (leverage memory savings)
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --batch-size 32 --max-samples 5000

# 7. Hybrid DP×TP + CUDA fused with larger batch
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --batch-size 32 --max-samples 5000
```

Results are saved to `results/` as JSON files with metrics including loss, accuracy, throughput, and MFU.

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

Performance results:

- **GELU operation**: 1.12-1.18× faster than PyTorch baseline
- **End-to-end MLP**: Minimal improvement because GELU is only ~2% of MLP compute time
- Linear layers (matmul) dominate at ~94% of MLP time

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

### Calculating Scaling Efficiency

```
Scaling Efficiency = T_N / (N × T_1) × 100%

Where:
- T_1 = Single GPU throughput
- T_N = N-GPU throughput
- N = Number of GPUs
```

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
3. Use Hybrid DP×TP mode for lower memory usage

## References

1. Z. Duan et al., "Research on Model Parallelism and Data Parallelism Optimization Methods in Large Language Model–Based Recommendation Systems," arXiv:2506.17551, 2025.

2. Microsoft, "DeepSpeed: Accelerating Deep Learning Training," GitHub repository, 2024.

3. S. Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," SC20, 2020.

4. M. Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," arXiv:1909.08053, 2019.

## Authors

- M. Akram Bari (ma9091@nyu.edu)
- Yashas Harisha (yh5569@nyu.edu)

New York University - High Performance Machine Learning (HPML)
