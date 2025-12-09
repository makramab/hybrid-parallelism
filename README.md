# Hybrid Parallelism for Efficient Multi-GPU Training

This project investigates hybrid parallelism techniques for improving the performance and efficiency of large-scale language model training on multi-GPU systems.

## Overview

We train **GPT-2 Large (774M parameters)** on **WikiText-103** using 4 NVIDIA A100 GPUs, comparing different parallelism strategies:

| Configuration | Description | Memory | MFU |
|---------------|-------------|--------|-----|
| **Single GPU** | Baseline for scaling efficiency | ~21 GB | ~45% |
| **Baseline DDP** | PyTorch Distributed Data Parallel | ~21 GB | ~43% |
| **Hybrid ZeRO-3** | DeepSpeed ZeRO-3 optimizer sharding | ~20 GB | ~26% |
| **Hybrid DP×TP** | ZeRO-3 + Tensor Parallelism (DP=2, TP=2) | **~6 GB** | ~19-52%* |

*MFU varies with batch size - larger batches leverage memory savings for higher MFU.

## Key Results

- **70% Memory Reduction**: Hybrid DP×TP reduces memory from 21GB to 6GB per GPU
- **Batch Size Scaling**: With lower memory, batch size can be increased 4× for better MFU
- **52% MFU Achieved**: With batch size 32, Hybrid DP×TP reaches 52% MFU (vs 19% with batch 8)

## Project Structure

```
hpml/
├── train.py              # Main training script with all modes
├── ds_config_zero3.json  # DeepSpeed ZeRO-3 configuration
├── setup_and_run.sh      # Helper script for NYU HPC
├── CONTEXT.md            # Project context and background
├── REPORT_DRAFT.tex      # IEEE format report draft
├── README.md             # This file
└── results/              # Training results (JSON files)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support)
- DeepSpeed
- Transformers (HuggingFace)
- 4× NVIDIA A100 GPUs (40GB each)

### Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tqdm accelerate deepspeed
```

## Usage

### Quick Start (NYU HPC)

```bash
# Enter Singularity container
/scratch/work/public/singularity/run-cuda-12.2.2.bash

# Navigate to project
cd /path/to/hpml

# Source helper script
source setup_and_run.sh

# Run tests
test_data && test_model && test_ddp
```

### Training Modes

#### 1. Single GPU (Scaling Baseline)
```bash
python train.py --mode single --max-samples 5000
```

#### 2. Baseline DDP (4 GPUs)
```bash
torchrun --nproc_per_node=4 train.py --mode baseline --max-samples 5000
```

#### 3. Hybrid ZeRO-3 (4 GPUs)
```bash
deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 5000
```

#### 4. Hybrid DP×TP (DP=2, TP=2)
```bash
# Default batch size (memory efficient)
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 5000

# Larger batch size (higher MFU)
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --batch-size 32 --max-samples 5000
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Training mode: `single`, `baseline`, `hybrid` | `baseline` |
| `--tp-size` | Tensor parallel size (1=off, 2=split across 2 GPUs) | `1` |
| `--batch-size` | Batch size per GPU | `8` |
| `--max-samples` | Limit training samples for quick tests | None (full) |
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

### Memory Optimization Stack

1. **FP16 Mixed Precision**: Reduces memory by 50%
2. **Gradient Checkpointing**: Trades compute for memory in transformer blocks
3. **Flash Attention**: O(n) memory instead of O(n²) for attention
4. **ZeRO-3**: Shards parameters, gradients, and optimizer states
5. **Tensor Parallelism**: Splits model layers across GPUs

## Metrics

The training script tracks:

- **Throughput**: Tokens/second and samples/second
- **MFU (Model FLOPs Utilization)**: Achieved FLOPs / Theoretical peak FLOPs
- **Peak Memory**: GPU memory usage
- **Scaling Efficiency**: Multi-GPU throughput vs single-GPU baseline

Results are saved to `results/` as JSON files.

## Full Comparison (For Report)

Run all configurations with the same settings:

```bash
# 1. Single GPU baseline
python train.py --mode single --max-samples 5000

# 2. Baseline DDP (4 GPUs)
torchrun --nproc_per_node=4 train.py --mode baseline --max-samples 5000

# 3. Hybrid ZeRO-3 (4 GPUs)
deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 5000

# 4. Hybrid DP×TP with default batch
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 5000

# 5. Hybrid DP×TP with larger batch (leverage memory savings)
deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --batch-size 32 --max-samples 5000
```

### Calculating Scaling Efficiency

```
Scaling Efficiency = T_N / (N × T_1) × 100%

Where:
- T_1 = Single GPU throughput
- T_N = N-GPU throughput
- N = Number of GPUs
```

## References

1. Z. Duan et al., "Research on Model Parallelism and Data Parallelism Optimization Methods in Large Language Model–Based Recommendation Systems," arXiv:2506.17551, 2025.

2. Microsoft, "DeepSpeed: Accelerating Deep Learning Training," GitHub repository, 2024.

## Authors

- M. Akram Bari (ma9091@nyu.edu)
- Yashas Harisha (yh5569@nyu.edu)

New York University - High Performance Machine Learning (HPML)
