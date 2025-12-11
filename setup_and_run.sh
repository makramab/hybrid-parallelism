#!/bin/bash
# =============================================================================
# NYU HPC Setup and Run Script for Hybrid Parallelism Training
# =============================================================================
#
# This script provides commands for setting up and running the training
# experiments on NYU HPC with Singularity containers.
#
# IMPORTANT: Run these commands INSIDE the Singularity container
# =============================================================================

echo "=================================================="
echo "Hybrid Parallelism Training - NYU HPC Setup"
echo "=================================================="

# -----------------------------------------------------------------------------
# STEP 1: Enter Singularity Container (run this on HPC login node)
# -----------------------------------------------------------------------------
# /scratch/work/public/singularity/run-cuda-12.2.2.bash

# -----------------------------------------------------------------------------
# STEP 2: Install Python Dependencies (inside Singularity)
# -----------------------------------------------------------------------------
install_dependencies() {
  echo ""
  echo "Installing Python dependencies..."
  echo ""

  pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install --user transformers datasets tqdm accelerate
  pip install --user deepspeed
  pip install --user triton # For Triton fused kernels

  echo ""
  echo "Dependencies installed!"
  echo ""
}

# -----------------------------------------------------------------------------
# STEP 3: Test Data Loading
# -----------------------------------------------------------------------------
test_data() {
  echo ""
  echo "Testing data loading..."
  echo ""

  python train.py --test-data
}

# -----------------------------------------------------------------------------
# STEP 4: Test Model Creation (Single GPU)
# -----------------------------------------------------------------------------
test_model() {
  echo ""
  echo "Testing model creation..."
  echo ""

  python train.py --test-model
}

# -----------------------------------------------------------------------------
# STEP 5: Test DDP Training (4 GPUs)
# -----------------------------------------------------------------------------
test_ddp() {
  echo ""
  echo "Testing DDP training with 4 GPUs..."
  echo ""

  torchrun --nproc_per_node=4 train.py --test-ddp
}

# -----------------------------------------------------------------------------
# STEP 6: Single GPU Training (for scaling efficiency)
# -----------------------------------------------------------------------------
run_single_gpu() {
  echo ""
  echo "Running single GPU training (scaling baseline)..."
  echo ""

  python train.py --mode single --max-samples 5000
}

# -----------------------------------------------------------------------------
# STEP 7: Full Baseline DDP Training (4 GPUs)
# -----------------------------------------------------------------------------
run_baseline() {
  echo ""
  echo "Running full baseline DDP training..."
  echo ""

  torchrun --nproc_per_node=4 train.py --mode baseline
}

# -----------------------------------------------------------------------------
# STEP 8: Baseline Training (Test Mode - Small Model/Data)
# -----------------------------------------------------------------------------
run_baseline_test() {
  echo ""
  echo "Running baseline training in test mode..."
  echo ""

  torchrun --nproc_per_node=4 train.py --mode baseline --test
}

# -----------------------------------------------------------------------------
# STEP 9: Hybrid ZeRO-3 Training
# -----------------------------------------------------------------------------
run_hybrid() {
  echo ""
  echo "Running hybrid DeepSpeed ZeRO-3 training..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid
}

# -----------------------------------------------------------------------------
# STEP 10: Hybrid ZeRO-3 Training (Quick Test)
# -----------------------------------------------------------------------------
run_hybrid_test() {
  echo ""
  echo "Running hybrid training with limited samples..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 1000
}

# -----------------------------------------------------------------------------
# STEP 11: Full Hybrid DP×TP Training (DP=2, TP=2)
# -----------------------------------------------------------------------------
run_hybrid_tp() {
  echo ""
  echo "Running full hybrid DP×TP training (DP=2, TP=2)..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2
}

# -----------------------------------------------------------------------------
# STEP 12: Hybrid DP×TP Training (Quick Test)
# -----------------------------------------------------------------------------
run_hybrid_tp_test() {
  echo ""
  echo "Running hybrid DP×TP training with limited samples..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 1000
}

# -----------------------------------------------------------------------------
# STEP 13: Hybrid DP×TP + Triton Fused Kernels (NOVELTY)
# -----------------------------------------------------------------------------
run_hybrid_tp_triton() {
  echo ""
  echo "Running hybrid DP×TP training with Triton fused kernels..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-triton
}

# -----------------------------------------------------------------------------
# STEP 14: Hybrid DP×TP + Triton (Quick Test)
# -----------------------------------------------------------------------------
run_hybrid_tp_triton_test() {
  echo ""
  echo "Running hybrid DP×TP + Triton with limited samples..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-triton --max-samples 1000
}

# -----------------------------------------------------------------------------
# STEP 15: Hybrid DP×TP + Triton with Larger Batch
# -----------------------------------------------------------------------------
run_hybrid_tp_triton_batch32() {
  echo ""
  echo "Running hybrid DP×TP + Triton with batch size 32..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-triton --batch-size 32 --max-samples 5000
}

# -----------------------------------------------------------------------------
# STEP 16: Hybrid DP×TP + CUDA Fused Kernels (NOVELTY)
# -----------------------------------------------------------------------------
run_hybrid_tp_fused() {
  echo ""
  echo "Running hybrid DP×TP training with CUDA fused bias+GELU kernel..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel
}

# -----------------------------------------------------------------------------
# STEP 17: Hybrid DP×TP + CUDA Fused Kernels (Quick Test)
# -----------------------------------------------------------------------------
run_hybrid_tp_fused_test() {
  echo ""
  echo "Running hybrid DP×TP + CUDA fused kernel with limited samples..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --max-samples 1000
}

# -----------------------------------------------------------------------------
# STEP 18: Hybrid DP×TP + CUDA Fused with Larger Batch
# -----------------------------------------------------------------------------
run_hybrid_tp_fused_batch32() {
  echo ""
  echo "Running hybrid DP×TP + CUDA fused kernel with batch size 32..."
  echo ""

  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --batch-size 32 --max-samples 5000
}

# -----------------------------------------------------------------------------
# CUDA Fused Kernel Tests
# -----------------------------------------------------------------------------
test_fused_kernel() {
  echo ""
  echo "Testing CUDA fused bias+GELU kernel correctness..."
  echo ""

  python tests/test_fused_kernel.py
}

# -----------------------------------------------------------------------------
# Triton Kernel Tests
# -----------------------------------------------------------------------------
test_triton_kernels() {
  echo ""
  echo "Testing Triton fused kernel correctness..."
  echo ""

  python tests/test_triton_kernels.py
}

# -----------------------------------------------------------------------------
# Microbenchmark (Detailed Operation Profiling)
# -----------------------------------------------------------------------------
run_microbenchmark() {
  echo ""
  echo "Running microbenchmark for detailed performance analysis..."
  echo ""

  python benchmarks/microbenchmark.py
}

# -----------------------------------------------------------------------------
# Microbenchmark with PyTorch Profiler
# -----------------------------------------------------------------------------
run_microbenchmark_detailed() {
  echo ""
  echo "Running microbenchmark with PyTorch profiler..."
  echo ""

  python benchmarks/microbenchmark.py --detailed
}

# -----------------------------------------------------------------------------
# Export Microbenchmark to CSV
# -----------------------------------------------------------------------------
run_microbenchmark_csv() {
  echo ""
  echo "Running microbenchmark and exporting to CSV..."
  echo ""

  mkdir -p results
  python benchmarks/microbenchmark.py --export-csv results/microbench.csv
}

# -----------------------------------------------------------------------------
# Full Comparison (for report)
# -----------------------------------------------------------------------------
run_full_comparison() {
  echo ""
  echo "Running full comparison (all configurations)..."
  echo "This will take a while..."
  echo ""

  echo "1/7: Single GPU baseline..."
  python train.py --mode single --max-samples 5000

  echo "2/7: Baseline DDP (4 GPUs)..."
  torchrun --nproc_per_node=4 train.py --mode baseline --max-samples 5000

  echo "3/7: Hybrid ZeRO-3 (4 GPUs)..."
  deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 5000

  echo "4/7: Hybrid DP×TP (DP=2, TP=2)..."
  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 5000

  echo "5/7: Hybrid DP×TP + CUDA Fused Kernel..."
  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --max-samples 5000

  echo "6/7: Hybrid DP×TP + CUDA Fused (batch 32)..."
  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --batch-size 32 --max-samples 5000

  echo "7/7: Hybrid DP×TP baseline (batch 32) - for comparison..."
  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --batch-size 32 --max-samples 5000

  echo ""
  echo "Full comparison complete! Check results/ directory for JSON files."
  echo ""
}

# -----------------------------------------------------------------------------
# Full Benchmark Suite (Training + Microbenchmark)
# -----------------------------------------------------------------------------
run_full_benchmark() {
  echo ""
  echo "Running full benchmark suite..."
  echo ""

  # Run all training configurations
  run_full_comparison

  # Run microbenchmark
  echo "Running microbenchmark..."
  python benchmarks/microbenchmark.py --export-csv results/microbench.csv

  echo ""
  echo "Full benchmark complete!"
  echo "Results saved to results/ directory"
  echo ""
}

# -----------------------------------------------------------------------------
# Print Usage
# -----------------------------------------------------------------------------
usage() {
  echo ""
  echo "Usage: source setup_and_run.sh [command]"
  echo ""
  echo "Commands (run as functions after sourcing):"
  echo ""
  echo "  SETUP:"
  echo "    install_dependencies  - Install required Python packages (incl. Triton)"
  echo ""
  echo "  TESTING:"
  echo "    test_data            - Test WikiText-103 loading"
  echo "    test_model           - Test GPT-2 model creation (single GPU)"
  echo "    test_ddp             - Test DDP training (4 GPUs)"
  echo "    test_fused_kernel    - Test CUDA fused bias+GELU kernel"
  echo "    test_triton_kernels  - Test Triton kernel correctness"
  echo ""
  echo "  TRAINING:"
  echo "    run_single_gpu       - Single GPU training (for scaling efficiency)"
  echo "    run_baseline_test    - Run baseline training with small model"
  echo "    run_baseline         - Run full baseline DDP training (4 GPUs)"
  echo "    run_hybrid_test      - Run hybrid ZeRO-3 with limited samples"
  echo "    run_hybrid           - Run full hybrid ZeRO-3 training"
  echo "    run_hybrid_tp_test   - Run hybrid DP×TP (DP=2, TP=2) with limited samples"
  echo "    run_hybrid_tp        - Run full hybrid DP×TP training"
  echo ""
  echo "  CUDA FUSED KERNEL (NOVELTY):"
  echo "    run_hybrid_tp_fused_test   - Run DP×TP + CUDA fused kernel (quick test)"
  echo "    run_hybrid_tp_fused        - Run full DP×TP + CUDA fused kernel"
  echo "    run_hybrid_tp_fused_batch32 - Run DP×TP + CUDA fused (batch 32)"
  echo ""
  echo "  TRITON (NOVELTY):"
  echo "    run_hybrid_tp_triton_test  - Run DP×TP + Triton with limited samples"
  echo "    run_hybrid_tp_triton       - Run full DP×TP + Triton training"
  echo "    run_hybrid_tp_triton_batch32 - Run DP×TP + Triton with batch 32"
  echo ""
  echo "  MICROBENCHMARKS:"
  echo "    run_microbenchmark         - Run detailed operation profiling"
  echo "    run_microbenchmark_detailed - Run with PyTorch profiler"
  echo "    run_microbenchmark_csv     - Export results to CSV"
  echo ""
  echo "  FULL COMPARISON:"
  echo "    run_full_comparison  - Run all 7 training configurations"
  echo "    run_full_benchmark   - Run training + microbenchmark (full suite)"
  echo ""
  echo "Example workflow:"
  echo "  1. /scratch/work/public/singularity/run-cuda-12.2.2.bash"
  echo "  2. cd /path/to/your/project"
  echo "  3. source setup_and_run.sh"
  echo "  4. install_dependencies"
  echo "  5. test_data && test_model && test_ddp"
  echo "  6. test_fused_kernel               # Test CUDA fused kernel"
  echo "  7. run_microbenchmark              # Run detailed benchmarks"
  echo "  8. run_full_comparison             # Run all training configs"
  echo ""
  echo "Individual Training Commands:"
  echo "  python train.py --mode single --max-samples 5000"
  echo "  torchrun --nproc_per_node=4 train.py --mode baseline --max-samples 5000"
  echo "  deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 5000"
  echo "  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 5000"
  echo "  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --max-samples 5000"
  echo "  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --use-fused-kernel --batch-size 32 --max-samples 5000"
  echo ""
}

# Show usage if script is sourced
usage
