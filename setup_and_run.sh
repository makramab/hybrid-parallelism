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
# STEP 7: Baseline Training (Test Mode - Small Model/Data)
# -----------------------------------------------------------------------------
run_baseline_test() {
    echo ""
    echo "Running baseline training in test mode..."
    echo ""

    torchrun --nproc_per_node=4 train.py --mode baseline --test
}

# -----------------------------------------------------------------------------
# STEP 8: Hybrid ZeRO-3 Training
# -----------------------------------------------------------------------------
run_hybrid() {
    echo ""
    echo "Running hybrid DeepSpeed ZeRO-3 training..."
    echo ""

    deepspeed --num_gpus=4 train.py --mode hybrid
}

# -----------------------------------------------------------------------------
# STEP 9: Hybrid ZeRO-3 Training (Quick Test)
# -----------------------------------------------------------------------------
run_hybrid_test() {
    echo ""
    echo "Running hybrid training with limited samples..."
    echo ""

    deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 1000
}

# -----------------------------------------------------------------------------
# STEP 10: Full Hybrid DP×TP Training (DP=2, TP=2)
# -----------------------------------------------------------------------------
run_hybrid_tp() {
    echo ""
    echo "Running full hybrid DP×TP training (DP=2, TP=2)..."
    echo ""

    deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2
}

# -----------------------------------------------------------------------------
# STEP 11: Hybrid DP×TP Training (Quick Test)
# -----------------------------------------------------------------------------
run_hybrid_tp_test() {
    echo ""
    echo "Running hybrid DP×TP training with limited samples..."
    echo ""

    deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 1000
}

# -----------------------------------------------------------------------------
# Print Usage
# -----------------------------------------------------------------------------
usage() {
    echo ""
    echo "Usage: source setup_and_run.sh [command]"
    echo ""
    echo "Commands (run as functions after sourcing):"
    echo "  install_dependencies  - Install required Python packages"
    echo "  test_data            - Test WikiText-103 loading"
    echo "  test_model           - Test GPT-2 model creation (single GPU)"
    echo "  test_ddp             - Test DDP training (4 GPUs)"
    echo "  run_single_gpu       - Single GPU training (for scaling efficiency)"
    echo "  run_baseline_test    - Run baseline training with small model"
    echo "  run_baseline         - Run full baseline DDP training (4 GPUs)"
    echo "  run_hybrid_test      - Run hybrid ZeRO-3 with limited samples"
    echo "  run_hybrid           - Run full hybrid ZeRO-3 training"
    echo "  run_hybrid_tp_test   - Run hybrid DP×TP (DP=2, TP=2) with limited samples"
    echo "  run_hybrid_tp        - Run full hybrid DP×TP training"
    echo ""
    echo "Example workflow:"
    echo "  1. /scratch/work/public/singularity/run-cuda-12.2.2.bash"
    echo "  2. cd /path/to/your/project"
    echo "  3. source setup_and_run.sh"
    echo "  4. install_dependencies"
    echo "  5. test_data && test_model && test_ddp"
    echo ""
    echo "Full Comparison (for report):"
    echo "  python train.py --mode single --max-samples 5000      # 1 GPU baseline"
    echo "  torchrun --nproc_per_node=4 train.py --mode baseline --max-samples 5000"
    echo "  deepspeed --num_gpus=4 train.py --mode hybrid --max-samples 5000"
    echo "  deepspeed --num_gpus=4 train.py --mode hybrid --tp-size 2 --max-samples 5000"
    echo ""
}

# Show usage if script is sourced
usage
