#!/usr/bin/env python3
"""
Hybrid Parallelism for Efficient Multi-GPU Training
=====================================================
GPT-2 Large (774M params) training with:
  - Baseline: PyTorch DDP
  - Hybrid: DeepSpeed ZeRO-3 (DP=2) + Tensor Parallelism (TP=2)

NYU HPC Setup:
  1. Start Singularity: /scratch/work/public/singularity/run-cuda-12.2.2.bash
  2. Install dependencies: pip install torch deepspeed transformers datasets tqdm
  3. Run tests below

Usage:
  # Test data loading only
  python train.py --test-data

  # Test model creation (single GPU)
  python train.py --test-model

  # Test baseline DDP (4 GPUs, small data)
  torchrun --nproc_per_node=4 train.py --test-ddp

  # Full baseline training
  torchrun --nproc_per_node=4 train.py --mode baseline

  # Hybrid DP×TP training (to be implemented)
  # deepspeed train.py --mode hybrid
"""

import os
import sys
import math
import time
import json
import argparse
from datetime import datetime
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint

# =============================================================================
# SECTION 1: Configuration
# =============================================================================

# GPT-2 Large configuration (774M parameters)
GPT2_LARGE_CONFIG = {
    "vocab_size": 50257,
    "max_position_embeddings": 1024,
    "hidden_size": 1280,
    "num_layers": 36,
    "num_heads": 20,
    "intermediate_size": 1280 * 4,  # 4x hidden for FFN
    "dropout": 0.1,
}

# Smaller config for testing
GPT2_TEST_CONFIG = {
    "vocab_size": 50257,
    "max_position_embeddings": 256,
    "hidden_size": 256,
    "num_layers": 4,
    "num_heads": 4,
    "intermediate_size": 256 * 4,
    "dropout": 0.1,
}

# Training defaults
TRAINING_CONFIG = {
    "batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-4,
    "max_seq_length": 1024,
    "num_epochs": 1,
    "warmup_steps": 100,
    "log_interval": 10,
    "checkpoint_interval": 1000,
}


# =============================================================================
# SECTION 2: Dataset Loading (WikiText-103)
# =============================================================================


def load_wikitext103(tokenizer, max_length=1024, test_mode=False):
    """
    Load and tokenize WikiText-103 dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        test_mode: If True, only load a small subset for testing

    Returns:
        train_dataset, val_dataset
    """
    from datasets import load_dataset

    print("Loading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    if test_mode:
        # Use only 1000 samples for testing
        dataset["train"] = dataset["train"].select(
            range(min(1000, len(dataset["train"])))
        )
        dataset["validation"] = dataset["validation"].select(
            range(min(100, len(dataset["validation"])))
        )
        print(
            f"Test mode: Using {len(dataset['train'])} train, {len(dataset['validation'])} val samples"
        )

    def tokenize_function(examples):
        """Tokenize and chunk text into max_length sequences."""
        # Concatenate all texts
        concatenated = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )

        # Flatten all input_ids
        all_ids = []
        for ids in concatenated["input_ids"]:
            all_ids.extend(ids)

        # Chunk into max_length sequences
        total_length = (len(all_ids) // max_length) * max_length
        all_ids = all_ids[:total_length]

        # Split into chunks
        result = {
            "input_ids": [
                all_ids[i : i + max_length] for i in range(0, total_length, max_length)
            ]
        }
        return result

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
        num_proc=4 if not test_mode else 1,
    )

    # Flatten the chunked sequences
    tokenized = tokenized.map(
        lambda x: {"input_ids": x["input_ids"]},
        batched=True,
        desc="Flattening",
    )

    print(
        f"Dataset ready: {len(tokenized['train'])} train, {len(tokenized['validation'])} val sequences"
    )

    return tokenized["train"], tokenized["validation"]


class WikiTextDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for tokenized WikiText."""

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        return {
            "input_ids": input_ids[:-1],
            "labels": input_ids[1:],
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


# =============================================================================
# SECTION 3: GPT-2 Model Implementation
# =============================================================================


class GPT2Attention(nn.Module):
    """Multi-head self-attention with Flash Attention (PyTorch 2.0+)."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout_p = config["dropout"]
        assert self.head_dim * self.num_heads == self.hidden_size

        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.resid_dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_size, dim=2)

        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention via PyTorch's scaled_dot_product_attention
        # - is_causal=True applies causal mask efficiently without materializing it
        # - Automatically uses Flash Attention kernel when available (CUDA, contiguous, etc.)
        # - Memory: O(seq_len) instead of O(seq_len^2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # Reshape back: (B, num_heads, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return out


class GPT2MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.c_proj = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["hidden_size"])
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config["hidden_size"])
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    """
    GPT-2 Language Model.

    This is a minimal implementation designed for easy modification
    for tensor parallelism experiments.
    """

    def __init__(self, config, use_gradient_checkpointing=True):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Token and position embeddings
        self.wte = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.wpe = nn.Embedding(
            config["max_position_embeddings"], config["hidden_size"]
        )
        self.drop = nn.Dropout(config["dropout"])

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [GPT2Block(config) for _ in range(config["num_layers"])]
        )

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config["hidden_size"])
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        ckpt_status = "enabled" if use_gradient_checkpointing else "disabled"
        print(
            f"GPT-2 model initialized with {self.n_params / 1e6:.1f}M parameters (gradient checkpointing: {ckpt_status})"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.size()
        assert T <= self.config["max_position_embeddings"], (
            f"Sequence length {T} exceeds max {self.config['max_position_embeddings']}"
        )

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks with optional gradient checkpointing
        # Checkpointing trades compute for memory: recomputes activations during backward
        # instead of storing them, reducing memory by ~70% at cost of ~30% more compute
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                # use_reentrant=False is the recommended setting for newer PyTorch
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        accuracy = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            # Compute top-1 token prediction accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == labels).float()
                accuracy = correct.mean() * 100

        return {"loss": loss, "logits": logits, "accuracy": accuracy}


# =============================================================================
# SECTION 3.5: Tensor Parallelism Components
# =============================================================================


class TPGroup:
    """Manages Tensor Parallel process groups."""

    _instance = None

    def __init__(self, tp_size=2):
        self.tp_size = tp_size
        self.tp_group = None
        self.tp_rank = 0
        self.dp_group = None
        self.dp_rank = 0
        self.world_size = 1
        self.initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TPGroup()
        return cls._instance

    def initialize(self, tp_size=2):
        """Initialize TP and DP process groups.

        For 4 GPUs with TP=2:
        - TP groups: [0,1], [2,3] - GPUs that share tensor parallel work
        - DP groups: [0,2], [1,3] - GPUs that share data parallel work
        """
        if self.initialized:
            return

        self.world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        self.tp_size = tp_size
        dp_size = self.world_size // tp_size

        assert self.world_size % tp_size == 0, (
            f"World size {self.world_size} must be divisible by TP size {tp_size}"
        )

        # Create TP groups: GPUs within a node that split model layers
        # For TP=2, world=4: groups are [0,1] and [2,3]
        for i in range(dp_size):
            tp_ranks = list(range(i * tp_size, (i + 1) * tp_size))
            group = torch.distributed.new_group(tp_ranks)
            if global_rank in tp_ranks:
                self.tp_group = group
                self.tp_rank = tp_ranks.index(global_rank)

        # Create DP groups: GPUs that process different data batches
        # For TP=2, world=4: groups are [0,2] and [1,3]
        for i in range(tp_size):
            dp_ranks = list(range(i, self.world_size, tp_size))
            group = torch.distributed.new_group(dp_ranks)
            if global_rank in dp_ranks:
                self.dp_group = group
                self.dp_rank = dp_ranks.index(global_rank)

        self.initialized = True

        if global_rank == 0:
            print(f"\nTP Groups initialized:")
            print(f"  World size: {self.world_size}")
            print(f"  TP size: {tp_size}, DP size: {dp_size}")
            print(
                f"  TP groups: {[list(range(i * tp_size, (i + 1) * tp_size)) for i in range(dp_size)]}"
            )
            print(
                f"  DP groups: {[list(range(i, self.world_size, tp_size)) for i in range(tp_size)]}"
            )

    def cleanup(self):
        """Clean up TP process groups."""
        if not self.initialized:
            return

        # Destroy process groups
        if self.tp_group is not None:
            torch.distributed.destroy_process_group(self.tp_group)
            self.tp_group = None
        if self.dp_group is not None:
            torch.distributed.destroy_process_group(self.dp_group)
            self.dp_group = None

        self.initialized = False
        TPGroup._instance = None


def get_tp_group():
    """Get the TP group instance."""
    return TPGroup.get_instance()


# -----------------------------------------------------------------------------
# Autograd Functions for Tensor Parallelism
# These ensure proper gradient flow through TP communication operations
# -----------------------------------------------------------------------------


class _CopyToTPRegion(torch.autograd.Function):
    """
    Pass input forward unchanged, all-reduce gradients backward.
    Used at the START of a TP region (before ColumnParallelLinear).
    """

    @staticmethod
    def forward(ctx, input_, tp_group):
        ctx.tp_group = tp_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # Clone to avoid in-place modification, then all-reduce gradients
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, group=ctx.tp_group)
        return grad_input, None


class _ReduceFromTPRegion(torch.autograd.Function):
    """
    All-reduce input forward, pass gradients backward unchanged.
    Used at the END of a TP region (after RowParallelLinear matmul).
    """

    @staticmethod
    def forward(ctx, input_, tp_group):
        ctx.tp_group = tp_group
        # Clone to avoid in-place modification on computation graph
        output = input_.clone()
        # All-reduce to sum partial results
        torch.distributed.all_reduce(output, group=tp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Gradients pass through unchanged (each GPU gets full gradient)
        return grad_output, None


class _GatherFromTPRegion(torch.autograd.Function):
    """
    All-gather input forward, split gradients backward.
    Used when we need full tensor from split tensor (e.g., gather output).
    """

    @staticmethod
    def forward(ctx, input_, tp_group, tp_size, tp_rank):
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        ctx.tp_rank = tp_rank

        # All-gather: collect tensors from all TP ranks
        output_list = [torch.empty_like(input_) for _ in range(tp_size)]
        torch.distributed.all_gather(output_list, input_, group=tp_group)
        output = torch.cat(output_list, dim=-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Split gradient and keep only this rank's portion
        tp_size = ctx.tp_size
        tp_rank = ctx.tp_rank

        # Split along last dimension
        dim_size = grad_output.shape[-1] // tp_size
        grad_input = grad_output[..., tp_rank * dim_size : (tp_rank + 1) * dim_size]
        return grad_input.contiguous(), None, None, None


class _ScatterToTPRegion(torch.autograd.Function):
    """
    Split input forward, all-gather gradients backward.
    Used when we need to scatter full tensor to TP ranks.
    """

    @staticmethod
    def forward(ctx, input_, tp_group, tp_size, tp_rank):
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size

        # Split along last dimension and keep this rank's portion
        dim_size = input_.shape[-1] // tp_size
        output = input_[..., tp_rank * dim_size : (tp_rank + 1) * dim_size]
        return output.contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        # All-gather gradients from all TP ranks
        tp_group = ctx.tp_group
        tp_size = ctx.tp_size

        grad_list = [torch.empty_like(grad_output) for _ in range(tp_size)]
        torch.distributed.all_gather(grad_list, grad_output, group=tp_group)
        grad_input = torch.cat(grad_list, dim=-1)
        return grad_input, None, None, None


# Convenience functions to apply autograd functions
def copy_to_tp_region(input_, tp_group):
    """Copy input to TP region (identity forward, all-reduce backward)."""
    return _CopyToTPRegion.apply(input_, tp_group)


def reduce_from_tp_region(input_, tp_group):
    """Reduce from TP region (all-reduce forward, identity backward)."""
    return _ReduceFromTPRegion.apply(input_, tp_group)


def gather_from_tp_region(input_, tp_group, tp_size, tp_rank):
    """Gather from TP region (all-gather forward, split backward)."""
    return _GatherFromTPRegion.apply(input_, tp_group, tp_size, tp_rank)


def scatter_to_tp_region(input_, tp_group, tp_size, tp_rank):
    """Scatter to TP region (split forward, all-gather backward)."""
    return _ScatterToTPRegion.apply(input_, tp_group, tp_size, tp_rank)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    Splits the weight matrix along the output dimension:
    Y = XA where A is partitioned column-wise across GPUs.

    Each GPU computes Y_i = X @ A_i (partial output)
    Output needs All-Gather if the next layer needs full tensor.

    Used for: QKV projections, MLP up-projection (fc1)
    """

    def __init__(self, in_features, out_features, bias=True, gather_output=False):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.tp_size
        self.tp_rank = tp.tp_rank
        self.tp_group = tp.tp_group
        self.gather_output = gather_output

        # Each GPU handles out_features // tp_size columns
        assert out_features % self.tp_size == 0
        self.out_features_per_partition = out_features // self.tp_size
        self.in_features = in_features
        self.out_features = out_features

        # Local weight: [in_features, out_features_per_partition]
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, seq, in_features]
        # output: [batch, seq, out_features_per_partition] or [batch, seq, out_features] if gather
        output = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            # All-gather outputs from all TP ranks (with proper autograd)
            output = gather_from_tp_region(
                output, self.tp_group, self.tp_size, self.tp_rank
            )

        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    Splits the weight matrix along the input dimension:
    Y = XA where A is partitioned row-wise across GPUs.

    Each GPU computes Y_i = X_i @ A_i (partial result)
    Output needs All-Reduce to sum partial results.

    Used for: Attention output projection, MLP down-projection (fc2)
    """

    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.tp_size
        self.tp_rank = tp.tp_rank
        self.tp_group = tp.tp_group
        self.input_is_parallel = input_is_parallel

        # Each GPU handles in_features // tp_size rows
        assert in_features % self.tp_size == 0
        self.in_features_per_partition = in_features // self.tp_size
        self.in_features = in_features
        self.out_features = out_features

        # Local weight: [out_features, in_features_per_partition]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            # Bias is not partitioned - only rank 0 adds it after reduce
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, seq, in_features_per_partition] (if input_is_parallel)
        #    or [batch, seq, in_features] (if not input_is_parallel, need to scatter)

        if not self.input_is_parallel:
            # Scatter input across TP ranks (with proper autograd)
            x = scatter_to_tp_region(x, self.tp_group, self.tp_size, self.tp_rank)

        # Local matmul: [batch, seq, in_features_per_partition] @ [in_features_per_partition, out_features]
        output = F.linear(x, self.weight)

        # All-reduce to sum partial results across TP ranks (with proper autograd)
        output = reduce_from_tp_region(output, self.tp_group)

        # Add bias after reduce (avoid double-counting)
        if self.bias is not None:
            output = output + self.bias

        return output


class TPAttention(nn.Module):
    """
    Tensor Parallel Multi-head Attention.

    - Heads are split across TP ranks
    - QKV projection: ColumnParallel (each GPU computes subset of heads)
    - Output projection: RowParallel (each GPU has subset, reduce to full)
    """

    def __init__(self, config):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.tp_size

        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout_p = config["dropout"]

        # Ensure heads can be evenly split
        assert self.num_heads % self.tp_size == 0
        self.num_heads_per_partition = self.num_heads // self.tp_size
        self.hidden_size_per_partition = self.num_heads_per_partition * self.head_dim

        # QKV: ColumnParallel - splits 3*hidden into 3*hidden/tp_size per GPU
        # Each GPU handles num_heads/tp_size heads
        self.c_attn = ColumnParallelLinear(
            self.hidden_size,
            3 * self.hidden_size,
            bias=True,
            gather_output=False,  # Keep split for attention computation
        )

        # Output: RowParallel - each GPU has hidden/tp_size, reduce to hidden
        self.c_proj = RowParallelLinear(
            self.hidden_size, self.hidden_size, bias=True, input_is_parallel=True
        )

        self.resid_dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection (column parallel)
        # Output: [B, T, 3 * hidden_size_per_partition]
        qkv = self.c_attn(x)

        # Split into Q, K, V for this partition
        q, k, v = qkv.split(self.hidden_size_per_partition, dim=2)

        # Reshape for multi-head attention
        # [B, T, hidden_per_part] -> [B, num_heads_per_part, T, head_dim]
        q = q.view(B, T, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads_per_partition, self.head_dim).transpose(1, 2)

        # Flash attention on local heads
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # Reshape back: [B, num_heads_per_part, T, head_dim] -> [B, T, hidden_per_part]
        out = (
            out.transpose(1, 2).contiguous().view(B, T, self.hidden_size_per_partition)
        )

        # Output projection (row parallel) - reduces across TP ranks
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return out


class TPMLP(nn.Module):
    """
    Tensor Parallel MLP.

    - c_fc (up-projection): ColumnParallel - split intermediate_size
    - c_proj (down-projection): RowParallel - reduce back to hidden_size
    """

    def __init__(self, config, use_fused_kernel=False):
        super().__init__()
        tp = get_tp_group()
        self.tp_size = tp.tp_size
        self.use_fused_kernel = use_fused_kernel

        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]

        assert intermediate_size % self.tp_size == 0

        # Up-projection: ColumnParallel
        # When using fused kernel, we split bias handling
        self.c_fc = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=not use_fused_kernel,  # No bias if using fused kernel
            gather_output=False,
        )

        # Separate bias for fused kernel
        if use_fused_kernel:
            # Bias is partitioned same as c_fc output
            self.c_fc_bias = nn.Parameter(
                torch.zeros(intermediate_size // self.tp_size)
            )

        # Down-projection: RowParallel
        self.c_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=True, input_is_parallel=True
        )

        self.dropout = nn.Dropout(config["dropout"])

        # Load fused kernel if requested
        if use_fused_kernel:
            try:
                from fused_kernels.fused_bias_gelu import fused_bias_gelu

                self._fused_bias_gelu = fused_bias_gelu
            except Exception as e:
                print(f"Warning: Could not load fused kernel: {e}")
                print("Falling back to standard implementation.")
                self.use_fused_kernel = False

    def forward(self, x):
        x = self.c_fc(x)
        if self.use_fused_kernel:
            # Use fused bias + GELU kernel
            x = self._fused_bias_gelu(x, self.c_fc_bias)
        else:
            x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TPBlock(nn.Module):
    """Tensor Parallel Transformer block."""

    def __init__(self, config, use_fused_kernel=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["hidden_size"])
        self.attn = TPAttention(config)
        self.ln_2 = nn.LayerNorm(config["hidden_size"])
        self.mlp = TPMLP(config, use_fused_kernel=use_fused_kernel)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TPModel(nn.Module):
    """
    Tensor Parallel GPT-2 Model.

    Combines TP for attention/MLP with replicated embeddings.
    Embeddings are replicated (small relative to attention/MLP).
    """

    def __init__(self, config, use_gradient_checkpointing=True, use_fused_kernel=False):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_fused_kernel = use_fused_kernel

        tp = get_tp_group()
        self.tp_rank = tp.tp_rank

        # Embeddings are replicated across TP ranks
        self.wte = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.wpe = nn.Embedding(
            config["max_position_embeddings"], config["hidden_size"]
        )
        self.drop = nn.Dropout(config["dropout"])

        # TP transformer blocks
        self.blocks = nn.ModuleList(
            [
                TPBlock(config, use_fused_kernel=use_fused_kernel)
                for _ in range(config["num_layers"])
            ]
        )

        # Final layer norm (replicated)
        self.ln_f = nn.LayerNorm(config["hidden_size"])

        # LM head (replicated for simplicity - could also be ColumnParallel)
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters (local to this TP rank)
        n_params = sum(p.numel() for p in self.parameters())
        # Estimate total params across TP ranks (rough approximation)
        # Most params are in attention/MLP which are split
        total_params_estimate = n_params * tp.tp_size * 0.6 + n_params * 0.4

        ckpt_status = "enabled" if use_gradient_checkpointing else "disabled"
        fused_status = "enabled" if use_fused_kernel else "disabled"
        print(
            f"TP-GPT-2 initialized: ~{total_params_estimate / 1e6:.1f}M total params "
            f"({n_params / 1e6:.1f}M local), TP rank {tp.tp_rank}/{tp.tp_size} "
            f"(gradient checkpointing: {ckpt_status}, fused kernel: {fused_status})"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        device = input_ids.device
        B, T = input_ids.size()

        # Position IDs
        position_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)

        # Embeddings (replicated)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(position_ids)
        x = self.drop(tok_emb + pos_emb)

        # TP Transformer blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        accuracy = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            # Compute top-1 token prediction accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == labels).float()
                accuracy = correct.mean() * 100

        return {"loss": loss, "logits": logits, "accuracy": accuracy}


# =============================================================================
# SECTION 4: Training Infrastructure
# =============================================================================


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ:
        # Launched with torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        # Single GPU
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def calculate_model_flops(config, batch_size, seq_length):
    """
    Calculate theoretical FLOPs for a single forward+backward pass of GPT-2.

    Based on the Chinchilla paper and PaLM paper methodology:
    - Forward pass: ~2 * N * tokens (where N = parameters)
    - Backward pass: ~4 * N * tokens (2x forward for gradients)
    - Total: ~6 * N * tokens

    More detailed breakdown per layer:
    - Attention QKV projection: 2 * 3 * seq * hidden^2
    - Attention scores: 2 * seq^2 * hidden
    - Attention output projection: 2 * seq * hidden^2
    - FFN: 2 * 2 * seq * hidden * intermediate (up + down projections)
    - Factor of 2 for forward, 4 for backward = 6 total
    """
    hidden = config["hidden_size"]
    num_layers = config["num_layers"]
    vocab_size = config["vocab_size"]
    intermediate = config["intermediate_size"]

    # Per-layer FLOPs (forward pass)
    # Attention: QKV projection + attention scores + output projection
    attn_qkv = 2 * 3 * seq_length * hidden * hidden  # Q, K, V projections
    attn_scores = 2 * seq_length * seq_length * hidden  # Attention computation
    attn_out = 2 * seq_length * hidden * hidden  # Output projection

    # FFN: two linear layers
    ffn = 2 * 2 * seq_length * hidden * intermediate  # Up and down projections

    # Per layer total
    per_layer = attn_qkv + attn_scores + attn_out + ffn

    # All layers
    all_layers = per_layer * num_layers

    # Embeddings and final projection
    embedding = 2 * seq_length * hidden  # Token + position
    final_proj = 2 * seq_length * hidden * vocab_size  # LM head

    # Total forward pass FLOPs
    forward_flops = all_layers + embedding + final_proj

    # Backward pass is approximately 2x forward
    backward_flops = 2 * forward_flops

    # Total FLOPs per sample
    total_flops_per_sample = forward_flops + backward_flops

    # Total FLOPs for batch
    total_flops = total_flops_per_sample * batch_size

    return total_flops


def get_gpu_utilization():
    """Get GPU utilization percentage using torch.cuda (simpler than pynvml)."""
    try:
        # This requires CUDA 11.0+ and gives memory utilization
        # For SM utilization, we'd need pynvml, but memory util is a good proxy
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_memory = torch.cuda.get_device_properties(0).total_memory
        return {
            "memory_utilization": allocated / max_memory * 100,
            "memory_reserved_utilization": reserved / max_memory * 100,
        }
    except Exception:
        return {"memory_utilization": 0, "memory_reserved_utilization": 0}


# A100 40GB theoretical peak FLOPS
A100_PEAK_FLOPS_FP16 = 312e12  # 312 TFLOPS for FP16 Tensor Core operations


class MetricsTracker:
    """Track and log training metrics including MFU."""

    def __init__(self, model_config, batch_size, seq_length, world_size, rank=0):
        self.rank = rank
        self.model_config = model_config
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.world_size = world_size

        # Calculate FLOPs per step (per GPU)
        self.flops_per_step = calculate_model_flops(
            model_config, batch_size, seq_length
        )

        # Theoretical peak for all GPUs
        self.theoretical_peak = A100_PEAK_FLOPS_FP16 * world_size

        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.accuracy_sum = 0.0
        self.tokens_sum = 0
        self.samples_sum = 0
        self.step_times = []
        self.start_time = time.time()

    def update(self, loss, accuracy, num_tokens, num_samples, step_time):
        self.loss_sum += loss
        self.accuracy_sum += accuracy if accuracy is not None else 0.0
        self.tokens_sum += num_tokens
        self.samples_sum += num_samples
        self.step_times.append(step_time)

    def get_metrics(self, steps):
        avg_loss = self.loss_sum / steps if steps > 0 else 0
        avg_accuracy = self.accuracy_sum / steps if steps > 0 else 0
        total_time = sum(self.step_times) if self.step_times else 1

        throughput_tokens = self.tokens_sum / total_time
        throughput_samples = self.samples_sum / total_time
        avg_step_time = total_time / len(self.step_times) if self.step_times else 0

        # Calculate MFU
        # FLOPs achieved = FLOPs per step * steps / total time
        total_flops = self.flops_per_step * len(self.step_times) * self.world_size
        achieved_flops_per_sec = total_flops / total_time
        mfu = (achieved_flops_per_sec / self.theoretical_peak) * 100

        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "perplexity": math.exp(avg_loss) if avg_loss < 100 else float("inf"),
            "throughput_tokens_per_sec": throughput_tokens,
            "throughput_samples_per_sec": throughput_samples,
            "avg_step_time_ms": avg_step_time * 1000,
            "mfu_percent": mfu,
            "achieved_tflops": achieved_flops_per_sec / 1e12,
        }

    def log(self, step, total_steps, metrics):
        if self.rank == 0:
            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"Step {step}/{total_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.1f}% | "
                f"Throughput: {metrics['throughput_tokens_per_sec']:.0f} tok/s | "
                f"MFU: {metrics['mfu_percent']:.1f}% | "
                f"Step time: {metrics['avg_step_time_ms']:.1f}ms | "
                f"GPU mem: {mem_used:.2f}GB"
            )


class MetricsLogger:
    """Save metrics to JSON file for comparison between runs."""

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.metrics_history = []

    def log_config(self, mode, model_config, train_config, world_size):
        """Log configuration at start of training."""
        self.config = {
            "mode": mode,
            "model": {
                "vocab_size": model_config["vocab_size"],
                "hidden_size": model_config["hidden_size"],
                "num_layers": model_config["num_layers"],
                "num_heads": model_config["num_heads"],
                "params_millions": sum(
                    [
                        model_config["vocab_size"] * model_config["hidden_size"],  # wte
                        model_config["max_position_embeddings"]
                        * model_config["hidden_size"],  # wpe
                        model_config["num_layers"]
                        * (
                            4
                            * model_config["hidden_size"]
                            * model_config["hidden_size"]  # attn
                            + 2
                            * model_config["hidden_size"]
                            * model_config["intermediate_size"]  # ffn
                        ),
                    ]
                )
                / 1e6,
            },
            "training": {
                "batch_size_per_gpu": train_config["batch_size_per_gpu"],
                "global_batch_size": train_config["batch_size_per_gpu"] * world_size,
                "seq_length": train_config["max_seq_length"],
                "learning_rate": train_config["learning_rate"],
                "gradient_accumulation_steps": train_config[
                    "gradient_accumulation_steps"
                ],
            },
            "hardware": {
                "num_gpus": world_size,
                "gpu_type": "A100-40GB",
                "theoretical_peak_tflops": A100_PEAK_FLOPS_FP16 * world_size / 1e12,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def log_step(self, step, metrics):
        """Log metrics for a single step."""
        self.metrics_history.append({"step": step, **metrics})

    def save(self, final_metrics):
        """Save all metrics to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config['mode']}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        output = {
            "config": self.config,
            "final_results": final_metrics,
            "step_history": self.metrics_history,
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        return filepath


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    model_config,
    train_config,
    epoch,
    rank=0,
    world_size=1,
    metrics_logger=None,
):
    """Train for one epoch."""
    model.train()

    batch_size = train_config["batch_size_per_gpu"]
    seq_length = train_config["max_seq_length"]
    metrics = MetricsTracker(model_config, batch_size, seq_length, world_size, rank)

    total_steps = len(dataloader)
    accumulation_steps = train_config["gradient_accumulation_steps"]
    log_interval = train_config["log_interval"]

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        step_start = time.time()

        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        # Mixed precision forward pass
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] / accumulation_steps
            accuracy = outputs["accuracy"]

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        step_time = time.time() - step_start
        num_tokens = input_ids.numel() * world_size
        num_samples = input_ids.size(0) * world_size
        current_loss = loss.item() * accumulation_steps
        current_accuracy = accuracy.item() if accuracy is not None else 0.0
        metrics.update(
            current_loss, current_accuracy, num_tokens, num_samples, step_time
        )

        # Log periodically
        if (step + 1) % log_interval == 0:
            last_metrics = metrics.get_metrics(log_interval)
            metrics.log(step + 1, total_steps, last_metrics)

            # Log to metrics logger if provided
            if metrics_logger is not None and rank == 0:
                metrics_logger.log_step(step + 1, last_metrics)

            metrics.reset()

    # Get final metrics - use current accumulated metrics if available, otherwise last logged
    steps_since_reset = (step + 1) % log_interval
    if steps_since_reset > 0:
        final_metrics = metrics.get_metrics(steps_since_reset)
        # Print final step
        metrics.log(step + 1, total_steps, final_metrics)
    else:
        final_metrics = (
            last_metrics
            if "last_metrics" in locals()
            else metrics.get_metrics(log_interval)
        )

    return final_metrics


# =============================================================================
# SECTION 5: Test Functions
# =============================================================================


def test_data_loading():
    """Test dataset download and tokenization."""
    print("=" * 60)
    print("TEST: Data Loading")
    print("=" * 60)

    from transformers import GPT2TokenizerFast

    # Load tokenizer
    print("\n1. Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Load dataset (test mode = small subset)
    print("\n2. Loading WikiText-103 (test mode)...")
    train_data, val_data = load_wikitext103(tokenizer, max_length=256, test_mode=True)

    # Create PyTorch dataset
    print("\n3. Creating PyTorch DataLoader...")
    train_dataset = WikiTextDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Test iteration
    print("\n4. Testing batch iteration...")
    batch = next(iter(train_loader))
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   Sample tokens: {batch['input_ids'][0, :10].tolist()}")
    print(f"   Decoded: {tokenizer.decode(batch['input_ids'][0, :20])[:50]}...")

    print("\n" + "=" * 60)
    print("SUCCESS: Data loading test passed!")
    print("=" * 60)


def test_model_creation():
    """Test model creation and forward pass."""
    print("=" * 60)
    print("TEST: Model Creation (Single GPU)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Using device: {device}")

    # Create small model for testing
    print("\n2. Creating test GPT-2 model...")
    model = GPT2Model(GPT2_TEST_CONFIG).to(device)

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(
        0, GPT2_TEST_CONFIG["vocab_size"], (batch_size, seq_len), device=device
    )
    labels = torch.randint(
        0, GPT2_TEST_CONFIG["vocab_size"], (batch_size, seq_len), device=device
    )

    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(input_ids, labels=labels)

    print(f"   Loss: {outputs['loss'].item():.4f}")
    print(f"   Logits shape: {outputs['logits'].shape}")

    # Test backward pass
    print("\n4. Testing backward pass...")
    outputs["loss"].backward()
    print("   Gradients computed successfully")

    # Memory usage
    mem_used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n5. Peak GPU memory: {mem_used:.2f} GB")

    print("\n" + "=" * 60)
    print("SUCCESS: Model creation test passed!")
    print("=" * 60)


def test_ddp_training():
    """Test DDP training with small model and data."""
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print(f"TEST: DDP Training ({world_size} GPUs)")
        print("=" * 60)

    try:
        from transformers import GPT2TokenizerFast

        # Load tokenizer and data
        if rank == 0:
            print("\n1. Loading tokenizer and data...")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        train_data, _ = load_wikitext103(tokenizer, max_length=256, test_mode=True)
        train_dataset = WikiTextDataset(train_data)

        # Distributed sampler
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        # Create model
        if rank == 0:
            print("\n2. Creating model with DDP...")
        model = GPT2Model(GPT2_TEST_CONFIG).cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

        # Optimizer and scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        # Train a few steps
        if rank == 0:
            print("\n3. Running 10 training steps...")

        model.train()
        for step, batch in enumerate(train_loader):
            if step >= 10:
                break

            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if rank == 0:
                print(f"   Step {step + 1}: loss = {loss.item():.4f}")

        # Memory stats
        torch.distributed.barrier()
        mem_used = torch.cuda.max_memory_allocated() / 1024**3
        if rank == 0:
            print(f"\n4. Peak GPU memory (rank 0): {mem_used:.2f} GB")
            print("\n" + "=" * 60)
            print("SUCCESS: DDP training test passed!")
            print("=" * 60)

    finally:
        cleanup_distributed()


# =============================================================================
# SECTION 6: Main Training Functions
# =============================================================================


def train_single_gpu(args):
    """Single GPU training for scaling efficiency measurement."""
    print("=" * 60)
    print("SINGLE GPU TRAINING (Scaling Baseline)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from transformers import GPT2TokenizerFast

        # Config
        model_config = GPT2_LARGE_CONFIG if not args.test else GPT2_TEST_CONFIG
        train_config = TRAINING_CONFIG.copy()
        if args.test:
            train_config["max_seq_length"] = 256

        # Override batch size if specified
        if args.batch_size is not None:
            train_config["batch_size_per_gpu"] = args.batch_size

        # Load data
        print("\nLoading tokenizer and dataset...")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        train_data, val_data = load_wikitext103(
            tokenizer,
            max_length=train_config["max_seq_length"],
            test_mode=args.test,
        )

        # Limit training samples if specified
        if args.max_samples is not None and args.max_samples < len(train_data):
            train_data = train_data.select(range(args.max_samples))
            print(f"Limited training data to {args.max_samples} samples")

        train_dataset = WikiTextDataset(train_data)

        # DataLoader (no distributed sampler)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config["batch_size_per_gpu"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # Model (no DDP wrapper)
        print("\nCreating GPT-2 model...")
        model = GPT2Model(model_config).to(device)

        # Optimizer, scheduler, scaler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=0.01,
        )

        total_steps = len(train_loader) * train_config["num_epochs"]
        scheduler = get_lr_scheduler(
            optimizer, train_config["warmup_steps"], total_steps
        )
        scaler = torch.amp.GradScaler("cuda")

        print(
            f"\nTraining for {train_config['num_epochs']} epoch(s), {total_steps} steps"
        )
        print(f"Batch size: {train_config['batch_size_per_gpu']}")

        # Calculate theoretical FLOPs info
        flops_per_step = calculate_model_flops(
            model_config,
            train_config["batch_size_per_gpu"],
            train_config["max_seq_length"],
        )
        print(f"FLOPs per step: {flops_per_step / 1e12:.2f} TFLOPs")
        print(f"Theoretical peak (1x A100): {A100_PEAK_FLOPS_FP16 / 1e12:.0f} TFLOPs")

        # Initialize metrics
        metrics_logger = MetricsLogger("single_gpu")
        metrics = MetricsTracker(
            model_config,
            train_config["batch_size_per_gpu"],
            train_config["max_seq_length"],
            world_size=1,  # Single GPU
            rank=0,
        )

        # Training loop
        model.train()
        accumulation_steps = train_config["gradient_accumulation_steps"]
        log_interval = 10

        start_time = time.time()

        for epoch in range(train_config["num_epochs"]):
            print(f"\n--- Epoch {epoch + 1}/{train_config['num_epochs']} ---")

            for step, batch in enumerate(train_loader):
                step_start = time.time()

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass with mixed precision
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"] / accumulation_steps
                    accuracy = outputs["accuracy"]

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                step_time = time.time() - step_start
                num_tokens = input_ids.numel()
                num_samples = input_ids.size(0)
                current_loss = loss.item() * accumulation_steps
                current_accuracy = accuracy.item() if accuracy is not None else 0.0
                metrics.update(
                    current_loss, current_accuracy, num_tokens, num_samples, step_time
                )

                # Log periodically
                if (step + 1) % log_interval == 0:
                    last_metrics = metrics.get_metrics(log_interval)
                    metrics.log(step + 1, total_steps, last_metrics)
                    metrics_logger.log_step(step + 1, last_metrics)
                    metrics.reset()

        training_time = time.time() - start_time
        # Get final metrics - use current accumulated metrics if available, otherwise last logged
        steps_since_reset = (step + 1) % log_interval
        if steps_since_reset > 0:
            final_metrics = metrics.get_metrics(steps_since_reset)
            # Print final step
            metrics.log(step + 1, total_steps, final_metrics)
        else:
            final_metrics = (
                last_metrics
                if "last_metrics" in locals()
                else metrics.get_metrics(log_interval)
            )
        mem_used = torch.cuda.max_memory_allocated() / 1024**3
        gpu_util = get_gpu_utilization()

        # Prepare final results
        final_results = {
            "final_loss": final_metrics["loss"],
            "accuracy": final_metrics["accuracy"],
            "throughput_tokens_per_sec": final_metrics["throughput_tokens_per_sec"],
            "throughput_samples_per_sec": final_metrics["throughput_samples_per_sec"],
            "mfu_percent": final_metrics["mfu_percent"],
            "achieved_tflops": final_metrics["achieved_tflops"],
            "peak_memory_gb": mem_used,
            "memory_utilization_percent": gpu_util["memory_utilization"],
            "total_training_time_sec": training_time,
            "total_steps": total_steps,
            "num_gpus": 1,
        }

        # Save to JSON
        results_file = metrics_logger.save(final_results)

        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE (Single GPU)")
        print(f"{'=' * 60}")
        print(f"Final loss:           {final_metrics['loss']:.4f}")
        print(f"Accuracy:             {final_metrics['accuracy']:.1f}%")
        print(
            f"Throughput:           {final_metrics['throughput_tokens_per_sec']:.0f} tokens/sec ({final_metrics['throughput_samples_per_sec']:.1f} samples/sec)"
        )
        print(f"MFU:                  {final_metrics['mfu_percent']:.1f}%")
        print(f"Achieved TFLOPs:      {final_metrics['achieved_tflops']:.1f}")
        print(f"Peak GPU memory:      {mem_used:.2f} GB")
        print(f"Total training time:  {training_time:.1f} sec")
        print(f"Results saved to:     {results_file}")
        print("=" * 60)

    except Exception as e:
        print(f"Error during single GPU training: {e}")
        raise
    finally:
        torch.cuda.empty_cache()


def train_baseline_ddp(args):
    """Full baseline DDP training."""
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("BASELINE DDP TRAINING")
        print(f"World size: {world_size} GPUs")
        print("=" * 60)

    try:
        from transformers import GPT2TokenizerFast

        # Config
        model_config = GPT2_LARGE_CONFIG if not args.test else GPT2_TEST_CONFIG
        train_config = TRAINING_CONFIG.copy()
        if args.test:
            train_config["max_seq_length"] = 256

        # Override batch size if specified
        if args.batch_size is not None:
            train_config["batch_size_per_gpu"] = args.batch_size

        # Load data
        if rank == 0:
            print("\nLoading tokenizer and dataset...")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        train_data, val_data = load_wikitext103(
            tokenizer,
            max_length=train_config["max_seq_length"],
            test_mode=args.test,
        )

        # Limit training samples if specified
        if args.max_samples is not None and args.max_samples < len(train_data):
            train_data = train_data.select(range(args.max_samples))
            if rank == 0:
                print(f"Limited training data to {args.max_samples} samples")

        train_dataset = WikiTextDataset(train_data)

        # DataLoader with distributed sampler
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config["batch_size_per_gpu"],
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # Model
        if rank == 0:
            print("\nCreating GPT-2 model...")
        model = GPT2Model(model_config).cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

        # Optimizer, scheduler, scaler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=0.01,
        )

        total_steps = len(train_loader) * train_config["num_epochs"]
        scheduler = get_lr_scheduler(
            optimizer, train_config["warmup_steps"], total_steps
        )
        scaler = torch.amp.GradScaler("cuda")

        if rank == 0:
            print(
                f"\nTraining for {train_config['num_epochs']} epoch(s), {total_steps} steps"
            )
            print(f"Batch size per GPU: {train_config['batch_size_per_gpu']}")
            print(
                f"Global batch size: {train_config['batch_size_per_gpu'] * world_size}"
            )

            # Calculate theoretical FLOPs info
            flops_per_step = calculate_model_flops(
                model_config,
                train_config["batch_size_per_gpu"],
                train_config["max_seq_length"],
            )
            print(f"FLOPs per step (per GPU): {flops_per_step / 1e12:.2f} TFLOPs")
            print(
                f"Theoretical peak (4x A100): {A100_PEAK_FLOPS_FP16 * world_size / 1e12:.0f} TFLOPs"
            )

        # Initialize metrics logger (only on rank 0)
        metrics_logger = None
        if rank == 0:
            metrics_logger = MetricsLogger(output_dir="results")
            metrics_logger.log_config(
                "baseline", model_config, train_config, world_size
            )

        # Training loop
        training_start_time = time.time()
        for epoch in range(train_config["num_epochs"]):
            sampler.set_epoch(epoch)

            if rank == 0:
                print(f"\n--- Epoch {epoch + 1}/{train_config['num_epochs']} ---")

            final_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                model_config,
                train_config,
                epoch,
                rank,
                world_size,
                metrics_logger,
            )

        training_time = time.time() - training_start_time

        # Final stats
        torch.distributed.barrier()
        if rank == 0:
            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            gpu_util = get_gpu_utilization()

            # Prepare final results
            final_results = {
                "final_loss": final_metrics["loss"],
                "accuracy": final_metrics["accuracy"],
                "throughput_tokens_per_sec": final_metrics["throughput_tokens_per_sec"],
                "throughput_samples_per_sec": final_metrics[
                    "throughput_samples_per_sec"
                ],
                "mfu_percent": final_metrics["mfu_percent"],
                "achieved_tflops": final_metrics["achieved_tflops"],
                "peak_memory_gb": mem_used,
                "memory_utilization_percent": gpu_util["memory_utilization"],
                "total_training_time_sec": training_time,
                "total_steps": total_steps,
            }

            # Save to JSON
            results_file = metrics_logger.save(final_results)

            print(f"\n{'=' * 60}")
            print("TRAINING COMPLETE")
            print(f"{'=' * 60}")
            print(f"Final loss:           {final_metrics['loss']:.4f}")
            print(f"Accuracy:             {final_metrics['accuracy']:.1f}%")
            print(
                f"Throughput:           {final_metrics['throughput_tokens_per_sec']:.0f} tokens/sec ({final_metrics['throughput_samples_per_sec']:.1f} samples/sec)"
            )
            print(f"MFU:                  {final_metrics['mfu_percent']:.1f}%")
            print(f"Achieved TFLOPs:      {final_metrics['achieved_tflops']:.1f}")
            print(f"Peak GPU memory:      {mem_used:.2f} GB")
            print(f"Total training time:  {training_time:.1f} sec")
            print(f"Results saved to:     {results_file}")
            print("=" * 60)

    finally:
        cleanup_distributed()


def train_hybrid(args):
    """Hybrid training with DeepSpeed ZeRO-3 (and optionally Tensor Parallelism)."""
    import deepspeed
    from deepspeed import comm as dist

    # Initialize distributed (DeepSpeed handles this)
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Get TP size from args
    tp_size = getattr(args, "tp_size", 1)
    use_tp = tp_size > 1

    # Initialize TP groups if using tensor parallelism
    if use_tp:
        tp_group = get_tp_group()
        tp_group.initialize(tp_size=tp_size)
        dp_size = world_size // tp_size
    else:
        dp_size = world_size

    if rank == 0:
        print("=" * 60)
        if use_tp:
            print(f"HYBRID TRAINING (DeepSpeed ZeRO-3 + TP={tp_size})")
            print(f"World size: {world_size} GPUs (DP={dp_size} × TP={tp_size})")
        else:
            print("HYBRID TRAINING (DeepSpeed ZeRO-3)")
            print(f"World size: {world_size} GPUs")
        print("=" * 60)

    try:
        from transformers import GPT2TokenizerFast

        # Config
        model_config = GPT2_LARGE_CONFIG if not args.test else GPT2_TEST_CONFIG
        train_config = TRAINING_CONFIG.copy()
        if args.test:
            train_config["max_seq_length"] = 256

        # Override batch size if specified
        if args.batch_size is not None:
            train_config["batch_size_per_gpu"] = args.batch_size

        # Load data
        if rank == 0:
            print("\nLoading tokenizer and dataset...")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        train_data, val_data = load_wikitext103(
            tokenizer,
            max_length=train_config["max_seq_length"],
            test_mode=args.test,
        )

        # Limit training samples if specified
        if args.max_samples is not None and args.max_samples < len(train_data):
            train_data = train_data.select(range(args.max_samples))
            if rank == 0:
                print(f"Limited training data to {args.max_samples} samples")

        train_dataset = WikiTextDataset(train_data)

        # Create model (without DDP wrapper - DeepSpeed handles this)
        if rank == 0:
            print("\nCreating GPT-2 model...")

        # Check for optimization modes
        use_triton = getattr(args, "use_triton", False)
        use_fused_kernel = getattr(args, "use_fused_kernel", False)

        # Use TP model if tensor parallelism is enabled
        if use_tp:
            if use_triton:
                # Use Triton fused kernels for TP layers (novelty feature)
                from triton_tp_layers import TritonTPModel

                if rank == 0:
                    print("Using Triton fused kernels for TP layers...")
                model = TritonTPModel(model_config, use_gradient_checkpointing=True)
            elif use_fused_kernel:
                # Use custom CUDA fused bias+GELU kernel
                if rank == 0:
                    print("Using custom CUDA fused bias+GELU kernel...")
                model = TPModel(
                    model_config, use_gradient_checkpointing=True, use_fused_kernel=True
                )
            else:
                model = TPModel(model_config, use_gradient_checkpointing=True)
        else:
            if use_triton and rank == 0:
                print("Warning: --use-triton requires --tp-size > 1, ignoring...")
            if use_fused_kernel and rank == 0:
                print("Warning: --use-fused-kernel requires --tp-size > 1, ignoring...")
            # Enable gradient checkpointing alongside ZeRO-3 for maximum memory efficiency
            model = GPT2Model(model_config, use_gradient_checkpointing=True)

        # Load DeepSpeed config
        ds_config_path = (
            args.deepspeed_config if args.deepspeed_config else "ds_config_zero3.json"
        )

        # Update DeepSpeed config with actual training params
        with open(ds_config_path, "r") as f:
            ds_config = json.load(f)

        ds_config["train_micro_batch_size_per_gpu"] = train_config["batch_size_per_gpu"]
        ds_config["gradient_accumulation_steps"] = train_config[
            "gradient_accumulation_steps"
        ]

        # Calculate total steps for scheduler
        total_samples = len(train_dataset)
        total_steps = (
            total_samples // (train_config["batch_size_per_gpu"] * world_size)
        ) * train_config["num_epochs"]

        # Update warmup steps (WarmupLR doesn't need total_num_steps)
        ds_config["scheduler"]["params"]["warmup_num_steps"] = min(
            train_config["warmup_steps"], max(1, total_steps // 10)
        )

        # Initialize DeepSpeed
        model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
            training_data=train_dataset,
            collate_fn=collate_fn,
        )

        if rank == 0:
            print(f"\nTraining for {train_config['num_epochs']} epoch(s)")
            print(f"Batch size per GPU: {train_config['batch_size_per_gpu']}")
            print(
                f"Global batch size: {train_config['batch_size_per_gpu'] * world_size}"
            )
            print(f"ZeRO Stage: {ds_config['zero_optimization']['stage']}")

            # Calculate theoretical FLOPs info
            flops_per_step = calculate_model_flops(
                model_config,
                train_config["batch_size_per_gpu"],
                train_config["max_seq_length"],
            )
            print(f"FLOPs per step (per GPU): {flops_per_step / 1e12:.2f} TFLOPs")
            print(
                f"Theoretical peak (4x A100): {A100_PEAK_FLOPS_FP16 * world_size / 1e12:.0f} TFLOPs"
            )

        # Initialize metrics logger (only on rank 0)
        metrics_logger = None
        if rank == 0:
            metrics_logger = MetricsLogger(output_dir="results")
            metrics_logger.log_config(
                "hybrid_zero3", model_config, train_config, world_size
            )

        # Training loop
        training_start_time = time.time()
        total_steps_done = 0
        log_interval = train_config["log_interval"]

        batch_size = train_config["batch_size_per_gpu"]
        seq_length = train_config["max_seq_length"]
        metrics = MetricsTracker(model_config, batch_size, seq_length, world_size, rank)

        for epoch in range(train_config["num_epochs"]):
            if rank == 0:
                print(f"\n--- Epoch {epoch + 1}/{train_config['num_epochs']} ---")

            model_engine.train()

            for step, batch in enumerate(train_loader):
                step_start = time.time()

                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()

                # Forward pass (DeepSpeed handles mixed precision)
                outputs = model_engine(input_ids, labels=labels)
                loss = outputs["loss"]
                accuracy = outputs["accuracy"]

                # Backward pass (DeepSpeed handles gradient scaling)
                model_engine.backward(loss)

                # Optimizer step (DeepSpeed handles gradient accumulation)
                model_engine.step()

                step_time = time.time() - step_start
                num_tokens = input_ids.numel() * world_size
                num_samples = input_ids.size(0) * world_size
                current_loss = loss.item()
                current_accuracy = accuracy.item() if accuracy is not None else 0.0
                metrics.update(
                    current_loss, current_accuracy, num_tokens, num_samples, step_time
                )
                total_steps_done += 1

                # Log periodically
                if (step + 1) % log_interval == 0:
                    last_metrics = metrics.get_metrics(log_interval)
                    metrics.log(step + 1, len(train_loader), last_metrics)

                    if metrics_logger is not None and rank == 0:
                        metrics_logger.log_step(total_steps_done, last_metrics)

                    metrics.reset()

        training_time = time.time() - training_start_time
        # Get final metrics - use current accumulated metrics if available, otherwise last logged
        steps_since_reset = (step + 1) % log_interval
        if steps_since_reset > 0:
            final_metrics = metrics.get_metrics(steps_since_reset)
            # Print final step
            metrics.log(step + 1, len(train_loader), final_metrics)
        else:
            final_metrics = (
                last_metrics
                if "last_metrics" in locals()
                else metrics.get_metrics(log_interval)
            )

        # Final stats
        dist.barrier()
        if rank == 0:
            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            gpu_util = get_gpu_utilization()

            # Prepare final results
            final_results = {
                "final_loss": final_metrics["loss"],
                "accuracy": final_metrics["accuracy"],
                "throughput_tokens_per_sec": final_metrics["throughput_tokens_per_sec"],
                "throughput_samples_per_sec": final_metrics[
                    "throughput_samples_per_sec"
                ],
                "mfu_percent": final_metrics["mfu_percent"],
                "achieved_tflops": final_metrics["achieved_tflops"],
                "peak_memory_gb": mem_used,
                "memory_utilization_percent": gpu_util["memory_utilization"],
                "total_training_time_sec": training_time,
                "total_steps": total_steps_done,
                "zero_stage": ds_config["zero_optimization"]["stage"],
                "tp_size": tp_size,
                "dp_size": dp_size,
                "use_tp": use_tp,
                "use_triton": use_triton,
                "use_fused_kernel": use_fused_kernel,
            }

            # Save to JSON
            results_file = metrics_logger.save(final_results)

            if use_tp:
                mode_str = f"Hybrid ZeRO-3 + TP={tp_size}"
                if use_triton:
                    mode_str += " + Triton"
                elif use_fused_kernel:
                    mode_str += " + CUDA Fused"
            else:
                mode_str = "Hybrid ZeRO-3"
            print(f"\n{'=' * 60}")
            print(f"TRAINING COMPLETE ({mode_str})")
            print(f"{'=' * 60}")
            print(f"Final loss:           {final_metrics['loss']:.4f}")
            print(f"Accuracy:             {final_metrics['accuracy']:.1f}%")
            print(
                f"Throughput:           {final_metrics['throughput_tokens_per_sec']:.0f} tokens/sec ({final_metrics['throughput_samples_per_sec']:.1f} samples/sec)"
            )
            print(f"MFU:                  {final_metrics['mfu_percent']:.1f}%")
            print(f"Achieved TFLOPs:      {final_metrics['achieved_tflops']:.1f}")
            print(f"Peak GPU memory:      {mem_used:.2f} GB")
            print(f"Total training time:  {training_time:.1f} sec")
            if use_tp:
                print(f"Parallelism:          DP={dp_size} × TP={tp_size}")
            if use_triton:
                print(
                    f"Triton kernels:       Enabled (FusedLinearGELU, FusedLinearDropout)"
                )
            if use_fused_kernel:
                print(f"CUDA fused kernel:    Enabled (Fused Bias+GELU)")
            print(f"Results saved to:     {results_file}")
            print("=" * 60)

    except Exception as e:
        print(f"Error during hybrid training: {e}")
        raise
    finally:
        # TP cleanup (if used)
        if use_tp:
            tp_group = get_tp_group()
            tp_group.cleanup()

        # DeepSpeed cleanup
        if "model_engine" in locals():
            del model_engine
        torch.cuda.empty_cache()


# =============================================================================
# SECTION 7: Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="GPT-2 Hybrid Parallelism Training")

    # Test modes
    parser.add_argument(
        "--test-data", action="store_true", help="Test data loading only"
    )
    parser.add_argument(
        "--test-model", action="store_true", help="Test model creation (single GPU)"
    )
    parser.add_argument(
        "--test-ddp", action="store_true", help="Test DDP training (multi-GPU)"
    )

    # Training modes
    parser.add_argument(
        "--mode",
        choices=["baseline", "hybrid", "single"],
        default="baseline",
        help="Training mode: single (1 GPU), baseline (DDP), or hybrid (DP×TP)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Use small model/data for testing"
    )

    # Data options
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (e.g., 10000 for quick runs)",
    )

    # Training options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per GPU (default: 8). Increase for better MFU with TP.",
    )

    # DeepSpeed (for hybrid mode)
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for DeepSpeed"
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default="ds_config_zero3.json",
        help="Path to DeepSpeed config file",
    )

    # Tensor Parallelism
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (1=disabled, 2=split across 2 GPUs)",
    )

    # Triton fused kernels (novelty feature)
    parser.add_argument(
        "--use-triton",
        action="store_true",
        help="Use Triton fused kernels for TP layers (requires --tp-size > 1)",
    )

    # Custom CUDA fused kernels (novelty feature)
    parser.add_argument(
        "--use-fused-kernel",
        action="store_true",
        help="Use custom CUDA fused bias+GELU kernel for TP layers (requires --tp-size > 1)",
    )

    args = parser.parse_args()

    # Run appropriate mode
    if args.test_data:
        test_data_loading()
    elif args.test_model:
        test_model_creation()
    elif args.test_ddp:
        test_ddp_training()
    elif args.mode == "single":
        train_single_gpu(args)
    elif args.mode == "baseline":
        train_baseline_ddp(args)
    elif args.mode == "hybrid":
        train_hybrid(args)


if __name__ == "__main__":
    main()
