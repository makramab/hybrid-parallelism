"""
fused_kernels/fused_bias_gelu.py
Python wrapper for fused bias + GELU CUDA kernel with autograd support.

This module provides:
- fused_bias_gelu(x, bias): Drop-in replacement for F.gelu(x + bias)
- BiasGeluFn: Autograd function for custom backward pass
"""

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os

# JIT compile the CUDA extension
_ext = None

def _get_extension():
    """Lazily load the CUDA extension."""
    global _ext
    if _ext is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="bias_gelu_ext",
            sources=[
                os.path.join(module_dir, "bias_gelu.cpp"),
                os.path.join(module_dir, "bias_gelu_cuda.cu"),
            ],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _ext


class BiasGeluFn(Function):
    """
    Autograd function for fused bias + GELU.

    Forward: y = GELU(x + bias)
    Backward: grad_x = grad_out * GELU'(x + bias)
              grad_bias = sum(grad_x, dim=batch)
    """

    @staticmethod
    def forward(ctx, x, bias):
        """
        Forward pass: compute y = GELU(x + bias)

        Args:
            x: Input tensor of shape [B, T, H] or [N, H]
            bias: Bias tensor of shape [H]

        Returns:
            y: Output tensor of same shape as x
        """
        ext = _get_extension()
        y, xb = ext.forward(x.contiguous(), bias.contiguous())
        ctx.save_for_backward(xb)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass: compute gradients.

        Args:
            grad_out: Gradient of loss w.r.t. output y

        Returns:
            grad_x: Gradient w.r.t. input x
            grad_bias: Gradient w.r.t. bias
        """
        xb, = ctx.saved_tensors
        ext = _get_extension()
        grad_x, grad_bias = ext.backward(grad_out.contiguous(), xb)
        return grad_x, grad_bias


def fused_bias_gelu(x, bias):
    """
    Fused bias + GELU activation.

    Computes y = GELU(x + bias) in a single CUDA kernel,
    avoiding intermediate tensor allocation for (x + bias).

    This is a drop-in replacement for:
        x = x + bias
        x = F.gelu(x)

    Args:
        x: Input tensor of shape [B, T, H] or [N, H]
        bias: Bias tensor of shape [H]

    Returns:
        Output tensor of same shape as x

    Example:
        # Instead of:
        x = self.c_fc(input)  # Linear without bias
        x = F.gelu(x + self.c_fc.bias)

        # Use:
        x = F.linear(input, self.c_fc.weight)  # No bias
        x = fused_bias_gelu(x, self.c_fc.bias)
    """
    return BiasGeluFn.apply(x, bias)


# Fallback implementation for CPU or when CUDA is not available
def fused_bias_gelu_fallback(x, bias):
    """
    Fallback implementation using standard PyTorch ops.
    Used when CUDA extension is not available.
    """
    import torch.nn.functional as F
    return F.gelu(x + bias, approximate='tanh')


def get_fused_bias_gelu():
    """
    Get the appropriate fused_bias_gelu function.
    Returns CUDA version if available, otherwise fallback.
    """
    if torch.cuda.is_available():
        try:
            # Test if extension can be loaded
            _get_extension()
            return fused_bias_gelu
        except Exception as e:
            print(f"Warning: Could not load fused_bias_gelu CUDA extension: {e}")
            print("Falling back to PyTorch implementation.")
            return fused_bias_gelu_fallback
    else:
        return fused_bias_gelu_fallback
