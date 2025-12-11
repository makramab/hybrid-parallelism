# Fused CUDA Kernels for Tensor Parallel GPT-2
# Custom bias+GELU fusion for reduced memory traffic

from .fused_bias_gelu import fused_bias_gelu, BiasGeluFn

__all__ = ['fused_bias_gelu', 'BiasGeluFn']
