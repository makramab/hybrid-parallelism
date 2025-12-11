// fused_kernels/bias_gelu_cuda.cu
// Fused Bias + GELU CUDA Kernel
// Computes y = GELU(x + bias) in a single kernel pass
// Reduces memory traffic by avoiding intermediate tensor materialization

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU forward using tanh approximation (matches PyTorch's approximate='tanh')
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
__device__ __forceinline__ float gelu(float x) {
    const float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/π)
    const float kCoeff = 0.044715f;
    float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// GELU backward (derivative of tanh approximation)
// Let inner = sqrt(2/π) * (x + 0.044715 * x³)
// GELU = 0.5 * x * (1 + tanh(inner))
// d(GELU)/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech²(inner) * d(inner)/dx
// where d(inner)/dx = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
__device__ __forceinline__ float gelu_grad(float x) {
    const float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/π)
    const float kCoeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * x_cubed);
    float tanh_inner = tanhf(inner);
    float sech2_inner = 1.0f - tanh_inner * tanh_inner;  // sech²(x) = 1 - tanh²(x)
    float inner_grad = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * x * x);
    return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2_inner * inner_grad;
}

// Forward kernel: y = GELU(x + bias), also stores (x + bias) for backward
template <typename T>
__global__ void bias_gelu_fwd_kernel(
    const T* __restrict__ x,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ xb,  // stores x + bias for backward
    int n_rows,
    int hidden
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * hidden;

    if (idx >= total) return;

    int h = idx % hidden;  // which hidden dimension (for bias indexing)

    // Compute x + bias
    float v = (float)x[idx] + (float)bias[h];

    // Store for backward pass
    xb[idx] = (T)v;

    // Apply GELU and store output
    y[idx] = (T)gelu(v);
}

// Backward kernel: compute grad_x = grad_out * gelu'(x + bias)
template <typename T>
__global__ void bias_gelu_bwd_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ xb,  // stored (x + bias) from forward
    T* __restrict__ grad_x,
    int n_rows,
    int hidden
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * hidden;

    if (idx >= total) return;

    float v = (float)xb[idx];           // x + bias (saved from forward)
    float go = (float)grad_out[idx];    // upstream gradient

    // grad_x = grad_out * gelu'(x + bias)
    grad_x[idx] = (T)(go * gelu_grad(v));
}

// C++ wrapper for forward pass
std::vector<torch::Tensor> bias_gelu_forward_cuda(
    torch::Tensor x,
    torch::Tensor bias
) {
    // x: [B, T, H] or [N, H] where N = B*T
    // bias: [H]

    auto sizes = x.sizes();
    int ndim = sizes.size();

    int H = sizes[ndim - 1];  // hidden dimension (last dim)
    int N = x.numel() / H;    // total number of rows

    // Verify bias shape
    TORCH_CHECK(bias.size(0) == H, "Bias size must match hidden dimension");

    // Allocate outputs
    auto y = torch::empty_like(x);
    auto xb = torch::empty_like(x);  // store x + bias for backward

    // Launch kernel
    int block = 256;
    int grid = (N * H + block - 1) / block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "bias_gelu_forward_cuda", [&] {
            bias_gelu_fwd_kernel<scalar_t>
                <<<grid, block>>>(
                    x.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    y.data_ptr<scalar_t>(),
                    xb.data_ptr<scalar_t>(),
                    N, H
                );
        }
    );

    return {y, xb};
}

// C++ wrapper for backward pass
std::vector<torch::Tensor> bias_gelu_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor xb
) {
    // grad_out: [B, T, H] or [N, H]
    // xb: [B, T, H] or [N, H] - saved (x + bias) from forward

    auto sizes = grad_out.sizes();
    int ndim = sizes.size();

    int H = sizes[ndim - 1];
    int N = grad_out.numel() / H;

    // Allocate grad_x
    auto grad_x = torch::empty_like(grad_out);

    // Launch kernel
    int block = 256;
    int grid = (N * H + block - 1) / block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_out.scalar_type(), "bias_gelu_backward_cuda", [&] {
            bias_gelu_bwd_kernel<scalar_t>
                <<<grid, block>>>(
                    grad_out.data_ptr<scalar_t>(),
                    xb.data_ptr<scalar_t>(),
                    grad_x.data_ptr<scalar_t>(),
                    N, H
                );
        }
    );

    // Compute grad_bias by reducing grad_x over the batch dimension
    // grad_bias[h] = sum over all positions of grad_x[..., h]
    // This is equivalent to: grad_bias = grad_x.view(-1, H).sum(0)
    auto grad_bias = grad_x.view({N, H}).sum(0);

    return {grad_x, grad_bias};
}
