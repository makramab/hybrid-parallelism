// fused_kernels/bias_gelu.cpp
// C++ binding for fused bias + GELU CUDA kernel

#include <torch/extension.h>

// Forward declarations of CUDA functions
std::vector<torch::Tensor> bias_gelu_forward_cuda(
    torch::Tensor x,
    torch::Tensor bias
);

std::vector<torch::Tensor> bias_gelu_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor xb
);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bias_gelu_forward_cuda, "Fused bias + GELU forward (CUDA)");
    m.def("backward", &bias_gelu_backward_cuda, "Fused bias + GELU backward (CUDA)");
}
