#include <torch/extension.h>

// CUDA forward declarations
torch::Tensor my_relu_cuda_forward(torch::Tensor input);
torch::Tensor my_relu_cuda_backward(torch::Tensor d_input, torch::Tensor input);
// CPU forward declarations
torch::Tensor my_relu_cpu_forward(torch::Tensor input);
torch::Tensor my_relu_cpu_backward(torch::Tensor d_input, torch::Tensor input);


// Input checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")


// C++ Implementations

torch::Tensor my_relu_forward(torch::Tensor input) {
    if(input.is_cuda())
        return my_relu_cuda_forward(input);
    return my_relu_cpu_forward(input);
}

torch::Tensor my_relu_backward(torch::Tensor d_input, torch::Tensor input) {
    TORCH_CHECK(d_input.device() == input.device(), "Inputs must be on the same device, but got '",
                                                    d_input.device().str(), "' for the first input and '",
                                                    input.device().str(), "' for the second input");
    if(d_input.is_cuda())
        return my_relu_cuda_backward(d_input, input);
    return my_relu_cpu_backward(d_input, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &my_relu_forward, "my_relu forward");
    m.def("backward", &my_relu_backward, "my_relu backward");
}