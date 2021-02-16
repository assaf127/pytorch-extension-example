#include <torch/extension.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// Templated element-wise device function
template <typename scalar_t>
__forceinline__ GPU_LAMBDA scalar_t my_relu_cuda_forward_kernel(scalar_t input_val) {
    return (input_val > 0) ? input_val : 0;
}
// Specialization example
template <>
__forceinline__ GPU_LAMBDA float my_relu_cuda_forward_kernel<float>(float input_val) {
    return 1 + ((input_val > 0) ? input_val : 0);
}


template <typename scalar_t>
__forceinline__ GPU_LAMBDA scalar_t my_relu_cuda_backward_kernel(scalar_t d_input_val, scalar_t input_val) {
    return (input_val > 0) ? d_input_val : 0;
}

// This function replaces torch::empty_like which did not work for some reason
inline torch::Tensor init_like(const torch::Tensor &other) {
    auto options = torch::TensorOptions()
        .dtype(other.dtype())
        .layout(other.layout())
        .device(other.device());
    return torch::empty(other.sizes(), options);
}


// The host functions to export

torch::Tensor my_relu_cuda_forward(torch::Tensor input) {
    auto res = init_like(input);

    auto iter = at::TensorIteratorConfig().add_output(res).add_input(input).build();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_relu_cuda_forward", [&] {
        at::native::gpu_kernel(iter, [] GPU_LAMBDA (scalar_t input_val) -> scalar_t {
            return my_relu_cuda_forward_kernel<scalar_t>(input_val);
        });
    });

    return res;
}

torch::Tensor my_relu_cuda_backward(torch::Tensor d_input, torch::Tensor input) {
    auto res = init_like(input);

    auto iter = at::TensorIteratorConfig().add_output(res).add_input(d_input).add_input(input).build();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_relu_cuda_backward", [&] {
        at::native::gpu_kernel(iter, [] GPU_LAMBDA (scalar_t d_input_val, scalar_t input_val) -> scalar_t {
            return my_relu_cuda_backward_kernel<scalar_t>(d_input_val, input_val);
        });
    });

    return res;
}

