#include <torch/extension.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

inline torch::Tensor init_like(const torch::Tensor &other) {
    auto options = torch::TensorOptions()
        .dtype(other.dtype())
        .layout(other.layout())
        .device(other.device());
    return torch::empty(other.sizes(), options);
}

torch::Tensor my_relu_cpu_forward(torch::Tensor input) {
    auto res = init_like(input);

    auto iter = at::TensorIteratorConfig().add_output(res).add_input(input).build();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_relu", [&] {
        at::native::cpu_kernel(iter, [] (scalar_t input_val) {
            return (input_val > 0) ? input_val : 0;
        });
    });

    return res;
}

torch::Tensor my_relu_cpu_backward(torch::Tensor d_input, torch::Tensor input) {
    auto res = init_like(input);

    auto iter = at::TensorIteratorConfig().add_output(res).add_input(d_input).add_input(input).build();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "d_my_relu", [&] {
        at::native::cpu_kernel(iter, [] (scalar_t d_input_val, scalar_t input_val) {
            return (input_val > 0) ? d_input_val : 0;
        });
    });

    return res;
}

