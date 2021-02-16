import torch
# import my_relu

# JIT Compilation
import os
from torch.utils.cpp_extension import load
build_dir = os.path.join(os.getcwd(), 'my_relu')
os.makedirs(build_dir, exist_ok=True)
my_relu = load(name='my_relu', sources=['my_relu.cpp', 'my_relu_cpu.cpp', 'my_relu_cuda.cu'],
               extra_cuda_cflags=['--extended-lambda'], extra_cflags=['-fopenmp', '-O3'],
               build_directory=os.path.join(os.getcwd(), 'my_relu'), verbose=True)

# A simple test
x = torch.tensor([[1, 2], [-1, 5]], dtype=torch.double, device=torch.device('cuda'))
print(x)

y = my_relu.forward(x)  # Note that
print(y)

t = torch.tensor([[1, 1], [1, 2]], dtype=torch.double, device=torch.device('cuda'))
z = my_relu.backward(t, x)
print(z)

x = torch.tensor([[1, 2], [-1, 5]], dtype=torch.float, device=torch.device('cuda'))
print(x)

y = my_relu.forward(x)  # Note that the result is different due to the template specialization
print(y)

t = torch.tensor([[1, 1], [1, 2]], dtype=torch.float, device=torch.device('cuda'))
z = my_relu.backward(t, x)
print(z)