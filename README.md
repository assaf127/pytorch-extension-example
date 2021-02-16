# PyTorch Extension Example

An example of a custom C++/CUDA extension for pytorch for element-wise operations on tensors

The code demonstrates how to write a custom PyTorch C++/CUDA extension for element-wise operators on tensors.
As an example, the ReLU function and its backward (gradient) version are implemented, both for the CPU and for the GPU (using CUDA).

The input and output tensors are iterated over using a TensorIterator object, and custom CPU/GPU element-wise functions are called using at::native::gpu_kernel / at::native::cpu_kernel.

## Build
The extension can be compiled using:
```
python setup.py build_ext --inplace
```
and then imported with: 
```
import my_relu
```

Alternatively, the extension can be compiled Just-In-Time (JIT), as demonstrated in [test.py](test.py).
