from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='my_relu',
      ext_modules=[CUDAExtension(name='my_relu',
                                 sources=['my_relu.cpp', 'my_relu_cpu.cpp', 'my_relu_cuda.cu'],
                                 extra_compile_args={'cxx': ['-fopenmp', '-O3'], 'nvcc': ['--extended-lambda']})],
      cmdclass={'build_ext': BuildExtension})
