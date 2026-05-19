from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='agoge_dummy_cuda',
    ext_modules=[
        CUDAExtension(
            name='agoge_dummy_cuda',
            sources=[
                'kernels/agoge_dummy_kernel.cpp',
                'kernels/agoge_dummy_kernel.cu',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
