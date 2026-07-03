#!/usr/bin/env python

import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import setup

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = [os.path.join(extensions_dir, "vision.cpp")]
    source_cuda = [
        os.path.join(extensions_dir, "cuda", "dcn_v2_cuda.cu"),
        os.path.join(extensions_dir, "cuda", "dcn_v2_im2col_cuda.cu"),
    ]
    
    os.environ["CC"] = "g++"
    sources = main_file
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    
    if CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        #raise NotImplementedError('Cuda is not available')
        pass
    

    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "_ext",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="DCNv2",
    version="0.1",
    author="charlesshang",
    url="https://github.com/charlesshang/DCNv2",
    description="deformable convolutional networks",
    packages=["DCNv2"],
    package_dir={"DCNv2": "."},
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
