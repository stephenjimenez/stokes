"""Tools for compiling C libraries."""

import numpy as np
from setuptools import Extension, setup

# Compiler/linker flags (adjust for OpenMP if needed)
extra_compile_args = ["-O3", "-std=c99"]
extra_link_args = []

ext = Extension(
    name="stokes._c_assembly",
    sources=["csrc/assembly.c", "csrc/python_module.c"],
    include_dirs=["csrc", np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(ext_modules=[ext])
