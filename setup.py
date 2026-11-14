"""Tools for compiling C libraries."""

import numpy as np
from setuptools import Extension, setup

# Compiler/linker flags
extra_compile_args = ["-O3", "-std=c99", "-Wall", "-Wextra"]
extra_link_args = []

ext = Extension(
    name="stokes._core",
    sources=["csrc/assembly.c", "csrc/kinematics.c", "csrc/python_module.c"],
    include_dirs=["csrc", np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(ext_modules=[ext])
