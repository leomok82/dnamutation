# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import sys

# Define extra compile and link arguments based on the platform
extra_compile_args = ['-O3']
# extra_compile_args = ['-g', '-O0', '-fno-omit-frame-pointer'] #(for debugging)
extra_link_args = []
if sys.platform == 'win32':
    # For Windows, use optimization flag
    extra_compile_args += ['/O2']
else:
    # For Unix-like systems, enable architecture-specific optimizations
    extra_compile_args += ['-march=native', '-fopenmp']
    extra_link_args += ['-fopenmp']
setup(
    name='mutation_new',
    version='0.1',

    ext_modules=[
        CppExtension(
            name='mutation_new',               # Name of the Python module
            sources=['mutation.cpp'],      # Source file(s)
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            # include_dirs can be omitted as CppExtension handles PyTorch and Pybind11 includes
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
