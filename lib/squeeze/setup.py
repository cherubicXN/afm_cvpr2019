import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy as np

def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

class custom_build_ext(build_ext):
    def build_extensions(self):
        # customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

#     def build_extensions(self):
#         build_ext.build_extensions(self)

ext_modules = [
    Extension("squeeze",
              ["kernel.cpp", "squeeze.pyx"],
              include_dirs=[numpy_include],
              language='c++',
              extra_compile_args=["-Wno-unused-function"]
              # extra_compile_args={
              #     'g++': ["-Wno-unused-function"]},
              ),                      
]

setup(#name='lsd',
    ext_modules=ext_modules,
      cmdclass={'build_ext': custom_build_ext})