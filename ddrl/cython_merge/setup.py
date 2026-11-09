# Build the greedy-merge extension in place:
#     cd ddrl/cython_merge && python setup.py build_ext --inplace
# (adapted from DIFUSCO, difusco/utils/cython_merge/setup.py, MIT License)
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(ext_modules=cythonize(
    [Extension("cython_merge", ["cython_merge.pyx"], include_dirs=[numpy.get_include()])],
    language_level=3))
