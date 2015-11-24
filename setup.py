from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize(
           "sp_map.pyx",
           ### sources=[...],  # additional source file(s)
      ))
