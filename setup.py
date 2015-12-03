from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import sys
import numpy


fromfile = "sp_map.pyx.in"
# taken from scipy's cythonize.py
from Cython import Tempita as tempita
from_filename = tempita.Template.from_filename
template = from_filename(fromfile, encoding=sys.getdefaultencoding())
pyxcontent = template.substitute()
assert fromfile.endswith('.pyx.in')
pyxfile = fromfile[:-len('.pyx.in')] + '.pyx'
with open(pyxfile, "w") as f:
    f.write(pyxcontent)

ext = Extension("sp_map", ["sp_map.pyx"],
                include_dirs = [numpy.get_include()],
                language="c++",)
                
setup(ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})
