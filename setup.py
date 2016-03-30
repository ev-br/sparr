from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
import sys
import numpy


# are we compiling in the debugging mode? 
try:
    if os.environ['SPARR_DEBUG']:
        extra_compile_args = ['-UNDEBUG']
except KeyError:
    extra_compile_args = []

def process_tempita_pyx():
    # taken from scipy's cythonize.py
    fromfile = os.path.join("sparr", "sp_map.pyx.in")
    from Cython import Tempita as tempita
    from_filename = tempita.Template.from_filename
    template = from_filename(fromfile, encoding=sys.getdefaultencoding())
    pyxcontent = template.substitute()
    assert fromfile.endswith('.pyx.in')
    pyxfile = fromfile[:-len('.pyx.in')] + '.pyx'
    with open(pyxfile, "w") as f:
        f.write(pyxcontent)

process_tempita_pyx()

ext = Extension("sparr._sp_map", [os.path.join("sparr", "sp_map.pyx")],
                include_dirs = [numpy.get_include()],
                extra_compile_args = extra_compile_args,
                language="c++",)
                
setup(name="sparr",
      packages=["sparr", "sparr.tests"],
      version="0.0.1",
      license="BSD",

      ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})
