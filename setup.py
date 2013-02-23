from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext as build_pyx
import numpy

setup(name = 'prisoner_c', include_dirs=[numpy.get_include()], ext_modules=[Extension('prisoner_c', ['prisoner_c.pyx'])], cmdclass = { 'build_ext': build_pyx })