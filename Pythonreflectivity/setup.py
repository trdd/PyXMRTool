#import numpy
#
#
#from distutils.core import setup
#from distutils.extension import Extension
#
#from Cython.Build import cythonize
#setup(ext_modules = cythonize(Extension(
#           "*",                                # the extension name
#           sources=["*.pyx"],
#           language="c++",
#           include_dirs=[numpy.get_include()]
#      )))
#

import numpy

##try:
##    from setuptools import setup
##    from setuptools import Extension
#except ImportError:
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
setup(ext_modules = cythonize(Extension(
           "*",                                # the extension name
           sources=["*.pyx"],
           language="c++",
           extra_compile_args = ["-O3"],
           include_dirs=[numpy.get_include()]
      )))


