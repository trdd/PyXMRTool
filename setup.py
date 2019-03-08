import numpy
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

setup(
    name='PyXMRTool',
    version='0.9dev',
    packages=['PyXMRTool'],
    ext_modules = cythonize(Extension(
           "Pythonreflectivity/*",                                # the extension name
           sources=["Pythonreflectivity/*.pyx"],
           language="c++",
           extra_compile_args = ["-O3"],
           include_dirs=[numpy.get_include()]
      )),
    license='GNU Lesser General Public License v3.0',
    long_description=open('README.md').read(),
    package_data={'PyXMRTool': ['resources/ChantlerTables/*.cff','resources/ChantlerTables/*.pyt']},
    install_requires=['numpy','scipy','matplotlib','joblib','sklearn']
)
