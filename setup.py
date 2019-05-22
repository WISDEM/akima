# setup.py
# only if building in place: ``python setup.py build_ext --inplace``

from setuptools import setup
from numpy.distutils.core import setup, Extension
import os, platform

if platform.system() == 'Windows':
    # Note: must use mingw compiler on windows or a Visual C++ compiler version that supports std=c++11
    arglist = ['-O3','-std=gnu++11','-fPIC']
else:
    arglist = ['-O3','-std=c++11','-fPIC']

setup(
    name='akima',
    version='1.0.0',
    description='Akima spline interpolation (and derivatives)',
    author='NREL WISDEM Team',
    author_email='systems.engineering@nrel.gov',
    package_dir={'': 'src'},
    py_modules=['akima'],
    license='Apache License, Version 2.0',
    ext_modules=[Extension('_akima', sources=[os.path.join('src','akima.cpp')], extra_compile_args=arglist, include_dirs=[os.path.join('src','include')])]
)
