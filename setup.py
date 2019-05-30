# setup.py
# only if building in place: ``python setup.py build_ext --inplace``

from setuptools import setup
from numpy.distutils.core import setup, Extension

setup(
    name='akima',
    version='1.0.0',
    description='Akima spline interpolation (and derivatives)',
    author='NREL WISDEM Team',
    package_dir={'': 'src'},
    py_modules=['akima'],
    license='Apache License, Version 2.0',
    ext_modules=[Extension('_akima', ['src/akima.f90'], extra_f90_compile_args=['-O3','-fPIC','-shared'], extra_link_args=['-shared'])]
)
