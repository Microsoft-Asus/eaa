from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy as np


setup(
	cmdclass = {'build_ext': build_ext},
	ext_modules = [Extension("definitions", ["definitions.pyx"],
				libraries=cython_gsl.get_libraries() + ["gmp", "mpfr"],
				library_dirs=[cython_gsl.get_library_dir()],
				include_dirs=[np.get_include(), cython_gsl.get_include()],
				)]
)

setup(
	include_dirs = [cython_gsl.get_include()],
	cmdclass = {'build_ext': build_ext},
	ext_modules = [Extension("froutines", ["froutines.pyx"],
				libraries=cython_gsl.get_libraries() + ["gmp", "mpfr"],
				library_dirs=[cython_gsl.get_library_dir()],
				include_dirs=[np.get_include(), cython_gsl.get_include()],
				)]
)

setup(
	cmdclass = {'build_ext': build_ext},
	ext_modules = [Extension("approx_routines", ["approx_routines.pyx"],
				libraries=cython_gsl.get_libraries() + ["gmp", "mpfr"],
				library_dirs=[cython_gsl.get_library_dir()],
				include_dirs=[np.get_include(), cython_gsl.get_include()],
				)]
)

setup(
	cmdclass = {'build_ext': build_ext},
	ext_modules = [Extension("misc", ["misc.pyx"],
				libraries=cython_gsl.get_libraries(),
				library_dirs=[cython_gsl.get_library_dir()],
				include_dirs=[np.get_include(), cython_gsl.get_include()])
				])


setup(
	cmdclass = {'build_ext': build_ext},
	ext_modules = [Extension("eaamodel", ["eaamodel.pyx"],
				libraries=cython_gsl.get_libraries() + ["gmp", "mpfr"],
				library_dirs=[cython_gsl.get_library_dir()],
				include_dirs=[np.get_include(), cython_gsl.get_include()],
				)]
)