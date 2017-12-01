from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

exts = [
    Extension(
        "occupancy_grid",
        ["occupancy_grid.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['/openmp']
    )
]

setup(
    name = "occupancy_grid_display",
    ext_modules = cythonize(exts)
)
