"""
setup file for mclController
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("MicroDrive_v1.pyx")
)

#setup(name='mclController_customized_module',
#      packages=['mclController_customized_module'],
#     )