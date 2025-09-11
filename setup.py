from setuptools import setup, Extension
import sys
import sysconfig
import os

# pybind11 is a build dependency
try:
    import pybind11
except Exception as e:
    raise RuntimeError("pybind11 must be installed before building. Try: pip install pybind11") from e

# numpy headers are also required for pybind11's numpy bindings
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("numpy must be installed before building. Try: pip install numpy") from e

cxx_flags = ["-O3"]
if sys.platform == "win32":
    cxx_flags = ["/O2"]

ext_modules = [
    Extension(
        name="mbd_core",
        sources=["mbd_core.cpp"],
        include_dirs=[pybind11.get_include(), np.get_include()],
        language="c++",
        extra_compile_args=cxx_flags,
    )
]

setup(
    name="mbd_core",
    version="0.1.0",
    description="C++ core for Minimum Barrier Distance",
    ext_modules=ext_modules,
)
