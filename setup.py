"""
Setup script for building the Minimum Barrier Distance C++ extension
"""
from setuptools import setup, Extension, find_packages
import sys
import sysconfig
import os
import platform

# Check Python version
if sys.version_info < (3, 9):
    sys.exit('Python >= 3.9 is required')

# Check for build dependencies
def check_dependency(package, name=None):
    name = name or package
    try:
        return __import__(package)
    except ImportError as e:
        raise RuntimeError(
            f"{name} is required for building. "
            f"Please install it first: pip install {package}"
        ) from e

pybind11 = check_dependency("pybind11")
np = check_dependency("numpy")

# Configure platform-specific compilation flags
if sys.platform == "win32":
    cxx_flags = ["/O2", "/std:c++14", "/EHsc"]
else:
    cxx_flags = ["-O3", "-std=c++14", "-fvisibility=hidden"]
    # Add platform-specific optimizations
    if platform.machine() in ("x86_64", "AMD64"):
        cxx_flags.extend(["-march=native"])

ext_modules = [
    Extension(
        name="mbd_core",
        sources=["mbd_core.cpp"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
        ],
        language="c++",
        extra_compile_args=cxx_flags,
    )
]

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="mc_mbd",
    version="0.1.0",
    author="Mvzvrt",
    description="Multiclass Minimum Barrier Distance segmentation algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Include both the new package name and the legacy package name for compatibility
    packages=find_packages(),
    ext_modules=ext_modules,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
        "pillow>=8.0.0",  # Added PIL dependency
    ],
    project_urls={
        "Source": "https://github.com/Mvzvrt/minimum-barrier-distance"
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
