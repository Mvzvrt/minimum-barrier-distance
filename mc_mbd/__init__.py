"""
mc_mbd package shim for Multiclass Minimum Barrier Distance

This package provides the same public API as the old `mbd` package but
uses the import name `mc_mbd` so it matches the distribution name.
"""

from .core import segment_image, process_image_file

__version__ = "0.1.3"
__all__ = ["segment_image", "process_image_file"]
