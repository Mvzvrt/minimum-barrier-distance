"""
Minimum Barrier Distance (MBD) Segmentation
-----------------------------------------
A high-performance implementation of seeded image segmentation using the Minimum
Barrier Distance algorithm.

Example:
    >>> import numpy as np
    >>> from mbd import segment_image
    >>> 
    >>> # Load your grayscale image as a numpy array (float32 in [0,1])
    >>> image = ...  # Your image loading code here
    >>> 
    >>> # Create seeds array (0=unlabeled, 1=background, 2+=objects)
    >>> seeds = np.zeros(image.shape, dtype=np.int32)
    >>> seeds[10:20, 10:20] = 1  # Background region
    >>> seeds[30:40, 30:40] = 2  # Object region
    >>> 
    >>> # Run segmentation
    >>> labels = segment_image(image, seeds)
"""

from .core import segment_image, process_image_file

__version__ = "0.1.0"
__all__ = ["segment_image", "process_image_file"]