#!/usr/bin/env python3
"""
Minimum Barrier Distance (MBD) Segmentation
-----------------------------------------
A high-performance implementation of seeded image segmentation using the Minimum
Barrier Distance algorithm. This implementation provides an optimized C++ core for
efficient segmentation.

The Minimum Barrier Distance measures the difficulty of reaching a pixel from seed
points by considering the range of intensity values along the path. For a path π,
the barrier cost is:
    Φ(π) = max(π) - min(π)

The minimum barrier distance from a seed set S to a pixel c is:
    Φ(c,S) = min_{π ∈ Π} Φ(π)
where Π is the set of all paths from S to c.

Usage:
-----
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

Reference:
@article{Strand_2014,
  title={The Minimum Barrier Distance},
  author={Strand, Robin and Malmberg, Filip and Saha, Punam K.},
  year={2014}
}

License: MIT License
Copyright (c) 2025 Mvzvrt
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import mbd_core  # C++ extension


def load_image_grayscale(path: str) -> np.ndarray:
    """
    Load an image and convert it to normalized grayscale format.
    
    Parameters:
    ----------
    path : str
        Path to the input image file
        
    Returns:
    -------
    np.ndarray
        Grayscale image as float32 array normalized to [0,1] range
    """
    img = Image.open(path)
    if img.mode not in ("L", "I;16", "I", "F"):
        img = img.convert("L")
    arr = np.asarray(img)
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32)
        vmin, vmax = float(arr.min()), float(arr.max())
        arr = np.zeros_like(arr, dtype=np.float32) if vmax <= vmin else (arr - vmin) / (vmax - vmin)
    return arr


def segment_image(image: np.ndarray, seeds: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """
    Perform image segmentation using Minimum Barrier Distance.

    Parameters:
    ----------
    image : np.ndarray
        Input grayscale image as float32 array in range [0,1]
        Shape: [height, width]
        
    seeds : np.ndarray
        Seed mask as int32 array with labels:
        - 0: unlabeled regions
        - 1: background
        - 2+: object segments
        Must have same shape as image.
        
    connectivity : int, optional
        Neighborhood connectivity, either 4 or 8
        Default: 4

    Returns:
    -------
    np.ndarray
        Label map where each pixel is assigned to the closest seed point
        according to the minimum barrier distance.
        Shape: same as input image
        dtype: int32
    """
    # Input validation
    if image.shape != seeds.shape:
        raise ValueError(f"Image and seeds must have same shape, got {image.shape} vs {seeds.shape}")
    if connectivity not in (4, 8):
        raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")
    
    # Ensure proper data types
    image = np.ascontiguousarray(image, dtype=np.float32)
    seeds = np.ascontiguousarray(seeds, dtype=np.int32)
    
    # Normalize image to [0,1] if needed
    if image.min() < 0 or image.max() > 1:
        image = (image - image.min()) / (image.max() - image.min())
    
    # Run MBD propagation (labels, distances, pops count)
    labels, _, _ = mbd_core.run_mbd_label_propagation(image, seeds, connectivity)
    return labels


def process_image_file(image_path: str, seeds: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """
    Load an image file and perform segmentation using Minimum Barrier Distance.

    Parameters:
    ----------
    image_path : str
        Path to the input image file
        
    seeds : np.ndarray
        Seed mask as int32 array with labels:
        - 0: unlabeled regions
        - 1: background
        - 2+: object segments
        
    connectivity : int, optional
        Neighborhood connectivity, either 4 or 8
        Default: 4

    Returns:
    -------
    np.ndarray
        Label map from segmentation
    """
    # Load and convert image to grayscale
    img = Image.open(image_path)
    if img.mode not in ("L", "I;16", "I", "F"):
        img = img.convert("L")
    
    # Convert to normalized float32
    arr = np.asarray(img)
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32)
        vmin, vmax = float(arr.min()), float(arr.max())
        arr = np.zeros_like(arr, dtype=np.float32) if vmax <= vmin else (arr - vmin) / (vmax - vmin)
    
    return segment_image(arr, seeds, connectivity)

# Make the API functions available at package level
__all__ = ['segment_image', 'process_image_file']
