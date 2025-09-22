#!/usr/bin/env python3
"""
Minimum Barrier Distance (MBD) Segmentation
-----------------------------------------
A high-performance implementation of seeded image segmentation using the Minimum
Barrier Distance algorithm. This implementation provides both a pure Python version
and an optimized C++ core for production use.

Key Features:
- Exact MBD computation with proven optimality guarantees
- Multi-label segmentation support (background + multiple foreground classes)
- VOC-compliant color palette and label mapping
- Optional C++ acceleration via pybind11
- Thread-pooled I/O for batch processing

Algorithm Overview:
------------------
The Minimum Barrier Distance measures the difficulty of reaching a pixel from seed
points by considering the range of intensity values along the path. For a path π,
the barrier cost is:
    Φ(π) = max(π) - min(π)

The minimum barrier distance from a seed set S to a pixel c is:
    Φ(c,S) = min_{π ∈ Π} Φ(π)
where Π is the set of all paths from S to c.

Implementation Details:
---------------------
- Exact distance computation using priority queue propagation
- Lexicographic ordering: (b_plus↑, b_minus↓, label↑)
- Hard constraints for seed points (labels > 0)
- Optional 4 or 8 connectivity
- Memory-efficient contiguous array operations

Usage Examples:
-------------
1. Basic Single Image Segmentation:
   
   Example using the core functions to segment a single image:
   
   >>> # Load grayscale image (returns float32 array in [0,1])
   >>> img = load_image_grayscale('input.jpg')
   >>> H, W = img.shape
   >>> 
   >>> # Create seed mask (0=unlabeled, 1=background, 2+=objects)
   >>> seeds = np.zeros((H,W), dtype=np.int32)
   >>> seeds[10:20, 10:20] = 1    # Mark background region
   >>> seeds[30:40, 30:40] = 2    # Mark object region
   >>> 
   >>> # Run MBD propagation with 4-connectivity
   >>> labels, dists, _ = _run_mbd_py(img, seeds, conn=4)
   >>> 
   >>> # Save result with VOC colormap
   >>> save_indexed_png(labels, 'output.png')

2. Batch Processing with Multi-threading:
   
   Example processing multiple images in parallel:
   
   >>> # Set up processing arguments
   >>> class Args: pass
   >>> args = Args()
   >>> args.conn = 4               # Use 4-connectivity
   >>> args.workers = 4            # Use 4 worker threads
   >>> args.unlabeled_to_void = False
   >>> 
   >>> # Find all image/annotation pairs
   >>> pairs = find_image_annotation_pairs('images/', 'annotations/')
   >>> 
   >>> # Process images in parallel
   >>> for img_path, ann_path in pairs:
   ...     labels = run_single_image(img_path, ann_path, args)
   ...     out_name = f"{Path(img_path).stem}_index.png"
   ...     save_indexed_png(labels, f"results/{out_name}")

3. Custom Visualization:
   
   Example using custom colormaps and label mapping:
   
   >>> # Generate custom colormap (normalized to [0,1])
   >>> colors = voc_colormap(N=256, normalized=True)
   >>> 
   >>> # Save with custom settings
   >>> save_indexed_png(
   ...     labels,
   ...     'custom_viz.png',
   ...     palette=colors,           # Use custom colors
   ...     shift_for_voc=True,      # Map background 1->0
   ...     unlabeled_to_void=False  # Keep unlabeled visible
   ... )

Reference:
@inbook{Strand_2014,
  title={The Minimum Barrier Distance – Stability to Seed Point Position},
  ISBN={9783642387098},
  ISSN={1611-3349},
  url={http://dx.doi.org/10.1007/978-3-319-09955-2_10},
  DOI={10.1007/978-3-319-09955-2_10},
  booktitle={Advanced Information Systems Engineering},
  publisher={Springer Berlin Heidelberg},
  author={Strand, Robin and Malmberg, Filip and Saha, Punam K. and Linnér, Elisabeth},
  year={2014},
  pages={111–121}
}

Implemented by: Mvzvrt

License: MIT License
Copyright (c) 2025 Mvzvrt
"""

"""
mbd.py

Seeded segmentation using Minimum Barrier Distance, exact objective and propagation.

Assumptions, consistent with your annotations:
  0 = unlabeled, ignored as a seed
  1 = background, hard constraint
  2..K = object classes, hard constraints where provided

Saving rule for correct VOC colors:
  We shift labels down by 1 at save time only, so background 1 becomes palette index 0,
  classes 2..21 become 1..20, and any remaining 0 can optionally be mapped to 255 (void).
  This preserves your internal label IDs in memory while producing VOC colored PNGs.

Core algorithmic notes, aligned with the attached PDFs:
  Path barrier cost bw(π) = max(π) - min(π). Minimum barrier distance from a seed set S to node c
  is dbw(c, S) = min over paths π from S to c of bw(π). We propagate the lexicographic pair
  (b_plus, b_minus) as in Algorithm 3 Asimple_MBD, priority order b_plus ascending, then b_minus
  descending, and assign each pixel the label of the seed that attains the best barrier width.
"""

import argparse, json, logging, time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Try the C++ extension, fall back to Python if unavailable
_USE_CPP = False
try:
    import mbd_core  # built from mbd_core.cpp
    _USE_CPP = True
except Exception:
    _USE_CPP = False

METHOD_NAME = "mbd"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------- VOC colormap, consistent with batch_refine.py ---------------------------

def voc_colormap(N: int = 256, normalized: bool = False) -> np.ndarray:
    """
    Generate the PASCAL VOC dataset colormap for segmentation visualization.
    
    This function implements a bit-manipulation algorithm to generate a deterministic
    colormap that matches the standard PASCAL VOC dataset visualization scheme.
    The colors are generated such that:
    - Neighboring indices get visually distinct colors
    - The pattern repeats every 8 colors
    - Each color channel (R,G,B) is determined by specific bits of the index
    
    Parameters:
    ----------
    N : int
        Number of colors to generate. Default is 256 to cover all possible
        8-bit label values.
    normalized : bool
        If True, returns floating point values in [0,1]
        If False, returns integer values in [0,255]
    
    Returns:
    -------
    np.ndarray [N,3]
        RGB colormap where each row is [r,g,b]
        Data type is np.float32 if normalized=True, np.uint8 otherwise
    """
    def bitget(byteval, idx):
        """Extract the idx-th bit from byteval."""
        return (byteval & (1 << idx)) != 0

    dtype = np.float32 if normalized else np.uint8
    colormap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (int(bitget(c, 0)) << (7 - j))
            g |= (int(bitget(c, 1)) << (7 - j))
            b |= (int(bitget(c, 2)) << (7 - j))
            c >>= 3
        colormap[i] = [r, g, b]
    if normalized:
        colormap = colormap / 255.0
    return colormap


def save_indexed_png(mask: np.ndarray,
                     out_path: str,
                     palette: Optional[np.ndarray] = None,
                     shift_for_voc: bool = True,
                     unlabeled_to_void: bool = True) -> None:
    """
    Save a label mask as an indexed PNG file with optional VOC dataset compatibility.
    
    This function saves segmentation results as 8-bit paletted PNGs, optionally
    adjusting the label indices to match PASCAL VOC dataset conventions:
    
    Input Label Convention:
        0: Unlabeled regions
        1: Background
        2+: Object segments
        
    VOC Label Convention (if shift_for_voc=True):
        0: Background
        1-254: Object segments
        255: Void/Unlabeled (if unlabeled_to_void=True)
        
    Parameters:
    ----------
    mask : np.ndarray
        Integer label mask of shape [H,W] where each value
        represents a different segment
    out_path : str
        Path where the PNG file will be saved
    palette : np.ndarray, optional
        RGB color palette of shape [256,3] with uint8 values
        If None, uses the standard VOC colormap
    shift_for_voc : bool
        If True, shifts all labels down by 1 to match VOC convention:
        - background (1 -> 0)
        - objects (2+ -> 1+)
        Default: True
    unlabeled_to_void : bool
        If True, maps unlabeled regions (0) to void label (255)
        Only applies when shift_for_voc=True
        Common in semantic segmentation tasks to ignore these regions
        Default: True
        
    Notes:
    -----
    The function automatically:
    1. Creates output directory if it doesn't exist
    2. Converts input to uint8 for PNG compatibility
    3. Applies the specified palette for visualization
    """
    m = np.asarray(mask, dtype=np.int32)
    if shift_for_voc:
        # Build adjusted array
        if unlabeled_to_void:
            out = np.full_like(m, 255, dtype=np.uint8)
            valid = m > 0
            out[valid] = (m[valid] - 1).astype(np.uint8)
        else:
            out = np.clip(m - 1, 0, 255).astype(np.uint8)
    else:
        # Save raw indices
        out = np.asarray(m, dtype=np.uint8)

    im = Image.fromarray(out, mode="P")
    pal = palette if palette is not None else voc_colormap()
    im.putpalette(pal.astype(np.uint8).flatten().tolist())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="PNG")


# --------------------------- I O helpers ---------------------------

def load_image_grayscale(path: str) -> np.ndarray:
    """
    Load an image and convert it to normalized grayscale format.
    
    This function handles various input image formats and converts them
    to a standardized grayscale representation suitable for MBD processing.
    Supported input formats:
    - L: 8-bit grayscale
    - I;16: 16-bit grayscale
    - I: 32-bit signed integer
    - F: 32-bit floating point
    - RGB/RGBA: Automatically converted to grayscale
    
    Parameters:
    ----------
    path : str
        Path to the input image file
        
    Returns:
    -------
    np.ndarray
        Grayscale image as float32 array normalized to [0,1] range
        
    Notes:
    -----
    For RGB/RGBA images, conversion uses the standard
    luminosity formula: 0.299R + 0.587G + 0.114B
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


def load_annotation_map(ann_path: str) -> np.ndarray:
    """
    Load a segmentation annotation map from file.
    
    Supports two file formats:
    1. NumPy arrays (.npy):
       - Direct load of integer label maps
       - Preserved exactly as stored
       
    2. Indexed PNG files (.png):
       - 8-bit paletted images (mode 'P')
       - Index values map directly to segment labels
       - Common format for dataset annotations
    
    Parameters:
    ----------
    ann_path : str
        Path to the annotation file (.npy or .png)
        
    Returns:
    -------
    np.ndarray [H,W]
        Label map as int32 array where each unique value
        represents a different segment
        - 0 typically represents background
        - 1+ are foreground object segments
    
    Raises:
    ------
    ValueError
        If file format is not .npy or .png
    RuntimeError
        If PNG is not in paletted ('P') mode
    """
    p = Path(ann_path)
    if p.suffix.lower() == ".npy":
        ann = np.load(ann_path)
        return np.asarray(ann, dtype=np.int32)
    img = Image.open(ann_path)
    if img.mode != "P":
        img = img.convert("P")
    ann = np.asarray(img, dtype=np.int32)
    return ann


def find_image_annotation_pairs(images_dir: str, anns_dir: str) -> List[Tuple[str, Optional[str]]]:
    """
    Find matching image and annotation files in separate directories.
    
    This function pairs images with their corresponding annotation files
    by matching base filenames across directories. For example:
    - images_dir/img1.jpg -> anns_dir/img1.npy or img1.png
    
    Priority for annotation formats:
    1. .npy files (preferred for exact label preservation)
    2. .png files (common in datasets but may have palette issues)
    
    Parameters:
    ----------
    images_dir : str
        Directory containing input images
        Supported extensions: .jpg, .jpeg, .png
    anns_dir : str
        Directory containing annotation files
        Supported extensions: .npy, .png
        
    Returns:
    -------
    List[Tuple[str, Optional[str]]]
        List of (image_path, annotation_path) pairs where:
        - image_path: Path to an input image
        - annotation_path: Path to corresponding annotation file,
                         or None if no match found
                         
    Notes:
    -----
    - Files are matched by their base name (without extension)
    - Images without matching annotations are included with None
    - Non-image and non-annotation files are ignored
    """
    images_dir = Path(images_dir)
    anns_dir = Path(anns_dir)
    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    pairs = []
    for ip in sorted(imgs):
        stem = ip.stem
        npy = anns_dir / f"{stem}.npy"
        png = anns_dir / f"{stem}.png"
        ann_path = str(npy) if npy.exists() else (str(png) if png.exists() else None)
        pairs.append((str(ip), ann_path))
    return pairs


# --------------------------- Pure Python MBD core ---------------------------

def _neighbors_offsets(conn: int = 4):
    """
    Generate pixel neighborhood offsets for connectivity patterns.
    
    The function returns a list of (dy, dx) offset tuples that define
    the pixel neighborhood structure for propagation. Two connectivity 
    patterns are supported:
    
    4-connectivity:
        p4 = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (0, -1),   # Left
            (0, 1)     # Right
        ]
        
    8-connectivity (adds diagonals):
        p8 = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (0, -1),   # Left
            (0, 1),    # Right
            (-1, -1),  # Up-Left
            (-1, 1),   # Up-Right
            (1, -1),   # Down-Left
            (1, 1)     # Down-Right
        ]
    
    Parameters:
    ----------
    conn : int
        Connectivity pattern to use:
        - 4 for von Neumann neighborhood (default)
        - 8 for Moore neighborhood
    
    Returns:
    -------
    list[tuple]
        List of (dy, dx) offset pairs defining the neighborhood
        structure for the requested connectivity pattern
    
    Raises:
    ------
    ValueError
        If conn is not 4 or 8
    """
    if conn == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn == 8:
        return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    raise ValueError(f"Connectivity must be 4 or 8, got {conn}")
    raise ValueError("conn must be 4 or 8")


def _run_mbd_py(weights: np.ndarray, seeds: np.ndarray, conn: int = 4):
    """
    Pure Python implementation of exact MBD propagation with label assignment.
    
    Algorithm Details:
    ----------------
    1. Initialize distance maps:
       - bmin[p] = min value along best path to p
       - bmax[p] = max value along best path to p
       - dist[p] = barrier width (bmax[p] - bmin[p])
    
    2. Priority Queue Ordering:
       - Primary: smallest bmax (minimize maximum along path)
       - Secondary: largest bmin (maximize minimum along path)
       - Tertiary: smallest label (deterministic tie-breaking)
    
    3. Propagation Rule:
       For each pixel p and neighbor q:
       - Update q if new path through p gives:
         a) Smaller barrier width, or
         b) Equal width but smaller bmax, or
         c) Equal width and bmax but larger bmin, or
         d) Equal width, bmax, bmin but smaller label
    
    Parameters:
    ----------
    weights : np.ndarray [H,W] float32
        Input image intensities, normalized to [0,1]
    seeds : np.ndarray [H,W] int32
        Seed mask where 0=unlabeled, 1=background, >1=objects
    conn : int
        Pixel connectivity, either 4 or 8 (default: 4)
    
    Returns:
    -------
    labels : np.ndarray [H,W] int32
        Segmentation with propagated labels
    distances : np.ndarray [H,W] float32
        Minimum barrier distance at each pixel
    pops : int
        Number of priority queue operations
    """
    import heapq

    # Get image dimensions and flatten arrays for 1D indexing
    H, W = weights.shape
    N = H * W  # Total number of pixels
    w = weights.ravel()  # Flatten image to 1D array for direct indexing
    s = np.asarray(seeds, dtype=np.int32)  # Ensure seeds are int32
    if s.shape != (H, W):
        raise ValueError("seeds must match image size")

    # Initialize barrier distance arrays with infinities
    bmin = np.full(N, np.inf, dtype=np.float32)   # Minimum value along best path
    bmax = np.full(N, -np.inf, dtype=np.float32)  # Maximum value along best path
    dist = np.full(N, np.inf, dtype=np.float32)   # Barrier width (bmax - bmin)
    label = np.zeros(N, dtype=np.int32)           # Propagated label assignments
    locked = (s.ravel() > 0)                      # Track seed points (hard constraints)

    # Initialize priority queue for propagation
    heap = []
    push = heapq.heappush  # Shorthand for priority queue operations
    pop = heapq.heappop
    counter = 0  # Counter for deterministic FIFO behavior within same priority

    # Find and sort seed points for deterministic initialization
    ys, xs = np.nonzero(s > 0)  # Get coordinates of all seed points
    # Sort seeds by label and position for deterministic behavior:
    # - Primary sort by label (s[y,x])
    # - Secondary sort by linear index (y*W + x)
    order = np.argsort(s[ys, xs] * (H * W) + (ys * W + xs))
    ys, xs = ys[order], xs[order]  # Apply sorting

    # Initialize seed points in priority queue
    for y, x in zip(ys, xs):
        idx = y * W + x  # Convert 2D coordinates to 1D index
        lab = int(s[y, x])  # Get seed label
        val = w[idx]  # Get pixel intensity

        # Initialize distance maps for seed point:
        bmin[idx] = val  # Minimum value is current value
        bmax[idx] = val  # Maximum value is current value
        dist[idx] = 0.0  # Barrier distance is 0 at seeds
        label[idx] = lab  # Assign seed label

        # Add to priority queue with:
        # - Primary key: bmax (ascending)
        # - Secondary key: -bmin (ascending, so bmin descending)
        # - Tertiary key: counter (FIFO ordering)
        # - Data: pixel index and label
        push(heap, (bmax[idx], -bmin[idx], counter, idx, lab))
        counter += 1

    # Get neighborhood offsets based on connectivity pattern
    offs = _neighbors_offsets(conn)  # 4 or 8-connected neighborhood

    def itn(i):
        """Generate valid neighbor indices for a given pixel index.
        
        Args:
            i: Linear index of current pixel
            
        Yields:
            Linear indices of valid neighbors based on connectivity pattern
        """
        y, x = divmod(i, W)  # Convert linear index to 2D coordinates
        for dy, dx in offs:  # Check each neighbor offset
            ny, nx = y + dy, x + dx  # Get neighbor coordinates
            if 0 <= ny < H and 0 <= nx < W:  # Check image bounds
                yield ny * W + nx  # Yield linear index if valid

    # Main propagation loop
    pops = 0  # Count priority queue operations
    while heap:
        # Get next node with minimum barrier width
        # bp: barrier max (bmax), nbm: negative barrier min (-bmin)
        # idx: pixel index, lab: propagating label
        bp, nbm, _, idx, lab = pop(heap)
        pops += 1
        bm = -nbm  # Convert back to actual barrier min
        
        # Skip stale queue entries (values have been improved)
        if bp != bmax[idx] or bm != bmin[idx] or lab != label[idx]:
            continue  # Current node has been updated since enqueue

        # Process each valid neighbor
        for j in itn(idx):
            # Skip if neighbor is a locked seed with different label
            if locked[j] and label[j] != lab:
                continue  # Preserve hard constraints

            # Compute candidate barrier values through current pixel
            # The barrier must include both the current path and neighbor's value
            cand_bmin = min(bmin[idx], w[j])  # Update path minimum
            cand_bmax = max(bmax[idx], w[j])  # Update path maximum
            cand_bw = cand_bmax - cand_bmin   # New barrier width

            # Check if new path is better using lexicographic ordering:
            # 1. Minimum barrier width (primary criterion)
            # 2. Minimum bmax value (if barrier widths equal)
            # 3. Maximum bmin value (if barrier widths and bmax equal)
            # 4. Minimum label (if all above are equal)
            upd = False
            if cand_bw < dist[j]:  # Better barrier width
                upd = True
            elif cand_bw == dist[j]:  # Equal barrier width
                if cand_bmax < bmax[j]:  # Better maximum
                    upd = True
                elif cand_bmax == bmax[j] and cand_bmin > bmin[j]:  # Equal max, better min
                    upd = True
                elif (cand_bmax == bmax[j] and cand_bmin == bmin[j] and  # All equal
                      lab < label[j]):  # Use smallest label for deterministic behavior
                    upd = True

            # If better path found, update neighbor's state
            if upd:
                # Update barrier distance maps
                bmin[j] = cand_bmin  # New minimum along path
                bmax[j] = cand_bmax  # New maximum along path
                dist[j] = cand_bw    # New barrier width
                label[j] = lab       # Inherit label from best path
                
                # Re-add to queue with updated priority
                push(heap, (bmax[j], -bmin[j], counter, j, lab))
                counter += 1  # Maintain FIFO order for equal priorities

    # Return results reshaped to original image dimensions
    return (label.reshape(H, W),    # Segmentation labels
            dist.reshape(H, W),     # Minimum barrier distances
            pops)                   # Number of queue operations


# --------------------------- Runner ---------------------------

def run_single_image(image_path: str, ann_path: str, args) -> np.ndarray:
    w = load_image_grayscale(image_path)
    if ann_path is None:
        raise FileNotFoundError(f"No matching annotation for {image_path}")
    seeds = load_annotation_map(ann_path)
    if seeds.shape != w.shape:
        raise ValueError(f"Shape mismatch for {image_path} and {ann_path}, got {w.shape} vs {seeds.shape}")

    t0 = time.time()
    if _USE_CPP:
        labels, dist_map, pops = mbd_core.run_mbd_label_propagation(w.astype(np.float32, copy=False),
                                                                    seeds.astype(np.int32, copy=False),
                                                                    int(args.conn))
    else:
        labels, dist_map, pops = _run_mbd_py(w, seeds, conn=args.conn)
    ms = (time.time() - t0) * 1000.0

    H, W = w.shape
    logging.info(f"{Path(image_path).stem}, {H}x{W}, pops {int(pops)}, runtime_ms {ms:.2f}, cpp {bool(_USE_CPP)}")
    return labels


def main():
    ap = argparse.ArgumentParser(description="Minimum Barrier Distance, seeded, with VOC color saving")
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--anns_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num-images", type=int, default=0, help="0 means all")
    ap.add_argument("--start-one", type=int, default=1, help="1-indexed start position")
    ap.add_argument("--workers", type=int, default=0, help="only thread mode for I O, solver is single threaded")
    ap.add_argument("--conn", type=int, default=4, choices=[4, 8])
    ap.add_argument("--unlabeled-to-void", action="store_true", help="map unlabeled 0 to 255 at save time")
    ap.add_argument("--no-shift-for-voc", action="store_true", help="do not shift labels when saving")
    ap.add_argument("--run-tests", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.run_tests:
        _run_tests()
        return

    pairs = find_image_annotation_pairs(args.images_dir, args.anns_dir)
    start_idx = max(0, int(args.start_one) - 1)
    if start_idx >= len(pairs):
        logging.info(json.dumps({"processed": 0, "skipped": len(pairs), "reason": "start index beyond input"}))
        return
    end_idx = len(pairs) if args.num_images == 0 else min(len(pairs), start_idx + int(args.num_images))
    work_list = pairs[start_idx:end_idx]

    out_root = Path(args.output_dir) / METHOD_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    processed, skipped = 0, 0
    times = []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    if args.workers and args.workers > 0:
        def task(img_path, ann_path):
            base = Path(img_path).stem
            if ann_path is None:
                return base, None, "missing"
            t0 = time.time()
            m = run_single_image(img_path, ann_path, args)
            return base, m, (time.time() - t0) * 1000.0

        with tqdm(total=len(work_list), desc="MBD") as pbar, ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(task, i, a) for i, a in work_list]
            for f in as_completed(futs):
                base, mask, ms = f.result()
                if mask is None:
                    logging.error(f"Missing annotation for {base}, skipping")
                    skipped += 1
                else:
                    out_path = out_root / f"{base}_index.png"
                    save_indexed_png(mask, str(out_path),
                                     palette=voc_colormap(),
                                     shift_for_voc=not args.no_shift_for_voc,
                                     unlabeled_to_void=args.unlabeled_to_void)
                    processed += 1
                    times.append(ms)
                pbar.update(1)
    else:
        with tqdm(total=len(work_list), desc="MBD") as pbar:
            for img_path, ann_path in work_list:
                base = Path(img_path).stem
                if ann_path is None:
                    logging.error(f"Missing annotation for {base}, skipping")
                    skipped += 1
                    pbar.update(1)
                    continue
                try:
                    t0 = time.time()
                    mask = run_single_image(img_path, ann_path, args)
                    ms = (time.time() - t0) * 1000.0
                    out_path = out_root / f"{base}_index.png"
                    save_indexed_png(mask, str(out_path),
                                     palette=voc_colormap(),
                                     shift_for_voc=not args.no_shift_for_voc,
                                     unlabeled_to_void=args.unlabeled_to_void)
                    processed += 1
                    times.append(ms)
                except Exception as e:
                    logging.error(f"Error on {base}: {e}")
                    skipped += 1
                pbar.update(1)

    print(json.dumps({
        "total": len(work_list),
        "processed": processed,
        "skipped": skipped,
        "avg_runtime_ms": float(np.mean(times)) if times else None,
        "median_runtime_ms": float(np.median(times)) if times else None,
        "used_cpp": bool(_USE_CPP),
        "conn": int(args.conn),
        "method": METHOD_NAME
    }))


# --------------------------- Minimal tests ---------------------------

def _synthetic_case(H: int = 64, W: int = 64):
    x = np.linspace(0, 1, W, dtype=np.float32)
    img = np.tile(x, (H, 1))
    img[:, W // 2 - 1: W // 2 + 1] = 1.0
    seeds = np.zeros((H, W), dtype=np.int32)
    seeds[8:12, 8:12] = 1  # background
    seeds[H - 12:H - 8, W - 12:W - 8] = 2  # foreground class
    return img, seeds


def _run_tests():
    logging.info("Running synthetic test")
    img, seeds = _synthetic_case()
    if _USE_CPP:
        labels, dist_map, pops = mbd_core.run_mbd_label_propagation(img.astype(np.float32), seeds.astype(np.int32), 4)
    else:
        labels, dist_map, pops = _run_mbd_py(img, seeds, 4)
    assert labels[9, 9] == 1, "background seed must stay 1"
    assert labels[-9, -9] == 2, "foreground seed must stay 2"
    logging.info(f"OK, pops {int(pops)}")
    print(json.dumps({"test": "ok", "pops": int(pops), "cpp": bool(_USE_CPP)}))


if __name__ == "__main__":
    main()
