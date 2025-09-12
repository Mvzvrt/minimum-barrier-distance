#!/usr/bin/env python3
"""
Programmer Description:
-------------------------
This file, fh_mbd.py, is an experimental implementation of the Minimum Barrier Distance (MBD) segmentation algorithm that integrates seed densification using Felzenszwalb's superpixel segmentation.

Purpose:
- Enhance sparse scribble annotations by expanding them to superpixel regions based on area fraction and color similarity.
- Provide an experimental variant of the standard MBD pipeline (see minimum_barrier_distance.py) to investigate potential improvements in segmentation quality.

Note:
- This approach is experimental and may be less stable than the production version.
- More detailed documentation on the scribbles format and parameter tuning will be provided in future updates.

License: MIT License
Copyright (c) 2025 Mvzvrt

For further details, refer to the accompanying documentation or contact the maintainers.
"""

# pylint: disable=import-error

import argparse, json, logging, time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from tqdm import tqdm

# Optional C++ core
_USE_CPP = False
try:
    import mbd_core  # from mbd_core.cpp built via setup.py
    _USE_CPP = True
except Exception:
    _USE_CPP = False

METHOD_NAME = "minimum_barrier_distance"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------- VOC colormap and saving ---------------------------

def voc_colormap(N: int = 256, normalized: bool = False) -> np.ndarray:
    """PASCAL VOC colormap, same bit trick as common implementations."""
    def bitget(byteval, idx):
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
        colormap /= 255.0
    return colormap


def save_indexed_png(mask: np.ndarray,
                     out_path: str,
                     palette: Optional[np.ndarray] = None,
                     shift_for_voc: bool = True,
                     unlabeled_to_void: bool = False) -> None:
    """
    Save H x W integer mask as indexed PNG with VOC palette by default.

    If shift_for_voc:
      your labels: 0 unlabeled, 1 background, 2..K objects
      saved mask:  255 (void) for unlabeled if unlabeled_to_void,
                   else keep 0,
                   background -> 0, objects -> 1..K-1
    """
    m = np.asarray(mask, dtype=np.int32)
    if shift_for_voc:
        if unlabeled_to_void:
            out = np.full_like(m, 255, dtype=np.uint8)
            valid = m > 0
            out[valid] = (m[valid] - 1).astype(np.uint8)
        else:
            out = np.clip(m - 1, 0, 255).astype(np.uint8)
    else:
        out = np.asarray(m, dtype=np.uint8)

    im = Image.fromarray(out, mode="P")
    pal = palette if palette is not None else voc_colormap()
    im.putpalette(pal.astype(np.uint8).flatten().tolist())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="PNG")


# --------------------------- I O helpers ---------------------------

def load_image_rgb(path: str) -> np.ndarray:
    """Return H x W x 3 uint8 RGB."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_image_grayscale(path: str) -> np.ndarray:
    """Return float32 grayscale in [0, 1]."""
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
    """Load .npy or paletted .png as H x W int32."""
    p = Path(ann_path)
    if p.suffix.lower() == ".npy":
        return np.asarray(np.load(ann_path), dtype=np.int32)
    img = Image.open(ann_path)
    if img.mode != "P":
        img = img.convert("P")
    return np.asarray(img, dtype=np.int32)


def find_image_annotation_pairs(images_dir: str, anns_dir: str) -> List[Tuple[str, Optional[str]]]:
    """Pair images with annotations by basename. Prefer .npy, else .png, else None."""
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


# --------------------------- Seed densification (FH based) ---------------------------

# Using scikit-image for FH segmentation and LAB color
from skimage import segmentation, color  # type: ignore

def augment_anns_with_fh(
    image_rgb_u8: np.ndarray,
    anns: np.ndarray,
    *,
    fh_scale: int = 100,
    fh_sigma: float = 0.8,
    fh_min_size: int = 20,
    min_region_frac: float = 0.20,
    grow_color_thresh: float = 25.0
) -> np.ndarray:
    """
    Densify sparse scribbles using Felzenszwalb regions, then per-region rules.

    Steps
    1. Segment image into regions with felzenszwalb(image, scale, sigma, min_size).
    2. For each region:
       a) Count labeled pixels (>0). If dominant class covers at least
          min_region_frac of the region, claim the entire region.
       b) Else compute mean LAB color of region and mean LAB color of scribbled
          pixels of the dominant class within the region. If their distance is
          below grow_color_thresh, claim the region for that class.

    Implementation details
    - image is converted to float in [0,1] for skimage ops, but anns is kept as int32.
    - Boolean indexing bug fixed by building coordinates inside each region, then
      selecting only coords of pixels having the dominant class label.
    """
    H, W = anns.shape
    # Convert image to float [0,1] for skimage, and to LAB
    image_float = image_rgb_u8.astype(np.float32) / 255.0
    img_lab = color.rgb2lab(image_float)

    # Superpixel segmentation
    seg = segmentation.felzenszwalb(image_float, scale=fh_scale, sigma=fh_sigma, min_size=fh_min_size)
    user_anns_aug = anns.copy()

    region_ids = np.unique(seg)
    for rid in region_ids:
        region_mask = (seg == rid)
        region_size = int(region_mask.sum())
        if region_size == 0:
            continue

        # Scribbles present in this region
        region_scribbles = anns[region_mask]
        labeled = region_scribbles[region_scribbles > 0]
        if labeled.size == 0:
            continue

        classes, counts = np.unique(labeled, return_counts=True)
        dominant_class = int(classes[np.argmax(counts)])
        dominant_count = int(counts.max())

        # a) claim by fraction
        if dominant_count >= int(min_region_frac * region_size):
            user_anns_aug[region_mask] = dominant_class
            continue

        # b) color similarity growth, compute mean color of region and of scribbled pixels for the dominant class
        # build coordinates of region, then filter to dominant class coords
        region_coords = np.argwhere(region_mask)  # shape M x 2
        dom_mask = (region_scribbles == dominant_class)
        if dom_mask.any():
            dom_coords = region_coords[dom_mask]
            # handle degenerate safety
            if dom_coords.size > 0:
                mean_color_class = img_lab[dom_coords[:, 0], dom_coords[:, 1]].mean(axis=0)
            else:
                mean_color_class = img_lab[region_mask].mean(axis=0)
        else:
            mean_color_class = img_lab[region_mask].mean(axis=0)

        mean_color_region = img_lab[region_mask].mean(axis=0)
        # Euclidean distance in LAB
        color_dist = float(np.linalg.norm(mean_color_region - mean_color_class))
        if color_dist < grow_color_thresh:
            user_anns_aug[region_mask] = dominant_class

    return user_anns_aug


# --------------------------- Python fallback MBD ---------------------------

def _neighbors_offsets(conn: int = 4):
    if conn == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn == 8:
        return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    raise ValueError("conn must be 4 or 8")


def _run_mbd_py(weights: np.ndarray, seeds: np.ndarray, conn: int = 4):
    """
    Exact MBD propagation in Python.
    Seeds > 0 are locked. 0 is unlabeled and not locked.
    """
    import heapq

    H, W = weights.shape
    N = H * W
    w = weights.ravel()
    s = np.asarray(seeds, dtype=np.int32)
    if s.shape != (H, W):
        raise ValueError("seeds must match image size")

    bmin = np.full(N, np.inf, dtype=np.float32)
    bmax = np.full(N, -np.inf, dtype=np.float32)
    dist = np.full(N, np.inf, dtype=np.float32)
    label = np.zeros(N, dtype=np.int32)
    locked = (s.ravel() > 0)

    heap = []
    push = heapq.heappush
    pop = heapq.heappop
    counter = 0

    ys, xs = np.nonzero(s > 0)
    order = np.argsort(s[ys, xs] * (H * W) + (ys * W + xs))
    ys, xs = ys[order], xs[order]

    for y, x in zip(ys, xs):
        idx = y * W + x
        lab = int(s[y, x])
        val = w[idx]
        bmin[idx] = val
        bmax[idx] = val
        dist[idx] = 0.0
        label[idx] = lab
        push(heap, (bmax[idx], -bmin[idx], counter, idx, lab))
        counter += 1

    offs = _neighbors_offsets(conn)

    def itn(i):
        y, x = divmod(i, W)
        for dy, dx in offs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny * W + nx

    pops = 0
    while heap:
        bp, nbm, _, idx, lab = pop(heap)
        pops += 1
        bm = -nbm
        if bp != bmax[idx] or bm != bmin[idx] or lab != label[idx]:
            continue

        for j in itn(idx):
            if locked[j] and label[j] != lab:
                continue
            cand_bmin = bmin[idx] if bmin[idx] < w[j] else w[j]
            cand_bmax = bmax[idx] if bmax[idx] > w[j] else w[j]
            cand_bw = cand_bmax - cand_bmin

            upd = False
            if cand_bw < dist[j]:
                upd = True
            elif cand_bw == dist[j]:
                if cand_bmax < bmax[j]:
                    upd = True
                elif cand_bmax == bmax[j] and cand_bmin > bmin[j]:
                    upd = True
                elif cand_bmax == bmax[j] and cand_bmin == bmin[j] and lab < label[j]:
                    upd = True

            if upd:
                bmin[j] = cand_bmin
                bmax[j] = cand_bmax
                dist[j] = cand_bw
                label[j] = lab
                push(heap, (bmax[j], -bmin[j], counter, j, lab))
                counter += 1

    return label.reshape(H, W), dist.reshape(H, W), pops


# --------------------------- Runner ---------------------------

def run_single_image(image_path: str, ann_path: str, args) -> np.ndarray:
    # Load inputs
    img_rgb_u8 = load_image_rgb(image_path)
    seeds = load_annotation_map(ann_path)
    if seeds.shape[:2] != img_rgb_u8.shape[:2]:
        raise ValueError(f"Shape mismatch for {image_path} and {ann_path}: {img_rgb_u8.shape[:2]} vs {seeds.shape}")

    # Densify seeds via FH, unless disabled
    if not args.no_densify_fh:
        seeds_aug = augment_anns_with_fh(
            img_rgb_u8, seeds,
            fh_scale=args.fh_scale,
            fh_sigma=args.fh_sigma,
            fh_min_size=args.fh_min_size,
            min_region_frac=args.fh_min_region_frac,
            grow_color_thresh=args.fh_grow_color_thresh
        )
    else:
        seeds_aug = seeds

    # Weights are grayscale intensities in [0,1]
    w = load_image_grayscale(image_path)

    # Run MBD
    t0 = time.time()
    if _USE_CPP:
        labels, dist_map, pops = mbd_core.run_mbd_label_propagation(
            w.astype(np.float32, copy=False),
            seeds_aug.astype(np.int32, copy=False),
            int(args.conn)
        )
    else:
        labels, dist_map, pops = _run_mbd_py(w, seeds_aug, conn=args.conn)
    ms = (time.time() - t0) * 1000.0

    H, W = w.shape
    logging.info(f"{Path(image_path).stem}, {H}x{W}, pops {int(pops)}, {ms:.2f} ms, cpp {bool(_USE_CPP)}")
    return labels


def main():
    ap = argparse.ArgumentParser(description="MBD seeded segmentation with FH seed densification")
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--anns_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num-images", type=int, default=0, help="0 means all")
    ap.add_argument("--start-one", type=int, default=1, help="1-indexed starting position")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--conn", type=int, default=4, choices=[4, 8])
    ap.add_argument("--unlabeled-to-void", action="store_true", help="map 0 to 255 at save time")
    ap.add_argument("--no-shift-for-voc", action="store_true", help="do not shift label indices when saving")
    # FH densification controls
    ap.add_argument("--no-densify-fh", action="store_true", help="disable FH seed densification")
    ap.add_argument("--fh-scale", type=int, default=100)
    ap.add_argument("--fh-sigma", type=float, default=0.8)
    ap.add_argument("--fh-min-size", type=int, default=20)
    ap.add_argument("--fh-min-region-frac", type=float, default=0.20)
    ap.add_argument("--fh-grow-color-thresh", type=float, default=25.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pairs = find_image_annotation_pairs(args.images_dir, args.anns_dir)
    start_idx = max(0, int(args.start_one) - 1)
    if start_idx >= len(pairs):
        logging.info(json.dumps({"processed": 0, "skipped": len(pairs), "reason": "start index beyond input"}))
        return
    end_idx = len(pairs) if args.num_images == 0 else min(len(pairs), start_idx + int(args.num_images))
    work_list = pairs[start_idx:end_idx]

    out_root = Path(args.output_dir) / METHOD_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    times = []

    # single threaded saving and solving, optional workers would be for I O only
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
                times.append(ms)
                out_path = out_root / f"{base}_index.png"
                save_indexed_png(
                    mask,
                    str(out_path),
                    palette=voc_colormap(),
                    shift_for_voc=not args.no_shift_for_voc,
                    unlabeled_to_void=args.unlabeled_to_void
                )
                processed += 1
            except Exception as e:
                logging.error(f"Error on {base}: {e}")
                skipped += 1
            pbar.update(1)

    summary = {
        "total": len(work_list),
        "processed": processed,
        "skipped": skipped,
        "avg_runtime_ms": float(np.mean(times)) if times else None,
        "median_runtime_ms": float(np.median(times)) if times else None,
        "used_cpp": bool(_USE_CPP),
        "conn": int(args.conn),
        "method": METHOD_NAME
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()