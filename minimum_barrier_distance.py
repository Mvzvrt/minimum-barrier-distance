#!/usr/bin/env python3
"""
minimum_barrier_distance.py

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

METHOD_NAME = "minimum_barrier_distance"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------- VOC colormap, consistent with batch_refine.py ---------------------------

def voc_colormap(N: int = 256, normalized: bool = False) -> np.ndarray:
    """Return the standard VOC colormap. Matches the bit trick used in batch_refine.py."""
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
        colormap = colormap / 255.0
    return colormap


def save_indexed_png(mask: np.ndarray,
                     out_path: str,
                     palette: Optional[np.ndarray] = None,
                     shift_for_voc: bool = True,
                     unlabeled_to_void: bool = True) -> None:
    """
    Save H x W int label mask as an indexed PNG. To match VOC colors, shift labels down by 1.
    Your labels: 0 unlabeled, 1 background, 2..K objects.
    After shift: background -> 0, objects -> 1..K-1, unlabeled stays 255 if unlabeled_to_void else stays 0 before shift.

    Parameters
    ----------
    mask : np.ndarray of int
    out_path : output file path
    palette : np.ndarray palette shape 256 x 3 uint8
    shift_for_voc : if True, apply the described shift before saving
    unlabeled_to_void : if True, put unlabeled as 255, commonly used as ignore index
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
        ann = np.load(ann_path)
        return np.asarray(ann, dtype=np.int32)
    img = Image.open(ann_path)
    if img.mode != "P":
        img = img.convert("P")
    ann = np.asarray(img, dtype=np.int32)
    return ann


def find_image_annotation_pairs(images_dir: str, anns_dir: str) -> List[Tuple[str, Optional[str]]]:
    """Pair images with annotations by basename. Prefer .npy over .png, skip if missing."""
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
    if conn == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn == 8:
        return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    raise ValueError("conn must be 4 or 8")


def _run_mbd_py(weights: np.ndarray, seeds: np.ndarray, conn: int = 4):
    """
    Exact MBD with Algorithm 3 like propagation in Python.
    Seeds greater than 0 are hard constraints, 0 is unlabeled and not locked.
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
