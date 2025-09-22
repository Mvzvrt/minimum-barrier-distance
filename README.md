# Minimum Barrier Distance, seeded image segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org)
[![scikit-image](https://img.shields.io/badge/scikit--image-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-image.org)
[![Pillow](https://img.shields.io/badge/Pillow-%23EE4C2C.svg?logo=python&logoColor=white)](https://python-pillow.org)

A high-performance implementation of the Minimum Barrier Distance (MBD) algorithm for seeded image segmentation. This project provides both a pure Python implementation and an optimized C++ version using pybind11 for efficient multi-label segmentation, along with an experimental seed densification feature. The pipeline accepts per-image annotations, produces VOC-compliant indexed PNG masks, and follows a simple, reproducible command line interface.

## Features

- Exact Minimum Barrier Distance objective, multi label seeded propagation
- Annotation convention, class 0 is unlabeled, class 1 is background, classes greater than 1 are foreground categories
- Seed densification using Felzenszwalb regions, configurable by flags
- VOC colormap saving, background is shown with palette index 0
- Fast path via a C++ extension, Python fallback is available

## Repository layout

```
minimum-barrier-distance/
  mbd.py
  mbd_core.cpp
  setup.py
  README.md
  requirements.txt
```

## Requirements

- **Python:** 3.9 or newer
- **Core Libraries:**
  - `numpy`
  - `pillow`
  - `scikit-image`
  - `tqdm`
- **C++ Dependencies:**
  - Windows: Microsoft C++ Build Tools
  - macOS: Xcode Command Line Tools
  - Linux: build-essential and gcc/clang

## Installation

### 1. Set Up C++ Build Environment

**Windows:**

- Install Visual Studio Build Tools (includes C++ build tools)
- Open Developer PowerShell for VS

**macOS:**

```bash
xcode-select --install
```

**Linux:**

```bash
sudo apt install build-essential  # For Ubuntu/Debian
```

### 2. Create and Activate Virtual Environment

**Windows:**

```powershell
python -m venv .mbd
.mbd\Scripts\Activate.ps1
```

**Unix/macOS:**

```bash
python -m venv .mbd
source .mbd/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the C++ Core

The C++ extension accelerates the inner loop. The Python code will automatically fall back if the extension is not present.

```bash
python setup.py build_ext --inplace
```

This produces a platform specific module named `mbd_core.*` in the project folder. The Python script imports this module if available. The `setup.py` provided here is cross platform, not Windows only, it uses `pybind11` and `numpy` headers and builds a single extension named `mbd_core`.

## Prepare your data

Input folders are matched by basename, images against annotations.

- Images folder, contains `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif` files
- Annotations folder, contains either a `.npy` array with shape H×W and integer dtype, or a paletted `.png` mask
- Basename pairing, `images_dir/foo.jpg` pairs with `anns_dir/foo.npy` or `anns_dir/foo.png`

### Annotation rules, very important

DISCLAIMER: More information on the scribbles format will be provided in the future for easier reference.

- Class 0 is unlabeled, these pixels are not used as seeds
- Class 1 is background, these pixels are hard constrained to background
- Classes greater than 1 are foreground, these pixels are hard constrained to their class

For convenience, here are the Pascal VOC classes used commonly with a VOC colormap.

```json
{
  "1": "background",
  "2": "aeroplane",
  "3": "bicycle",
  "4": "bird",
  "5": "boat",
  "6": "bottle",
  "7": "bus",
  "8": "car",
  "9": "cat",
  "10": "chair",
  "11": "cow",
  "12": "diningtable",
  "13": "dog",
  "14": "horse",
  "15": "motorbike",
  "16": "person",
  "17": "pottedplant",
  "18": "sheep",
  "19": "sofa",
  "20": "train",
  "21": "tvmonitor"
}
```

## Run the segmenter

Output mask files are saved under `output_dir/mbd_masks/<basename>_index.png`, using a VOC palette, with label indices shifted on save so background 1 becomes palette index 0.

Basic example, single thread, four connectivity, seed densification enabled by default, full masks shown, do not map unlabeled to void.

### Windows, PowerShell

```powershell
python minimum_barrier_distance.py --images_dir <your_image_directory> --anns_dir <your_annotation_directory> --output_dir <your_output_directory> --num-images 0 --start-one 1 --workers 0 --conn 4
```

### macOS or Linux

```bash
python mbd.py --images_dir <your_image_directory> --anns_dir <your_annotation_directory> --output_dir <your_output_directory> --num-images 0 --start-one 1 --workers 0 --conn 4
```

### Important note about the `--unlabeled-to-void` flag

If you want to see full segmentation masks, do not set `--unlabeled-to-void`. Leaving it off keeps unlabeled pixels at index 0 before the save time shift, which then become palette index 0 after the shift. If you set `--unlabeled-to-void`, unlabeled class 0 will be saved as 255, many VOC tools treat this as ignore. For visualizing full masks, leave this flag out.

## Command line reference

- `--images_dir`, required, folder containing input images
- `--anns_dir`, required, folder containing per image annotations
- `--output_dir`, required, folder where results are saved
- `--num-images`, integer, default 0, zero means process all
- `--start-one`, integer, default 1, one indexed start offset
- `--workers`, integer, default 0, nonzero enables threaded I O in the outer loop, compute is single threaded
- `--conn`, integer, 4 or 8, pixel connectivity for MBD
- `--unlabeled-to-void`, flag, if set, saves unlabeled 0 as 255 in the PNG
- `--no-shift-for-voc`, flag, if set, disables the save time index shift, keep this off to match VOC colors
- Densification flags, see the previous section

## Output format and colors

- One file per input image, path `output_dir/minimum_barrier_distance/<basename>_index.png`
- PNG is paletted with the VOC colormap
- Labels are shifted by one at save time, to align with the VOC palette indexing
  - background class 1 becomes palette index 0
  - classes greater than 1 become indices greater than or equal to 1
  - unlabeled class 0 is left as 0 unless you set `--unlabeled-to-void`, in that case it becomes 255

## Troubleshooting

- Build fails for the C++ core on Windows, install Visual Studio Build Tools, select C++ build tools, then reopen the developer PowerShell and try again
- Build fails on macOS, run `xcode-select --install`, then rebuild
- Build fails on Linux, install `build-essential` with your package manager, for example `sudo apt install build-essential`, then rebuild
- Import error for `mbd_core`, the Python script will fall back to the pure Python solver, confirm that `python setup.py build_ext --inplace` succeeded and that `mbd_core.*` exists in the project folder
- Palette colors look shifted, do not use `--no-shift-for-voc`, also confirm that you did not set `--unlabeled-to-void` if you want full masks

## Performance Notes

The C++ core significantly accelerates the Minimum Barrier Distance propagation. The Python wrapper handles I/O, seed densification, and image saving. On typical VOC-sized images, the end-to-end processing time is minimal. The pipeline leverages:

- Optimized C++ core for MBD computation
- Efficient NumPy operations for preprocessing
- Multi-threaded I/O capabilities (optional)

## References

The Minimum Barrier Distance algorithm is based on the work:

```bibtex
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
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The scikit-image library for image processing functionality
- The pybind11 project for C++/Python bindings
- The NumPy community for numerical computing support

---

Copyright © 2025 Mvzvrt. All rights reserved.
