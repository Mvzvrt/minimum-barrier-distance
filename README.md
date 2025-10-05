# mc_mbd: Multiclass Minimum Barrier Distance Segmentation

<p align="left">
  <a href="https://pypi.org/project/mc_mbd/"><img src="https://img.shields.io/pypi/v/mc_mbd.svg" alt="PyPI version"></a>
  <a href="https://github.com/Mvzvrt/minimum-barrier-distance/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Mvzvrt/minimum-barrier-distance.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>

</p>

> Strand, R., Malmberg, F., Saha, P. K., & Linnér, E. (2014). The Minimum Barrier Distance – Stability to Seed Point Position. In _Advanced Information Systems Engineering_ (pp. 111–121). Springer Berlin Heidelberg. [https://doi.org/10.1007/978-3-319-09955-2_10](https://doi.org/10.1007/978-3-319-09955-2_10)

**mc-mbd** is a fast, cross-platform implementation of the Multiclass Minimum Barrier Distance (MBD) algorithm for seeded image segmentation. It supports multi-label propagation and is accelerated by a C++ core with Python fallback. The package is installable via PyPI and works out-of-the-box for research and practical segmentation tasks.

## Installation (pip)

Install from PyPI (package name uses an underscore so it's importable):

```bash
pip install mc-mbd
```

After installing via pip you can import the library in Python (package name stays `mbd`):

```python
from mc_mbd import segment_image
```

## For Development (build from source)

To build and develop locally:

```bash
# Create and activate a virtual environment
python -m venv .mc_mbd
# On Windows (PowerShell):
.\.mc_mbd\Scripts\Activate.ps1
# On Windows (cmd):
.\.mc_mbd\Scripts\activate.bat
# On macOS/Linux:
source .mc_mbd/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build the C++ core extension in-place
python setup.py build_ext --inplace
```

## Performance

**Segmentation accuracy:**
mc-mbd achieves a mean intersection-over-union (mIoU) of 55.5% on the PASCAL VOC 2012 dataset using the ScribblesForAll dataset for initial seeds, without iterative refinement (one-shot segmentation).

**Speed:**
On standard 480×480 pixel images, mc-mbd segments each image in about 0.16 seconds on average, enabling fast and efficient large-scale image processing.

## Quick Usage Example

```python
# If installed from PyPI:
from mc_mbd import segment_image
import numpy as np
# image: 2D numpy array, seeds: integer mask (0=unlabeled, 1=background, 2+=foreground)
labels = segment_image(image, seeds)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Copyright © 2025 Mvzvrt. All rights reserved.
