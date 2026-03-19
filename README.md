# Dense Multi-View 3D Reconstruction Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![COLMAP](https://img.shields.io/badge/COLMAP-3.8+-green.svg)](https://colmap.github.io/)
[![Open3D](https://img.shields.io/badge/Open3D-0.17+-orange.svg)](http://www.open3d.org/)

A production-ready **Structure from Motion (SfM) + Multi-View Stereo (MVS)** pipeline with classical and deep feature matching, config-driven automation, and interactive 3D visualisation.

---

## 🏗️ Pipeline Architecture

```
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Input       │     │  Feature          │     │  Feature        │
│  Images      │────▶│  Extraction       │────▶│  Matching       │
│  (data/)     │     │  (SIFT/ORB/       │     │  (BF/FLANN/     │
│              │     │   SuperPoint)     │     │   SuperGlue)    │
└──────────────┘     └───────────────────┘     └────────┬────────┘
                                                        │
                     ┌───────────────────┐              │
                     │  Structure from   │◀─────────────┘
                     │  Motion (SfM)     │
                     │  • Camera poses   │
                     │  • Sparse cloud   │
                     └────────┬──────────┘
                              │
                     ┌────────▼──────────┐
                     │  Multi-View       │
                     │  Stereo (MVS)     │
                     │  • Depth maps     │
                     │  • Dense cloud    │
                     └────────┬──────────┘
                              │
                ┌─────────────▼─────────────┐
                │  Surface Reconstruction   │
                │  • Poisson / Ball-Pivot   │
                │  • Mesh generation        │
                └─────────────┬─────────────┘
                              │
                     ┌────────▼──────────┐
                     │  Visualisation    │
                     │  • 3D viewer      │
                     │  • Camera traj.   │
                     │  • Match preview  │
                     └──────────────────┘
```

---

## 📖 SfM vs MVS

| Aspect | Structure from Motion (SfM) | Multi-View Stereo (MVS) |
|--------|---------------------------|------------------------|
| **Goal** | Estimate camera poses + sparse 3D points | Dense depth per pixel |
| **Input** | Images + feature correspondences | Images + known camera poses |
| **Output** | Sparse point cloud (thousands of pts) | Dense point cloud (millions of pts) |
| **Method** | Triangulation + bundle adjustment | Patch-match stereo + depth fusion |
| **Speed** | Fast (minutes) | Slow (hours for large sets) |
| **Dependency** | Runs first | Requires SfM output |

**SfM** solves the *where are the cameras?* problem by finding feature correspondences across images, estimating relative poses, and refining everything with bundle adjustment.

**MVS** takes the solved cameras and computes *per-pixel depth maps* using photometric consistency across views, producing point clouds orders of magnitude denser than SfM.

---

## 🔍 Classical vs Deep Feature Matching

| Method | Keypoints | Descriptors | Matching | Strengths | Weaknesses |
|--------|-----------|-------------|----------|-----------|------------|
| **SIFT** | DoG detector | 128-D float | BF / FLANN | Scale/rotation invariant, reliable | Slow, patent (expired) |
| **ORB** | FAST + orientation | 32-byte binary | BF Hamming | Very fast, real-time | Less robust to scale |
| **SuperPoint** | CNN detector | 256-D float | SuperGlue / BF | Handles textureless, repeatable | Needs GPU, model weights |

> SuperPoint+SuperGlue typically achieves **~15% higher inlier ratio** on textureless indoor scenes versus classical SIFT, making it the preferred choice for challenging datasets.

---

## 📁 Project Structure

```
Dense-3D-Reconstruction/
├── configs/
│   └── default.yaml           # Central pipeline configuration
├── data/
│   └── images/                # Place your multi-view images here
├── outputs/                   # Pipeline results
├── src/
│   ├── features/              # Feature extraction (SIFT, ORB, SuperPoint)
│   ├── matching/              # Feature matching (BF, FLANN, SuperGlue)
│   ├── sfm/                   # Structure from Motion (COLMAP wrapper)
│   ├── mvs/                   # Multi-View Stereo (COLMAP + Open3D)
│   ├── reconstruction/        # Mesh generation + point cloud processing
│   ├── visualization/         # 3D viewer, camera trajectory, match viz
│   ├── utils/                 # Logger, config, timer, I/O
│   └── pipeline.py            # End-to-end orchestrator
├── scripts/
│   ├── run_pipeline.py        # CLI entry point
│   └── benchmark.py           # Feature comparison benchmark
├── benchmarks/                # Benchmark results
├── CMakeLists.txt             # C++ build (optional accelerators)
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚙️ Setup

### 1. Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Install COLMAP

COLMAP is required for the SfM and MVS stages.

**Windows:**
```bash
# Download pre-built binaries from:
# https://github.com/colmap/colmap/releases
# Extract and add to PATH
```

**Ubuntu/Debian:**
```bash
sudo apt-get install colmap
```

**macOS (Homebrew):**
```bash
brew install colmap
```

**Verify installation:**
```bash
colmap help
```

### 3. (Optional) SuperPoint/SuperGlue Weights

Place pretrained weights in `models/`:
```
models/
├── superpoint_v1.pth
└── superglue_outdoor.pth
```

> Without weights, the pipeline gracefully falls back to SIFT-based extraction.

---

## 🚀 Usage

### End-to-End Pipeline

```bash
# Full pipeline with default settings
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --images data/images

# Use ORB features, skip dense MVS
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --images data/images \
    --feature-type orb --skip-mvs

# SuperPoint + SuperGlue deep matching
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --images data/images \
    --feature-type superpoint --matcher-type superglue

# Custom output, no visualisation
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --images data/images \
    --output outputs/experiment_01 --no-viz
```

### CLI Options

```
python scripts/run_pipeline.py --help

Arguments:
  --config            YAML config file path
  --images            Input image directory
  --output            Output directory
  --feature-type      sift | orb | superpoint
  --max-keypoints     Max keypoints per image
  --matcher-type      bf | flann | superglue
  --ratio-test        Lowe's ratio threshold
  --skip-mvs          Skip MVS stage
  --skip-mesh         Skip mesh generation
  --no-viz            Disable visualisation
  --log-level         DEBUG | INFO | WARNING | ERROR
```

### Benchmarking

```bash
python scripts/benchmark.py \
    --images data/images \
    --output benchmarks/ \
    --max-images 10
```

### Python API

```python
from src.pipeline import ReconstructionPipeline

pipe = ReconstructionPipeline("configs/default.yaml")
metrics = pipe.run(image_dir="data/images", skip_mvs=False)
print(metrics)
```

---

## 📊 Metrics

The pipeline collects and reports:

| Metric | Description |
|--------|-------------|
| **Feature inlier ratio** | Percentage of matches surviving RANSAC |
| **Reconstruction completeness** | Fraction of input images registered |
| **Processing time** | Per-stage and total wall-clock time |
| **Keypoint count** | Total and per-image keypoints |
| **Point cloud density** | Sparse vs dense point counts |

Metrics are saved to `outputs/metrics.json` after each run.

---

## 📸 Sample Outputs

After running the pipeline you will find:

| Output | Location |
|--------|----------|
| Sparse point cloud | `outputs/colmap_ws/sparse/` |
| Dense point cloud | `outputs/colmap_ws/dense/fused.ply` |
| Triangle mesh | `outputs/mesh.ply` |
| Match preview | `outputs/matches_preview.png` |
| Camera trajectory | `outputs/camera_trajectory.png` |
| Pipeline log | `outputs/pipeline.log` |
| Metrics JSON | `outputs/metrics.json` |

---

## 🔧 Configuration

All parameters are controlled via `configs/default.yaml`:

```yaml
features:
  type: "sift"          # sift | orb | superpoint
  max_keypoints: 8192

matching:
  type: "bf"            # bf | flann | superglue
  ratio_test: 0.75

sfm:
  camera_model: "SIMPLE_RADIAL"
  single_camera: true

mvs:
  enabled: true
  patch_match:
    max_image_size: 2000
    geom_consistency: true

reconstruction:
  mesh_method: "poisson"
```

CLI arguments override config values for quick experimentation.

---

## 🏗️ C++ Components (Optional)

```bash
mkdir build && cd build
cmake .. -DBUILD_FILTERS=ON
cmake --build .
```

Provides optional C++ accelerators for point cloud filtering.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
