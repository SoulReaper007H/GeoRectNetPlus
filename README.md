# GeoRectNetPlus

GeoRectNetPlus is a weakly supervised building-footprint segmentation framework that reduces pseudo-label noise and improves boundary accuracy using uncertainty estimation, attention, geometry, wavelets, and 7-gate filtering.

## Key Features

* Weakly supervised building-footprint segmentation
* Evidential Dirichlet Learning (EDL)
* Cross-Layer Attention Agreement Maps (CLAAM)
* Signed Distance Regression (SDR)
* Haar–Hadamard Wavelet Branch
* 7-Gate Pseudo-Label Acceptance Pipeline
* Designed for limited labeled satellite imagery

## Dataset

* WHU Building Dataset
* Inria Aerial Image Labeling Dataset

## Results

| Metric       |    Score |
| ------------ | -------: |
| IoU          |   87.38% |
| Boundary-IoU |   73.91% |
| Hausdorff-95 | 24.17 px |
| ECE          |   0.0104 |

## Tech Stack

Python • PyTorch • GDAL • Vision Transformer

## Project Structure

```text
GeoRectNetPlus/
├── datasets/
├── models/
├── modules/
├── training/
├── configs/
├── requirements.txt
└── train.py
```

## Phase II

Phase II extends GeoRectNet with uncertainty estimation, structural consistency, geometric supervision, and frequency-based boundary analysis. These components are integrated through a 7-gate pseudo-label acceptance pipeline to improve the reliability of weakly supervised training.

## License

For academic and research purposes.
