# GPU-Accelerated DAS Ambient Noise Processing and EGF Enhancement

This repository provides an end-to-end GPU-accelerated workflow for processing distributed acoustic sensing (DAS) ambient noise data, with enhancements for empirical Green’s function (EGF) extraction and cross-correlation stacking.

## ✨ Key Features

- **GPU-accelerated cross-correlation** based on [SeismicAmbientNoiseDAS](https://github.com/yanyangg/SeismicAmbientNoiseDAS)
- **Bin-stack-based EGF enhancement** to improve temporal stability and signal-to-noise ratio
- **Phase-weighted stacking (PWS)** for robust cross-correlation aggregation
- Modular pipeline for DAS data pre-processing, slicing, and stacking
- Scriptable and adaptable to large-scale continuous datasets

## 🧪 Dependencies

- Python 3.10
- CPU or CUDA-compatible GPU
