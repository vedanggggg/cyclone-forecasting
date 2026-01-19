# Replication Notes: Tropical Cyclone Forecast with Video Diffusion

**Paper Title:** Improving Tropical Cyclone Forecasting with Video Diffusion Models (ICLR 2025 Workshop)  
**Original Implementation:** [GitHub Repo](https://github.com/Ren-creater/forecast-video-diffmodels)  
**Current Implementation:** Kaggle Notebook (`training-clone-fc.ipynb`)

---

## 1. Executive Summary
This document outlines the discrepancies between the official research paper, the provided code repository, and the current Kaggle replication attempt. 

**Status:** The current Kaggle implementation is a **"Video-Only" adaptation**. It successfully ports the 3D architecture to low-memory environments (16GB VRAM) but currently skips the critical "Stage 1" (Spatial Pre-training) described in the paper. This may result in lower fidelity for individual frames compared to the paper's reported metrics.

---

## 2. Critical Discrepancies

| Feature | [cite_start]**Research Paper** [cite: 166, 167] | **Original Repo Code** | **Kaggle Replication** |
| :--- | :--- | :--- | :--- |
| **Training Strategy** | **Two-Stage**<br>1. Spatial (Image) <br>2. Temporal (Video) | **Hybrid / Early Switch**<br>Starts on images, switches to video at `epoch // 8`. | **Single-Stage (Video)**<br>Training starts immediately on video sequences (0 pre-training). |
| **Stage 1 Duration** | 100 Epochs (25% of total) | ~12% of total epochs | **0 Epochs** |
| **Learning Rate** | $3 \times 10^{-4}$ (`3e-4`) | `1e-4` | **`1e-4`** (Mismatch)<br>*Variable defined as `3e-4` but hardcoded in optimizer.* |
| **Batch Size** | 1 | 1 | **1 (Effective)**<br>*Simulated Batch Size of 8 via Gradient Accumulation.* |
| **Precision** | FP32 (Implied) | FP32 | **FP16 (Mixed Precision)**<br>*Necessary for 16GB VRAM limit.* |
| **Model Capacity** | Dim = 64 | Dim = 32 | **Dim = 128**<br>*Significantly larger capacity than repo baseline.* |

---

## 3. Detailed Analysis

### A. Training Strategy (The "Two-Stage" Issue)
[cite_start]The paper explicitly attributes its high SSIM and PSNR scores to a curriculum learning approach [cite: 64-75]:
1.  **Stage 1 (Spatial):** The model learns to generate high-quality static images of cyclones.
2.  **Stage 2 (Temporal):** The model learns the motion/evolution of the cyclone.

**Current Gap:** The Kaggle implementation trains the video model from scratch.
* **Risk:** The model must learn *what* a cyclone looks like and *how* it moves simultaneously. This often leads to "blurry" frames, even if the motion is smooth.

### B. Architecture Improvements
The Kaggle implementation actually improves upon the repository's baseline in two ways:
1.  **3D Convolutional Projector:** Instead of flattening ERA5 weather data, a `ConvERA5Projector` is used. This preserves the spatial structure of the atmospheric data before injection.
2.  **Flash Attention:** The use of `memory_efficient=True` (implied/recommended) allows for a larger model (`dim=128`) on smaller hardware.

### C. Hardware Workarounds (16GB VRAM)
To replicate the paper's results (originally on 48GB L40 GPU) on a 16GB T4/P100:
* **Mixed Precision (FP16):** Enabled (cuts memory by ~50%).
* **Gradient Accumulation:** Set to `8` steps. This mathematically simulates a larger batch size, stabilizing the gradients for the complex 3D U-Net.

---



