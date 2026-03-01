# Deep U-Net for Faint Stellar Stream Detection amidst Galactic Cirrus

This repository contains the complete Deep Learning pipeline for detecting **Low-Surface Brightness (LSB)** structures —stellar streams and galactic cirrus— in multi-band astronomical imagery. By combining physics-informed synthetic data generation with a Bayesian-optimized U-Net, the system achieves a **Stream IoU > 0.84** and **Recall > 0.93**, successfully disentangling faint stellar emission from interstellar dust contamination.

## 🚀 Key Achievements
* **High-Fidelity Detection:** Achieved **IoU > 0.84** on stellar streams at critically low signal-to-noise ratios.
* **Robust Disentanglement:** Simultaneous 3-class segmentation (Background, Stream, Cirrus) with **Recall > 0.93** for streams.
* **Physics-Informed Generation:** Created a 20,000-sample synthetic dataset with procedural $1/f^\beta$ fractal noise and real SDSS/WISE backgrounds.
* **Hard Example Mining:** 40% of the training data consists of forced edge cases (high-contrast cirrus masking faint streams) to push model robustness.
* **Test-Time Augmentation (TTA):** 4× geometric augmentation at inference for smoother, more reliable segmentation masks.

---

## 📂 Project Pipeline

### 1. [Astrophysical Data Generator](Notebooks/astrophysical-generator-stellar-streams-cirrus.ipynb)
This notebook implements a physics-informed synthetic data pipeline that generates 20,000 multi-band FITS images (SDSS g/r + WISE W1/W2) with pixel-level ground truth masks. Stellar streams are modeled using quadratic Bézier curves with stochastic $1/f^\beta$ fractal noise ($\beta \in [1.5, 2.5]$), while galactic cirrus is generated using Brownian motion fields ($\beta \in [2.5, 3.5]$). A Hard Example Mining strategy forces 8,000 samples into extreme conditions: high-contrast cirrus-stream overlaps and critically low SNR streams.

### 2. [Deep U-Net Training Pipeline](Notebooks/deep-u-net-for-faint-stellar-stream-detection.ipynb)
This notebook trains a Deep Dynamic U-Net (Depth 4, 64 start filters) for 3-class semantic segmentation of astronomical scenes. The architecture uses a weighted hybrid loss function (90% Dice + 10% CCE) to prioritize structural recovery of faint, curvilinear features. Training employs a "Slow Cooking" strategy with ReduceLROnPlateau and Early Stopping (patience=40), running under Mixed Precision (float16) on NVIDIA P100 for accelerated convergence.

### 3. [Bayesian Hyperparameter Optimization](Notebooks/hyperparameter-tuning-stellar-streams-optuna.ipynb)
This notebook executes a Bayesian hyperparameter search using the Optuna framework (TPE sampler + Hyperband pruning) to tune 5 critical parameters: network depth (3–5), start filters (32/64), dropout rate, learning rate, and Dice loss weight. The search runs on a representative 4,000-sample subset for up to 30 trials (7-hour timeout), identifying the optimal configuration that maximizes validation Dice coefficient.

### 4. [Inference & Error Analysis](Notebooks/stream-detection-inference-error-analysis.ipynb)
This notebook performs the final evaluation of the trained model on an unseen test set. It implements a 4× Test-Time Augmentation (TTA) engine to improve boundary precision, computes per-class IoU, Precision, Recall, and F1-Score, and generates a normalized confusion matrix. A qualitative hard-mining analysis identifies the best and worst predictions to understand model limitations and inform future improvements.

---

## 📊 Dataset
The training data is generated synthetically by injecting procedural structures into real astronomical backgrounds downloaded from the Virtual Observatory (HiPS/Aladin). The dataset contains:
* **20,000 FITS images** — 4 channels each (SDSS g, SDSS r, WISE W1, WISE W2).
* **20,000 pixel-level masks** — 3 classes (0: Background, 1: Stream, 2: Cirrus).
* **Hard Mining Split:** 12,000 normal samples + 8,000 edge cases.

---

## 🛠️ Methodology & Astrophysical Context
Stellar streams are the fossil remnants of tidally disrupted dwarf galaxies and globular clusters. Their detection is a cornerstone of **Near-Field Cosmology**, providing constraints on dark matter halo shapes and accretion histories. However, these structures are extremely faint (low surface brightness) and frequently overlap spatially with **galactic cirrus** — diffuse interstellar dust clouds that emit strongly in the infrared.

Our approach exploits the **spectral signature difference** between the two phenomena:
1. **Visible Bands (g/r):** Stellar streams emit predominantly in the optical, as they consist of resolved/unresolved stellar populations.
2. **Infrared Bands (W1/W2):** Galactic cirrus dominates in the mid-infrared due to thermal emission from dust grains.
3. **Multi-band Fusion:** The 4-channel U-Net simultaneously processes all bands, learning to disentangle the two morphologically similar but spectrally distinct structures.

---

## 📈 Performance Summary

| Metric | Stream (Class 1) | Cirrus (Class 2) | Background (Class 0) |
| :--- | :---: | :---: | :---: |
| **IoU** | **> 0.84** | > 0.88 | > 0.97 |
| **Precision** | > 0.89 | > 0.91 | > 0.98 |
| **Recall** | **> 0.93** | > 0.95 | > 0.98 |

*Results reported with 4× Test-Time Augmentation on the held-out 15% test set.*

---

## 🏗️ Model

The trained model (`best_model_final.keras`) is included in the [`model/`](model/) directory. It can be loaded with:

```python
from tensorflow.keras.models import load_model

model = load_model('model/best_model_final.keras', custom_objects={
    'dice_metric': dice_coef,
    'hybrid_loss': final_loss
})
```

---
*Author: Ibón García Gómez · Master's Thesis (TFM) Research · 2026*
