# VAE-based Anomaly Detection for Predictive Maintenance in Electric Motors

## Project Overview

This repository contains the research code and resources associated with our scientific article on **"Robust Anomaly Detection in Electric Motors using Genetically Optimized Variational Autoencoders and Mahalanobis Distance for Embedded Predictive Maintenance"**. The primary goal of this research is to develop and validate a computationally efficient, VAE-based anomaly detection system suitable for deployment on low-cost embedded platforms (e.g., ESP32) for predictive maintenance (PdM) in the context of Industry 4.0.

The system learns a model of normal operational behavior from vibration data (specifically, its spectral characteristics) of an electric motor. Anomalies are then detected as significant deviations from this learned normality, quantified using the Mahalanobis distance in the VAE's regularized latent space.

## Key Features and Contributions

*   **Variational Autoencoder (VAE):** A VAE with dense layers is employed to learn a compressed and regularized representation (latent space) of normal motor vibration signals.
*   **Genetic Algorithm (GA) Optimization:** The architecture of the VAE (number of layers, units per layer, latent dimension) and key training hyperparameters are optimized using a genetic algorithm to find a balance between detection performance and model compactness.
*   **Mahalanobis Distance for Anomaly Scoring:** Anomaly scores are calculated based on the Mahalanobis distance of a new signal's latent representation from the distribution of normal data in the latent space. Robust estimation (Minimum Covariance Determinant - MCD) is used for the normal distribution parameters.
*   **Focus on Embedded Deployment:** The optimization process prioritizes models with low computational footprints (e.g., targeting < 60,000 parameters for the encoder) suitable for microcontrollers like the ESP32.
*   **Generalization Assessment:** The system's ability to generalize is rigorously tested using the public **MAFAULDA (Machinery Fault Database)**, evaluating its performance on a wide range of fault types and severities from a different machine setup.
*   **Unsupervised Training:** The VAE is trained solely on data from normal operating conditions, alleviating the need for extensive labeled fault datasets.

## Repository Structure (Preliminary)

```
.
├── data/                     # Placeholder for datasets (proprietary and processed MAFAULDA)
│   ├── proprietary_data/
│   └── mafaulda_processed/
├── notebooks/                # Jupyter notebooks for experimentation, visualization, etc.
├── src/                      # Source code
│   ├── preprocessing.py      # Signal preprocessing pipeline (FFT, scaling)
│   ├── vae_model.py          # VAE factory, custom layers (Sampling, KLDivergence)
│   ├── mahalanobis_detector.py # Mahalanobis distance detector
│   ├── ga_optimizer.py       # Genetic algorithm implementation for VAE optimization
│   ├── train_evaluate.py     # Scripts for training VAEs and evaluating individuals in GA
│   └── process_mafaulda.py   # Script for processing the raw MAFAULDA dataset
├── models/                   # Saved trained models (the "Small Model" and "Medium Model")
│   └── optimized_vaes/
├── results/                  # Figures, tables, and logs from experiments
│   └── ga_logs/
├── article/                  # PDF of the associated scientific article
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

*(This structure is a suggestion and will be populated and refined as the project progresses.)*

## Methodology Highlights

1.  **Data Acquisition:**
    *   Proprietary dataset: Vibration signals from a lab motor (ADXL345 sensor, ESP32, 3200 Hz sampling).
    *   MAFAULDA dataset: Public benchmark (radial vibrations, 50 kHz original, processed to 3125 Hz, 625 samples/window).
2.  **Signal Preprocessing (`preprocessing.py`):**
    *   Windowing (625 samples).
    *   FFT (256 points, specific frequency bins selected).
    *   L2 normalization of the spectrum.
    *   Feature scaling (`RobustScaler` fitted on proprietary normal data).
3.  **VAE Model (`vae_model.py`):**
    *   Encoder maps preprocessed features to a latent space (`z_mean`, `z_log_var`).
    *   Decoder reconstructs features from latent samples.
    *   Loss: MSE (reconstruction) + KL-Divergence (regularization).
    *   Regularization: L2, Batch Normalization, Dropout.
4.  **Anomaly Detection (`mahalanobis_detector.py`):**
    *   Encoder (from trained VAE) generates `z_mean` for new signals.
    *   Mahalanobis distance of `z_mean` from the distribution of normal `z_mean`s (calibrated with MCD).
5.  **Optimization (`ga_optimizer.py`, `train_evaluate.py`):**
    *   GA optimizes VAE architecture and hyperparameters.
    *   Fitness function balances detection performance (AUC-ROC, Youden's J) and model efficiency (parameters, latent dimension).

## Getting Started (Placeholder)

*(This section will be updated with instructions on how to set up the environment, prepare data, and run the code.)*

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AxelSkrauba/vae-anomaly-detection.git
    cd vae-anomaly-detection
    ```
2.  **Set up Python environment and install dependencies:**
    ```bash
    # It's recommended to use a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Data Preparation:**
    *   Instructions for obtaining/placing proprietary data.
    *   Instructions for downloading and processing MAFAULDA using `src/process_mafulda.py`.
4.  **Running Experiments:**
    *   Instructions on how to run the GA optimization.
    *   Instructions on how to load a trained model and evaluate it.

## Citation

If you use this work or code, please cite our upcoming article:

*   [PLACEHOLDER: Full citation...]
