# Requirements by Script

This document explains which Python packages are required by each main script in the project. The exact package versions were derived from the active virtual environment used during development.

## Recommended Installation

Create and activate a virtual environment, then install the main requirements:

```bash
pip install -r requirements.txt
```

The project was tested with a CUDA-enabled PyTorch build:

```text
torch==2.12.0+cu126
torchvision==0.27.0+cu126
torchaudio==2.11.0+cu126
```

If this exact CUDA build is not available on another machine, install the appropriate PyTorch version for that machine from the official PyTorch installation selector.

---

## 1. BaselineAutoencoderExperiments.py

### Purpose
Runs the first baseline comparison across:

- SlowFast + Hand Pose pooled autoencoder
- SlowFast + Hand Pose per-camera autoencoder + fusion
- TSM + Hand Pose pooled autoencoder
- TSM + Hand Pose per-camera autoencoder + fusion

This script was used to decide whether pooled multi-view learning or per-camera decision-level fusion was more suitable.

### Required packages

```text
numpy==2.4.4
matplotlib==3.10.9
scikit-learn==1.8.0
joblib==1.5.3
torch==2.12.0+cu126
torchvision==0.27.0+cu126
torchaudio==2.11.0+cu126
```

### Main outputs

```text
BaselineAE_Results/
├── baseline_summary.csv
├── experiment folders
├── autoencoder_best.pth
├── scaler.pkl
├── threshold.npy
├── metrics.json
├── test_results.csv
└── error_distribution.png
```

---

## 2. OptimizedPerCameraAEExperiments.py

### Purpose
After pooled experiments were found to be weaker, this script focuses only on per-camera methods. It compares SlowFast and TSM feature families with different autoencoder sizes, threshold percentiles, and fusion strategies.

### Required packages

```text
numpy==2.4.4
matplotlib==3.10.9
scikit-learn==1.8.0
joblib==1.5.3
torch==2.12.0+cu126
torchvision==0.27.0+cu126
torchaudio==2.11.0+cu126
```

### Main outputs

```text
OptimizedPerCameraAE_Results/
├── optimized_per_camera_summary.csv
├── feature family folders
├── camera_results.json
├── test_results.csv
└── fusion_distribution.png
```

---

## 3. FinalTSMPerCameraTuning.py

### Purpose
Runs the final tuning stage on the selected feature family:

```text
TSM + Hand Pose / Per-camera Autoencoder / Decision-level Fusion
```

It tests final model variants, threshold percentiles, and fusion strategies. The selected final model is:

```text
small_no_dropout + max fusion + threshold percentile 75
```

### Required packages

```text
numpy==2.4.4
pandas==3.0.3
matplotlib==3.10.9
scikit-learn==1.8.0
joblib==1.5.3
torch==2.12.0+cu126
torchvision==0.27.0+cu126
torchaudio==2.11.0+cu126
```

### Main outputs

```text
FinalTSMPerCameraAE_Results/
├── final_tuning_summary.csv
├── final_best_config.json
├── model folders
├── autoencoder_best.pth
├── scaler.pkl
├── training_curve.png
├── error_distribution.png
├── fusion_test_results.csv
└── fusion_distribution.png
```

---

## 4. demo_app.py

### Purpose
Runs the Streamlit demo application. It does not train the model and does not perform feature extraction. Instead, it visualizes precomputed model outputs for a selected video:

- video playback from local path
- reconstruction error timeline
- anomaly threshold
- detected anomaly intervals
- TP / FP / FN / TN window-level correctness
- window-level result table

### Required packages

```text
streamlit==1.57.0
pandas==3.0.3
matplotlib==3.10.9
opencv-python==4.13.0.92
scikit-learn==1.8.0
```

### Main inputs

```text
single_video_poster_outputs_for_demo_app/
├── 9064-c13a_9064_C10095_timeline.csv
├── video_download_link.txt
└── 9064C10095_rgb.mp4   # downloaded manually; not stored in Git
```

### Run command

```bash
streamlit run demo_app.py
```

---

## 5. Optional plotting scripts

If extra plotting utilities are used for ROC curve, metric tables, or poster figures, they usually require:

```text
numpy==2.4.4
pandas==3.0.3
matplotlib==3.10.9
scikit-learn==1.8.0
```

---

## Full environment snapshot

A full virtual environment snapshot is also kept as:

```text
requirements_full.txt
```

This file contains all packages installed in the development environment. It is more detailed than `requirements.txt`, but may include packages not strictly required for the final autoencoder and demo scripts.
