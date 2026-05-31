# TestEnvironment – Final TSM Per-Camera Autoencoder Inference Guide

This folder contains a lightweight test environment for running inference with the final trained autoencoder models. It is designed for testing already-extracted TSM + 3D hand pose feature files without retraining the models.

## 1. Purpose

The main project tested several anomaly detection strategies for Assembly101 assembly videos. After comparing pooled models, per-camera models, SlowFast features, TSM features, different autoencoder architectures, threshold percentiles, and fusion methods, the final selected system was:

| Component | Final Choice |
|---|---|
| Feature type | TSM visual features + 3D hand pose features |
| Input dimension | 4064 |
| Model strategy | One autoencoder per camera |
| Autoencoder architecture | 4064 → 256 → 64 → 16 → 64 → 256 → 4064 |
| Dropout | Not used in final selected model |
| Threshold strategy | Validation normal reconstruction error, 75th percentile |
| Fusion method | Max fusion over available camera scores |

This test environment uses the trained final models and applies them to test feature files. It does not extract features from raw videos and does not train new autoencoders.

## 2. Folder Structure

The expected folder structure is:

```text
TestEnvironment/
│
├── TestSingleVideoWithFinalAE.py
├── README.md
├── requirements.txt
│
├── FinalTSMPerCameraAE_Results/
│   ├── final_tuning_summary.csv
│   └── small_no_dropout/
│       ├── C10095/
│       │   ├── autoencoder_best.pth
│       │   └── scaler.pkl
│       ├── C10118/
│       │   ├── autoencoder_best.pth
│       │   └── scaler.pkl
│       ├── C10119/
│       │   ├── autoencoder_best.pth
│       │   └── scaler.pkl
│       ├── C10390/
│       │   ├── autoencoder_best.pth
│       │   └── scaler.pkl
│       └── C10404/
│           ├── autoencoder_best.pth
│           └── scaler.pkl
│
├── per_camera_tsm/
│   ├── test_features_tsm_C10095.npy
│   ├── test_features_tsm_C10118.npy
│   ├── test_features_tsm_C10119.npy
│   ├── test_features_tsm_C10390.npy
│   ├── test_features_tsm_C10404.npy
│   ├── test_features_tsm_window_labels_C10095.npy
│   ├── test_features_tsm_window_labels_C10118.npy
│   ├── test_features_tsm_window_labels_C10119.npy
│   ├── test_features_tsm_window_labels_C10390.npy
│   └── test_features_tsm_window_labels_C10404.npy
│
└── single_video_inference_outputs/
    └── generated automatically after running the script
```

## 3. Required Input Files

### 3.1 Final model files

For each camera, the script expects:

```text
FinalTSMPerCameraAE_Results/small_no_dropout/<CAMERA>/autoencoder_best.pth
FinalTSMPerCameraAE_Results/small_no_dropout/<CAMERA>/scaler.pkl
```

These files are required because each camera has its own trained autoencoder and scaler. A feature extracted from camera `C10095` must be tested with the `C10095` scaler and autoencoder. Camera models should not be mixed.

### 3.2 Threshold file

The script also requires:

```text
FinalTSMPerCameraAE_Results/final_tuning_summary.csv
```

This file is used to read the final threshold values for each camera. The selected final threshold setting is `p=75`.

### 3.3 Test feature files

The test features must be TSM + 3D hand pose features with shape:

```text
(number_of_windows, 4064)
```

Expected names:

```text
per_camera_tsm/test_features_tsm_C10095.npy
per_camera_tsm/test_features_tsm_C10118.npy
per_camera_tsm/test_features_tsm_C10119.npy
per_camera_tsm/test_features_tsm_C10390.npy
per_camera_tsm/test_features_tsm_C10404.npy
```

The script can work with one camera, multiple cameras, or all five cameras:

| Available feature files | Inference mode |
|---|---|
| 1 camera | Single-camera inference |
| 2–4 cameras | Partial-camera max fusion |
| 5 cameras | Full final max fusion |

For the full final system, all five cameras should be available.

### 3.4 Label files

If label files are available, the script calculates Accuracy, Precision, Recall, F1-score, AUROC, and confusion matrix.

Expected names:

```text
per_camera_tsm/test_features_tsm_window_labels_C10095.npy
per_camera_tsm/test_features_tsm_window_labels_C10118.npy
per_camera_tsm/test_features_tsm_window_labels_C10119.npy
per_camera_tsm/test_features_tsm_window_labels_C10390.npy
per_camera_tsm/test_features_tsm_window_labels_C10404.npy
```

Label meaning:

```text
0 = Normal / correct assembly window
1 = Anomaly / mistake or correction window
```

If labels are missing, the script still produces anomaly predictions and timeline outputs, but it cannot calculate evaluation metrics.

## 4. How the Script Works

The script follows this inference pipeline:

```text
TSM + 3D hand pose feature files
        ↓
Detect available camera files
        ↓
Load the matching scaler.pkl for each camera
        ↓
Load the matching autoencoder_best.pth for each camera
        ↓
Compute reconstruction error for every window
        ↓
Normalize each camera score by its own threshold
        ↓
Apply max fusion over available cameras
        ↓
Classify windows as Normal or Anomaly
        ↓
Save CSV, JSON, and PNG output files
```

The per-camera normalized score is computed as:

```text
normalized_score = reconstruction_error / camera_threshold
```

The final max-fusion score is:

```text
fusion_score = max(camera_normalized_scores)
```

The final decision rule is:

```text
fusion_score > 1.0  →  Anomaly
fusion_score <= 1.0 →  Normal
```

## 5. Running the Script

Open PowerShell in the `TestEnvironment` folder.

Activate the virtual environment if needed:

```powershell
.\venv\Scripts\activate
```

Then run:

```powershell
python TestSingleVideoWithFinalAE.py
```

Expected terminal output includes:

```text
DEVICE: cuda
Camera: C10095
Feature shape: (839, 4064)
Detected anomaly windows: ...
Available cameras: ['C10095', 'C10118', 'C10119', 'C10390', 'C10404']
Inference mode: full final max fusion
```

If CUDA is available, the script will use the GPU. Otherwise, it will run on CPU.

## 6. Current Verified Test Example

In the verified test run, the feature files had the following shapes:

```text
test_features_tsm_C10095.npy (839, 4064)
test_features_tsm_C10118.npy (839, 4064)
test_features_tsm_C10119.npy (839, 4064)
test_features_tsm_C10390.npy (839, 4064)
test_features_tsm_C10404.npy (839, 4064)
```

The label files also had matching shapes:

```text
test_features_tsm_window_labels_C10095.npy (839,)
test_features_tsm_window_labels_C10118.npy (839,)
test_features_tsm_window_labels_C10119.npy (839,)
test_features_tsm_window_labels_C10390.npy (839,)
test_features_tsm_window_labels_C10404.npy (839,)
```

The verified full-fusion performance was:

| Metric | Value |
|---|---:|
| Accuracy | 91.06% |
| Precision | 76.95% |
| Recall | 92.49% |
| F1-score | 84.01% |
| AUROC | 95.73% |

Confusion matrix values:

|  | Predicted Normal | Predicted Anomaly |
|---|---:|---:|
| True Normal | 567 | 59 |
| True Anomaly | 16 | 197 |

Interpretation:

- The model correctly detected 197 anomaly windows.
- It missed 16 anomaly windows.
- It correctly classified 567 normal windows.
- It produced 59 false alarm windows.
- The high recall value means the system is strong at catching anomaly windows.

## 7. Generated Outputs

After running the script, outputs are saved in:

```text
single_video_inference_outputs/
```

### 7.1 CSV and JSON outputs

| Output file | Description |
|---|---|
| `C10095_camera_results.csv` | Per-window reconstruction results for C10095 |
| `C10118_camera_results.csv` | Per-window reconstruction results for C10118 |
| `C10119_camera_results.csv` | Per-window reconstruction results for C10119 |
| `C10390_camera_results.csv` | Per-window reconstruction results for C10390 |
| `C10404_camera_results.csv` | Per-window reconstruction results for C10404 |
| `fusion_timeline.csv` | Final fused per-window prediction timeline |
| `fusion_test_results.csv` | Same final fused prediction table for analysis/demo use |
| `detected_intervals.csv` | Consecutive detected anomaly windows grouped as intervals |
| `performance_metrics.csv` | Accuracy, Precision, Recall, F1, AUROC, TP/FP/FN/TN |
| `performance_metrics.json` | Same metrics in JSON format |
| `inference_mode.json` | Records single-camera / partial-fusion / full-fusion mode |

### 7.2 PNG visual outputs

| Output image | What it shows | Recommended use |
|---|---|---|
| `fusion_score_distribution.png` | Normal and anomaly score distributions with threshold | Report / presentation |
| `confusion_matrix.png` | TP, FP, FN, TN counts | Report / presentation |
| `roc_curve.png` | ROC curve and AUROC | Report / presentation |
| `performance_metrics_table.png` | Main metric values in table form | Presentation / poster |
| `fusion_timeline_plot.png` | Final fusion score over test windows | Report / demo explanation |
| `fusion_timeline_detailed.png` | TP, FP, and FN points over test windows | Detailed analysis / appendix |
| `per_camera_detected_anomalies.png` | Number of detected anomaly windows per camera | Multi-view behavior explanation |

Interval bar charts were intentionally removed because they were visually crowded and provided less useful information than the timeline, distribution, ROC, and confusion matrix plots.

## 8. Important Notes

### 8.1 This script does not train a model

This script is only for inference/testing. It does not use train features and does not update the model weights.

For retraining or repeating the final tuning experiment, use:

```text
FinalTSMPerCameraTuning.py
```

### 8.2 This script does not extract features from video

Raw `.mp4` video files cannot be passed directly to this script. Feature extraction must already be completed.

Correct workflow:

```text
Raw video
   ↓
Feature extraction script
   ↓
TSM + hand pose .npy feature files
   ↓
TestSingleVideoWithFinalAE.py
   ↓
Anomaly predictions and visual outputs
```

### 8.3 Feature dimension must be 4064

If a feature file has a different second dimension, the script will stop with an error. This is intentional because the final autoencoder was trained with 4064-dimensional TSM + hand pose vectors.

### 8.4 Camera ID must be known

A feature file must include the camera ID in its filename, such as:

```text
test_features_tsm_C10095.npy
```

The script uses the camera ID to select the correct scaler and autoencoder.

### 8.5 Full final result requires all five cameras

If all five camera files are available, the script runs the final max-fusion system. If fewer cameras are available, it still runs, but the output should be interpreted as single-camera or partial-camera inference, not the complete final system.

## 9. Quick Troubleshooting

### Problem: `Feature klasörü bulunamadı`

Check that this folder exists:

```text
per_camera_tsm/
```

### Problem: `model dosyası yok` or `scaler dosyası yok`

Check that each camera folder contains:

```text
autoencoder_best.pth
scaler.pkl
```

### Problem: feature dimension error

Expected shape:

```text
(number_of_windows, 4064)
```

If the second dimension is not 4064, the feature extraction output does not match the final model.

### Problem: fusion cannot be performed because lengths are different

All available camera feature arrays must have the same number of windows. For example:

```text
C10095: (839, 4064)
C10118: (839, 4064)
```

If one camera has fewer or more windows, the feature extraction/windowing step may not be aligned.

### Problem: metrics are not produced

Metrics require label files. Without labels, the script only generates predictions and timelines.

## 10. Short Explanation for Presentations

This test environment uses the final trained TSM + hand pose per-camera autoencoder models. Each camera feature file is normalized with its own scaler and passed through its own autoencoder. The reconstruction error is divided by that camera's threshold to obtain a normalized anomaly score. Available camera scores are combined with max fusion. If the final fusion score exceeds 1.0, the corresponding window is classified as anomaly. The script saves both numerical outputs and visual evidence such as ROC curve, confusion matrix, score distribution, timeline, and metric table.
