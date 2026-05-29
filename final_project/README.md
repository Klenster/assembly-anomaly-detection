# Assembly Video Anomaly Detection with Per-Camera Autoencoders

This repository contains the autoencoder-based anomaly detection part of an Assembly101 video anomaly detection project. The goal is to detect abnormal or faulty assembly actions from multi-view assembly videos by using pre-extracted video and hand-pose features.

The final system uses **TSM + Hand Pose features**, trains a separate **autoencoder for each camera angle**, and combines camera-level anomaly scores with **decision-level max fusion**.

---

## 1. Project Motivation

Assembly processes are often recorded from multiple camera angles. A mistake may be clearly visible from one camera but less visible from another. Therefore, instead of treating all camera views as one large pooled dataset, this project evaluates each camera angle separately and then combines the anomaly scores at the decision level.

The main idea is:

1. Learn the normal assembly pattern from correct assembly windows.
2. Reconstruct normal feature vectors with low error.
3. Produce higher reconstruction error for abnormal windows.
4. Detect anomalies when the reconstruction error exceeds a threshold.
5. Combine anomaly evidence from multiple camera views.

---

## 2. Dataset and Feature Representation

The project uses feature files extracted from Assembly101 videos. The raw videos are not directly passed into the autoencoder. Instead, the feature extraction stage produces numerical feature vectors from fixed-length video windows.

The final selected feature representation is:

- **TSM visual features**
- **3D hand pose / hand landmark features**
- **Window-based representation**
- **Per-camera feature files**

The final per-camera TSM feature files are expected under:

```text
per_camera_tsm/
├── train_features_correct_tsm_C10095.npy
├── train_features_correct_tsm_C10118.npy
├── train_features_correct_tsm_C10119.npy
├── train_features_correct_tsm_C10390.npy
├── train_features_correct_tsm_C10404.npy
├── test_features_tsm_C10095.npy
├── test_features_tsm_C10118.npy
├── test_features_tsm_C10119.npy
├── test_features_tsm_C10390.npy
├── test_features_tsm_C10404.npy
├── test_features_tsm_window_labels_C10095.npy
├── test_features_tsm_window_labels_C10118.npy
├── test_features_tsm_window_labels_C10119.npy
├── test_features_tsm_window_labels_C10390.npy
└── test_features_tsm_window_labels_C10404.npy
```

In the final setup:

- `correct` windows are treated as normal.
- `mistake` and `correction` windows are treated as anomaly during evaluation.
- The model is trained only on correct/normal windows.

---

## 3. Overall Pipeline

The final anomaly detection pipeline is:

```text
Assembly video
    ↓
Window-based feature extraction
    ↓
TSM + Hand Pose feature vector
    ↓
Separate autoencoder per camera
    ↓
Reconstruction error per camera
    ↓
Threshold comparison per camera
    ↓
Normalized anomaly score per camera
    ↓
Max fusion across cameras
    ↓
Normal / Anomaly decision
```

The final system is **window-based**, not frame-based. In the demo, each prediction is mapped to an approximate frame/time range for visualization.

---

## 4. Why Autoencoder?

The project uses an autoencoder because anomaly data can be limited and diverse. Instead of training a supervised classifier with many examples of every possible error type, the autoencoder learns only the normal assembly pattern.

During testing:

```text
Low reconstruction error  → Normal
High reconstruction error → Anomaly
```

The model does not learn anomaly classes directly. It learns to reconstruct normal feature patterns. Mistake and correction samples are used only for evaluation.

---

## 5. Why Per-Camera Models?

At first, pooled multi-view experiments were tested. In the pooled setup, features from different cameras were placed into a single training pool and one autoencoder tried to learn all camera distributions together.

However, each camera angle has its own visual distribution. A single pooled model can make the normal region too broad and reduce anomaly separation.

Therefore, the final system uses:

```text
C10095  → Autoencoder 1
C10118  → Autoencoder 2
C10119  → Autoencoder 3
C10390  → Autoencoder 4
C10404  → Autoencoder 5
```

At test time, every camera produces its own anomaly score. These scores are normalized by the camera-specific threshold and combined using fusion.

The final selected fusion strategy is:

```text
max fusion
```

This means that if any camera strongly detects an anomaly, the final system marks the window as anomalous.

---

## 6. Experimental Development Path

The project was developed in three experimental stages.

### 6.1 Stage 1 — Baseline Comparison

File:

```text
BaselineAutoencoderExperiments.py
```

Purpose:

- Compare pooled and per-camera autoencoder strategies.
- Compare SlowFast + Hand Pose and TSM + Hand Pose features.
- Test basic fusion methods: mean, max, and top-2 mean.

This stage showed that:

- Pooled models were weaker.
- Per-camera autoencoders performed better.
- TSM + Hand Pose features were more suitable than SlowFast + Hand Pose for the final autoencoder setup.

Main output directory:

```text
BaselineAE_Results/
```

Main output summary:

```text
BaselineAE_Results/baseline_summary.csv
```

---

### 6.2 Stage 2 — Optimized Per-Camera Experiments

File:

```text
OptimizedPerCameraAEExperiments.py
```

Purpose:

- Remove pooled experiments from the main search.
- Focus only on per-camera SlowFast and per-camera TSM data.
- Test different autoencoder sizes and regularization settings.
- Test multiple threshold percentiles and fusion methods.

This stage confirmed that TSM per-camera models were the strongest direction.

Main output directory:

```text
OptimizedPerCameraAE_Results/
```

Main output summary:

```text
OptimizedPerCameraAE_Results/optimized_per_camera_summary.csv
```

---

### 6.3 Stage 3 — Final TSM Per-Camera Tuning

File:

```text
FinalTSMPerCameraTuning.py
```

Purpose:

- Focus only on the selected final data type: `per_camera_tsm`.
- Test final autoencoder variants.
- Test threshold percentiles: 70, 75, 80, 85, 90.
- Test fusion methods: mean, max, top-2 mean.
- Save the best final configuration.

The best final configuration was:

```text
Feature type: TSM + Hand Pose
Model type: Per-camera Autoencoder
Architecture: small_no_dropout
Fusion: max
Threshold percentile: 75
```

Final performance:

| Metric | Value |
|---|---:|
| Accuracy | 91.18% |
| Precision | 77.25% |
| Recall | 92.49% |
| F1-score | 84.19% |
| AUROC | 95.73% |

Final confusion matrix values:

| Value | Meaning | Count |
|---|---|---:|
| TP | Correctly detected anomaly windows | 197 |
| FP | Normal windows incorrectly detected as anomaly | 58 |
| FN | Missed anomaly windows | 16 |
| TN | Correctly detected normal windows | 568 |

Main output directory:

```text
FinalTSMPerCameraAE_Results/
```

Important final files:

```text
FinalTSMPerCameraAE_Results/final_tuning_summary.csv
FinalTSMPerCameraAE_Results/final_best_config.json
FinalTSMPerCameraAE_Results/small_no_dropout/fusion/max_p75/fusion_test_results.csv
FinalTSMPerCameraAE_Results/small_no_dropout/fusion/max_p75/fusion_distribution.png
```

---

## 7. Important Terms and Metrics

### Reconstruction Error

The reconstruction error measures how different the reconstructed feature vector is from the input feature vector.

```text
reconstruction_error = mean((input_feature - reconstructed_feature)^2)
```

A low reconstruction error usually means the sample is similar to the normal training data. A high reconstruction error suggests abnormal behavior.

### Threshold

The threshold is used to decide whether a window is normal or anomalous.

```text
reconstruction_error > threshold → anomaly
reconstruction_error ≤ threshold → normal
```

In this project, thresholds are computed from validation normal reconstruction errors. This prevents the model from being tuned directly on test anomalies.

### Precision

Precision measures how reliable the anomaly predictions are.

```text
Precision = TP / (TP + FP)
```

High precision means the model produces fewer false alarms.

### Recall

Recall measures how many true anomalies are detected.

```text
Recall = TP / (TP + FN)
```

High recall means fewer anomalies are missed.

### F1-score

F1-score balances precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

This is useful when both false alarms and missed anomalies matter.

### AUROC

AUROC measures how well the anomaly score separates normal and anomaly samples across different thresholds. A value close to 1.0 means strong separation. A value around 0.5 means random-level separation.

---

## 8. Demo Application

File:

```text
demo_app.py
```

The Streamlit demo visualizes a single-video anomaly detection result. It does not retrain the model and does not run feature extraction again. It reads a precomputed timeline CSV and displays:

- the input video,
- reconstruction error timeline,
- detected anomaly intervals,
- TP / FP / FN / TN values for the selected video,
- interval-level correctness,
- window-level prediction results.

### Demo Folder

The demo files are expected under:

```text
single_video_poster_outputs_for_demo_app/
```

Expected structure:

```text
single_video_poster_outputs_for_demo_app/
├── 9064-c13a_9064_C10095_timeline.csv
├── video_download_link.txt
└── 9064C10095_rgb.mp4
```

The video file is not included directly in the repository because of file size limitations.

To run the demo correctly:

1. Open:

```text
single_video_poster_outputs_for_demo_app/
```

2. Open the text file containing the video download link.

3. Download the video from the provided link.

4. Place the video in the same folder with this exact name:

```text
9064C10095_rgb.mp4
```

5. Start the demo:

```bash
streamlit run demo_app.py
```

6. In the sidebar:

- The default video path should point to `single_video_poster_outputs_for_demo_app/9064C10095_rgb.mp4`.
- Upload the timeline CSV file from the same folder.
- Keep FPS as `60`.
- Keep window duration as `2.0` seconds.

---

## 9. Installation

Python 3.11 is recommended.

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If CUDA-enabled PyTorch is needed, install PyTorch from the official PyTorch CUDA wheel page according to the local GPU/CUDA setup.

For the environment used in this project, CUDA-enabled PyTorch was used successfully with an NVIDIA GeForce RTX GPU.

---

## 10. Running the Experiments

The scripts should be run from the project root directory, where the feature folders are located.

### Stage 1 — Baseline

```bash
python BaselineAutoencoderExperiments.py
```

Expected main output:

```text
BaselineAE_Results/baseline_summary.csv
```

### Stage 2 — Optimized Per-Camera Experiments

```bash
python OptimizedPerCameraAEExperiments.py
```

Expected main output:

```text
OptimizedPerCameraAE_Results/optimized_per_camera_summary.csv
```

### Stage 3 — Final TSM Per-Camera Tuning

```bash
python FinalTSMPerCameraTuning.py
```

Expected main output:

```text
FinalTSMPerCameraAE_Results/final_tuning_summary.csv
FinalTSMPerCameraAE_Results/final_best_config.json
```

### Demo

```bash
streamlit run demo_app.py
```

---

## 11. Repository Contents

Recommended final repository structure:

```text
project_root/
├── BaselineAutoencoderExperiments.py
├── OptimizedPerCameraAEExperiments.py
├── FinalTSMPerCameraTuning.py
├── demo_app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── FullSequenceTest/
├── Test&Train-MultiplePOV+newHands/
├── TrainSequenced/
├── TSMTrain(FlatveSekans)&Test(sekans)/
├── per_camera_slowfast/
├── per_camera_tsm/
│
├── single_video_poster_outputs_for_demo_app/
│   ├── 9064-c13a_9064_C10095_timeline.csv
│   └── video_download_link.txt
│
├── BaselineAutoencoderExperimentsResults.txt
├── OptimizedPerCameraAEExperimentsResults.txt
├── FinalTSMPerCameraTuningResults.txt
│
├── BaselineAE_Results.zip
├── OptimizedPerCameraAE_Results.zip
└── FinalTSMPerCameraAE_Results.zip
```

If the `.zip` result archives are too large for GitHub, keep them outside the repository and provide download links instead.

---

## 12. Notes and Limitations

- The final model is not a full real-time pipeline.
- Feature extraction is performed before autoencoder testing.
- The demo app visualizes precomputed anomaly scores.
- Detection is window-level rather than exact frame-level.
- The frame index shown in demo graphs is approximate because it is derived from window index, FPS, and window duration.
- The current final model is optimized on a limited Assembly101 subset.
- More videos and camera views could improve robustness.
- Future work may include sequence-level LSTM/Transformer autoencoders and real-time feature extraction.

---

## 13. Summary for Presentation

This project developed a multi-view assembly anomaly detection system using TSM + Hand Pose features and per-camera autoencoders. The system was tested through three experimental stages. First, pooled and per-camera strategies were compared. Then, per-camera SlowFast and TSM models were optimized. Finally, the selected TSM per-camera method was tuned in detail. The final model used a small no-dropout autoencoder with max fusion and achieved 84.19% F1-score and 95.73% AUROC. A Streamlit demo was also created to visualize detected anomaly intervals on a sample video.
