from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


# ============================================================
# 0. CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_DIR = BASE_DIR / "per_camera_tsm"
MODEL_ROOT = BASE_DIR / "FinalTSMPerCameraAE_Results"
MODEL_NAME = "small_no_dropout"
SUMMARY_PATH = MODEL_ROOT / "final_tuning_summary.csv"

OUTPUT_DIR = BASE_DIR / "single_video_inference_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CAMERAS = [
    "C10095",
    "C10118",
    "C10119",
    "C10390",
    "C10404",
]

THRESHOLD_PERCENTILE = 75

FPS = 60
WINDOW_DURATION_SEC = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("BASE_DIR:", BASE_DIR)
print("FEATURE_DIR:", FEATURE_DIR)
print("MODEL_ROOT:", MODEL_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("DEVICE:", DEVICE)


# ============================================================
# 1. FINAL AUTOENCODER MODEL
# Must match the final trained model:
# 4064 -> 256 -> 64 -> 16 -> 64 -> 256 -> 4064
# Dropout layers are kept for state_dict compatibility.
# dropout=0.0 means they do not affect inference.
# ============================================================

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),

            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_threshold(summary_df, camera):
    """
    Reads the selected camera threshold from final_tuning_summary.csv.
    Expected final setup:
    model_name = small_no_dropout
    camera = C10095, ...
    threshold_percentile = 75
    """

    df = summary_df.copy()

    camera_col_candidates = ["camera", "Camera"]
    threshold_col_candidates = ["threshold", "Threshold"]
    model_col_candidates = ["model_name", "model", "Model"]
    percentile_col_candidates = [
        "threshold_percentile",
        "percentile",
        "p",
        "Threshold Percentile",
    ]

    camera_col = next((c for c in camera_col_candidates if c in df.columns), None)
    threshold_col = next((c for c in threshold_col_candidates if c in df.columns), None)
    model_col = next((c for c in model_col_candidates if c in df.columns), None)
    percentile_col = next((c for c in percentile_col_candidates if c in df.columns), None)

    if camera_col is None:
        raise ValueError("final_tuning_summary.csv içinde camera sütunu bulunamadı.")

    if threshold_col is None:
        raise ValueError("final_tuning_summary.csv içinde threshold sütunu bulunamadı.")

    rows = df[df[camera_col].astype(str) == camera]

    if model_col is not None:
        rows = rows[rows[model_col].astype(str) == MODEL_NAME]

    if percentile_col is not None:
        rows = rows[rows[percentile_col].astype(float) == float(THRESHOLD_PERCENTILE)]

    if len(rows) == 0:
        raise ValueError(
            f"{camera} için threshold bulunamadı. "
            f"CSV sütunlarını ve p={THRESHOLD_PERCENTILE} satırlarını kontrol et."
        )

    threshold = float(rows.iloc[0][threshold_col])
    return threshold


def compute_reconstruction_errors(model, X_scaled):
    tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor)
        errors = torch.mean((reconstructed - tensor) ** 2, dim=1).cpu().numpy()

    return errors


def load_label_for_camera(camera):
    label_path = FEATURE_DIR / f"test_features_tsm_window_labels_{camera}.npy"

    if label_path.exists():
        labels = np.load(label_path, allow_pickle=True).astype(int)
        return labels

    return None


def save_detected_intervals(timeline_df):
    anomaly_df = timeline_df[timeline_df["prediction"] == 1]

    if anomaly_df.empty:
        intervals = pd.DataFrame(columns=[
            "start_window",
            "end_window",
            "start_time_sec",
            "end_time_sec",
            "approx_start_frame",
            "approx_end_frame",
            "max_score",
        ])
        intervals.to_csv(OUTPUT_DIR / "detected_intervals.csv", index=False)
        return intervals

    windows = anomaly_df["window_index"].tolist()

    groups = []
    start = windows[0]
    prev = windows[0]

    for w in windows[1:]:
        if w == prev + 1:
            prev = w
        else:
            groups.append((start, prev))
            start = w
            prev = w

    groups.append((start, prev))

    rows = []
    for s, e in groups:
        part = timeline_df[
            (timeline_df["window_index"] >= s) &
            (timeline_df["window_index"] <= e)
        ]

        start_time = s * WINDOW_DURATION_SEC
        end_time = (e + 1) * WINDOW_DURATION_SEC

        row = {
            "start_window": s,
            "end_window": e,
            "start_time_sec": round(start_time, 2),
            "end_time_sec": round(end_time, 2),
            "approx_start_frame": int(start_time * FPS),
            "approx_end_frame": int(end_time * FPS),
            "max_score": round(float(part["fusion_score"].max()), 6),
        }

        if "true_label" in timeline_df.columns:
            row["true_anomaly_windows"] = int((part["true_label"] == 1).sum())
            row["false_alarm_windows"] = int(
                ((part["true_label"] == 0) & (part["prediction"] == 1)).sum()
            )
            row["correct_detected_windows"] = int(
                ((part["true_label"] == 1) & (part["prediction"] == 1)).sum()
            )

        rows.append(row)

    intervals = pd.DataFrame(rows)
    intervals.to_csv(OUTPUT_DIR / "detected_intervals.csv", index=False)
    return intervals


def plot_timeline(timeline_df):
    plt.figure(figsize=(14, 5))

    x = timeline_df["window_index"]
    y = timeline_df["fusion_score"]

    plt.plot(x, y, linewidth=2, label="Normalized Fusion Score")

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=2,
        label="Anomaly Threshold = 1.0"
    )

    detected = timeline_df[timeline_df["prediction"] == 1]
    plt.scatter(
        detected["window_index"],
        detected["fusion_score"],
        marker="x",
        s=60,
        label="Detected Anomaly"
    )

    if "true_label" in timeline_df.columns:
        true_anom = timeline_df[timeline_df["true_label"] == 1]
        plt.scatter(
            true_anom["window_index"],
            true_anom["fusion_score"],
            marker="o",
            s=30,
            alpha=0.6,
            label="Ground Truth Anomaly"
        )

    plt.xlabel("Window Index")
    plt.ylabel("Normalized Anomaly Score")
    plt.title("Final AE Inference Timeline over Test Windows")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / "fusion_timeline_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Timeline plot saved:", out_path)


def save_metrics(timeline_df):
    if "true_label" not in timeline_df.columns:
        print("Label bulunamadı. Metrikler hesaplanmadı.")
        return None

    y_true = timeline_df["true_label"].astype(int).values
    y_pred = timeline_df["prediction"].astype(int).values
    scores = timeline_df["fusion_score"].astype(float).values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auroc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_windows": int(len(y_true)),
        "detected_anomaly_windows": int(y_pred.sum()),
        "true_anomaly_windows": int(y_true.sum()),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUTPUT_DIR / "performance_metrics.csv", index=False)

    with open(OUTPUT_DIR / "performance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n=== PERFORMANCE METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics

def plot_confusion_matrix_png(timeline_df):
    if "true_label" not in timeline_df.columns:
        return

    y_true = timeline_df["true_label"].astype(int).values
    y_pred = timeline_df["prediction"].astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold"
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion matrix saved:", out_path)


def plot_roc_curve_png(timeline_df):
    if "true_label" not in timeline_df.columns:
        return

    y_true = timeline_df["true_label"].astype(int).values
    scores = timeline_df["fusion_score"].astype(float).values

    if len(np.unique(y_true)) < 2:
        print("ROC curve oluşturulamadı: true_label tek sınıf içeriyor.")
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random Classifier")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve - Final Fusion Scores", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / "roc_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("ROC curve saved:", out_path)


def plot_performance_metrics_table_png(metrics):
    if metrics is None:
        return

    display_metrics = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1-score", metrics["f1_score"]),
        ("AUROC", metrics["auroc"]),
    ]

    table_data = [
        [name, f"{value * 100:.2f}%" if not np.isnan(value) else "N/A"]
        for name, value in display_metrics
    ]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.2, 1.7)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_text_props(fontweight="normal")

    ax.set_title("Performance Metrics", fontsize=16, fontweight="bold", pad=12)

    out_path = OUTPUT_DIR / "performance_metrics_table.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Performance metrics table saved:", out_path)


def plot_fusion_score_distribution(timeline_df):
    plt.figure(figsize=(9, 5))

    if "true_label" in timeline_df.columns:
        normal = timeline_df[timeline_df["true_label"] == 0]["fusion_score"]
        anomaly = timeline_df[timeline_df["true_label"] == 1]["fusion_score"]

        plt.hist(normal, bins=35, alpha=0.65, label="Ground Truth Normal")
        plt.hist(anomaly, bins=35, alpha=0.65, label="Ground Truth Anomaly")
    else:
        plt.hist(timeline_df["fusion_score"], bins=35, alpha=0.75, label="Fusion Score")

    plt.axvline(1.0, linestyle="--", linewidth=2, label="Threshold = 1.0")

    plt.xlabel("Normalized Fusion Score")
    plt.ylabel("Count")
    plt.title("Fusion Score Distribution", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / "fusion_score_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Fusion score distribution saved:", out_path)


def plot_per_camera_comparison(camera_results, available_cameras):
    rows = []

    for camera in available_cameras:
        scores = camera_results[camera]["normalized_scores"]
        preds = camera_results[camera]["prediction"]
        errors = camera_results[camera]["errors"]
        threshold = camera_results[camera]["threshold"]

        rows.append({
            "camera": camera,
            "detected_anomaly_windows": int(preds.sum()),
            "mean_normalized_score": float(np.mean(scores)),
            "max_normalized_score": float(np.max(scores)),
            "mean_reconstruction_error": float(np.mean(errors)),
            "threshold": float(threshold),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "per_camera_comparison.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(df["camera"], df["detected_anomaly_windows"])
    plt.xlabel("Camera")
    plt.ylabel("Detected Anomaly Windows")
    plt.title("Per-Camera Detected Anomaly Count", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "per_camera_detected_anomalies.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Per-camera comparison saved:", out_path)


def plot_detailed_timeline(timeline_df):
    plt.figure(figsize=(14, 5.5))

    x = timeline_df["window_index"]
    y = timeline_df["fusion_score"]

    plt.plot(x, y, linewidth=2, label="Normalized Fusion Score")
    plt.axhline(1.0, linestyle="--", linewidth=2, label="Threshold = 1.0")

    if "true_label" in timeline_df.columns:
        tp = timeline_df[
            (timeline_df["true_label"] == 1) &
            (timeline_df["prediction"] == 1)
        ]

        fp = timeline_df[
            (timeline_df["true_label"] == 0) &
            (timeline_df["prediction"] == 1)
        ]

        fn = timeline_df[
            (timeline_df["true_label"] == 1) &
            (timeline_df["prediction"] == 0)
        ]

        plt.scatter(
            tp["window_index"],
            tp["fusion_score"],
            marker="o",
            s=45,
            label="TP - Correct Anomaly"
        )

        plt.scatter(
            fp["window_index"],
            fp["fusion_score"],
            marker="x",
            s=65,
            label="FP - False Alarm"
        )

        plt.scatter(
            fn["window_index"],
            fn["fusion_score"],
            marker="s",
            s=45,
            label="FN - Missed Anomaly"
        )

    else:
        detected = timeline_df[timeline_df["prediction"] == 1]
        plt.scatter(
            detected["window_index"],
            detected["fusion_score"],
            marker="x",
            s=65,
            label="Detected Anomaly"
        )

    plt.xlabel("Window Index")
    plt.ylabel("Normalized Anomaly Score")
    plt.title("Detailed Final AE Timeline over Test Windows")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / "fusion_timeline_detailed.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Detailed timeline saved:", out_path)


# ============================================================
# 3. MAIN INFERENCE
# ============================================================

if not FEATURE_DIR.exists():
    raise FileNotFoundError(f"Feature klasörü bulunamadı: {FEATURE_DIR}")

if not MODEL_ROOT.exists():
    raise FileNotFoundError(f"Model klasörü bulunamadı: {MODEL_ROOT}")

if not SUMMARY_PATH.exists():
    raise FileNotFoundError(f"final_tuning_summary.csv bulunamadı: {SUMMARY_PATH}")

summary_df = pd.read_csv(SUMMARY_PATH)

camera_results = {}
label_reference = None
available_cameras = []

for camera in CAMERAS:
    feature_path = FEATURE_DIR / f"test_features_tsm_{camera}.npy"
    model_path = MODEL_ROOT / MODEL_NAME / camera / "autoencoder_best.pth"
    scaler_path = MODEL_ROOT / MODEL_NAME / camera / "scaler.pkl"

    if not feature_path.exists():
        print(f"Skipping {camera}: feature dosyası yok.")
        continue

    if not model_path.exists():
        print(f"Skipping {camera}: model dosyası yok.")
        continue

    if not scaler_path.exists():
        print(f"Skipping {camera}: scaler dosyası yok.")
        continue

    print("\n" + "=" * 80)
    print("Camera:", camera)
    print("Feature:", feature_path.name)

    X = np.load(feature_path, allow_pickle=True).astype(np.float32)

    if X.ndim != 2:
        raise ValueError(f"{feature_path.name} 2 boyutlu olmalı. Gelen shape: {X.shape}")

    if X.shape[1] != 4064:
        raise ValueError(
            f"{feature_path.name} feature boyutu 4064 olmalı. Gelen shape: {X.shape}"
        )

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X).astype(np.float32)

    input_dim = X_scaled.shape[1]
    model = SimpleAutoencoder(input_dim=input_dim, dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    threshold = load_threshold(summary_df, camera)
    errors = compute_reconstruction_errors(model, X_scaled)
    normalized_scores = errors / threshold
    camera_prediction = (normalized_scores > 1.0).astype(int)

    labels = load_label_for_camera(camera)

    if labels is not None:
        if len(labels) != len(X):
            raise ValueError(
                f"{camera} label uzunluğu feature uzunluğu ile uyuşmuyor: "
                f"labels={len(labels)}, features={len(X)}"
            )

        if label_reference is None:
            label_reference = labels
        else:
            if not np.array_equal(label_reference, labels):
                print(
                    f"Warning: {camera} label dosyası önceki kamera label'ı ile aynı değil. "
                    "Fusion metriğinde ilk label referansı kullanılacak."
                )

    camera_df = pd.DataFrame({
        "window_index": np.arange(len(X)),
        "reconstruction_error": errors,
        "camera_threshold": threshold,
        "normalized_score": normalized_scores,
        "camera_prediction": camera_prediction,
    })

    if labels is not None:
        camera_df["true_label"] = labels

    camera_csv = OUTPUT_DIR / f"{camera}_camera_results.csv"
    camera_df.to_csv(camera_csv, index=False)

    camera_results[camera] = {
        "errors": errors,
        "threshold": threshold,
        "normalized_scores": normalized_scores,
        "prediction": camera_prediction,
    }

    available_cameras.append(camera)

    print("Feature shape:", X.shape)
    print("Threshold:", threshold)
    print("Detected anomaly windows:", int(camera_prediction.sum()))
    print("Saved:", camera_csv)


if len(available_cameras) == 0:
    raise RuntimeError("Hiçbir kamera için geçerli feature/model/scaler bulunamadı.")

print("\nAvailable cameras:", available_cameras)

# ============================================================
# 4. FUSION / SINGLE-CAMERA DECISION
# ============================================================

score_matrix = np.vstack([
    camera_results[camera]["normalized_scores"]
    for camera in available_cameras
])

# Check same number of windows
lengths = [len(camera_results[c]["normalized_scores"]) for c in available_cameras]
if len(set(lengths)) != 1:
    raise ValueError(
        f"Kamera feature uzunlukları eşit değil. Fusion yapılamaz. Lengths: {dict(zip(available_cameras, lengths))}"
    )

fusion_score = score_matrix.max(axis=0)
prediction = (fusion_score > 1.0).astype(int)

num_windows = len(fusion_score)
window_index = np.arange(num_windows)
approx_time_sec = window_index * WINDOW_DURATION_SEC
approx_frame = approx_time_sec * FPS

timeline_df = pd.DataFrame({
    "window_index": window_index,
    "approx_time_sec": approx_time_sec,
    "approx_frame": approx_frame.astype(int),
})

for camera in available_cameras:
    timeline_df[f"{camera}_score"] = camera_results[camera]["normalized_scores"]

timeline_df["fusion_score"] = fusion_score
timeline_df["prediction"] = prediction

if label_reference is not None:
    timeline_df["true_label"] = label_reference.astype(int)

timeline_csv = OUTPUT_DIR / "fusion_timeline.csv"
timeline_df.to_csv(timeline_csv, index=False)

fusion_csv = OUTPUT_DIR / "fusion_test_results.csv"
timeline_df.to_csv(fusion_csv, index=False)

print("\nFusion timeline saved:", timeline_csv)
print("Fusion test results saved:", fusion_csv)

# Inference mode info
if len(available_cameras) == 1:
    mode = "single-camera inference"
elif len(available_cameras) < len(CAMERAS):
    mode = "partial-camera max fusion"
else:
    mode = "full final max fusion"

mode_info = {
    "mode": mode,
    "available_cameras": available_cameras,
    "num_windows": int(num_windows),
    "threshold_rule": "normalized_score > 1.0",
    "fusion_method": "max",
}

with open(OUTPUT_DIR / "inference_mode.json", "w", encoding="utf-8") as f:
    json.dump(mode_info, f, indent=4)

print("\nInference mode:", mode)
print("Detected anomaly windows:", int(prediction.sum()))

# ============================================================
# 5. OUTPUTS
# ============================================================

intervals = save_detected_intervals(timeline_df)

# Existing basic timeline
plot_timeline(timeline_df)

# Metrics
metrics = save_metrics(timeline_df)

# Extra visual outputs
plot_confusion_matrix_png(timeline_df)
plot_roc_curve_png(timeline_df)
plot_performance_metrics_table_png(metrics)
plot_fusion_score_distribution(timeline_df)
plot_per_camera_comparison(camera_results, available_cameras)
plot_detailed_timeline(timeline_df)

print("\nDetected intervals saved:", OUTPUT_DIR / "detected_intervals.csv")
print("All outputs saved under:", OUTPUT_DIR)