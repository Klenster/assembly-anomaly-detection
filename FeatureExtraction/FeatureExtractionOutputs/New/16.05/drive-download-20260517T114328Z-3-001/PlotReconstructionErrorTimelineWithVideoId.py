import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ============================================================
# 1. AYARLAR
# ============================================================

BASE_DIR = Path.cwd()

SEQ_DIR = BASE_DIR / "TSMTrain(FlatveSekasn)&Test(sekans)"
RESULT_DIR = BASE_DIR / "FinalTSMPerCameraAE_Results"

CAMERA = "C10095"
MODEL_NAME = "small_no_dropout"
THRESHOLD_PERCENTILE = 75

# Approximate frame axis için:
FPS = 60
WINDOW_DURATION_SEC = 2.0

OUTPUT_DIR = RESULT_DIR / "single_video_poster_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


# ============================================================
# 2. FINAL MODEL CLASS
# !!! BURAYI kendi final autoencoder class'ın ile eşleştir !!!
# ============================================================

class SmallNoDropoutAE(nn.Module):
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
# 3. LABEL DÖNÜŞTÜRÜCÜ
# correct/normal = 0
# anomaly/mistake/correction = 1
# ============================================================

def to_binary_labels(arr):
    arr = np.array(arr, dtype=object)
    out = []

    for x in arr:
        if isinstance(x, (int, np.integer, float, np.floating)):
            out.append(int(x))
        else:
            s = str(x).strip().lower()
            if s in ["correct", "normal", "0"]:
                out.append(0)
            elif s in ["mistake", "anomaly", "correction", "1"]:
                out.append(1)
            else:
                raise ValueError(f"Bilinmeyen label: {x}")

    return np.array(out, dtype=int)


# ============================================================
# 4. VERİLERİ YÜKLE
# ============================================================

test_sequences = np.load(SEQ_DIR / "test_sequences_tsm.npy", allow_pickle=True)
test_ids = np.load(SEQ_DIR / "test_sequence_ids_tsm.npy", allow_pickle=True)
test_window_labels = np.load(SEQ_DIR / "test_sequence_window_labels_tsm.npy", allow_pickle=True)

print("Total sequence count:", len(test_sequences))
print("Total id count:", len(test_ids))
print("Total label sequence count:", len(test_window_labels))


# ============================================================
# 5. MODEL / SCALER / THRESHOLD YÜKLE
# ============================================================

# Burada final sonuçların olduğu yol varsayıldı:
# FinalTSMPerCameraAE_Results/small_no_dropout/C10095/autoencoder_best.pth
# FinalTSMPerCameraAE_Results/small_no_dropout/C10095/scaler.pkl

model_path = RESULT_DIR / MODEL_NAME / CAMERA / "autoencoder_best.pth"
scaler_path = RESULT_DIR / MODEL_NAME / CAMERA / "scaler.pkl"
summary_path = RESULT_DIR / "final_tuning_summary.csv"

if not model_path.exists():
    raise FileNotFoundError(f"Model bulunamadı: {model_path}")

if not scaler_path.exists():
    raise FileNotFoundError(f"Scaler bulunamadı: {scaler_path}")

if not summary_path.exists():
    raise FileNotFoundError(f"Summary bulunamadı: {summary_path}")

scaler = joblib.load(scaler_path)
summary_df = pd.read_csv(summary_path)

row = summary_df[
    (summary_df["model_name"] == MODEL_NAME) &
    (summary_df["camera"] == CAMERA) &
    (summary_df["threshold_percentile"] == THRESHOLD_PERCENTILE)
]

if len(row) == 0:
    raise ValueError("Threshold bilgisi final_tuning_summary.csv içinde bulunamadı.")

threshold = float(row.iloc[0]["threshold"])
print("Threshold:", threshold)


# ============================================================
# 6. SADECE C10095 SEQUENCE'LARINI DEĞERLENDİR
# ============================================================

camera_indices = [i for i, sid in enumerate(test_ids) if str(sid).endswith(CAMERA)]

print(f"{CAMERA} için sequence sayısı:", len(camera_indices))

if len(camera_indices) == 0:
    raise ValueError(f"{CAMERA} için sequence bulunamadı.")

# input dim'i ilk sequence'tan alalım
first_seq = np.array(test_sequences[camera_indices[0]], dtype=np.float32)
input_dim = first_seq.shape[1]

model = SmallNoDropoutAE(input_dim=input_dim, dropout=0.0).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

all_results = []
sequence_outputs = {}

for idx in camera_indices:
    seq_id = str(test_ids[idx])
    X_seq = np.array(test_sequences[idx], dtype=np.float32)
    y_seq = to_binary_labels(test_window_labels[idx])

    X_scaled = scaler.transform(X_seq).astype(np.float32)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((recon - X_tensor) ** 2, dim=1).cpu().numpy()

    y_pred = (errors > threshold).astype(int)

    acc = accuracy_score(y_seq, y_pred)
    prec = precision_score(y_seq, y_pred, zero_division=0)
    rec = recall_score(y_seq, y_pred, zero_division=0)
    f1 = f1_score(y_seq, y_pred, zero_division=0)

    if len(np.unique(y_seq)) > 1:
        auroc = roc_auc_score(y_seq, errors)
    else:
        auroc = np.nan

    anomaly_count = int((y_seq == 1).sum())
    predicted_count = int((y_pred == 1).sum())

    all_results.append({
        "sequence_id": seq_id,
        "num_windows": len(X_seq),
        "anomaly_windows": anomaly_count,
        "predicted_anomaly_windows": predicted_count,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc
    })

    sequence_outputs[seq_id] = {
        "X_seq": X_seq,
        "y_true": y_seq,
        "errors": errors,
        "y_pred": y_pred
    }

results_df = pd.DataFrame(all_results)

# en iyi sequence seçimi:
# önce F1, sonra recall, sonra AUROC, sonra accuracy
results_df = results_df.sort_values(
    by=["f1", "recall", "auroc", "accuracy"],
    ascending=False
).reset_index(drop=True)

csv_results = OUTPUT_DIR / f"{CAMERA}_sequence_metrics.csv"
results_df.to_csv(csv_results, index=False)

print("\n=== Sequence Metrics ===")
print(results_df)
print("\nCSV saved:", csv_results)


# ============================================================
# 7. EN İYİ VIDEOYU SEÇ
# ============================================================

best_seq_id = results_df.iloc[0]["sequence_id"]
print("\nBest sequence selected:", best_seq_id)

best_data = sequence_outputs[best_seq_id]
y_true = best_data["y_true"]
errors = best_data["errors"]
y_pred = best_data["y_pred"]


# ============================================================
# 8. WINDOW INDEX ve APPROX FRAME AXIS OLUŞTUR
# ============================================================

window_idx = np.arange(len(errors))
approx_frame = window_idx * WINDOW_DURATION_SEC * FPS

plot_df = pd.DataFrame({
    "window_index": window_idx,
    "approx_frame": approx_frame.astype(int),
    "true_label": y_true,
    "reconstruction_error": errors,
    "prediction": y_pred
})

csv_timeline = OUTPUT_DIR / f"{best_seq_id}_timeline.csv"
plot_df.to_csv(csv_timeline, index=False)

print("Timeline CSV saved:", csv_timeline)


# ============================================================
# 9. POSTER İÇİN GRAFİK
# ============================================================

plt.figure(figsize=(14, 5))

# ana çizgi
plt.plot(
    approx_frame,
    errors,
    linewidth=2,
    label="Reconstruction Error"
)

# threshold
plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold:.3f}"
)

# gerçek anomaly window'lar
true_anom_idx = np.where(y_true == 1)[0]
plt.scatter(
    approx_frame[true_anom_idx],
    errors[true_anom_idx],
    s=45,
    marker="o",
    label="Ground Truth Anomaly"
)

# modelin anomaly dediği yerler
pred_anom_idx = np.where(y_pred == 1)[0]
plt.scatter(
    approx_frame[pred_anom_idx],
    errors[pred_anom_idx],
    s=65,
    marker="x",
    label="Detected Anomaly"
)

plt.xlabel("Approximate Frame Index")
plt.ylabel("Reconstruction Error")
plt.title(f"Single-Video Reconstruction Error Timeline ({best_seq_id})", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

png_path = OUTPUT_DIR / f"{best_seq_id}_timeline_plot.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()

print("Plot saved:", png_path)


# ============================================================
# 10. EKSTRA: GRAFİKTE SADECE ALGILANAN ANOMALİLER VURGULU
# ============================================================

plt.figure(figsize=(14, 4.5))

plt.plot(
    approx_frame,
    errors,
    linewidth=2
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="Anomaly Threshold"
)

plt.scatter(
    approx_frame[pred_anom_idx],
    errors[pred_anom_idx],
    s=75,
    marker="x",
    label="Detected Anomaly"
)

plt.xlabel("Approximate Frame Index")
plt.ylabel("Reconstruction Error")
plt.title(f"Detected Anomaly Moments Over Time ({best_seq_id})", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

poster_png = OUTPUT_DIR / f"{best_seq_id}_poster_plot.png"
plt.savefig(poster_png, dpi=300, bbox_inches="tight")
plt.show()

print("Poster plot saved:", poster_png)