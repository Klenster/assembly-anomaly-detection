import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 0. CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

OUTPUT_DIR = BASE_DIR / "BaselineAE_Results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
THRESHOLD_PERCENTILE = 85

EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("DEVICE:", DEVICE)


# ============================================================
# 1. BASELINE AUTOENCODER
# Same structure for all feature sets
# input_dim -> 256 -> 64 -> 16 -> 64 -> 256 -> input_dim
# ============================================================

class BaselineAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),

            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_npy(path, allow_pickle=False):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    arr = np.load(path, allow_pickle=allow_pickle)
    print(f"Loaded: {path.name} | shape={arr.shape} | dtype={arr.dtype}")
    return arr


def check_features(train_X, test_X, test_y=None):
    print("\n--- Data Check ---")
    print("Train shape:", train_X.shape)
    print("Test shape :", test_X.shape)

    if train_X.shape[1] != test_X.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: train={train_X.shape[1]}, test={test_X.shape[1]}"
        )

    if np.isnan(train_X).any() or np.isinf(train_X).any():
        raise ValueError("Train feature içinde NaN veya Inf var.")

    if np.isnan(test_X).any() or np.isinf(test_X).any():
        raise ValueError("Test feature içinde NaN veya Inf var.")

    print("Train NaN/Inf: OK")
    print("Test NaN/Inf : OK")

    if test_y is not None:
        if len(test_y) != len(test_X):
            raise ValueError(
                f"Test label length mismatch: labels={len(test_y)}, test={len(test_X)}"
            )
        print("Test labels:", np.unique(test_y, return_counts=True))


def train_autoencoder(train_X, experiment_dir):
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_part, val_X = train_test_split(
        train_X,
        test_size=0.25,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_part).astype(np.float32)
    val_scaled = scaler.transform(val_X).astype(np.float32)

    joblib.dump(scaler, experiment_dir / "scaler.pkl")

    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
    val_tensor = torch.tensor(val_scaled, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    input_dim = train_tensor.shape[1]
    model = BaselineAutoencoder(input_dim).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_model_path = experiment_dir / "autoencoder_best.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for (batch,) in train_loader:
            batch = batch.to(DEVICE)

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(batch)

        train_loss = train_loss_sum / len(train_tensor)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)

                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)

                val_loss_sum += loss.item() * len(batch)

        val_loss = val_loss_sum / len(val_tensor)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    val_errors = get_errors(model, val_tensor)

    threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))
    np.save(experiment_dir / "threshold.npy", np.array(threshold))

    print("Best epoch:", best_epoch)
    print("Best val loss:", best_val_loss)
    print(f"Threshold ({THRESHOLD_PERCENTILE}th percentile):", threshold)

    return model, scaler, threshold, val_errors


def get_errors(model, data_tensor):
    loader = DataLoader(
        TensorDataset(data_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    errors = []

    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            reconstructed = model(batch)
            batch_errors = torch.mean((reconstructed - batch) ** 2, dim=1)
            errors.extend(batch_errors.cpu().numpy())

    return np.array(errors)


def compute_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (scores > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = None

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": None if auc is None else float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "flagged_as_anomaly": int(y_pred.sum()),
        "total_test": int(len(y_true)),
    }


def save_error_plot(experiment_dir, train_errors, val_errors, test_errors, test_y, threshold, title):
    plt.figure(figsize=(9, 5))

    plt.hist(train_errors, bins=40, alpha=0.45, label="Train correct")
    plt.hist(val_errors, bins=40, alpha=0.45, label="Validation correct")

    if test_y is not None:
        test_y = np.asarray(test_y).astype(int)
        plt.hist(test_errors[test_y == 0], bins=40, alpha=0.55, label="Test correct")
        plt.hist(test_errors[test_y == 1], bins=40, alpha=0.55, label="Test anomaly")
    else:
        plt.hist(test_errors, bins=40, alpha=0.55, label="Test")

    plt.axvline(threshold, linestyle="--", label=f"Threshold {THRESHOLD_PERCENTILE}%")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()

    plot_path = experiment_dir / "error_distribution.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Plot saved:", plot_path)


def save_test_results(experiment_dir, y_true, scores, threshold):
    y_pred = (scores > threshold).astype(int)

    csv_path = experiment_dir / "test_results.csv"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_id,true_label,reconstruction_error,prediction\n")
        for i in range(len(scores)):
            f.write(f"{i},{int(y_true[i])},{scores[i]:.8f},{int(y_pred[i])}\n")

    print("Results saved:", csv_path)


def run_pooled_experiment(experiment_name, train_path, test_path, label_path):
    print("\n" + "=" * 90)
    print("POOLED EXPERIMENT:", experiment_name)
    print("=" * 90)

    experiment_dir = OUTPUT_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_X = load_npy(train_path)
    test_X = load_npy(test_path)
    test_y = load_npy(label_path).astype(int)

    check_features(train_X, test_X, test_y)

    model, scaler, threshold, val_errors = train_autoencoder(train_X, experiment_dir)

    train_part, _ = train_test_split(
        train_X,
        test_size=0.25,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_scaled = scaler.transform(train_part).astype(np.float32)
    test_scaled = scaler.transform(test_X).astype(np.float32)

    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

    train_errors = get_errors(model, train_tensor)
    test_errors = get_errors(model, test_tensor)

    metrics = compute_metrics(test_y, test_errors, threshold)

    result = {
        "experiment": experiment_name,
        "mode": "pooled",
        "input_dim": int(train_X.shape[1]),
        "train_count": int(len(train_X)),
        "test_count": int(len(test_X)),
        "test_correct_count": int((test_y == 0).sum()),
        "test_anomaly_count": int((test_y == 1).sum()),
        "threshold": float(threshold),
        **metrics,
    }

    save_error_plot(
        experiment_dir,
        train_errors,
        val_errors,
        test_errors,
        test_y,
        threshold,
        title=experiment_name
    )

    save_test_results(experiment_dir, test_y, test_errors, threshold)

    with open(experiment_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("\nMetrics:")
    print(json.dumps(result, indent=4))

    return result


def run_single_camera_experiment(experiment_name, camera, train_path, test_path, label_path):
    print("\n" + "-" * 90)
    print(f"CAMERA EXPERIMENT: {experiment_name} | {camera}")
    print("-" * 90)

    experiment_dir = OUTPUT_DIR / experiment_name / camera
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_X = load_npy(train_path)
    test_X = load_npy(test_path)
    test_y = load_npy(label_path).astype(int)

    check_features(train_X, test_X, test_y)

    model, scaler, threshold, val_errors = train_autoencoder(train_X, experiment_dir)

    train_part, _ = train_test_split(
        train_X,
        test_size=0.25,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_scaled = scaler.transform(train_part).astype(np.float32)
    test_scaled = scaler.transform(test_X).astype(np.float32)

    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

    train_errors = get_errors(model, train_tensor)
    test_errors = get_errors(model, test_tensor)

    metrics = compute_metrics(test_y, test_errors, threshold)

    result = {
        "experiment": experiment_name,
        "mode": "per_camera",
        "camera": camera,
        "input_dim": int(train_X.shape[1]),
        "train_count": int(len(train_X)),
        "test_count": int(len(test_X)),
        "test_correct_count": int((test_y == 0).sum()),
        "test_anomaly_count": int((test_y == 1).sum()),
        "threshold": float(threshold),
        **metrics,
    }

    save_error_plot(
        experiment_dir,
        train_errors,
        val_errors,
        test_errors,
        test_y,
        threshold,
        title=f"{experiment_name} - {camera}"
    )

    save_test_results(experiment_dir, test_y, test_errors, threshold)

    with open(experiment_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    # Normalized score: error / threshold
    normalized_scores = test_errors / threshold

    return result, test_y, normalized_scores


def compute_fusion_metrics(experiment_name, y_true, camera_scores_dict):
    fusion_dir = OUTPUT_DIR / experiment_name / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    cameras = list(camera_scores_dict.keys())
    score_matrix = np.vstack([camera_scores_dict[c] for c in cameras])

    # score_matrix shape: (num_cameras, num_samples)
    mean_scores = score_matrix.mean(axis=0)
    max_scores = score_matrix.max(axis=0)

    if score_matrix.shape[0] >= 2:
        sorted_scores = np.sort(score_matrix, axis=0)
        top2_mean_scores = sorted_scores[-2:, :].mean(axis=0)
    else:
        top2_mean_scores = max_scores

    fusion_methods = {
        "fusion_mean": mean_scores,
        "fusion_max": max_scores,
        "fusion_top2_mean": top2_mean_scores,
    }

    fusion_results = []

    for method_name, scores in fusion_methods.items():
        # Since scores are normalized by camera threshold,
        # threshold 1.0 means "above its camera threshold".
        threshold = 1.0

        metrics = compute_metrics(y_true, scores, threshold)

        result = {
            "experiment": experiment_name,
            "mode": method_name,
            "camera": "ALL",
            "input_dim": None,
            "train_count": None,
            "test_count": int(len(y_true)),
            "test_correct_count": int((y_true == 0).sum()),
            "test_anomaly_count": int((y_true == 1).sum()),
            "threshold": threshold,
            **metrics,
        }

        fusion_results.append(result)

        csv_path = fusion_dir / f"{method_name}_test_results.csv"
        y_pred = (scores > threshold).astype(int)

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("sample_id,true_label,normalized_score,prediction\n")
            for i in range(len(scores)):
                f.write(f"{i},{int(y_true[i])},{scores[i]:.8f},{int(y_pred[i])}\n")

        plt.figure(figsize=(9, 5))
        plt.hist(scores[y_true == 0], bins=40, alpha=0.55, label="Test correct")
        plt.hist(scores[y_true == 1], bins=40, alpha=0.55, label="Test anomaly")
        plt.axvline(threshold, linestyle="--", label="Fusion threshold = 1.0")
        plt.xlabel("Normalized Fusion Score")
        plt.ylabel("Count")
        plt.title(f"{experiment_name} - {method_name}")
        plt.legend()

        plot_path = fusion_dir / f"{method_name}_distribution.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

    with open(fusion_dir / "fusion_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fusion_results, f, indent=4)

    print("\nFusion metrics:")
    print(json.dumps(fusion_results, indent=4))

    return fusion_results


def run_per_camera_experiment(experiment_name, folder, cameras, feature_type):
    print("\n" + "=" * 90)
    print("PER-CAMERA EXPERIMENT:", experiment_name)
    print("=" * 90)

    folder = BASE_DIR / folder

    all_results = []
    camera_scores = {}
    reference_y = None

    for camera in cameras:
        if feature_type == "slowfast":
            train_path = folder / f"train_features_correct_{camera}.npy"
            test_path = folder / f"test_features_{camera}.npy"
            label_path = folder / f"test_features_window_labels_{camera}.npy"
        elif feature_type == "tsm":
            train_path = folder / f"train_features_correct_tsm_{camera}.npy"
            test_path = folder / f"test_features_tsm_{camera}.npy"
            label_path = folder / f"test_features_tsm_window_labels_{camera}.npy"
        else:
            raise ValueError("feature_type must be 'slowfast' or 'tsm'")

        if not train_path.exists():
            print(f"Skipping {camera}: train file not found.")
            continue

        result, y_true, normalized_scores = run_single_camera_experiment(
            experiment_name=experiment_name,
            camera=camera,
            train_path=train_path,
            test_path=test_path,
            label_path=label_path
        )

        all_results.append(result)
        camera_scores[camera] = normalized_scores

        if reference_y is None:
            reference_y = y_true
        else:
            if not np.array_equal(reference_y, y_true):
                raise ValueError(f"Label order mismatch detected for camera {camera}")

    fusion_results = compute_fusion_metrics(
        experiment_name=experiment_name,
        y_true=reference_y,
        camera_scores_dict=camera_scores
    )

    return all_results + fusion_results


# ============================================================
# 3. DEFINE EXPERIMENTS
# ============================================================

SLOWFAST_CAMERAS = [
    "C10095",
    "C10115",
    "C10118",
    "C10119",
    "C10390",
    "C10404",
]

TSM_CAMERAS = [
    "C10095",
    "C10118",
    "C10119",
    "C10390",
    "C10404",
]


# ============================================================
# 4. RUN ALL BASELINE EXPERIMENTS
# ============================================================

all_results = []

# ------------------------------------------------------------
# Experiment 1: SlowFast + Hand Pose / Pooled AE
# ------------------------------------------------------------

all_results.append(
    run_pooled_experiment(
        experiment_name="01_slowfast_handpose_pooled",
        train_path=BASE_DIR / "Test&Train-MultiplePOV+newHands" / "train_features_correct.npy",
        test_path=BASE_DIR / "FullSequenceTest" / "test_features.npy",
        label_path=BASE_DIR / "FullSequenceTest" / "test_window_labels.npy",
    )
)

# ------------------------------------------------------------
# Experiment 2: SlowFast + Hand Pose / Per-camera AE + Fusion
# ------------------------------------------------------------

all_results.extend(
    run_per_camera_experiment(
        experiment_name="02_slowfast_handpose_per_camera",
        folder="per_camera_slowfast",
        cameras=SLOWFAST_CAMERAS,
        feature_type="slowfast"
    )
)

# ------------------------------------------------------------
# Experiment 3: TSM + Hand Pose / Pooled AE
# ------------------------------------------------------------

all_results.append(
    run_pooled_experiment(
        experiment_name="03_tsm_handpose_pooled",
        train_path=BASE_DIR / "TSMTrain(FlatveSekasn)&Test(sekans)" / "train_features_correct_tsm.npy",
        test_path=BASE_DIR / "TSMTrain(FlatveSekasn)&Test(sekans)" / "test_features_tsm.npy",
        label_path=BASE_DIR / "TSMTrain(FlatveSekasn)&Test(sekans)" / "test_window_labels_tsm.npy",
    )
)

# ------------------------------------------------------------
# Experiment 4: TSM + Hand Pose / Per-camera AE + Fusion
# ------------------------------------------------------------

all_results.extend(
    run_per_camera_experiment(
        experiment_name="04_tsm_handpose_per_camera",
        folder="per_camera_tsm",
        cameras=TSM_CAMERAS,
        feature_type="tsm"
    )
)


# ============================================================
# 5. SAVE FINAL SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "baseline_summary.csv"

keys = [
    "experiment",
    "mode",
    "camera",
    "input_dim",
    "train_count",
    "test_count",
    "test_correct_count",
    "test_anomaly_count",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "false_positive_rate",
    "false_negative_rate",
    "flagged_as_anomaly",
    "total_test",
]

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(",".join(keys) + "\n")

    for result in all_results:
        row = []
        for key in keys:
            value = result.get(key, "")
            row.append("" if value is None else str(value))
        f.write(",".join(row) + "\n")

print("\n" + "=" * 90)
print("ALL BASELINE EXPERIMENTS FINISHED")
print("Summary saved:", summary_path)
print("=" * 90)

print("\nBest results by F1:")
sorted_results = sorted(all_results, key=lambda x: x.get("f1", 0), reverse=True)

for r in sorted_results[:10]:
    print(
        f"{r.get('experiment')} | "
        f"{r.get('mode')} | "
        f"camera={r.get('camera', '')} | "
        f"F1={r.get('f1'):.4f} | "
        f"Precision={r.get('precision'):.4f} | "
        f"Recall={r.get('recall'):.4f} | "
        f"AUC={r.get('auc')}"
    )