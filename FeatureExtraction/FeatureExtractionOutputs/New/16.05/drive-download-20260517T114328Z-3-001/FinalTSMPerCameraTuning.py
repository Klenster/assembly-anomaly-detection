import json
import random
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
FEATURE_DIR = BASE_DIR / "per_camera_tsm"

OUTPUT_DIR = BASE_DIR / "FinalTSMPerCameraAE_Results"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 160
LEARNING_RATE = 1e-3
PATIENCE = 25

THRESHOLD_PERCENTILES = [70, 75, 80, 85, 90]
FUSION_METHODS = ["mean", "max", "top2_mean"]

CAMERAS = [
    "C10095",
    "C10118",
    "C10119",
    "C10390",
    "C10404",
]

print("BASE_DIR:", BASE_DIR)
print("FEATURE_DIR:", FEATURE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("DEVICE:", DEVICE)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(RANDOM_STATE)


# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

class SimpleAutoencoder(nn.Module):
    """
    Kod 1 baseline modeline yakın yapı.
    Dropout sınırlı kullanılır.
    """
    def __init__(self, input_dim, dropout=0.2):
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


class FlexibleAutoencoder(nn.Module):
    """
    Daha kontrollü medium modeller için.
    Dropout her layer'a değil, isteğe bağlı olarak düzenli uygulanır.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.1):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


MODEL_CONFIGS = [
    {
        "model_name": "baseline_small_dropout02",
        "model_type": "simple",
        "dropout": 0.2,
        "weight_decay": 1e-4,
    },
    {
        "model_name": "small_dropout01",
        "model_type": "simple",
        "dropout": 0.1,
        "weight_decay": 1e-4,
    },
    {
        "model_name": "small_no_dropout",
        "model_type": "simple",
        "dropout": 0.0,
        "weight_decay": 1e-4,
    },
    {
        "model_name": "medium_dropout01",
        "model_type": "flexible",
        "hidden_dims": [512, 128],
        "latent_dim": 32,
        "dropout": 0.1,
        "weight_decay": 1e-4,
    },
    {
        "model_name": "medium_dropout015",
        "model_type": "flexible",
        "hidden_dims": [512, 128],
        "latent_dim": 32,
        "dropout": 0.15,
        "weight_decay": 1e-4,
    },
]


# ============================================================
# 3. HELPERS
# ============================================================

def load_npy(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    arr = np.load(path)
    print(f"Loaded: {path.name} | shape={arr.shape} | dtype={arr.dtype}")
    return arr


def build_model(input_dim, cfg):
    if cfg["model_type"] == "simple":
        return SimpleAutoencoder(
            input_dim=input_dim,
            dropout=cfg["dropout"],
        )

    if cfg["model_type"] == "flexible":
        return FlexibleAutoencoder(
            input_dim=input_dim,
            hidden_dims=cfg["hidden_dims"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
        )

    raise ValueError(f"Unknown model_type: {cfg['model_type']}")


def get_errors(model, data_tensor):
    loader = DataLoader(
        TensorDataset(data_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
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

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

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


def train_one_camera(train_X, cfg, camera_dir):
    set_seed(RANDOM_STATE)

    camera_dir.mkdir(parents=True, exist_ok=True)

    train_part, val_X = train_test_split(
        train_X,
        test_size=0.25,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_part).astype(np.float32)
    val_scaled = scaler.transform(val_X).astype(np.float32)

    joblib.dump(scaler, camera_dir / "scaler.pkl")

    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
    val_tensor = torch.tensor(val_scaled, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_dim = train_tensor.shape[1]

    model = build_model(input_dim, cfg).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=cfg["weight_decay"],
    )

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_model_path = camera_dir / "autoencoder_best.pth"

    train_history = []
    val_history = []

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

        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if epoch % 20 == 0:
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

    # Save training curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_history, label="Train Loss")
    plt.plot(val_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve - {camera_dir.name}")
    plt.legend()
    plt.savefig(camera_dir / "training_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "model": model,
        "scaler": scaler,
        "train_part": train_part,
        "val_X": val_X,
        "val_errors": val_errors,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def plot_error_distribution(path, train_errors, val_errors, test_errors, test_y, threshold, title):
    plt.figure(figsize=(9, 5))

    plt.hist(train_errors, bins=40, alpha=0.45, label="Train correct")
    plt.hist(val_errors, bins=40, alpha=0.45, label="Validation correct")
    plt.hist(test_errors[test_y == 0], bins=40, alpha=0.55, label="Test correct")
    plt.hist(test_errors[test_y == 1], bins=40, alpha=0.55, label="Test anomaly")

    plt.axvline(threshold, linestyle="--", label="Threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_csv(path, rows, keys):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")

        for row in rows:
            values = []
            for key in keys:
                value = row.get(key, "")
                value = "" if value is None else str(value)
                value = value.replace(",", ";")
                values.append(value)
            f.write(",".join(values) + "\n")


# ============================================================
# 4. FINAL TUNING LOOP
# ============================================================

all_results = []

for cfg in MODEL_CONFIGS:
    model_name = cfg["model_name"]

    print("\n" + "=" * 100)
    print("FINAL TUNING MODEL:", model_name)
    print("=" * 100)

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    camera_scores_by_percentile = {p: {} for p in THRESHOLD_PERCENTILES}
    camera_raw_errors = {}
    camera_thresholds = {p: {} for p in THRESHOLD_PERCENTILES}

    reference_y = None
    single_camera_results = []

    for camera in CAMERAS:
        print("\n" + "-" * 100)
        print(f"Training camera model: {camera}")
        print("-" * 100)

        train_path = FEATURE_DIR / f"train_features_correct_tsm_{camera}.npy"
        test_path = FEATURE_DIR / f"test_features_tsm_{camera}.npy"
        label_path = FEATURE_DIR / f"test_features_tsm_window_labels_{camera}.npy"

        train_X = load_npy(train_path)
        test_X = load_npy(test_path)
        test_y = load_npy(label_path).astype(int)

        if train_X.shape[1] != test_X.shape[1]:
            raise ValueError(f"Feature dim mismatch for camera {camera}")

        if len(test_X) != len(test_y):
            raise ValueError(f"Label length mismatch for camera {camera}")

        if reference_y is None:
            reference_y = test_y
        else:
            if not np.array_equal(reference_y, test_y):
                raise ValueError(f"Test label order mismatch for camera {camera}")

        camera_dir = model_dir / camera

        trained = train_one_camera(
            train_X=train_X,
            cfg=cfg,
            camera_dir=camera_dir,
        )

        model = trained["model"]
        scaler = trained["scaler"]
        val_errors = trained["val_errors"]

        train_scaled = scaler.transform(trained["train_part"]).astype(np.float32)
        test_scaled = scaler.transform(test_X).astype(np.float32)

        train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
        test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

        train_errors = get_errors(model, train_tensor)
        test_errors = get_errors(model, test_tensor)

        camera_raw_errors[camera] = test_errors

        for p in THRESHOLD_PERCENTILES:
            threshold = float(np.percentile(val_errors, p))
            camera_thresholds[p][camera] = threshold

            normalized_scores = test_errors / threshold
            camera_scores_by_percentile[p][camera] = normalized_scores

            metrics = compute_metrics(test_y, test_errors, threshold)

            row = {
                "model_name": model_name,
                "mode": "single_camera",
                "camera": camera,
                "fusion_method": "",
                "threshold_percentile": p,
                "input_dim": int(train_X.shape[1]),
                "train_count": int(len(train_X)),
                "test_count": int(len(test_X)),
                "test_correct_count": int((test_y == 0).sum()),
                "test_anomaly_count": int((test_y == 1).sum()),
                "threshold": threshold,
                "dropout": cfg["dropout"],
                "weight_decay": cfg["weight_decay"],
                "best_epoch": trained["best_epoch"],
                "best_val_loss": trained["best_val_loss"],
                **metrics,
            }

            single_camera_results.append(row)
            all_results.append(row)

            p_dir = camera_dir / f"threshold_p{p}"
            p_dir.mkdir(exist_ok=True)

            plot_error_distribution(
                p_dir / "error_distribution.png",
                train_errors=train_errors,
                val_errors=val_errors,
                test_errors=test_errors,
                test_y=test_y,
                threshold=threshold,
                title=f"{model_name} | {camera} | p={p}",
            )

    # Save single camera result table for this model
    single_keys = [
        "model_name", "mode", "camera", "fusion_method", "threshold_percentile",
        "input_dim", "train_count", "test_count", "test_correct_count", "test_anomaly_count",
        "threshold", "dropout", "weight_decay", "best_epoch", "best_val_loss",
        "accuracy", "precision", "recall", "f1", "auc",
        "tn", "fp", "fn", "tp", "false_positive_rate", "false_negative_rate",
        "flagged_as_anomaly", "total_test",
    ]

    save_csv(model_dir / "single_camera_results.csv", single_camera_results, single_keys)

    # Fusion evaluation
    fusion_results = []

    for p in THRESHOLD_PERCENTILES:
        score_matrix = np.vstack([camera_scores_by_percentile[p][cam] for cam in CAMERAS])
        sorted_scores = np.sort(score_matrix, axis=0)

        fusion_scores = {
            "mean": score_matrix.mean(axis=0),
            "max": score_matrix.max(axis=0),
            "top2_mean": sorted_scores[-2:, :].mean(axis=0),
        }

        for fusion_method in FUSION_METHODS:
            scores = fusion_scores[fusion_method]
            threshold = 1.0

            metrics = compute_metrics(reference_y, scores, threshold)

            row = {
                "model_name": model_name,
                "mode": "fusion",
                "camera": "ALL",
                "fusion_method": fusion_method,
                "threshold_percentile": p,
                "input_dim": "",
                "train_count": "",
                "test_count": int(len(reference_y)),
                "test_correct_count": int((reference_y == 0).sum()),
                "test_anomaly_count": int((reference_y == 1).sum()),
                "threshold": threshold,
                "dropout": cfg["dropout"],
                "weight_decay": cfg["weight_decay"],
                "best_epoch": "",
                "best_val_loss": "",
                **metrics,
            }

            fusion_results.append(row)
            all_results.append(row)

            fusion_dir = model_dir / "fusion" / f"{fusion_method}_p{p}"
            fusion_dir.mkdir(parents=True, exist_ok=True)

            y_pred = (scores > threshold).astype(int)

            with open(fusion_dir / "fusion_test_results.csv", "w", encoding="utf-8") as f:
                f.write("sample_id,true_label,normalized_score,prediction\n")
                for i in range(len(scores)):
                    f.write(f"{i},{int(reference_y[i])},{scores[i]:.8f},{int(y_pred[i])}\n")

            plt.figure(figsize=(9, 5))
            plt.hist(scores[reference_y == 0], bins=40, alpha=0.55, label="Test correct")
            plt.hist(scores[reference_y == 1], bins=40, alpha=0.55, label="Test anomaly")
            plt.axvline(1.0, linestyle="--", label="Fusion threshold = 1")
            plt.xlabel("Normalized Fusion Score")
            plt.ylabel("Count")
            plt.title(f"{model_name} | {fusion_method} | p={p}")
            plt.legend()
            plt.savefig(fusion_dir / "fusion_distribution.png", dpi=200, bbox_inches="tight")
            plt.close()

    save_csv(model_dir / "fusion_results.csv", fusion_results, single_keys)


# ============================================================
# 5. SAVE FINAL SUMMARY
# ============================================================

summary_keys = [
    "model_name", "mode", "camera", "fusion_method", "threshold_percentile",
    "input_dim", "train_count", "test_count", "test_correct_count", "test_anomaly_count",
    "threshold", "dropout", "weight_decay", "best_epoch", "best_val_loss",
    "accuracy", "precision", "recall", "f1", "auc",
    "tn", "fp", "fn", "tp", "false_positive_rate", "false_negative_rate",
    "flagged_as_anomaly", "total_test",
]

save_csv(OUTPUT_DIR / "final_tuning_summary.csv", all_results, summary_keys)

sorted_results = sorted(all_results, key=lambda x: x.get("f1", 0), reverse=True)

print("\n" + "=" * 100)
print("FINAL TUNING FINISHED")
print("=" * 100)

print("\nBest results by F1:")
for r in sorted_results[:20]:
    print(
        f"{r['model_name']} | "
        f"{r['mode']} | "
        f"camera={r['camera']} | "
        f"fusion={r['fusion_method']} | "
        f"p={r['threshold_percentile']} | "
        f"F1={r['f1']:.4f} | "
        f"Precision={r['precision']:.4f} | "
        f"Recall={r['recall']:.4f} | "
        f"AUC={r['auc']}"
    )

best = sorted_results[0]

with open(OUTPUT_DIR / "final_best_config.json", "w", encoding="utf-8") as f:
    json.dump(best, f, indent=4)

print("\nBest config saved:", OUTPUT_DIR / "final_best_config.json")
print("Summary saved:", OUTPUT_DIR / "final_tuning_summary.csv")