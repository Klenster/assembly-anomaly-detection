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

OUTPUT_DIR = BASE_DIR / "OptimizedPerCameraAE_Results"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_STATE = 42
EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 20

THRESHOLD_PERCENTILES = [75, 80, 85, 90]

FUSION_METHODS = ["mean", "max", "top2_mean"]

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("DEVICE:", DEVICE)


# ============================================================
# 1. MODEL
# ============================================================

class FlexibleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.2):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ============================================================
# 2. EXPERIMENT CONFIGS
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

EXPERIMENTS = [
    {
        "feature_type": "slowfast",
        "folder": "per_camera_slowfast",
        "cameras": SLOWFAST_CAMERAS,
        "train_template": "train_features_correct_{camera}.npy",
        "test_template": "test_features_{camera}.npy",
        "label_template": "test_features_window_labels_{camera}.npy",
        "models": [
            {
                "model_name": "slowfast_small",
                "hidden_dims": [256, 64],
                "latent_dim": 16,
                "dropout": 0.2,
                "weight_decay": 1e-4,
            },
            {
                "model_name": "slowfast_medium",
                "hidden_dims": [512, 128],
                "latent_dim": 32,
                "dropout": 0.25,
                "weight_decay": 1e-4,
            },
            {
                "model_name": "slowfast_medium_regularized",
                "hidden_dims": [512, 128],
                "latent_dim": 32,
                "dropout": 0.3,
                "weight_decay": 1e-3,
            },
        ],
    },
    {
        "feature_type": "tsm",
        "folder": "per_camera_tsm",
        "cameras": TSM_CAMERAS,
        "train_template": "train_features_correct_tsm_{camera}.npy",
        "test_template": "test_features_tsm_{camera}.npy",
        "label_template": "test_features_tsm_window_labels_{camera}.npy",
        "models": [
            {
                "model_name": "tsm_small",
                "hidden_dims": [256, 64],
                "latent_dim": 16,
                "dropout": 0.2,
                "weight_decay": 1e-4,
            },
            {
                "model_name": "tsm_medium",
                "hidden_dims": [512, 128],
                "latent_dim": 32,
                "dropout": 0.25,
                "weight_decay": 1e-4,
            },
            {
                "model_name": "tsm_medium_regularized",
                "hidden_dims": [512, 128],
                "latent_dim": 32,
                "dropout": 0.3,
                "weight_decay": 1e-3,
            },
        ],
    },
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def load_npy(path, allow_pickle=False):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    arr = np.load(path, allow_pickle=allow_pickle)
    print(f"Loaded: {path.name} | shape={arr.shape} | dtype={arr.dtype}")
    return arr


def check_data(train_X, test_X, test_y):
    if train_X.shape[1] != test_X.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: train={train_X.shape[1]}, test={test_X.shape[1]}"
        )

    if len(test_X) != len(test_y):
        raise ValueError(
            f"Test label length mismatch: test={len(test_X)}, labels={len(test_y)}"
        )

    if np.isnan(train_X).any() or np.isinf(train_X).any():
        raise ValueError("Train içinde NaN veya Inf var.")

    if np.isnan(test_X).any() or np.isinf(test_X).any():
        raise ValueError("Test içinde NaN veya Inf var.")


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


def train_camera_model(train_X, model_cfg, experiment_dir):
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_part, val_X = train_test_split(
        train_X,
        test_size=0.25,
        random_state=RANDOM_STATE,
        shuffle=True,
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
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_dim = train_tensor.shape[1]

    model = FlexibleAutoencoder(
        input_dim=input_dim,
        hidden_dims=model_cfg["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],
        dropout=model_cfg["dropout"],
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=model_cfg["weight_decay"],
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

    return {
        "model": model,
        "scaler": scaler,
        "val_errors": val_errors,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_part": train_part,
        "val_X": val_X,
    }


def save_camera_plot(
    experiment_dir,
    train_errors,
    val_errors,
    test_errors,
    test_y,
    threshold,
    title,
):
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

    plot_path = experiment_dir / "error_distribution.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_single_camera(
    feature_type,
    camera,
    model_cfg,
    train_X,
    test_X,
    test_y,
    camera_dir,
):
    print("\n" + "-" * 90)
    print(f"{feature_type.upper()} | {model_cfg['model_name']} | Camera: {camera}")
    print("-" * 90)

    trained = train_camera_model(
        train_X=train_X,
        model_cfg=model_cfg,
        experiment_dir=camera_dir,
    )

    model = trained["model"]
    scaler = trained["scaler"]
    val_errors = trained["val_errors"]

    train_part = trained["train_part"]

    train_scaled = scaler.transform(train_part).astype(np.float32)
    test_scaled = scaler.transform(test_X).astype(np.float32)

    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

    train_errors = get_errors(model, train_tensor)
    test_errors = get_errors(model, test_tensor)

    camera_results = []
    normalized_scores_by_threshold = {}

    for p in THRESHOLD_PERCENTILES:
        threshold = float(np.percentile(val_errors, p))
        normalized_scores = test_errors / threshold

        metrics = compute_metrics(test_y, test_errors, threshold)

        result = {
            "feature_type": feature_type,
            "model_name": model_cfg["model_name"],
            "camera": camera,
            "mode": "single_camera",
            "threshold_percentile": p,
            "fusion_method": "",
            "input_dim": int(train_X.shape[1]),
            "train_count": int(len(train_X)),
            "test_count": int(len(test_X)),
            "test_correct_count": int((test_y == 0).sum()),
            "test_anomaly_count": int((test_y == 1).sum()),
            "threshold": threshold,
            "hidden_dims": str(model_cfg["hidden_dims"]),
            "latent_dim": int(model_cfg["latent_dim"]),
            "dropout": float(model_cfg["dropout"]),
            "weight_decay": float(model_cfg["weight_decay"]),
            "best_epoch": int(trained["best_epoch"]),
            "best_val_loss": float(trained["best_val_loss"]),
            **metrics,
        }

        camera_results.append(result)
        normalized_scores_by_threshold[p] = normalized_scores

        threshold_dir = camera_dir / f"threshold_{p}"
        threshold_dir.mkdir(parents=True, exist_ok=True)

        save_camera_plot(
            threshold_dir,
            train_errors,
            val_errors,
            test_errors,
            test_y,
            threshold,
            title=f"{feature_type} | {model_cfg['model_name']} | {camera} | p={p}",
        )

        y_pred = (test_errors > threshold).astype(int)
        with open(threshold_dir / "test_results.csv", "w", encoding="utf-8") as f:
            f.write("sample_id,true_label,reconstruction_error,prediction\n")
            for i in range(len(test_errors)):
                f.write(f"{i},{int(test_y[i])},{test_errors[i]:.8f},{int(y_pred[i])}\n")

    with open(camera_dir / "camera_results.json", "w", encoding="utf-8") as f:
        json.dump(camera_results, f, indent=4)

    return camera_results, normalized_scores_by_threshold


def compute_fusion_results(
    feature_type,
    model_name,
    threshold_percentile,
    y_true,
    camera_scores,
    fusion_dir,
    model_cfg,
):
    fusion_dir.mkdir(parents=True, exist_ok=True)

    cameras = list(camera_scores.keys())
    score_matrix = np.vstack([camera_scores[c] for c in cameras])

    sorted_scores = np.sort(score_matrix, axis=0)

    fusion_score_dict = {
        "mean": score_matrix.mean(axis=0),
        "max": score_matrix.max(axis=0),
        "top2_mean": sorted_scores[-2:, :].mean(axis=0)
        if score_matrix.shape[0] >= 2
        else score_matrix.max(axis=0),
    }

    fusion_results = []

    for fusion_method, scores in fusion_score_dict.items():
        threshold = 1.0

        metrics = compute_metrics(y_true, scores, threshold)

        result = {
            "feature_type": feature_type,
            "model_name": model_name,
            "camera": "ALL",
            "mode": "fusion",
            "threshold_percentile": threshold_percentile,
            "fusion_method": fusion_method,
            "input_dim": "",
            "train_count": "",
            "test_count": int(len(y_true)),
            "test_correct_count": int((y_true == 0).sum()),
            "test_anomaly_count": int((y_true == 1).sum()),
            "threshold": threshold,
            "hidden_dims": str(model_cfg["hidden_dims"]),
            "latent_dim": int(model_cfg["latent_dim"]),
            "dropout": float(model_cfg["dropout"]),
            "weight_decay": float(model_cfg["weight_decay"]),
            "best_epoch": "",
            "best_val_loss": "",
            **metrics,
        }

        fusion_results.append(result)

        method_dir = fusion_dir / f"{fusion_method}_p{threshold_percentile}"
        method_dir.mkdir(parents=True, exist_ok=True)

        y_pred = (scores > threshold).astype(int)

        with open(method_dir / "fusion_test_results.csv", "w", encoding="utf-8") as f:
            f.write("sample_id,true_label,normalized_score,prediction\n")
            for i in range(len(scores)):
                f.write(f"{i},{int(y_true[i])},{scores[i]:.8f},{int(y_pred[i])}\n")

        plt.figure(figsize=(9, 5))
        plt.hist(scores[y_true == 0], bins=40, alpha=0.55, label="Test correct")
        plt.hist(scores[y_true == 1], bins=40, alpha=0.55, label="Test anomaly")
        plt.axvline(threshold, linestyle="--", label="Fusion threshold = 1")
        plt.xlabel("Normalized Fusion Score")
        plt.ylabel("Count")
        plt.title(f"{feature_type} | {model_name} | {fusion_method} | p={threshold_percentile}")
        plt.legend()

        plt.savefig(method_dir / "fusion_distribution.png", dpi=200, bbox_inches="tight")
        plt.close()

    return fusion_results


def run_feature_family_experiment(exp_cfg):
    feature_type = exp_cfg["feature_type"]
    folder = BASE_DIR / exp_cfg["folder"]

    all_results = []

    print("\n" + "=" * 100)
    print(f"FEATURE FAMILY: {feature_type.upper()}")
    print("=" * 100)

    for model_cfg in exp_cfg["models"]:
        model_name = model_cfg["model_name"]

        print("\n" + "=" * 100)
        print(f"MODEL CONFIG: {feature_type} | {model_name}")
        print("=" * 100)

        model_dir = OUTPUT_DIR / feature_type / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        camera_scores_by_threshold = {p: {} for p in THRESHOLD_PERCENTILES}
        reference_y = None

        for camera in exp_cfg["cameras"]:
            train_path = folder / exp_cfg["train_template"].format(camera=camera)
            test_path = folder / exp_cfg["test_template"].format(camera=camera)
            label_path = folder / exp_cfg["label_template"].format(camera=camera)

            if not train_path.exists():
                print(f"Skipping {camera}: train file not found.")
                continue

            train_X = load_npy(train_path)
            test_X = load_npy(test_path)
            test_y = load_npy(label_path).astype(int)

            check_data(train_X, test_X, test_y)

            camera_dir = model_dir / camera

            camera_results, normalized_scores_by_threshold = evaluate_single_camera(
                feature_type=feature_type,
                camera=camera,
                model_cfg=model_cfg,
                train_X=train_X,
                test_X=test_X,
                test_y=test_y,
                camera_dir=camera_dir,
            )

            all_results.extend(camera_results)

            if reference_y is None:
                reference_y = test_y
            else:
                if not np.array_equal(reference_y, test_y):
                    raise ValueError(f"Label order mismatch for camera: {camera}")

            for p in THRESHOLD_PERCENTILES:
                camera_scores_by_threshold[p][camera] = normalized_scores_by_threshold[p]

        for p in THRESHOLD_PERCENTILES:
            fusion_dir = model_dir / "fusion"
            fusion_results = compute_fusion_results(
                feature_type=feature_type,
                model_name=model_name,
                threshold_percentile=p,
                y_true=reference_y,
                camera_scores=camera_scores_by_threshold[p],
                fusion_dir=fusion_dir,
                model_cfg=model_cfg,
            )

            all_results.extend(fusion_results)

    return all_results


def save_summary(all_results):
    summary_path = OUTPUT_DIR / "optimized_per_camera_summary.csv"

    keys = [
        "feature_type",
        "model_name",
        "mode",
        "camera",
        "threshold_percentile",
        "fusion_method",
        "input_dim",
        "train_count",
        "test_count",
        "test_correct_count",
        "test_anomaly_count",
        "threshold",
        "hidden_dims",
        "latent_dim",
        "dropout",
        "weight_decay",
        "best_epoch",
        "best_val_loss",
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
                value = "" if value is None else str(value)
                value = value.replace(",", ";")
                row.append(value)
            f.write(",".join(row) + "\n")

    print("\nSummary saved:", summary_path)

    sorted_results = sorted(all_results, key=lambda x: x.get("f1", 0), reverse=True)

    print("\nBest results by F1:")
    for r in sorted_results[:20]:
        print(
            f"{r.get('feature_type')} | "
            f"{r.get('model_name')} | "
            f"{r.get('mode')} | "
            f"camera={r.get('camera')} | "
            f"fusion={r.get('fusion_method')} | "
            f"p={r.get('threshold_percentile')} | "
            f"F1={r.get('f1'):.4f} | "
            f"Precision={r.get('precision'):.4f} | "
            f"Recall={r.get('recall'):.4f} | "
            f"AUC={r.get('auc')}"
        )


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":
    all_results = []

    for exp_cfg in EXPERIMENTS:
        results = run_feature_family_experiment(exp_cfg)
        all_results.extend(results)

    save_summary(all_results)

    print("\n" + "=" * 100)
    print("OPTIMIZED PER-CAMERA AE EXPERIMENTS FINISHED")
    print("=" * 100)