import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 0. CONFIG
# ============================================================

# Bu dosyayı repo kök klasöründe çalıştırırsan BASE_DIR aynı klasör olur.
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()

# Feature dosyaları repo root'ta veya Features klasöründe olabilir.
SEARCH_DIRS = [
    BASE_DIR
]

OUTPUT_DIR = BASE_DIR / "AutoencoderOutputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INCLUDE_CHECKPOINTS = True
REMOVE_DUPLICATES = True

THRESHOLD_PERCENTILE = 85

RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 20


# ============================================================
# 1. FILE HELPERS
# ============================================================

def find_existing_file(filename):
    for directory in SEARCH_DIRS:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def load_and_merge_npy(filenames, allow_pickle=False):
    arrays = []
    used_files = []

    for filename in filenames:
        path = find_existing_file(filename)
        if path is not None:
            arr = np.load(path, allow_pickle=allow_pickle)
            arrays.append(arr)
            used_files.append(path)
            print(f"Loaded: {path} | shape: {arr.shape}")

    if not arrays:
        return None, []

    if len(arrays) == 1:
        return arrays[0], used_files

    merged = np.concatenate(arrays, axis=0)
    return merged, used_files


def remove_duplicate_rows_with_labels(X, y=None):
    """
    Aynı feature satırı birden fazla geldiyse tekrarları siler.
    y varsa label hizasını da korur.
    """
    X_unique, unique_indices = np.unique(X, axis=0, return_index=True)
    order = np.sort(unique_indices)

    X_clean = X[order]

    if y is not None and len(y) == len(X):
        y_clean = y[order]
    else:
        y_clean = y

    return X_clean, y_clean


# ============================================================
# 2. LOAD FEATURES
# ============================================================

train_feature_names = [
    "train_features_correct_checkpoint.npy",
    "train_features_correct.npy",
] if INCLUDE_CHECKPOINTS else [
    "train_features_correct.npy",
]

test_feature_names = [
    "test_features_anomaly_checkpoint.npy",
    "test_features_anomaly.npy",
] if INCLUDE_CHECKPOINTS else [
    "test_features_anomaly.npy",
]

train_label_names = [
    "train_labels_checkpoint.npy",
    "train_labels.npy",
] if INCLUDE_CHECKPOINTS else [
    "train_labels.npy",
]

test_label_names = [
    "test_labels_checkpoint.npy",
    "test_labels.npy",
] if INCLUDE_CHECKPOINTS else [
    "test_labels.npy",
]

train_X, train_files = load_and_merge_npy(train_feature_names)
test_X, test_files = load_and_merge_npy(test_feature_names)

if train_X is None:
    raise FileNotFoundError("Train feature dosyası bulunamadı.")

if test_X is None:
    raise FileNotFoundError("Test feature dosyası bulunamadı.")

train_y, train_label_files = load_and_merge_npy(train_label_names, allow_pickle=True)
test_y, test_label_files = load_and_merge_npy(test_label_names, allow_pickle=True)

print("\n--- Loaded Dataset ---")
print("Train X:", train_X.shape)
print("Test X :", test_X.shape)

if train_y is not None:
    print("Train labels:", np.unique(train_y, return_counts=True))

if test_y is not None:
    print("Test labels:", np.unique(test_y, return_counts=True))
else:
    print("Test labels not found. Evaluation will be prediction-only.")

if train_X.shape[1] != test_X.shape[1]:
    raise ValueError(
        f"Feature dimension mismatch: train={train_X.shape[1]}, test={test_X.shape[1]}"
    )


# ============================================================
# 3. BASIC DATA CHECKS
# ============================================================

print("\n--- Data Checks ---")
print("Train NaN:", np.isnan(train_X).any())
print("Train Inf:", np.isinf(train_X).any())
print("Test NaN :", np.isnan(test_X).any())
print("Test Inf :", np.isinf(test_X).any())

if np.isnan(train_X).any() or np.isinf(train_X).any():
    raise ValueError("Train feature içinde NaN veya Inf var.")

if np.isnan(test_X).any() or np.isinf(test_X).any():
    raise ValueError("Test feature içinde NaN veya Inf var.")


# ============================================================
# 4. OPTIONAL DEDUPLICATION
# ============================================================

if REMOVE_DUPLICATES:
    before_train = len(train_X)
    before_test = len(test_X)

    train_X, train_y = remove_duplicate_rows_with_labels(train_X, train_y)
    test_X, test_y = remove_duplicate_rows_with_labels(test_X, test_y)

    print("\n--- Deduplication ---")
    print(f"Train: {before_train} -> {len(train_X)}")
    print(f"Test : {before_test} -> {len(test_X)}")


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

train_X_part, val_X = train_test_split(
    train_X,
    test_size=0.25,
    random_state=RANDOM_STATE,
    shuffle=True
)

print("\n--- Split ---")
print("Train part:", train_X_part.shape)
print("Validation:", val_X.shape)


# ============================================================
# 6. SCALER
# ============================================================

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_X_part).astype(np.float32)
val_scaled = scaler.transform(val_X).astype(np.float32)
test_scaled = scaler.transform(test_X).astype(np.float32)

scaler_path = OUTPUT_DIR / "scaler.pkl"
joblib.dump(scaler, scaler_path)

print("\nScaler saved:", scaler_path)


# ============================================================
# 7. TENSORS & DATALOADERS
# ============================================================

train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
val_tensor = torch.tensor(val_scaled, dtype=torch.float32)
test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

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
print("Input dim:", input_dim)


# ============================================================
# 8. AUTOENCODER MODEL
# ============================================================

class Autoencoder(nn.Module):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = Autoencoder(input_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

best_model_path = OUTPUT_DIR / "autoencoder_best.pth"

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0

    for (batch,) in train_loader:
        batch = batch.to(device)

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
            batch = batch.to(device)

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

print("\nBest epoch:", best_epoch)
print("Best val loss:", best_val_loss)
print("Best model saved:", best_model_path)


# ============================================================
# 10. LOAD BEST MODEL & ERROR FUNCTION
# ============================================================

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()


def get_errors(data_tensor):
    loader = DataLoader(
        TensorDataset(data_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_errors = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            errors = torch.mean((reconstructed - batch) ** 2, dim=1)
            all_errors.extend(errors.cpu().numpy())

    return np.array(all_errors)


train_errors = get_errors(train_tensor)
val_errors = get_errors(val_tensor)
test_errors = get_errors(test_tensor)

#==============================================================
print("\n--- Test Error Range ---")
print("Min:", test_errors.min())
print("Max:", test_errors.max())
print("Mean:", test_errors.mean())
print("Std:", test_errors.std())

print("\n--- Top 20 Highest Test Errors ---")
top_idx = np.argsort(test_errors)[-20:][::-1]

for idx in top_idx:
    if test_y is not None and len(test_y) == len(test_errors):
        print(f"Index {idx:03d} | Label: {test_y[idx]} | Error: {test_errors[idx]:.6f}")
    else:
        print(f"Index {idx:03d} | Error: {test_errors[idx]:.6f}")

print("\n--- Top 20 Lowest Test Errors ---")
low_idx = np.argsort(test_errors)[:20]

for idx in low_idx:
    if test_y is not None and len(test_y) == len(test_errors):
        print(f"Index {idx:03d} | Label: {test_y[idx]} | Error: {test_errors[idx]:.6f}")
    else:
        print(f"Index {idx:03d} | Error: {test_errors[idx]:.6f}")
#======================================================================================0


# ============================================================
# 11. THRESHOLD
# ============================================================

threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)

threshold_path = OUTPUT_DIR / "threshold.npy"
np.save(threshold_path, np.array(threshold))

print("\n--- Error Summary ---")
print("Train error mean     :", train_errors.mean())
print("Validation error mean:", val_errors.mean())
print("Test error mean      :", test_errors.mean())
print(f"Threshold ({THRESHOLD_PERCENTILE}th percentile):", threshold)
print("Threshold saved:", threshold_path)


# ============================================================
# 12. THRESHOLD SENSITIVITY
# ============================================================

print("\n--- Threshold Sensitivity Analysis ---")

for p in [75, 80, 85, 90, 95]:
    th = np.percentile(val_errors, p)
    preds = test_errors > th

    print(f"\nPercentile: {p}")
    print(f"Threshold: {th:.6f}")
    print(f"Flagged as anomaly: {preds.sum()} / {len(preds)}")

    if test_y is not None and len(test_y) == len(test_errors):
        for label in np.unique(test_y):
            idx = test_y == label
            print(f"{label}: {preds[idx].sum()} / {idx.sum()} flagged")


# ============================================================
# 13. TEST PREDICTIONS
# ============================================================

predictions = test_errors > threshold
pred_labels = np.where(predictions, "anomaly", "normal")

print("\n--- Test Results Summary ---")
print("Total test samples:", len(test_errors))
print("Flagged as anomaly:", predictions.sum(), "/", len(predictions))

if test_y is not None and len(test_y) == len(test_errors):
    print("\nLabel-based summary:")
    for label in np.unique(test_y):
        idx = test_y == label
        print(f"{label}: {predictions[idx].sum()} / {idx.sum()} flagged as anomaly")


# ============================================================
# 14. SAVE RESULTS CSV
# ============================================================

results_path = OUTPUT_DIR / "test_results.csv"

if test_y is not None and len(test_y) == len(test_errors):
    header = "sample_id,true_label,reconstruction_error,prediction"
    rows = [
        f"{i},{test_y[i]},{test_errors[i]:.8f},{pred_labels[i]}"
        for i in range(len(test_errors))
    ]
else:
    header = "sample_id,reconstruction_error,prediction"
    rows = [
        f"{i},{test_errors[i]:.8f},{pred_labels[i]}"
        for i in range(len(test_errors))
    ]

with open(results_path, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    f.write("\n".join(rows))

print("\nResults saved:", results_path)


# ============================================================
# 15. PLOT
# ============================================================

plt.figure(figsize=(9, 5))
plt.hist(train_errors, bins=30, alpha=0.5, label="Train Normal")
plt.hist(val_errors, bins=30, alpha=0.5, label="Validation Normal")
plt.hist(test_errors, bins=30, alpha=0.6, label="Test")
plt.axvline(threshold, linestyle="--", label=f"Threshold ({THRESHOLD_PERCENTILE}%)")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Autoencoder Test Results")
plt.legend()

plot_path = OUTPUT_DIR / "error_distribution.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()

print("Plot saved:", plot_path)