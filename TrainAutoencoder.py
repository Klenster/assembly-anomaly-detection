import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Autoencoder Model
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim=6080):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),   # bottleneck — compressed representation
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim),  # reconstruct original
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(feature_file, output_dir=".", epochs=100, batch_size=16,
                      learning_rate=1e-3, val_split=0.1):
    """
    Train an autoencoder on correct (normal) features only.
    Saves the trained model, scaler, and anomaly threshold.

    Args:
        feature_file  : path to train_features_correct.npy
        output_dir    : where to save model, scaler, threshold
        epochs        : number of training epochs
        batch_size    : batch size
        learning_rate : learning rate
        val_split     : fraction of data to use for validation
    """

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Load Data ---
    print(f"\nLoading features from: {feature_file}")
    data = np.load(feature_file)
    print(f"Loaded shape: {data.shape}")   # (num_windows, 6080)

    # --- 2. Normalize ---
    # StandardScaler normalizes ALL 6080 dims to zero mean, unit variance.
    # This is critical because visual dims (2048) are already normalized
    # but hand pose dims (4032) are raw MediaPipe coordinates.
    # Fitting only on correct (normal) data — same scaler used at inference.
    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data).astype(np.float32)

    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    # --- 3. Train / Validation Split ---
    n_total = len(data_normalized)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    # Shuffle before splitting
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_tensor = torch.tensor(data_normalized[train_idx])
    val_tensor   = torch.tensor(data_normalized[val_idx])

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size,
        shuffle=False
    )

    print(f"\nTrain windows: {n_train} | Val windows: {n_val}")

    # --- 4. Model ---
    input_dim = data_normalized.shape[1]   # 6080
    model     = Autoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"Input dim: {input_dim}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {learning_rate}")

    # --- 5. Training Loop ---
    print("\n" + "=" * 50)
    best_val_loss  = float('inf')
    best_model_path = os.path.join(output_dir, "autoencoder_best.pth")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item() * len(batch)
        val_loss /= n_val

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}"
                  + (" ← best" if val_loss == best_val_loss else ""))

    print("=" * 50)
    print(f"\nBest model saved: {best_model_path} (val loss: {best_val_loss:.6f})")

    # --- 6. Compute Anomaly Threshold ---
    # Load best model and compute reconstruction error on ALL training data.
    # Threshold = mean + 3 * std of reconstruction errors on normal data.
    # Windows above this threshold will be flagged as anomalies at inference.
    print("\nComputing anomaly threshold on training data...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_errors = []
    full_tensor = torch.tensor(data_normalized)
    full_loader = DataLoader(
        TensorDataset(full_tensor),
        batch_size=batch_size,
        shuffle=False
    )

    with torch.no_grad():
        for (batch,) in full_loader:
            batch         = batch.to(device)
            reconstructed = model(batch)
            # Per-sample reconstruction error (MSE)
            errors = ((reconstructed - batch) ** 2).mean(dim=1)
            all_errors.extend(errors.cpu().numpy())

    all_errors = np.array(all_errors)
    threshold  = float(all_errors.mean() + 3 * all_errors.std())

    threshold_path = os.path.join(output_dir, "threshold.npy")
    np.save(threshold_path, np.array(threshold))
    print(f"Reconstruction error — mean: {all_errors.mean():.6f} | "
          f"std: {all_errors.std():.6f}")
    print(f"Threshold (mean + 3*std): {threshold:.6f}")
    print(f"Threshold saved: {threshold_path}")

    # --- 7. Quick Sanity Check ---
    flagged = (all_errors > threshold).sum()
    print(f"\nSanity check: {flagged}/{len(all_errors)} training windows "
          f"flagged as anomaly ({flagged/len(all_errors)*100:.1f}%) "
          f"— should be close to 0%")

    print("\nDone. Files saved:")
    print(f"  {best_model_path}")
    print(f"  {scaler_path}")
    print(f"  {threshold_path}")

    return model, scaler, threshold


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    FEATURE_FILE = "train_features_correct.npy"
    OUTPUT_DIR   = "."   # saves model, scaler, threshold in current directory

    EPOCHS        = 100
    BATCH_SIZE    = 16
    LEARNING_RATE = 1e-3
    VAL_SPLIT     = 0.1   # 10% of training windows used for validation

    model, scaler, threshold = train_autoencoder(
        feature_file  = FEATURE_FILE,
        output_dir    = OUTPUT_DIR,
        epochs        = EPOCHS,
        batch_size    = BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        val_split     = VAL_SPLIT,
    )
