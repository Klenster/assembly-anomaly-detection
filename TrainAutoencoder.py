import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Model 1: Standard Autoencoder (pencere bazlı — sırasız)
# Kullanım: "shouldn't have happened", "wrong position" gibi görsel hatalar
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
            nn.Linear(512, 128),   # bottleneck
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Model 2: LSTM Autoencoder (sekans bazlı — sıralı)
# Kullanım: "wrong order" gibi sıra bazlı hatalar
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=6080, hidden_dim=256, num_layers=2):
        super().__init__()

        # Encoder: sekansı hidden_dim boyutlu gizli duruma sıkıştırır
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Decoder: gizli durumdan orijinal sekansı yeniden üretir
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

    def forward(self, x, lengths=None):
        # x: [batch, seq_len, input_dim]

        if lengths is not None:
            # Pack for efficiency with variable length sequences
            packed = pack_padded_sequence(x, lengths, batch_first=True,
                                          enforce_sorted=False)
            encoded_packed, (h, c) = self.encoder(packed)
            encoded, _ = pad_packed_sequence(encoded_packed, batch_first=True)
        else:
            encoded, (h, c) = self.encoder(x)

        # Decode from encoder output
        decoded, _ = self.decoder(encoded)
        # decoded: [batch, seq_len, input_dim]
        return decoded


# ---------------------------------------------------------------------------
# Training: Standard Autoencoder
# ---------------------------------------------------------------------------

def train_autoencoder(feature_file, output_dir=".", epochs=100, batch_size=16,
                      learning_rate=1e-3, val_split=0.1):
    """
    Train standard autoencoder on correct (normal) features.
    Uses flat format: (total_windows, 6080)

    Saves:
        autoencoder_best.pth
        feature_scaler.pkl
        threshold.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print("STANDARD AUTOENCODER EĞİTİMİ")
    print(f"{'='*50}")
    print(f"Device: {device}")

    # --- Load & Normalize ---
    print(f"\nLoading: {feature_file}")
    data = np.load(feature_file)
    print(f"Shape: {data.shape}")

    scaler          = StandardScaler()
    data_normalized = scaler.fit_transform(data).astype(np.float32)

    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    # --- Train / Val Split ---
    n_total = len(data_normalized)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    indices   = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(data_normalized[train_idx])),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(data_normalized[val_idx])),
        batch_size=batch_size, shuffle=False
    )

    print(f"Train windows: {n_train} | Val windows: {n_val}")

    # --- Model ---
    input_dim = data_normalized.shape[1]
    model     = Autoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"Input dim: {input_dim} | Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")

    # --- Training Loop ---
    best_val_loss   = float('inf')
    best_model_path = os.path.join(output_dir, "autoencoder_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            loss  = criterion(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch    = batch.to(device)
                val_loss += criterion(model(batch), batch).item() * len(batch)
        val_loss /= n_val

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f}"
                  + (" ← best" if val_loss == best_val_loss else ""))

    print(f"\nBest model: {best_model_path} (val loss: {best_val_loss:.6f})")

    # --- Threshold ---
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_errors  = []
    full_loader = DataLoader(
        TensorDataset(torch.tensor(data_normalized)),
        batch_size=batch_size, shuffle=False
    )
    with torch.no_grad():
        for (batch,) in full_loader:
            batch  = batch.to(device)
            errors = ((model(batch) - batch) ** 2).mean(dim=1)
            all_errors.extend(errors.cpu().numpy())

    all_errors = np.array(all_errors)
    threshold  = float(all_errors.mean() + 3 * all_errors.std())

    threshold_path = os.path.join(output_dir, "threshold.npy")
    np.save(threshold_path, np.array(threshold))

    print(f"Error — mean: {all_errors.mean():.6f} | std: {all_errors.std():.6f}")
    print(f"Threshold (mean+3*std): {threshold:.6f}")
    print(f"Threshold saved: {threshold_path}")

    flagged = (all_errors > threshold).sum()
    print(f"Sanity check: {flagged}/{len(all_errors)} training windows flagged "
          f"({flagged/len(all_errors)*100:.1f}%) — should be ~0%")

    return model, scaler, threshold


# ---------------------------------------------------------------------------
# Training: LSTM Autoencoder
# ---------------------------------------------------------------------------

def train_lstm_autoencoder(sequence_file, output_dir=".", epochs=100,
                            batch_size=8, learning_rate=1e-3, val_split=0.1):
    """
    Train LSTM autoencoder on correct (normal) action sequences.
    Uses sequence format: list of (num_windows, 6080) arrays.
    Each element is one full action clip with variable number of windows.

    Saves:
        lstm_autoencoder_best.pth
        lstm_feature_scaler.pkl
        lstm_threshold.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print("LSTM AUTOENCODER EĞİTİMİ")
    print(f"{'='*50}")
    print(f"Device: {device}")

    # --- Load sequences ---
    print(f"\nLoading: {sequence_file}")
    sequences = np.load(sequence_file, allow_pickle=True)
    print(f"Total clips: {len(sequences)}")
    lengths = [len(s) for s in sequences]
    print(f"Windows per clip — min: {min(lengths)} | max: {max(lengths)} | "
          f"avg: {np.mean(lengths):.1f}")

    # --- Normalize ---
    # Fit scaler on all windows flattened, then reshape back
    all_flat = np.concatenate(sequences, axis=0).astype(np.float32)
    scaler   = StandardScaler()
    scaler.fit(all_flat)

    scaler_path = os.path.join(output_dir, "lstm_feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    # Normalize each sequence
    sequences_norm = [
        scaler.transform(s.astype(np.float32))
        for s in sequences
    ]

    # --- Train / Val Split ---
    n_total = len(sequences_norm)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    indices        = np.random.permutation(n_total)
    train_seqs     = [sequences_norm[i] for i in indices[:n_train]]
    val_seqs       = [sequences_norm[i] for i in indices[n_train:]]

    print(f"Train clips: {n_train} | Val clips: {n_val}")

    def make_loader(seqs, shuffle):
        """Convert list of variable-length sequences to padded DataLoader."""
        tensors = [torch.tensor(s, dtype=torch.float32) for s in seqs]
        lengths = torch.tensor([len(t) for t in tensors])
        padded  = pad_sequence(tensors, batch_first=True)   # [N, max_len, dim]
        return DataLoader(
            TensorDataset(padded, lengths),
            batch_size=batch_size,
            shuffle=shuffle
        )

    train_loader = make_loader(train_seqs, shuffle=True)
    val_loader   = make_loader(val_seqs,   shuffle=False)

    # --- Model ---
    input_dim  = sequences_norm[0].shape[1]   # 6080
    model      = LSTMAutoencoder(input_dim=input_dim, hidden_dim=256, num_layers=2).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion  = nn.MSELoss()

    print(f"Input dim: {input_dim} | Hidden: 256 | Layers: 2")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")

    # --- Training Loop ---
    best_val_loss   = float('inf')
    best_model_path = os.path.join(output_dir, "lstm_autoencoder_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_windows = 0
        for (padded, lengths) in train_loader:
            padded  = padded.to(device)
            lengths = lengths.to(device)
            output  = model(padded, lengths.cpu())

            # Only compute loss on non-padded positions
            loss = 0.0
            for i, l in enumerate(lengths):
                loss += criterion(output[i, :l], padded[i, :l])
            loss /= len(lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss      += loss.item()
            n_train_windows += 1
        train_loss /= n_train_windows

        model.eval()
        val_loss = 0.0
        n_val_windows = 0
        with torch.no_grad():
            for (padded, lengths) in val_loader:
                padded = padded.to(device)
                output = model(padded, lengths)
                loss   = 0.0
                for i, l in enumerate(lengths):
                    loss += criterion(output[i, :l], padded[i, :l])
                loss /= len(lengths)
                val_loss      += loss.item()
                n_val_windows += 1
        val_loss /= n_val_windows

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f}"
                  + (" ← best" if val_loss == best_val_loss else ""))

    print(f"\nBest model: {best_model_path} (val loss: {best_val_loss:.6f})")

    # --- Threshold ---
    # Compute per-clip mean reconstruction error on all training sequences
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_clip_errors = []
    with torch.no_grad():
        for seq in sequences_norm:
            t      = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(t)
            error  = ((output - t) ** 2).mean().item()
            all_clip_errors.append(error)

    all_clip_errors = np.array(all_clip_errors)
    threshold       = float(all_clip_errors.mean() + 3 * all_clip_errors.std())

    threshold_path = os.path.join(output_dir, "lstm_threshold.npy")
    np.save(threshold_path, np.array(threshold))

    print(f"Error — mean: {all_clip_errors.mean():.6f} | std: {all_clip_errors.std():.6f}")
    print(f"Threshold (mean+3*std): {threshold:.6f}")
    print(f"Threshold saved: {threshold_path}")

    flagged = (all_clip_errors > threshold).sum()
    print(f"Sanity check: {flagged}/{len(all_clip_errors)} training clips flagged "
          f"({flagged/len(all_clip_errors)*100:.1f}%) — should be ~0%")

    return model, scaler, threshold


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    OUTPUT_DIR = "."

    # -----------------------------------------------------------------------
    # 1. Standard Autoencoder — pencere bazlı (görsel hatalar için)
    # -----------------------------------------------------------------------
    model_ae, scaler_ae, threshold_ae = train_autoencoder(
        feature_file  = "train_features_correct.npy",
        output_dir    = OUTPUT_DIR,
        epochs        = 100,
        batch_size    = 16,
        learning_rate = 1e-3,
        val_split     = 0.1,
    )

    # -----------------------------------------------------------------------
    # 2. LSTM Autoencoder — sekans bazlı (sıra hataları için)
    # -----------------------------------------------------------------------
    model_lstm, scaler_lstm, threshold_lstm = train_lstm_autoencoder(
        sequence_file = "train_features_correct_sequences.npy",
        output_dir    = OUTPUT_DIR,
        epochs        = 100,
        batch_size    = 8,
        learning_rate = 1e-3,
        val_split     = 0.1,
    )

    print("\n" + "="*50)
    print("Tüm modeller eğitildi. Kaydedilen dosyalar:")
    print("  autoencoder_best.pth       ← standard model")
    print("  feature_scaler.pkl         ← standard scaler")
    print("  threshold.npy              ← standard threshold")
    print("  lstm_autoencoder_best.pth  ← LSTM model")
    print("  lstm_feature_scaler.pkl    ← LSTM scaler")
    print("  lstm_threshold.npy         ← LSTM threshold")
    print("="*50)
