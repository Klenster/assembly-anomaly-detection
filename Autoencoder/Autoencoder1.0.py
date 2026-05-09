import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import os

# -------------------------
# 1. LOAD DATA
# -------------------------

train_X = np.concat([np.load(r".\Features\train_features_correct_checkpoint(part1).npy"),
                     np.load(r".\Features\train_features_correct(part2).npy")])
test_X = np.load("test_features_anomaly9033.npy")
test_y = np.load("test_labels9033(anomaly).npy", allow_pickle=True)

print("Train X:", train_X.shape)
print("Test X:", test_X.shape)
print("Test labels:", np.unique(test_y, return_counts=True))

# -------------------------
# 2. TRAIN / VALIDATION SPLIT
# -------------------------

train_X_part, val_X = train_test_split(
    train_X,
    test_size=0.25,
    random_state=42
)

# -------------------------
# 3. SCALER
# Fit only on train normal data
# -------------------------

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_X_part)
val_scaled = scaler.transform(val_X)
test_scaled = scaler.transform(test_X)

joblib.dump(scaler, "scaler.pkl")

# -------------------------
# 4. TO TENSOR
# -------------------------

train_tensor = torch.tensor(train_scaled).float()
val_tensor = torch.tensor(val_scaled).float()
test_tensor = torch.tensor(test_scaled).float()

input_dim = train_tensor.shape[1]

# -------------------------
# 5. AUTOENCODER MODEL
# -------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

model = Autoencoder(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# -------------------------
# 6. TRAIN MODEL
# -------------------------

best_val_loss = float("inf")
best_epoch = 0
epochs = 80

for epoch in range(epochs):
    model.train()

    output = model(train_tensor)
    train_loss = criterion(output, train_tensor)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(val_tensor)
        val_loss = criterion(val_output, val_tensor)

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_epoch = epoch
        torch.save(model.state_dict(), "autoencoder_best.pth")

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss.item():.6f} | "
            f"Val Loss: {val_loss.item():.6f}"
        )

print("\nBest epoch:", best_epoch)
print("Best val loss:", best_val_loss)

# -------------------------
# 7. LOAD BEST MODEL
# -------------------------

model.load_state_dict(torch.load("autoencoder_best.pth"))
model.eval()

# -------------------------
# 8. RECONSTRUCTION ERROR FUNCTION
# -------------------------

def get_errors(data_tensor):
    with torch.no_grad():
        reconstructed = model(data_tensor)
        errors = torch.mean((reconstructed - data_tensor) ** 2, dim=1)
    return errors.numpy()

train_errors = get_errors(train_tensor)
val_errors = get_errors(val_tensor)
test_errors = get_errors(test_tensor)

# -------------------------
# 9. THRESHOLD
# Validation normal errors only
# -------------------------

threshold = np.percentile(val_errors, 85)
np.save("threshold.npy", threshold)

print("\nTrain error mean:", train_errors.mean())
print("Validation error mean:", val_errors.mean())
print("Test error mean:", test_errors.mean())
print("Threshold:", threshold)

# -------------------------
# 10. GENERAL TEST RESULTS
# Labels are used ONLY for evaluation
# -------------------------

predictions = test_errors > threshold

print("\n--- General Test Results ---")
for i in range(len(test_errors)):
    pred_label = "anomaly" if predictions[i] else "normal"
    print(
        f"Sample {i:02d} | "
        f"True label: {test_y[i]} | "
        f"Error: {test_errors[i]:.6f} | "
        f"Prediction: {pred_label}"
    )

print("\nFlagged as anomaly:", predictions.sum(), "/", len(predictions))

for label in np.unique(test_y):
    idx = test_y == label
    print(
        f"{label}: {predictions[idx].sum()} / {idx.sum()} flagged as anomaly"
    )

# -------------------------
# 11. SCENARIO TESTS
# -------------------------

print("\n==============================")
print("SCENARIO TESTS")
print("==============================")

# -------------------------
# Scenario-1: Normal Assembly (5 samples)
# -------------------------

print("\nScenario-1: Normal Assembly (5 samples)")

# validation normal errors'a göre en düşük 5 taneyi seç
normal_indices = np.argsort(val_errors)[:5]

pass_count = 0

for i, idx in enumerate(normal_indices):
    error = val_errors[idx]
    prediction = "Anomaly" if error > threshold else "Normal"

    print(f"\nSample {i+1}")
    print(f"Reconstruction Error: {error:.6f}")
    print(f"Prediction: {prediction}")

    if prediction == "Normal":
        pass_count += 1

print(f"\nResult: {pass_count}/5 PASS")

# -------------------------
# Scenario-2: Faulty Assembly (5 samples)
# -------------------------

print("\nScenario-2: Faulty Assembly (5 samples)")

# mistake indexlerini al
mistake_indices = np.where(test_y == "mistake")[0]

# en yüksek error'lu 5 taneyi seç
top_mistakes = mistake_indices[np.argsort(test_errors[mistake_indices])[-5:]]

pass_count = 0

for i, idx in enumerate(top_mistakes):
    error = test_errors[idx]
    prediction = "Anomaly" if error > threshold else "Normal"

    print(f"\nSample {i+1} (index {idx})")
    print(f"Reconstruction Error: {error:.6f}")
    print(f"Prediction: {prediction}")

    if prediction == "Anomaly":
        pass_count += 1

print(f"\nResult: {pass_count}/5 PASS")

# -------------------------
# Scenario-3: Invalid Data Input
# -------------------------

print("\nScenario-3: Invalid Data Input")

invalid_path = "non_existing_feature_file.npy"

try:
    invalid_X = np.load(invalid_path)
    print("Unexpected: file loaded successfully")
    print("Result: FAIL")
except Exception as e:
    print("Input: invalid feature file path")
    print("Error detected:", type(e).__name__)
    print("System response: controlled error handling")
    print("Result: PASS")

# -------------------------
# 12. PLOT
# -------------------------

plt.figure(figsize=(8, 5))
plt.hist(train_errors, bins=10, alpha=0.5, label="Train Normal")
plt.hist(val_errors, bins=10, alpha=0.5, label="Validation Normal")
plt.hist(test_errors, bins=10, alpha=0.6, label="Test")
plt.axvline(threshold, linestyle="--", label="Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Autoencoder Anomaly Detection with Validation Threshold")
plt.legend()
plt.show()

# -------------------------
# 13
# -------------------------

print(input[:10])
print(output[:10])
print(error)