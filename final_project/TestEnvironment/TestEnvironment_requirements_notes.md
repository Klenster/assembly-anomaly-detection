# Requirements Notes for TestEnvironment

## Required Python Version

Recommended:

```text
Python 3.11.x
```

The project was tested inside a Python virtual environment.

## Required Packages by Usage

| Package | Used for |
|---|---|
| `numpy` | Loading `.npy` feature and label files, array operations |
| `pandas` | Saving CSV outputs and reading `final_tuning_summary.csv` |
| `matplotlib` | Generating timeline, ROC, confusion matrix, distribution, and metric table PNG files |
| `scikit-learn` | Accuracy, Precision, Recall, F1, AUROC, confusion matrix, ROC curve |
| `joblib` | Loading `scaler.pkl` files |
| `torch` | Loading trained autoencoder weights and running inference |
| `torchvision` | Included for PyTorch environment compatibility |
| `torchaudio` | Included for PyTorch environment compatibility |

## Installation

Create and activate a virtual environment:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
```

Install requirements:

```powershell
pip install -r requirements.txt
```

If the CUDA-specific PyTorch packages fail to install from `requirements.txt`, install PyTorch separately using the official PyTorch selector. Example for CUDA 12.6:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then install the remaining packages:

```powershell
pip install numpy pandas matplotlib scikit-learn joblib
```

## CPU vs GPU

The script automatically selects GPU if CUDA is available:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If CUDA is unavailable, it runs on CPU. Inference is usually still manageable, but GPU is faster.

## Do Not Commit the Virtual Environment

Do not push the `venv/` folder to Git. Keep only:

```text
requirements.txt
README.md
TestSingleVideoWithFinalAE.py
```

Recommended `.gitignore` entries:

```gitignore
venv/
.venv/
__pycache__/
*.pyc
```
