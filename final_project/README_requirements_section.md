## Requirements

The project uses Python with PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib, OpenCV, and Streamlit.

For the main project installation:

```bash
pip install -r requirements.txt
```

For detailed script-level dependency information, see:

```text
requirements_by_script.md
```

The exact development environment was exported to:

```text
requirements_full.txt
```

The development machine used a CUDA-enabled PyTorch build:

```text
torch==2.12.0+cu126
torchvision==0.27.0+cu126
torchaudio==2.11.0+cu126
```

If CUDA is not available or a different CUDA version is installed, PyTorch should be installed according to the target machine using the official PyTorch installation selector.
