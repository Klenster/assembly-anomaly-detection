import numpy as np
from pathlib import Path

folders = [
    r".\FullSequenceTest",
    r".\per_camera_slowfast",
    r".\per_camera_tsm",
    r".\Test&Train-MultiplePOV+newHands",
    r".\TrainSequenced",
    r".\TSMTrain(FlatveSekasn)&Test(sekans)",
]

for folder in folders:
    p = Path(folder)
    print("\n" + "="*80)
    print("FOLDER:", p)
    print("="*80)

    if not p.exists():
        print("not found")
        continue

    for f in sorted(p.glob("*.npy")):
        try:
            arr = np.load(f, allow_pickle=True)
            print(f"{f.name:55s} shape={arr.shape} dtype={arr.dtype}")
            if "label" in f.name:
                print("   unique:", np.unique(arr, return_counts=True))
        except Exception as e:
            print(f"{f.name:55s} ERROR: {e}")