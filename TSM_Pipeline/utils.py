# tsm_pipeline/utils.py
# ---------------------------------------------------------------------------
# Windowed feature extraction, checkpoint saving, STD validation tests.
# ---------------------------------------------------------------------------

import os
import gc
import torch
import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo

from .constants import VIDEO_FPS, NUM_FRAMES, FEATURE_DIM, VISUAL_DIM, POSE_DIM


# ---------------------------------------------------------------------------
# Windowed feature extraction
# ---------------------------------------------------------------------------

def extract_windowed_features(video_path, s_sec, e_sec, video_transform,
                               feature_extractor, device, window_size=2.0):
    """
    Extract TSM visual features for every non-overlapping window in [s_sec, e_sec].

    s_sec / e_sec must be in seconds (annotation frames / ANNOTATION_FPS).

    Returns:
        List of (window_start_sec, visual_feat [1, 2048]) tuples.
    """
    video         = EncodedVideo.from_path(video_path)
    file_duration = float(video.duration)
    num_windows   = max(1, int((e_sec - s_sec) / window_size))
    window_starts = [s_sec + i * window_size for i in range(num_windows)]

    results = []
    for ws in window_starts:
        ws = max(0.0, ws)
        we = min(ws + window_size, file_duration)
        if we - ws < 0.1:
            continue

        clip_data = video.get_clip(ws, we)
        if clip_data is None or clip_data.get("video") is None:
            print(f"  Warning: get_clip failed at {ws:.2f}s — skipping")
            continue

        clip_data   = video_transform(clip_data)
        clip_frames = clip_data["video"]

        if clip_frames.shape[1] != NUM_FRAMES:
            continue

        inp = clip_frames.unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(inp)  # (1, 2048)

        results.append((ws, feat))

    del video
    return results


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(all_features, all_labels, feature_filename,
                    label_filename, clip_idx):
    """Save partial flat extraction progress. Overwrites previous checkpoint."""
    try:
        m = np.concatenate(all_features, axis=0)
        l = np.array(all_labels, dtype=object)
        np.save(feature_filename.replace(".npy", "_checkpoint.npy"), m)
        np.save(label_filename.replace(".npy",  "_checkpoint.npy"), l)
        print(f"  [Checkpoint @ clip {clip_idx} — {m.shape[0]} windows]")
    except Exception as e:
        print(f"  [Checkpoint failed: {e}]")


# ---------------------------------------------------------------------------
# STD Validation Tests
# ---------------------------------------------------------------------------

def run_std_tests_flat(feature_file, label_file, expected_label_type='string'):
    """Validate a flat feature/label pair."""
    print("\n" + "=" * 55)
    print(f"📋 FLAT STD — {os.path.basename(feature_file)}")
    print("=" * 55)
    try:
        data   = np.load(feature_file)
        labels = np.load(label_file, allow_pickle=True)
        print(f"ℹ️  Shape: {data.shape} | Labels: {np.unique(labels)}")

        print(f"{'✅' if data.shape[1] == FEATURE_DIM else '❌'} Dim: {data.shape[1]}")
        clean = not np.isnan(data).any() and not np.isinf(data).any()
        print(f"{'✅' if clean else '❌'} NaN/Inf yok")
        print(f"{'✅' if data.shape[0] == labels.shape[0] else '❌'} "
              f"{data.shape[0]} feature == {labels.shape[0]} label")
        dead = (data == 0).all(axis=0).sum()
        print(f"{'✅' if dead == 0 else '⚠️'} Dead features: {dead}/{FEATURE_DIM}")
        zero = (data == 0).sum() / data.size
        print(f"ℹ️  Sıfır oranı: %{zero * 100:.2f}")

        visual = data[:, :VISUAL_DIM]
        pose   = data[:, VISUAL_DIM:]
        print(f"ℹ️  Visual({VISUAL_DIM}) mean:{visual.mean():.3f} std:{visual.std():.3f}")
        print(f"ℹ️  Pose  ({POSE_DIM})  mean:{pose.mean():.3f}   std:{pose.std():.3f}")

        if expected_label_type == 'binary':
            n_c = (labels == 0).sum()
            n_a = (labels == 1).sum()
            print(f"ℹ️  correct:{n_c} anomaly:{n_a} ({n_a / (n_c + n_a) * 100:.1f}%)")
            if n_c == 0: print("❌ Hiç correct window yok!")
            if n_a == 0: print("❌ Hiç anomaly window yok!")
    except Exception as e:
        print(f"❌ Hata: {e}")
    print("=" * 55 + "\n")


def run_std_tests_sequences(seq_feat_file, seq_lbl_file,
                             seq_window_lbl_file=None, seq_id_file=None):
    """Validate a sequence feature/label pair."""
    print("\n" + "=" * 55)
    print(f"📋 SEQUENCE STD — {os.path.basename(seq_feat_file)}")
    print("=" * 55)
    try:
        seqs   = np.load(seq_feat_file, allow_pickle=True)
        labels = np.load(seq_lbl_file,  allow_pickle=True)
        print(f"ℹ️  {len(seqs)} sekans | labels: {np.unique(labels)}")

        print(f"{'✅' if len(seqs) == len(labels) else '❌'} "
              f"Sekans/label: {len(seqs)}=={len(labels)}")
        bad = [i for i, s in enumerate(seqs) if s.shape[1] != FEATURE_DIM]
        print(f"{'✅' if not bad else '❌'} Dim {FEATURE_DIM}: "
              f"{'tümü doğru' if not bad else f'hatalı: {bad}'}")
        clean = not any(np.isnan(s).any() for s in seqs)
        print(f"{'✅' if clean else '❌'} NaN/Inf yok")

        lengths = [len(s) for s in seqs]
        print(f"ℹ️  Uzunluk min:{min(lengths)} max:{max(lengths)} "
              f"mean:{np.mean(lengths):.1f}")

        if seq_id_file and os.path.exists(seq_id_file):
            ids = np.load(seq_id_file, allow_pickle=True)
            for sid, lbl, ln in zip(ids, labels, lengths):
                print(f"     {sid} | {lbl} | {ln} windows")

        if seq_window_lbl_file and os.path.exists(seq_window_lbl_file):
            wlbls   = np.load(seq_window_lbl_file, allow_pickle=True)
            lmatch  = all(len(s) == len(wl) for s, wl in zip(seqs, wlbls))
            print(f"{'✅' if lmatch else '❌'} Window label uzunlukları eşleşiyor")
            total   = sum(len(wl) for wl in wlbls)
            anomaly = sum((wl == 1).sum() for wl in wlbls)
            print(f"ℹ️  correct:{total - anomaly} anomaly:{anomaly} "
                  f"({anomaly / total * 100:.1f}%)")
    except Exception as e:
        print(f"❌ Hata: {e}")
    print("=" * 55 + "\n")
