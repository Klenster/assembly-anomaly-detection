import torch
import torch.nn as nn
import torchvision.models as tv_models
import json
import os
import time
import numpy as np
import gc
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

# ---------------------------------------------------------------------------
# Frame Space Constants
# ---------------------------------------------------------------------------
# Assembly101 annotation frame numbers are in 30fps space.
# Raw videos and hand pose JSONs are in 60fps space.
#   annotation frame → seconds : frame / ANNOTATION_FPS
#   seconds → video/pose frame : seconds * VIDEO_FPS

ANNOTATION_FPS = 30    # annotation CSV frame numbers
VIDEO_FPS      = 60    # actual video and pose JSON frame rate
NUM_FRAMES     = 16    # frames sampled per 2-second window for TSM

# Feature dimensions:
#   Visual (TSM ResNet-50) : 2048
#   Hand pose              : NUM_FRAMES × 21 keypoints × 3 values × 2 hands
#                          : 16 × 21 × 3 × 2 = 2016
#   Total                  : 2048 + 2016 = 4064
VISUAL_DIM   = 2048
POSE_DIM     = NUM_FRAMES * 21 * 3 * 2   # 2016
FEATURE_DIM  = VISUAL_DIM + POSE_DIM     # 4064

# ---------------------------------------------------------------------------
# Camera Constants
# ---------------------------------------------------------------------------
ACTIVE_CAMERAS = [
    'C10095',
    'C10118',
    'C10119',
    'C10390',
    'C10404',
]

# ---------------------------------------------------------------------------
# TSM Feature Extractor
# ---------------------------------------------------------------------------

class TSMExtractor(nn.Module):
    """
    Temporal Shift Module feature extractor built on ResNet-50.

    Architecture:
        ResNet-50 (pretrained on ImageNet)
        → remove final FC layer
        → per-frame features (NUM_FRAMES, 2048)
        → temporal average pooling
        → (1, 2048)

    The temporal shift is approximated by processing all frames through
    the shared ResNet backbone and averaging — giving a temporally-aware
    representation without requiring the full TSM library.
    """
    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet50(pretrained=True)
        # Remove the final FC classification layer
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1],  # up to avgpool
            nn.Flatten()
        )

    def forward(self, x):
        """
        x: (1, C, T, H, W) — batch=1, channels, frames, height, width
        returns: (1, 2048)
        """
        B, C, T, H, W = x.shape
        # Reshape to process each frame independently
        x = x.permute(0, 2, 1, 3, 4)          # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)          # (B*T, C, H, W)
        x = self.feature_extractor(x)           # (B*T, 2048)
        x = x.reshape(B, T, VISUAL_DIM)        # (B, T, 2048)
        x = x.mean(dim=1)                       # (B, 2048) temporal average
        return x


# ---------------------------------------------------------------------------
# Annotation Loading
# ---------------------------------------------------------------------------

def load_annotations(csv_path):
    """
    Load an Assembly101 annotation CSV for a single video.

    CSV format (no header):
        start_frame, end_frame, action, object, target, label, note

    Frame space:
        Annotation frames → 30fps space (ANNOTATION_FPS)
        Video/pose frames → 60fps space (VIDEO_FPS)

    Labels:
        'correct'    → training (normal)
        'mistake'    → testing  (anomaly)
        'correction' → testing  (anomaly)
    """
    import pandas as pd

    df = pd.read_csv(
        csv_path, header=None,
        names=['start_frame','end_frame','action','object','target','label','note']
    )
    for col in ['action','object','target','label']:
        df[col] = df[col].str.strip()

    df['start_time'] = df['start_frame'] / ANNOTATION_FPS
    df['end_time']   = df['end_frame']   / ANNOTATION_FPS

    # Guarantee temporal order within each session
    df = df.sort_values('start_frame').reset_index(drop=True)

    basename   = os.path.splitext(os.path.basename(csv_path))[0]
    anchor     = 'nusar-2021_action_both_'
    anchor_idx = basename.find(anchor)
    if anchor_idx == -1:
        raise ValueError(f"Cannot extract video_id from: {basename}")
    remainder = basename[anchor_idx + len(anchor):]
    parts     = remainder.split('_')
    video_id  = f"{parts[0]}_{parts[1]}"
    df['video_id'] = video_id
    return df


def load_all_annotations(annotation_dir):
    import pandas as pd
    all_dfs = []
    for fname in sorted(os.listdir(annotation_dir)):
        if not fname.endswith('.csv'):
            continue
        try:
            df = load_annotations(os.path.join(annotation_dir, fname))
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: skipping {fname} — {e}")
    if not all_dfs:
        raise RuntimeError(f"No CSVs in: {annotation_dir}")
    return pd.concat(all_dfs, ignore_index=True)


def get_session_num(video_id):
    """'9033-c13a_9033' → '9033'"""
    return video_id.split('-')[0]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AssemblyHybridDataset(Dataset):
    def __init__(self, video_dir, json_dir, annotations):
        self.video_dir   = video_dir
        self.json_dir    = json_dir
        self.annotations = annotations.reset_index(drop=True)
        self._pose_cache = {}

        self.video_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    # Scale BEFORE subsampling — prevents 800MB RAM per window
                    ShortSideScale(size=256),
                    UniformTemporalSubsample(NUM_FRAMES),  # 16 frames for TSM
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(
                        mean=[0.45, 0.45, 0.45],
                        std=[0.225, 0.225, 0.225]
                    ),
                    CenterCropVideo(crop_size=224),
                ]),
            ),
        ])

    def _find_json_path(self, video_id):
        exact = os.path.join(self.json_dir, f"{video_id}.json")
        if os.path.exists(exact):
            return exact
        for fname in os.listdir(self.json_dir):
            if video_id in fname and fname.endswith('.json'):
                return os.path.join(self.json_dir, fname)
        raise FileNotFoundError(f"No JSON for: {video_id}")

    def _get_pose_lookup(self, video_id):
        if video_id not in self._pose_cache:
            json_path = self._find_json_path(video_id)
            if os.path.getsize(json_path) == 0:
                self._pose_cache[video_id] = {}
                return self._pose_cache[video_id]
            with open(json_path, 'r') as f:
                full_data = json.load(f)
            lookup = {}
            if isinstance(full_data, list):
                # Old format: [{frame_index, landmarks, ...}]
                for entry in full_data:
                    flat = []
                    for hand_key in ['0', '1']:
                        if hand_key in entry['landmarks']:
                            for point in entry['landmarks'][hand_key]:
                                flat.extend(point)
                        else:
                            flat.extend([0.0] * 63)  # 21 × 3
                    lookup[entry['frame_index']] = flat
            elif isinstance(full_data, dict):
                # New landmarks3D format: {"frame": {"0": [...], "1": [...]}}
                for frame_str, hands in full_data.items():
                    flat = []
                    for hand_key in ['0', '1']:
                        if hand_key in hands:
                            for point in hands[hand_key]:
                                flat.extend(point)
                        else:
                            flat.extend([0.0] * 63)  # 21 × 3
                    lookup[int(frame_str)] = flat
            self._pose_cache[video_id] = lookup
        return self._pose_cache[video_id]

    def load_hand_poses(self, video_id, start_frame_60fps):
        """
        Load NUM_FRAMES (16) consecutive hand pose frames.
        start_frame_60fps must be in VIDEO_FPS (60fps) space.
        Returns tensor of shape (NUM_FRAMES, 126)
        """
        lookup = self._get_pose_lookup(video_id)
        poses  = [
            # Fallback: 2 hands × 21 keypoints × 3 values = 126
            lookup.get(start_frame_60fps + i, [0.0] * 126)
            for i in range(NUM_FRAMES)
        ]
        return torch.tensor(poses, dtype=torch.float32)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.annotations.iloc[idx]


# ---------------------------------------------------------------------------
# Windowed Feature Extraction
# ---------------------------------------------------------------------------

def extract_windowed_features(video_path, s_sec, e_sec, video_transform,
                               feature_extractor, device, window_size=2.0):
    """
    Extract TSM visual features for every non-overlapping 2-second window.

    s_sec / e_sec must be in seconds (converted from 30fps annotation frames).

    TSM input: (1, C, T, H, W) where T=NUM_FRAMES=16
    TSM output: (1, 2048) after temporal average pooling

    Returns:
        List of (window_start_sec, visual_feat [1, 2048]) tuples.
    """
    video         = EncodedVideo.from_path(video_path)
    file_duration = float(video.duration)
    action_dur    = e_sec - s_sec
    num_windows   = max(1, int(action_dur / window_size))
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
        clip_frames = clip_data["video"]  # (C, T, H, W)

        if clip_frames.shape[1] != NUM_FRAMES:
            continue

        # TSM input: (1, C, T, H, W)
        inp = clip_frames.unsqueeze(0).to(device)

        with torch.no_grad():
            feat = feature_extractor(inp)  # (1, 2048)

        results.append((ws, feat))

    del video
    return results


# ---------------------------------------------------------------------------
# STD Tests
# ---------------------------------------------------------------------------

def run_std_tests_flat(feature_file, label_file, expected_label_type='string'):
    print("\n" + "="*55)
    print(f"📋 FLAT STD — {os.path.basename(feature_file)}")
    print("="*55)
    try:
        data   = np.load(feature_file)
        labels = np.load(label_file, allow_pickle=True)
        print(f"ℹ️  Shape        : {data.shape}")
        print(f"ℹ️  Unique labels: {np.unique(labels)}")

        print(f"{'✅' if data.shape[1]==FEATURE_DIM else '❌'} "
              f"Dim: {data.shape[1]} (beklenen {FEATURE_DIM})")
        clean = not np.isnan(data).any() and not np.isinf(data).any()
        print(f"{'✅' if clean else '❌'} NaN/Inf yok")
        match = data.shape[0] == labels.shape[0]
        print(f"{'✅' if match else '❌'} "
              f"{data.shape[0]} feature == {labels.shape[0]} label")
        dead  = (data == 0).all(axis=0).sum()
        print(f"{'✅' if dead==0 else '⚠️'} Dead features: {dead}/{FEATURE_DIM}")
        zero  = (data == 0).sum() / data.size
        print(f"ℹ️  Sıfır oranı : %{zero*100:.2f}")

        # Visual vs pose
        visual = data[:, :VISUAL_DIM]
        pose   = data[:, VISUAL_DIM:]
        print(f"ℹ️  Visual({VISUAL_DIM}) — "
              f"mean:{visual.mean():.3f} std:{visual.std():.3f} "
              f"min:{visual.min():.3f} max:{visual.max():.3f}")
        print(f"ℹ️  Pose  ({POSE_DIM})  — "
              f"mean:{pose.mean():.3f}   std:{pose.std():.3f}   "
              f"min:{pose.min():.3f}   max:{pose.max():.3f}")

        if expected_label_type == 'binary':
            n_c = (labels == 0).sum()
            n_a = (labels == 1).sum()
            print(f"ℹ️  correct:{n_c} anomaly:{n_a} "
                  f"({n_a/(n_c+n_a)*100:.1f}% anomaly)")
            if n_c == 0: print("❌ UYARI: Hiç correct window yok!")
            if n_a == 0: print("❌ UYARI: Hiç anomaly window yok!")
    except Exception as e:
        print(f"❌ Hata: {e}")
    print("="*55 + "\n")


def run_std_tests_sequences(seq_feat_file, seq_lbl_file,
                             seq_window_lbl_file=None, seq_id_file=None):
    print("\n" + "="*55)
    print(f"📋 SEQUENCE STD — {os.path.basename(seq_feat_file)}")
    print("="*55)
    try:
        seqs   = np.load(seq_feat_file, allow_pickle=True)
        labels = np.load(seq_lbl_file,  allow_pickle=True)
        print(f"ℹ️  {len(seqs)} sekans | labels: {np.unique(labels)}")

        match = len(seqs) == len(labels)
        print(f"{'✅' if match else '❌'} Sekans/label: {len(seqs)}=={len(labels)}")
        bad = [i for i,s in enumerate(seqs) if s.shape[1]!=FEATURE_DIM]
        print(f"{'✅' if not bad else '❌'} Dim {FEATURE_DIM}: "
              f"{'tümü doğru' if not bad else f'hatalı: {bad}'}")
        clean = not any(np.isnan(s).any() for s in seqs)
        print(f"{'✅' if clean else '❌'} NaN/Inf yok")
        lengths = [len(s) for s in seqs]
        print(f"ℹ️  Uzunluk — min:{min(lengths)} max:{max(lengths)} "
              f"mean:{np.mean(lengths):.1f} toplam:{sum(lengths)}")

        if seq_id_file and os.path.exists(seq_id_file):
            ids = np.load(seq_id_file, allow_pickle=True)
            for sid, lbl, ln in zip(ids, labels, lengths):
                print(f"     {sid} | {lbl} | {ln} windows")

        if seq_window_lbl_file and os.path.exists(seq_window_lbl_file):
            wlbls = np.load(seq_window_lbl_file, allow_pickle=True)
            lmatch = all(len(s)==len(wl) for s,wl in zip(seqs,wlbls))
            print(f"{'✅' if lmatch else '❌'} Window label uzunlukları eşleşiyor")
            total   = sum(len(wl) for wl in wlbls)
            anomaly = sum((wl==1).sum() for wl in wlbls)
            print(f"ℹ️  correct:{total-anomaly} anomaly:{anomaly} "
                  f"({anomaly/total*100:.1f}%)")
    except Exception as e:
        print(f"❌ Hata: {e}")
    print("="*55 + "\n")


# ---------------------------------------------------------------------------
# Train Extraction — correct clips only
# ---------------------------------------------------------------------------

def run_train_extractor(dataset, feature_filename, label_filename,
                        feature_extractor, device, window_size=2.0,
                        checkpoint_every=10, cameras=ACTIVE_CAMERAS):
    """
    Train: correct clips only.

    Saves:
        Flat      → (total_windows, 4064)  label='correct'
        Sequences → one per session+camera in temporal order
                    35 sequences (7 sessions × 5 cameras)

    Files:
        {feature_filename}
        {label_filename}
        {feature_filename}_sequences.npy
        {label_filename}_sequences.npy
        {feature_filename}_sequence_ids.npy
    """
    all_features           = []
    all_labels             = []
    session_camera_windows = {}   # {(video_id, camera): [np arrays]}

    total = len(dataset)
    print(f"\n--- TRAIN EXTRACTION (TSM {NUM_FRAMES} frames) ---")
    print(f"Device:{device} | Clips:{total} | Cameras:{cameras}")
    print(f"Feature dim: visual({VISUAL_DIM}) + pose({POSE_DIM}) = {FEATURE_DIM}")

    for idx in range(total):
        anno        = dataset.annotations.iloc[idx]
        video_id    = anno['video_id']
        label       = anno['label']
        session_num = get_session_num(video_id)
        s_sec = anno['start_frame'] / ANNOTATION_FPS
        e_sec = anno['end_frame']   / ANNOTATION_FPS
        clip_had_windows = False

        for camera in cameras:
            video_path = os.path.join(
                dataset.video_dir, f"{session_num}{camera}_rgb.mp4")
            if not os.path.exists(video_path):
                print(f"  Warning: missing {session_num}{camera}_rgb.mp4")
                continue

            windows = extract_windowed_features(
                video_path, s_sec, e_sec,
                dataset.video_transform, feature_extractor,
                device, window_size)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not windows:
                continue

            clip_had_windows = True
            key = (video_id, camera)
            if key not in session_camera_windows:
                session_camera_windows[key] = []

            for (ws, visual_feat) in windows:
                # Convert seconds → 60fps frame index for pose lookup
                wf60      = int(ws * VIDEO_FPS)
                hand_poses = dataset.load_hand_poses(video_id, wf60)
                hand_feat  = hand_poses.view(1, -1).to(device)
                # (1, NUM_FRAMES×126) = (1, 2016)

                # Fusion: visual(2048) + pose(2016) = 4064
                combined_np = torch.cat(
                    (visual_feat, hand_feat), dim=1).cpu().numpy()

                all_features.append(combined_np)
                all_labels.append(label)
                session_camera_windows[key].append(combined_np)

            print(f"  [{idx+1}/{total}] {video_id}|{camera} "
                  f"windows:{len(windows)} dur:{e_sec-s_sec:.1f}s")

        if not clip_had_windows:
            print(f"  Warning: no windows — {video_id}")
        if (idx+1) % checkpoint_every == 0:
            _save_checkpoint(all_features, all_labels,
                             feature_filename, label_filename, idx+1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save flat
    fm = np.concatenate(all_features, axis=0)   # (total_windows, 4064)
    fl = np.array(all_labels, dtype=object)
    np.save(feature_filename, fm)
    np.save(label_filename,   fl)
    print(f"\n✅ Flat: {feature_filename} | {fm.shape}")

    # Save sequences
    seqs, seq_lbls, seq_ids = [], [], []
    for (vid, cam), wins in session_camera_windows.items():
        if not wins: continue
        seqs.append(np.vstack(wins))
        seq_lbls.append('correct')
        seq_ids.append(f"{vid}_{cam}")

    sf = feature_filename.replace(".npy", "_sequences.npy")
    sl = label_filename.replace(".npy",   "_sequences.npy")
    si = feature_filename.replace(".npy",  "_sequence_ids.npy")
    np.save(sf, np.array(seqs,     dtype=object), allow_pickle=True)
    np.save(sl, np.array(seq_lbls, dtype=object), allow_pickle=True)
    np.save(si, np.array(seq_ids,  dtype=object), allow_pickle=True)

    lengths = [len(s) for s in seqs]
    print(f"✅ Sequences: {sf} | {len(seqs)} seqs | "
          f"min:{min(lengths)} max:{max(lengths)} mean:{np.mean(lengths):.1f}")


# ---------------------------------------------------------------------------
# Test Extraction — full sessions (correct + mistake + correction)
# ---------------------------------------------------------------------------

def run_test_extractor(full_df, video_dir, json_dir, feature_extractor,
                       device, video_transform, window_size=2.0,
                       cameras=ACTIVE_CAMERAS):
    """
    Test: full sessions in temporal order.

    Every session processed complete — correct + mistake + correction clips
    in start_frame order. Each window gets a binary label:
        0 = correct
        1 = mistake or correction (anomaly)

    Flat output:
        test_features.npy           → (N, 4064) all windows
        test_window_labels.npy      → (N,) binary 0/1

    Sequence output (per session+camera):
        test_sequences.npy                  → variable length sequences
        test_sequence_labels.npy            → 0=no anomaly, 1=has anomaly
        test_sequence_window_labels.npy     → binary per-window labels
        test_sequence_ids.npy               → video_id_camera strings
    """
    all_features      = []
    all_window_labels = []
    sequences               = []
    sequence_labels         = []
    sequence_window_labels  = []
    sequence_ids            = []
    sequence_window_times   = []

    pose_ds = AssemblyHybridDataset(video_dir, json_dir, full_df)

    print(f"\n--- TEST EXTRACTION (TSM {NUM_FRAMES} frames) ---")
    print(f"Feature dim: visual({VISUAL_DIM}) + pose({POSE_DIM}) = {FEATURE_DIM}")

    for video_id, session_df in full_df.groupby('video_id'):
        session_df  = session_df.sort_values('start_frame').reset_index(drop=True)
        session_num = get_session_num(video_id)
        has_anomaly = (session_df['label'] != 'correct').any()

        print(f"\n  Session:{video_id} | clips:{len(session_df)} "
              f"| has_anomaly:{has_anomaly}")

        for camera in cameras:
            video_path = os.path.join(
                video_dir, f"{session_num}{camera}_rgb.mp4")
            if not os.path.exists(video_path):
                print(f"    Warning: missing {session_num}{camera}_rgb.mp4")
                continue

            seq_wins  = []
            seq_wlbls = []
            seq_wtimes = []

            for _, anno in session_df.iterrows():
                s_sec        = anno['start_frame'] / ANNOTATION_FPS
                e_sec        = anno['end_frame']   / ANNOTATION_FPS
                window_label = 0 if anno['label'] == 'correct' else 1

                windows = extract_windowed_features(
                    video_path, s_sec, e_sec,
                    pose_ds.video_transform, feature_extractor,
                    device, window_size)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not windows:
                    continue

                for (ws, visual_feat) in windows:
                    wf60      = int(ws * VIDEO_FPS)
                    hand_poses = pose_ds.load_hand_poses(video_id, wf60)
                    hand_feat  = hand_poses.view(1, -1).to(device)

                    combined_np = torch.cat(
                        (visual_feat, hand_feat), dim=1).cpu().numpy()

                    # Flat
                    all_features.append(combined_np)
                    all_window_labels.append(window_label)

                    # Sequence
                    seq_wins.append(combined_np)
                    seq_wlbls.append(window_label)
                    seq_wtimes.append(ws)    

            if not seq_wins:
                continue

            seq_arr  = np.vstack(seq_wins)
            wlbl_arr = np.array(seq_wlbls, dtype=np.int32)

            sequences.append(seq_arr)
            sequence_labels.append(1 if has_anomaly else 0)
            sequence_window_labels.append(wlbl_arr)
            sequence_ids.append(f"{video_id}_{camera}")
            sequence_window_times.append(
                np.array(seq_wtimes, dtype=np.float32))

            n_c = (wlbl_arr == 0).sum()
            n_a = (wlbl_arr == 1).sum()
            print(f"    {camera} | total:{len(seq_wins)} "
                  f"correct:{n_c} anomaly:{n_a}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save flat
    fm  = np.concatenate(all_features, axis=0)   # (N, 4064)
    fwl = np.array(all_window_labels, dtype=np.int32)
    np.save("test_features_tsm.npy",      fm)
    np.save("test_window_labels_tsm.npy", fwl)
    print(f"\n✅ Flat: test_features_tsm.npy | {fm.shape}")
    print(f"   correct:{(fwl==0).sum()} anomaly:{(fwl==1).sum()}")

    # Save sequences
    np.save("test_sequences_tsm.npy",
            np.array(sequences,              dtype=object), allow_pickle=True)
    np.save("test_sequence_labels_tsm.npy",
            np.array(sequence_labels,        dtype=np.int32))
    np.save("test_sequence_window_labels_tsm.npy",
            np.array(sequence_window_labels, dtype=object), allow_pickle=True)
    np.save("test_sequence_ids_tsm.npy",
            np.array(sequence_ids,           dtype=object), allow_pickle=True)
    np.save("test_sequence_window_times.npy",
        np.array(sequence_window_times, dtype=object),allow_pickle=True)

    lengths = [len(s) for s in sequences]
    print(f"✅ Sequences: test_sequences_tsm.npy | {len(sequences)} seqs | "
          f"min:{min(lengths)} max:{max(lengths)} mean:{np.mean(lengths):.1f}")
    print(f"   Anomalili session: {sum(sequence_labels)}/{len(sequence_labels)}")


# ---------------------------------------------------------------------------
# Checkpoint Helper
# ---------------------------------------------------------------------------

def _save_checkpoint(all_features, all_labels, feature_filename,
                     label_filename, clip_idx):
    try:
        m = np.concatenate(all_features, axis=0)
        l = np.array(all_labels, dtype=object)
        np.save(feature_filename.replace(".npy","_checkpoint.npy"), m)
        np.save(label_filename.replace(".npy","_checkpoint.npy"),   l)
        print(f"  [Checkpoint @ clip {clip_idx} — {m.shape[0]} windows]")
    except Exception as e:
        print(f"  [Checkpoint failed: {e}]")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VIDEO_DIR      = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\videos"
    JSON_DIR       = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\HandPoses"
    ANNOTATION_DIR = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\annots"
    WINDOW_SIZE    = 2.0

    total_start = time.time()

    print(f"TSM Feature Extraction")
    print(f"Frames per window : {NUM_FRAMES}")
    print(f"Visual dim        : {VISUAL_DIM}")
    print(f"Pose dim          : {POSE_DIM}")
    print(f"Total feature dim : {FEATURE_DIM}")

    full_df  = load_all_annotations(ANNOTATION_DIR)
    train_df = full_df[full_df['label'] == 'correct'].copy()

    n_correct     = (full_df['label'] == 'correct').sum()
    n_mistakes    = (full_df['label'] == 'mistake').sum()
    n_corrections = (full_df['label'] == 'correction').sum()
    print(f"\nAnnotations — correct:{n_correct} "
          f"mistake:{n_mistakes} correction:{n_corrections}")

    assert len(train_df) > 0,              "train_df boş"
    assert n_mistakes + n_corrections > 0, "anomaly clip bulunamadı"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build TSM feature extractor
    feature_extractor = TSMExtractor().to(device)
    feature_extractor.eval()

    # -----------------------------------------------------------------------
    # Train extraction
    # -----------------------------------------------------------------------
    """"
    train_dataset = AssemblyHybridDataset(VIDEO_DIR, JSON_DIR, train_df)
    run_train_extractor(
        train_dataset,
        feature_filename  = "train_features_correct_tsm.npy",
        label_filename    = "train_labels_tsm.npy",
        feature_extractor = feature_extractor,
        device            = device,
        window_size       = WINDOW_SIZE,
        cameras           = ACTIVE_CAMERAS
    )

    print(f"\n⏱  Train: {time.time()-total_start:.1f}s")

    run_std_tests_flat(
        "train_features_correct_tsm.npy",
        "train_labels_tsm.npy",
        expected_label_type='string'
    )
    run_std_tests_sequences(
        "train_features_correct_tsm_sequences.npy",
        "train_labels_tsm_sequences.npy",
        seq_id_file="train_features_correct_tsm_sequence_ids.npy"
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Memory cleared.")
     """
    # -----------------------------------------------------------------------
    # Test extraction
    # -----------------------------------------------------------------------
    test_start = time.time()
    run_test_extractor(
        full_df           = full_df,
        video_dir         = VIDEO_DIR,
        json_dir          = JSON_DIR,
        feature_extractor = feature_extractor,
        device            = device,
        video_transform   = AssemblyHybridDataset(
                                VIDEO_DIR, JSON_DIR, full_df).video_transform,
        window_size       = WINDOW_SIZE,
        cameras           = ACTIVE_CAMERAS
    )

    print(f"\n⏱  Test: {time.time()-test_start:.1f}s")

    run_std_tests_flat(
        "test_features_tsm.npy",
        "test_window_labels_tsm.npy",
        expected_label_type='binary'
    )
    run_std_tests_sequences(
        "test_sequences_tsm.npy",
        "test_sequence_labels_tsm.npy",
        seq_window_lbl_file="test_sequence_window_labels_tsm.npy",
        seq_id_file="test_sequence_ids_tsm.npy"
    )

    print(f"\n⏱  Toplam: {time.time()-total_start:.1f}s")

    del feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Done.")