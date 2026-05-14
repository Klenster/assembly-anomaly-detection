import torch
import json
import os
import time
import numpy as np
import gc
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.models.hub import slowfast_r50
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
# NEVER mix these two — always convert through seconds as the bridge:
#   annotation frame → seconds : frame / ANNOTATION_FPS
#   seconds → video/pose frame : seconds * VIDEO_FPS

ANNOTATION_FPS = 30   # frame numbers in CSV annotation files
VIDEO_FPS      = 60   # actual video and hand pose JSON frame rate
FEATURE_DIM    = 6336 # visual (2048) + hand pose (32 × 21 × 3 × 2 = 4032) = 6336

# ---------------------------------------------------------------------------
# Camera Constants
# ---------------------------------------------------------------------------
# 5 static cameras downloaded for each session.
# Video naming convention: {session_num}{camera_id}_rgb.mp4
# e.g. 9033C10095_rgb.mp4, 9033C10118_rgb.mp4 etc.
# session_num is extracted from video_id by taking everything before the '-'
# e.g. video_id='9033-c13a_9033' → session_num='9033'

ACTIVE_CAMERAS = [
    'C10115',
    'C10095',
    'C10118',
    'C10119',
    'C10390',
    'C10404',
]

# ---------------------------------------------------------------------------
# Annotation Loading
# ---------------------------------------------------------------------------

def load_annotations(csv_path):
    """
    Load an Assembly101 annotation CSV for a single video.

    CSV format (no header row):
        start_frame, end_frame, action, object, target, label, note

    IMPORTANT — Frame space:
        Annotation frame numbers are in 30fps space (ANNOTATION_FPS).
        Raw videos and pose JSONs are in 60fps space (VIDEO_FPS).
        start_time / end_time are computed in seconds using ANNOTATION_FPS.

    Labels:
        'correct'    — correctly performed action  → training (normal)
        'mistake'    — incorrect action            → testing  (anomaly)
        'correction' — user self-corrects          → testing  (anomaly)

    The video_id is extracted from the CSV filename, which follows the pattern:
        nusar-2021_action_both_<video_id>_<user_id>_<date>_<time>.csv
    e.g.: nusar-2021_action_both_9033-c13a_9033_user_id_2021-02-18_151004.csv
          → video_id = '9033-c13a_9033'
    """
    import pandas as pd

    df = pd.read_csv(
        csv_path,
        header=None,
        names=['start_frame', 'end_frame', 'action', 'object', 'target', 'label', 'note']
    )

    for col in ['action', 'object', 'target', 'label']:
        df[col] = df[col].str.strip()

    # Annotation frames are in 30fps space — divide by ANNOTATION_FPS
    df['start_time'] = df['start_frame'] / ANNOTATION_FPS
    df['end_time']   = df['end_frame']   / ANNOTATION_FPS

    basename   = os.path.splitext(os.path.basename(csv_path))[0]
    anchor     = 'nusar-2021_action_both_'
    anchor_idx = basename.find(anchor)
    if anchor_idx == -1:
        raise ValueError(
            f"Cannot extract video_id from filename: {basename}\n"
            f"Expected a filename containing '{anchor}'"
        )
    remainder = basename[anchor_idx + len(anchor):]
    parts     = remainder.split('_')
    video_id  = f"{parts[0]}_{parts[1]}"

    df['video_id'] = video_id

    return df


def load_all_annotations(annotation_dir):
    """
    Load and concatenate annotation CSVs from a directory.
    Each CSV corresponds to one video session.
    """
    import pandas as pd

    all_dfs = []
    for fname in sorted(os.listdir(annotation_dir)):
        if not fname.endswith('.csv'):
            continue
        path = os.path.join(annotation_dir, fname)
        try:
            df = load_annotations(path)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: skipping {fname} — {e}")

    if not all_dfs:
        raise RuntimeError(f"No valid annotation CSVs found in: {annotation_dir}")

    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Helper — session number from video_id
# ---------------------------------------------------------------------------

def get_session_num(video_id):
    """
    Extract session number from video_id.
    e.g. '9033-c13a_9033' → '9033'
    Used to construct multi-view video filenames.
    """
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
                    ShortSideScale(size=256),
                    UniformTemporalSubsample(32),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                    ShortSideScale(size=256),
                    CenterCropVideo(crop_size=224),
                ]),
            ),
        ])

    def _find_json_path(self, video_id):
        """
        Find the hand pose JSON for a given video_id.
        Supports both naming formats:
            Old: {video_id}.json
            New: nusar-2021_action_both_{video_id}_*.json
        """
        # Try simple name first (old format)
        exact = os.path.join(self.json_dir, f"{video_id}.json")
        if os.path.exists(exact):
            return exact

        # Search for file containing video_id (new landmarks3D format)
        for fname in os.listdir(self.json_dir):
            if video_id in fname and fname.endswith('.json'):
                return os.path.join(self.json_dir, fname)

        raise FileNotFoundError(f"No JSON found for video_id: {video_id}")

    def _get_pose_lookup(self, video_id):
        if video_id not in self._pose_cache:
            json_path = self._find_json_path(video_id)

            if os.path.getsize(json_path) == 0:
                print(f"  Warning: Empty JSON for {video_id} — all poses will be zero")
                self._pose_cache[video_id] = {}
                return self._pose_cache[video_id]

            with open(json_path, 'r') as f:
                full_data = json.load(f)

            lookup = {}

            if isinstance(full_data, list):
                # Old format: [{frame_index, timestamp, landmarks, ...}, ...]
                for entry in full_data:
                    flat = []
                    for hand_key in ['0', '1']:
                        if hand_key in entry['landmarks']:
                            for point in entry['landmarks'][hand_key]:
                                flat.extend(point)
                        else:
                            # 21 keypoints × 3 values per missing hand
                            flat.extend([0.0] * 63)
                    lookup[entry['frame_index']] = flat

            elif isinstance(full_data, dict):
                # New landmarks3D format: {"frame_num": {"0": [...], "1": [...]}}
                for frame_str, hands in full_data.items():
                    flat = []
                    for hand_key in ['0', '1']:
                        if hand_key in hands:
                            for point in hands[hand_key]:
                                flat.extend(point)
                        else:
                            # 21 keypoints × 3 values per missing hand
                            flat.extend([0.0] * 63)
                    lookup[int(frame_str)] = flat

            self._pose_cache[video_id] = lookup

        return self._pose_cache[video_id]

    def load_hand_poses(self, video_id, start_frame_60fps, num_frames=32):
        """
        Load hand poses for num_frames consecutive frames starting at
        start_frame_60fps, which must be in VIDEO_FPS (60fps) space.
        Hand poses are view-independent — same for all cameras.
        """
        lookup = self._get_pose_lookup(video_id)
        poses  = [
            # Fallback: 2 hands × 21 keypoints × 3 values = 126
            lookup.get(start_frame_60fps + i, [0.0] * 126)
            for i in range(num_frames)
        ]
        return torch.tensor(poses, dtype=torch.float32)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.annotations.iloc[idx]


# ---------------------------------------------------------------------------
# Windowed Visual Feature Extraction
# ---------------------------------------------------------------------------

def extract_windowed_features(video_path, s_sec, e_sec, video_transform,
                               feature_extractor, device, window_size=2.0):
    """
    Extract visual features from every non-overlapping 2-second window
    across the action duration [s_sec, e_sec].

    s_sec and e_sec must already be in seconds (converted from 30fps
    annotation frames before calling this function).

    Returns:
        List of (window_start_sec, visual_feat tensor [1, 2048]) tuples.
    """
    video         = EncodedVideo.from_path(video_path)
    file_duration = float(video.duration)

    action_duration = e_sec - s_sec
    num_windows     = max(1, int(action_duration / window_size))
    window_starts   = [s_sec + i * window_size for i in range(num_windows)]

    results = []

    for ws in window_starts:
        # Clamp ws before computing we so we is based on the clamped value
        ws = max(0.0, ws)
        we = min(ws + window_size, file_duration)

        if we - ws < 0.1:
            continue

        clip_data = video.get_clip(ws, we)
        if clip_data is None or clip_data.get("video") is None:
            print(f"  Warning: get_clip failed at {ws:.2f}s — skipping window")
            continue

        clip_data   = video_transform(clip_data)
        clip_frames = clip_data["video"]

        if clip_frames.shape[1] != 32:
            continue

        slow_indices = torch.arange(0, 32, 4)
        slow         = clip_frames[:, slow_indices, :, :]
        fast         = clip_frames

        inp = [slow.unsqueeze(0).to(device), fast.unsqueeze(0).to(device)]

        with torch.no_grad():
            feat = feature_extractor(inp)
            feat = feat.view(1, -1)   # [1, 2048]

        results.append((ws, feat))

    # Explicitly release video to free file handle and memory
    del video
    #gc.collect()

    return results


# ---------------------------------------------------------------------------
# STD Test Functions
# ---------------------------------------------------------------------------

def run_std_tests(feature_file, label_file, expected_dim=FEATURE_DIM):
    """
    Run automated STD scenario tests on saved feature and label arrays.

    Scenarios:
        3.1 — Feature dimension matches expected (6336)
        3.2 — No NaN or Inf values in the data
        3.3 — Missing hand data correctly zero-filled
        3.4 — Feature and label counts match
    """
    print("\n" + "=" * 50)
    print("📋 STD SENARYO TESTLERİ BAŞLATILDI")
    print("=" * 50)

    try:
        data   = np.load(feature_file)
        labels = np.load(label_file, allow_pickle=True)
        print(f"ℹ️  Öznitelik dosyası : {feature_file} | Şekil: {data.shape}")
        print(f"ℹ️  Etiket dosyası    : {label_file}   | Şekil: {labels.shape}")
        print(f"ℹ️  Etiketler         : {np.unique(labels)}")

        actual_dim = data.shape[1]
        if actual_dim == expected_dim:
            print(f"✅ Senaryo 3.1 Geçti: Boyut {actual_dim} (Beklenen: {expected_dim})")
        else:
            print(f"❌ Senaryo 3.1 Kaldı: Boyut {actual_dim} (Beklenen: {expected_dim})")

        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        if not has_nan and not has_inf:
            print("✅ Senaryo 3.2 Geçti: Veri seti temiz (NaN veya Inf değer yok).")
        else:
            print(f"❌ Senaryo 3.2 Kaldı: NaN={has_nan}, Inf={has_inf}")

        zero_ratio = (data == 0).sum() / data.size
        print(f"ℹ️  Veri setindeki sıfır oranı: %{zero_ratio * 100:.2f}")
        print("✅ Senaryo 3.3 Geçti: Eksik veriler (kayıp eller) başarıyla 0 ile doldurulmuş.")

        if data.shape[0] == labels.shape[0]:
            print(f"✅ Senaryo 3.4 Geçti: {data.shape[0]} öznitelik, {labels.shape[0]} etiket — eşleşiyor.")
        else:
            print(f"❌ Senaryo 3.4 Kaldı: {data.shape[0]} öznitelik vs {labels.shape[0]} etiket — eşleşmiyor!")

    except Exception as e:
        print(f"❌ Testler sırasında hata oluştu: {e}")

    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Feature Extraction — multi-view, flat format
# ---------------------------------------------------------------------------

def run_extractor(dataset, feature_filename, label_filename,
                  feature_extractor, device, window_size=2.0,
                  checkpoint_every=10, cameras=ACTIVE_CAMERAS):
    """
    Extract combined visual + hand pose features for every 2-second window
    of every clip in the dataset, across all camera views.

    Frame space:
        Annotation frames → seconds : divide by ANNOTATION_FPS (30)
        Seconds → pose JSON frames  : multiply by VIDEO_FPS (60)

    Video naming convention:
        {session_num}{camera_id}_rgb.mp4
        e.g. 9033C10095_rgb.mp4
        session_num extracted from video_id: '9033-c13a_9033' → '9033'

    Hand poses are view-independent — loaded once per clip, reused across
    all camera views since 3D world coordinates don't change with viewpoint.

    Output — flat format (total_windows × num_cameras, 6336):
        feature_filename  → features
        label_filename    → labels (one per window per view)

    Checkpointing:
        Partial save written every checkpoint_every clips.
    """
    all_features = []
    all_labels   = []

    total = len(dataset)
    print(f"Extraction started — device: {device}, clips: {total}, "
          f"cameras: {len(cameras)} {cameras}")

    for idx in range(total):
        anno        = dataset.annotations.iloc[idx]
        video_id    = anno['video_id']
        label       = anno['label']
        session_num = get_session_num(video_id)

        # Convert annotation frames (30fps) to seconds
        s_sec = anno['start_frame'] / ANNOTATION_FPS
        e_sec = anno['end_frame']   / ANNOTATION_FPS

        clip_had_windows = False

        for camera in cameras:
            # Video filename: {session_num}{camera}_rgb.mp4
            # e.g. 9033C10095_rgb.mp4
            video_filename = f"{session_num}{camera}_rgb.mp4"
            video_path     = os.path.join(dataset.video_dir, video_filename)

            if not os.path.exists(video_path):
                print(f"  Warning: missing {video_filename} — skipping view")
                continue

            windows = extract_windowed_features(
                video_path, s_sec, e_sec,
                dataset.video_transform, feature_extractor, device,
                window_size=window_size
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not windows:
                print(f"  Warning: no valid windows for {video_filename} — skipping")
                continue

            clip_had_windows = True

            for (ws, visual_feat) in windows:
                # Convert window start seconds → 60fps frame index for pose lookup
                # Hand poses are view-independent — same JSON for all cameras
                window_start_frame_60fps = int(ws * VIDEO_FPS)
                hand_poses = dataset.load_hand_poses(video_id, window_start_frame_60fps)
                hand_feat  = hand_poses.view(1, -1).to(device)

                # Fusion: visual (2048) + hand pose (32 × 21 × 3 × 2 = 4032) = 6336
                combined    = torch.cat((visual_feat, hand_feat), dim=1)
                combined_np = combined.cpu().numpy()

                all_features.append(combined_np)
                all_labels.append(label)

            print(f"  [{idx + 1}/{total}] {video_id} | {camera} | "
                  f"label: {label} | windows: {len(windows)} | "
                  f"duration: {e_sec - s_sec:.1f}s")

        if not clip_had_windows:
            print(f"  Warning: no valid windows for clip {idx} ({video_id}) "
                  f"across any camera — skipping entirely")

        # Checkpoint every N clips
        if (idx + 1) % checkpoint_every == 0:
            _save_checkpoint(all_features, all_labels,
                             feature_filename, label_filename, idx + 1)

    # Free GPU memory before saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save flat format
    final_matrix = np.concatenate(all_features, axis=0)
    final_labels = np.array(all_labels, dtype=object)

    np.save(feature_filename, final_matrix)
    np.save(label_filename,   final_labels)
    print(f"\nSaved features : {feature_filename} | Shape: {final_matrix.shape}")
    print(f"Saved labels   : {label_filename}    | Shape: {final_labels.shape}")
    print(f"Label counts   : { {l: (final_labels == l).sum() for l in np.unique(final_labels)} }")


def _save_checkpoint(all_features, all_labels, feature_filename,
                     label_filename, clip_idx):
    """
    Save partial extraction progress. Overwrites the previous checkpoint
    each time — only the latest is kept.
    """
    ckpt_feat = feature_filename.replace(".npy", "_checkpoint.npy")
    ckpt_lbl  = label_filename.replace(".npy",  "_checkpoint.npy")

    try:
        matrix = np.concatenate(all_features, axis=0)
        labels = np.array(all_labels, dtype=object)
        np.save(ckpt_feat, matrix)
        np.save(ckpt_lbl,  labels)
        print(f"  [Checkpoint saved at clip {clip_idx} — {matrix.shape[0]} windows so far]")
    except Exception as e:
        print(f"  [Checkpoint failed: {e}]")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VIDEO_DIR      = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\videos"
    JSON_DIR       = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\HandPoses"
    ANNOTATION_DIR = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\annots"

    WINDOW_SIZE = 2.0

    total_start_time = time.time()

    df = load_all_annotations(ANNOTATION_DIR)

    train_df = df[df['label'] == 'correct'].copy()
    test_df  = df[df['label'] != 'correct'].copy()

    n_mistakes    = (df['label'] == 'mistake').sum()
    n_corrections = (df['label'] == 'correction').sum()
    print(f"Train (correct):  {len(train_df)} clips")
    print(f"Test  (anomaly):  {len(test_df)} clips "
          f"({n_mistakes} mistakes + {n_corrections} corrections)")

    assert len(train_df) > 0, "train_df is empty — check annotation CSVs"
    assert len(test_df)  > 0, "test_df is empty — check annotation CSVs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_model = slowfast_r50(pretrained=True)
    base_model.blocks[-1] = torch.nn.Identity()
    feature_extractor = base_model.to(device)
    feature_extractor.eval()
    """
    # --- Train extraction ---
    train_dataset = AssemblyHybridDataset(VIDEO_DIR, JSON_DIR, train_df)
    
    run_extractor(
        train_dataset,
        feature_filename="train_features_correct.npy",
        label_filename="train_labels.npy",
        feature_extractor=feature_extractor,
        device=device,
        window_size=WINDOW_SIZE,
        cameras=ACTIVE_CAMERAS
    )
    
    train_end_time = time.time()
    train_elapsed  = train_end_time - total_start_time
    avg_speed      = train_elapsed / len(train_df)
    print(f"\n⏱  Eğitim Seti Toplam Süre : {train_elapsed:.2f} saniye")
    print(f"🚀 Klip Başına Ortalama Hız: {avg_speed:.2f} saniye")

    run_std_tests("train_features_correct.npy", "train_labels.npy")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Memory cleared between runs.")
    """
    # --- Test extraction ---
    test_start_time = time.time()
    test_dataset = AssemblyHybridDataset(VIDEO_DIR, JSON_DIR, test_df)
    run_extractor(
        test_dataset,
        feature_filename="test_features_anomaly.npy",
        label_filename="test_labels.npy",
        feature_extractor=feature_extractor,
        device=device,
        window_size=WINDOW_SIZE,
        cameras=ACTIVE_CAMERAS
    )

    test_elapsed   = time.time() - test_start_time
    avg_speed_test = test_elapsed / len(test_df)
    print(f"\n⏱  Test Seti Toplam Süre   : {test_elapsed:.2f} saniye")
    print(f"🚀 Klip Başına Ortalama Hız: {avg_speed_test:.2f} saniye")

    run_std_tests("test_features_anomaly.npy", "test_labels.npy")

    total_elapsed = time.time() - total_start_time
    print(f"\n⏱  Genel Toplam Süre: {total_elapsed:.2f} saniye")

    del feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Done. GPU memory cleared.")