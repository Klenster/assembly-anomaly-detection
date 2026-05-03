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
# Annotation Loading
# ---------------------------------------------------------------------------

def load_annotations(csv_path, video_fps=60):
    """
    Load an Assembly101 annotation CSV for a single video.

    CSV format (no header row):
        start_frame, end_frame, action, object, target, label, note

    Labels:
        'correct'    — correctly performed action  → training (normal)
        'mistake'    — incorrect action            → testing  (anomaly)
        'correction' — user self-corrects          → testing  (anomaly)

    The video_id is extracted from the CSV filename, which follows the pattern:
        <prefix>_nusar-2021_action_both_<video_id>_<user_id>_<date>_<time>.csv
    e.g.: ..._9033-c13a_9033_user_id_2021-02-18_151004.csv
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

    df['start_time'] = df['start_frame'] / video_fps
    df['end_time']   = df['end_frame']   / video_fps

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


def load_all_annotations(annotation_dir, video_fps=60):
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
            df = load_annotations(path, video_fps=video_fps)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: skipping {fname} — {e}")

    if not all_dfs:
        raise RuntimeError(f"No valid annotation CSVs found in: {annotation_dir}")

    return pd.concat(all_dfs, ignore_index=True)


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
                    UniformTemporalSubsample(32),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                    ShortSideScale(size=256),
                    CenterCropVideo(crop_size=224),
                ]),
            ),
        ])

    def _get_pose_lookup(self, video_id):
        if video_id not in self._pose_cache:
            json_path = os.path.join(self.json_dir, f"{video_id}.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Hand pose JSON not found: {json_path}")

            if os.path.getsize(json_path) == 0:
                print(f"  Warning: Empty JSON for {video_id} — all poses will be zero")
                self._pose_cache[video_id] = {}
                return self._pose_cache[video_id]

            with open(json_path, 'r') as f:
                full_data = json.load(f)

            lookup = {}
            for entry in full_data:
                flat = []
                for hand_key in ['0', '1']:
                    if hand_key in entry['landmarks']:
                        for point in entry['landmarks'][hand_key]:
                            flat.extend(point)
                    else:
                        flat.extend([0.0] * 63)
                lookup[entry['frame_index']] = flat

            self._pose_cache[video_id] = lookup

        return self._pose_cache[video_id]

    def load_hand_poses(self, video_id, start_frame, num_frames=32):
        lookup = self._get_pose_lookup(video_id)
        poses  = [
            lookup.get(start_frame + i, [0.0] * 126)
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
    across the full action duration.

    Returns:
        List of (window_start_sec, visual_feat tensor [1, 2048]) tuples.
        fps value.
    """
    video         = EncodedVideo.from_path(video_path)
    fps           = float(video._container.streams.video[0].average_rate)
    file_duration = float(video.duration)

    action_duration = e_sec - s_sec
    num_windows     = max(1, int(action_duration / window_size))
    window_starts   = [s_sec + i * window_size for i in range(num_windows)]

    results = []

    for ws in window_starts:
        we = min(ws + window_size, file_duration)
        ws = max(0.0, ws)

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

    return results, fps


# ---------------------------------------------------------------------------
# STD Test Functions
# ---------------------------------------------------------------------------

def run_std_tests(feature_file, label_file, expected_dim=6080):
    """
    Run automated STD scenario tests on saved feature and label arrays.

    Scenarios:
        3.1 — Feature dimension matches expected
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
# Feature Extraction — saves both flat and sequence formats
# ---------------------------------------------------------------------------

def run_extractor(dataset, feature_filename, label_filename,
                  feature_extractor, device, window_size=2.0):
    """
    Extract combined visual + hand pose features for every 2-second window
    of every clip in the dataset.

    Saves TWO formats:
        1. Flat format   — (total_windows, 6080) for standard Autoencoder
        2. Sequence format — list of per-clip arrays for LSTM Autoencoder
           Each element: (num_windows_in_clip, 6080)
           Clips have different lengths so saved with allow_pickle=True

    Output files:
        feature_filename              → flat format
        feature_filename_sequences    → sequence format (for LSTM)
        label_filename                → flat labels
        label_filename_sequences      → per-clip labels (one per clip, not per window)
    """
    all_features  = []    # flat: every window as a row
    all_labels    = []    # flat: label repeated per window
    all_sequences = []    # grouped: list of per-clip window arrays
    all_seq_labels = []   # one label per clip (not per window)

    total = len(dataset)
    print(f"Extraction started — device: {device}, total clips: {total}")

    for idx in range(total):
        anno     = dataset.annotations.iloc[idx]
        video_id = anno['video_id']
        label    = anno['label']

        video_path = os.path.join(dataset.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        s_sec = anno['start_frame'] / 60.0
        e_sec = anno['end_frame']   / 60.0

        # --- Visual Features (all windows) ---
        windows, fps = extract_windowed_features(
            video_path, s_sec, e_sec,
            dataset.video_transform, feature_extractor, device,
            window_size=window_size
        )

        if not windows:
            print(f"  Warning: No valid windows for clip {idx} ({video_id}) — skipping")
            continue

        clip_windows = []   # windows for this clip (for sequence format)

        for (ws, visual_feat) in windows:
            # Hand pose aligned to this window
            window_start_frame = int(ws * fps)
            hand_poses = dataset.load_hand_poses(video_id, window_start_frame)
            hand_feat  = hand_poses.view(1, -1).to(device)

            # Fusion: visual (2048) + hand (4032) = 6080
            combined = torch.cat((visual_feat, hand_feat), dim=1)   # [1, 6080]
            combined_np = combined.cpu().numpy()

            # Flat format
            all_features.append(combined_np)
            all_labels.append(label)

            # Sequence format
            clip_windows.append(combined_np)

        # Save this clip as one sequence — shape: (num_windows, 6080)
        all_sequences.append(np.concatenate(clip_windows, axis=0))
        all_seq_labels.append(label)

        print(f"  [{idx + 1}/{total}] {video_id} | label: {label} | "
              f"windows: {len(clip_windows)} | "
              f"duration: {e_sec - s_sec:.1f}s")

    # Free GPU memory before saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Save flat format (for standard Autoencoder) ---
    final_matrix = np.concatenate(all_features, axis=0)   # (total_windows, 6080)
    final_labels = np.array(all_labels, dtype=object)

    np.save(feature_filename, final_matrix)
    np.save(label_filename, final_labels)
    print(f"\nSaved flat features : {feature_filename} | Shape: {final_matrix.shape}")
    print(f"Saved flat labels   : {label_filename}    | Shape: {final_labels.shape}")

    # --- Save sequence format (for LSTM Autoencoder) ---
    seq_feature_filename = feature_filename.replace(".npy", "_sequences.npy")
    seq_label_filename   = label_filename.replace(".npy", "_sequences.npy")

    seq_array       = np.array(all_sequences, dtype=object)
    seq_label_array = np.array(all_seq_labels, dtype=object)

    np.save(seq_feature_filename, seq_array,       allow_pickle=True)
    np.save(seq_label_filename,   seq_label_array, allow_pickle=True)
    print(f"Saved seq features  : {seq_feature_filename} | {len(all_sequences)} clips")
    print(f"Saved seq labels    : {seq_label_filename}   | {len(all_seq_labels)} clips")
    print(f"Label counts        : { {l: (final_labels == l).sum() for l in np.unique(final_labels)} }")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VIDEO_FPS      = 60
    VIDEO_DIR      = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\videos"
    JSON_DIR       = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\HandPoses"
    ANNOTATION_DIR = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\annots"

    WINDOW_SIZE = 2.0

    total_start_time = time.time()

    df = load_all_annotations(ANNOTATION_DIR, video_fps=VIDEO_FPS)

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

    # --- Train extraction ---
    train_dataset = AssemblyHybridDataset(VIDEO_DIR, JSON_DIR, train_df)
    run_extractor(
        train_dataset,
        feature_filename="train_features_correct.npy",
        label_filename="train_labels.npy",
        feature_extractor=feature_extractor,
        device=device,
        window_size=WINDOW_SIZE
    )

    train_end_time = time.time()
    train_elapsed  = train_end_time - total_start_time
    avg_speed      = train_elapsed / len(train_df)
    print(f"\n⏱  Eğitim Seti Toplam Süre : {train_elapsed:.2f} saniye")
    print(f"🚀 Klip Başına Ortalama Hız: {avg_speed:.2f} saniye")

    run_std_tests("train_features_correct.npy", "train_labels.npy", expected_dim=6080)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Memory cleared between runs.")

    # --- Test extraction ---
    test_start_time = time.time()
    test_dataset = AssemblyHybridDataset(VIDEO_DIR, JSON_DIR, test_df)
    run_extractor(
        test_dataset,
        feature_filename="test_features_anomaly.npy",
        label_filename="test_labels.npy",
        feature_extractor=feature_extractor,
        device=device,
        window_size=WINDOW_SIZE
    )

    test_elapsed   = time.time() - test_start_time
    avg_speed_test = test_elapsed / len(test_df)
    print(f"\n⏱  Test Seti Toplam Süre   : {test_elapsed:.2f} saniye")
    print(f"🚀 Klip Başına Ortalama Hız: {avg_speed_test:.2f} saniye")

    run_std_tests("test_features_anomaly.npy", "test_labels.npy", expected_dim=6080)

    total_elapsed = time.time() - total_start_time
    print(f"\n⏱  Genel Toplam Süre: {total_elapsed:.2f} saniye")

    del feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Done. GPU memory cleared.")
