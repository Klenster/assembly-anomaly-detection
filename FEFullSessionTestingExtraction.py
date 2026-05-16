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
ANNOTATION_FPS = 30
VIDEO_FPS      = 60
FEATURE_DIM    = 6336

# ---------------------------------------------------------------------------
# Camera Constants
# ---------------------------------------------------------------------------
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
    import pandas as pd
    df = pd.read_csv(
        csv_path, header=None,
        names=['start_frame','end_frame','action','object','target','label','note']
    )
    for col in ['action','object','target','label']:
        df[col] = df[col].str.strip()
    df['start_time'] = df['start_frame'] / ANNOTATION_FPS
    df['end_time']   = df['end_frame']   / ANNOTATION_FPS
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
        raise RuntimeError(f"No CSVs found in: {annotation_dir}")
    return pd.concat(all_dfs, ignore_index=True)


def get_session_num(video_id):
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
                    # Scale BEFORE subsampling to avoid 800MB RAM per window
                    ShortSideScale(size=256),
                    UniformTemporalSubsample(32),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225]),
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
        raise FileNotFoundError(f"No JSON for video_id: {video_id}")

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
                for entry in full_data:
                    flat = []
                    for hand_key in ['0','1']:
                        if hand_key in entry['landmarks']:
                            for point in entry['landmarks'][hand_key]:
                                flat.extend(point)
                        else:
                            flat.extend([0.0] * 63)
                    lookup[entry['frame_index']] = flat
            elif isinstance(full_data, dict):
                for frame_str, hands in full_data.items():
                    flat = []
                    for hand_key in ['0','1']:
                        if hand_key in hands:
                            for point in hands[hand_key]:
                                flat.extend(point)
                        else:
                            flat.extend([0.0] * 63)
                    lookup[int(frame_str)] = flat
            self._pose_cache[video_id] = lookup
        return self._pose_cache[video_id]

    def load_hand_poses(self, video_id, start_frame_60fps, num_frames=32):
        lookup = self._get_pose_lookup(video_id)
        poses  = [lookup.get(start_frame_60fps + i, [0.0]*126)
                  for i in range(num_frames)]
        return torch.tensor(poses, dtype=torch.float32)

    def __len__(self):  return len(self.annotations)
    def __getitem__(self, idx): return self.annotations.iloc[idx]


# ---------------------------------------------------------------------------
# Windowed Feature Extraction
# ---------------------------------------------------------------------------

def extract_windowed_features(video_path, s_sec, e_sec, video_transform,
                               feature_extractor, device, window_size=2.0):
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
            continue
        clip_data   = video_transform(clip_data)
        clip_frames = clip_data["video"]
        if clip_frames.shape[1] != 32:
            continue
        slow_indices = torch.arange(0, 32, 4)
        slow = clip_frames[:, slow_indices, :, :]
        fast = clip_frames
        inp  = [slow.unsqueeze(0).to(device), fast.unsqueeze(0).to(device)]
        with torch.no_grad():
            feat = feature_extractor(inp).view(1, -1)
        results.append((ws, feat))

    del video
    return results


# ---------------------------------------------------------------------------
# STD Tests
# ---------------------------------------------------------------------------

def run_std_tests_flat(feature_file, label_file):
    print("\n" + "="*50)
    print("📋 FLAT STD TESTLERİ")
    print("="*50)
    try:
        data   = np.load(feature_file)
        labels = np.load(label_file, allow_pickle=True)
        print(f"ℹ️  Features : {feature_file} | {data.shape}")
        print(f"ℹ️  Labels   : {label_file}   | {labels.shape}")
        print(f"ℹ️  Unique   : {np.unique(labels)}")
        print(f"{'✅' if data.shape[1]==FEATURE_DIM else '❌'} 3.1 Boyut: {data.shape[1]}")
        clean = not np.isnan(data).any() and not np.isinf(data).any()
        print(f"{'✅' if clean else '❌'} 3.2 NaN/Inf yok")
        print(f"ℹ️  Sıfır oranı: %{(data==0).sum()/data.size*100:.2f}")
        match = data.shape[0] == labels.shape[0]
        print(f"{'✅' if match else '❌'} 3.4 {data.shape[0]} feature, {labels.shape[0]} label")
    except Exception as e:
        print(f"❌ Hata: {e}")
    print("="*50 + "\n")


def run_std_tests_sequences(seq_feat_file, seq_lbl_file,
                             seq_window_lbl_file=None):
    print("\n" + "="*50)
    print("📋 SEQUENCE STD TESTLERİ")
    print("="*50)
    try:
        seqs   = np.load(seq_feat_file, allow_pickle=True)
        labels = np.load(seq_lbl_file,  allow_pickle=True)
        print(f"ℹ️  Sequences: {seq_feat_file} | {len(seqs)} sekans")
        print(f"ℹ️  Labels   : {seq_lbl_file}  | {len(labels)}")
        print(f"ℹ️  Unique   : {np.unique(labels)}")
        print(f"{'✅' if len(seqs)==len(labels) else '❌'} S.1 Sekans/label eşleşme")
        dims_ok = all(s.shape[1]==FEATURE_DIM for s in seqs)
        print(f"{'✅' if dims_ok else '❌'} S.2 Boyut {FEATURE_DIM}")
        lengths = [len(s) for s in seqs]
        print(f"ℹ️  Uzunluk — min:{min(lengths)} max:{max(lengths)} "
              f"ort:{np.mean(lengths):.1f}")
        clean = not any(np.isnan(s).any() for s in seqs)
        print(f"{'✅' if clean else '❌'} S.3 NaN/Inf yok")

        if seq_window_lbl_file and os.path.exists(seq_window_lbl_file):
            wlbls = np.load(seq_window_lbl_file, allow_pickle=True)
            match = all(len(s)==len(wl) for s,wl in zip(seqs,wlbls))
            print(f"{'✅' if match else '❌'} S.4 Window label uzunlukları eşleşiyor")
            total   = sum(len(wl) for wl in wlbls)
            anomaly = sum((wl==1).sum() for wl in wlbls)
            print(f"ℹ️  Windows — toplam:{total} correct:{total-anomaly} anomaly:{anomaly}")
    except Exception as e:
        print(f"❌ Hata: {e}")
    print("="*50 + "\n")


# ---------------------------------------------------------------------------
# Train Extraction — correct clips only
# ---------------------------------------------------------------------------

def run_train_extractor(dataset, feature_filename, label_filename,
                        feature_extractor, device, window_size=2.0,
                        checkpoint_every=10, cameras=ACTIVE_CAMERAS):
    """
    Train: correct clips only.

    Flat  → (total_windows, 6336), label='correct'
    Seqs  → one sequence per session+camera in temporal order
            35 sequences (7 sessions × 5 cameras)
    """
    all_features           = []
    all_labels             = []
    session_camera_windows = {}

    total = len(dataset)
    print(f"\n--- TRAIN EXTRACTION ---")
    print(f"Device:{device} | Clips:{total} | Cameras:{cameras}")

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
                wf60 = int(ws * VIDEO_FPS)
                hand_feat = dataset.load_hand_poses(
                    video_id, wf60).view(1,-1).to(device)
                combined_np = torch.cat(
                    (visual_feat, hand_feat), dim=1).cpu().numpy()
                all_features.append(combined_np)
                all_labels.append(label)
                session_camera_windows[key].append(combined_np)

            print(f"  [{idx+1}/{total}] {video_id}|{camera} "
                  f"windows:{len(windows)} dur:{e_sec-s_sec:.1f}s")

        if not clip_had_windows:
            print(f"  Warning: no windows for {video_id}")
        if (idx+1) % checkpoint_every == 0:
            _save_checkpoint(all_features, all_labels,
                             feature_filename, label_filename, idx+1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save flat
    fm = np.concatenate(all_features, axis=0)
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

    sf = feature_filename.replace(".npy","_sequences.npy")
    sl = label_filename.replace(".npy","_sequences.npy")
    si = feature_filename.replace(".npy","_sequence_ids.npy")
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

    Every session's clips (correct + mistake + correction) are extracted
    together in start_frame order. Each window gets a binary label:
        0 = correct
        1 = mistake or correction (anomaly)

    Flat output:
        test_features.npy          → (N, 6336) all windows mixed
        test_window_labels.npy     → (N,) binary 0/1 per window

    Sequence output (per session+camera):
        test_sequences.npy                  → variable length sequences
        test_sequence_labels.npy            → 0=no anomaly, 1=has anomaly
        test_sequence_window_labels.npy     → binary labels per window
        test_sequence_ids.npy               → video_id_camera strings

    This structure allows:
        Flat AE  → histogram correct vs anomaly reconstruction errors
        LSTM AE  → temporal localization of anomalies within sequences
    """
    all_features      = []
    all_window_labels = []
    sequences               = []
    sequence_labels         = []
    sequence_window_labels  = []
    sequence_ids            = []

    pose_ds = AssemblyHybridDataset(video_dir, json_dir, full_df)

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
                    wf60 = int(ws * VIDEO_FPS)
                    hand_feat = pose_ds.load_hand_poses(
                        video_id, wf60).view(1,-1).to(device)
                    combined_np = torch.cat(
                        (visual_feat, hand_feat), dim=1).cpu().numpy()
                    all_features.append(combined_np)
                    all_window_labels.append(window_label)
                    seq_wins.append(combined_np)
                    seq_wlbls.append(window_label)

            if not seq_wins:
                continue

            seq_arr  = np.vstack(seq_wins)
            wlbl_arr = np.array(seq_wlbls, dtype=np.int32)
            sequences.append(seq_arr)
            sequence_labels.append(1 if has_anomaly else 0)
            sequence_window_labels.append(wlbl_arr)
            sequence_ids.append(f"{video_id}_{camera}")

            n_c = (wlbl_arr==0).sum()
            n_a = (wlbl_arr==1).sum()
            print(f"    {camera} | total:{len(seq_wins)} "
                  f"correct:{n_c} anomaly:{n_a}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save flat
    fm  = np.concatenate(all_features, axis=0)
    fwl = np.array(all_window_labels, dtype=np.int32)
    np.save("test_features.npy",      fm)
    np.save("test_window_labels.npy", fwl)
    print(f"\n✅ Flat: test_features.npy | {fm.shape}")
    print(f"   correct:{(fwl==0).sum()} anomaly:{(fwl==1).sum()}")

    # Save sequences
    np.save("test_sequences.npy",
            np.array(sequences,             dtype=object), allow_pickle=True)
    np.save("test_sequence_labels.npy",
            np.array(sequence_labels,       dtype=np.int32))
    np.save("test_sequence_window_labels.npy",
            np.array(sequence_window_labels,dtype=object), allow_pickle=True)
    np.save("test_sequence_ids.npy",
            np.array(sequence_ids,          dtype=object), allow_pickle=True)

    lengths = [len(s) for s in sequences]
    print(f"✅ Sequences: test_sequences.npy | {len(sequences)} seqs | "
          f"min:{min(lengths)} max:{max(lengths)} mean:{np.mean(lengths):.1f}")
    print(f"   Sessions with anomaly: {sum(sequence_labels)}/{len(sequence_labels)}")


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

    full_df  = load_all_annotations(ANNOTATION_DIR)
    train_df = full_df[full_df['label'] == 'correct'].copy()

    n_correct     = (full_df['label'] == 'correct').sum()
    n_mistakes    = (full_df['label'] == 'mistake').sum()
    n_corrections = (full_df['label'] == 'correction').sum()
    print(f"Annotations — correct:{n_correct} mistake:{n_mistakes} "
          f"correction:{n_corrections}")

    assert len(train_df) > 0,              "train_df boş"
    assert n_mistakes + n_corrections > 0, "anomaly clip bulunamadı"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_model = slowfast_r50(pretrained=True)
    base_model.blocks[-1] = torch.nn.Identity()
    feature_extractor = base_model.to(device)
    feature_extractor.eval()
    """
    # --- Train ---
    train_dataset = AssemblyHybridDataset(VIDEO_DIR, JSON_DIR, train_df)
    run_train_extractor(
        train_dataset,
        feature_filename = "train_features_correct.npy",
        label_filename   = "train_labels.npy",
        feature_extractor= feature_extractor,
        device           = device,
        window_size      = WINDOW_SIZE,
        cameras          = ACTIVE_CAMERAS
    )
    print(f"\n⏱  Train: {time.time()-total_start:.1f}s")
    run_std_tests_flat("train_features_correct.npy", "train_labels.npy")
    run_std_tests_sequences("train_features_correct_sequences.npy",
                            "train_labels_sequences.npy")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    """
    # --- Test ---
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
    run_std_tests_flat("test_features.npy", "test_window_labels.npy")
    run_std_tests_sequences(
        "test_sequences.npy",
        "test_sequence_labels.npy",
        seq_window_lbl_file="test_sequence_window_labels.npy"
    )

    print(f"\n⏱  Toplam: {time.time()-total_start:.1f}s")
    del feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Done.")