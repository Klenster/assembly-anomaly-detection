# tsm_pipeline/train.py
# ---------------------------------------------------------------------------
# Train feature extraction — correct clips only.
# ---------------------------------------------------------------------------

import gc
import torch
import numpy as np

from .constants import ANNOTATION_FPS, VIDEO_FPS, VISUAL_DIM, ACTIVE_CAMERAS
from .dataset import get_session_num
from .utils import extract_windowed_features, save_checkpoint


def run_train_extractor(dataset, feature_filename, label_filename,
                        feature_extractor, device, window_size=2.0,
                        checkpoint_every=10, cameras=ACTIVE_CAMERAS):
    """
    Extract features from correct clips only.

    Saves flat format and sequence format simultaneously:
        {feature_filename}                   → (total_windows, 4064)
        {label_filename}                     → all 'correct'
        {feature_filename}_sequences.npy     → per session+camera sequences
        {label_filename}_sequences.npy       → all 'correct'
        {feature_filename}_sequence_ids.npy  → video_id_camera strings
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
            video_path = f"{dataset.video_dir}/{session_num}{camera}_rgb.mp4"
            if not __import__('os').path.exists(video_path):
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
                wf60       = int(ws * VIDEO_FPS)
                hand_feat  = dataset.load_hand_poses(
                    video_id, wf60).view(1, -1).to(device)
                combined_np = torch.cat(
                    (visual_feat, hand_feat), dim=1).cpu().numpy()

                all_features.append(combined_np)
                all_labels.append(label)
                session_camera_windows[key].append(combined_np)

            print(f"  [{idx+1}/{total}] {video_id}|{camera} "
                  f"windows:{len(windows)} dur:{e_sec-s_sec:.1f}s")

        if not clip_had_windows:
            print(f"  Warning: no windows — {video_id}")
        if (idx + 1) % checkpoint_every == 0:
            save_checkpoint(all_features, all_labels,
                            feature_filename, label_filename, idx + 1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save flat
    fm = np.concatenate(all_features, axis=0)
    fl = np.array(all_labels, dtype=object)
    import numpy as np_
    np_.save(feature_filename, fm)
    np_.save(label_filename,   fl)
    print(f"\n✅ Flat: {feature_filename} | {fm.shape}")

    # Save sequences
    seqs, seq_lbls, seq_ids = [], [], []
    for (vid, cam), wins in session_camera_windows.items():
        if not wins:
            continue
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
