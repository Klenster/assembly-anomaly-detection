# tsm_pipeline/test.py
# ---------------------------------------------------------------------------
# Test feature extraction — full sessions (correct + mistake + correction).
# ---------------------------------------------------------------------------

import gc
import torch
import numpy as np

from .constants import ANNOTATION_FPS, VIDEO_FPS, ACTIVE_CAMERAS
from .dataset import AssemblyHybridDataset, get_session_num
from .utils import extract_windowed_features


def run_test_extractor(full_df, video_dir, json_dir, feature_extractor,
                       device, video_transform, window_size=2.0,
                       cameras=ACTIVE_CAMERAS):
    """
    Extract features for all sessions in temporal order.

    Correct and anomaly clips are processed together so that:
      - Flat format: correct/anomaly windows in the same table with binary labels
      - Sequence format: full session sequences with per-window binary labels
        enabling temporal anomaly localization

    Window labels:
        0 = correct
        1 = mistake or correction (anomaly)

    Saved files:
        test_features_tsm.npy
        test_window_labels_tsm.npy
        test_sequences_tsm.npy
        test_sequence_labels_tsm.npy
        test_sequence_window_labels_tsm.npy
        test_sequence_ids_tsm.npy
        test_sequence_window_times.npy       ← window start seconds
    """
    all_features      = []
    all_window_labels = []
    sequences               = []
    sequence_labels         = []
    sequence_window_labels  = []
    sequence_ids            = []
    sequence_window_times   = []

    pose_ds = AssemblyHybridDataset(video_dir, json_dir, full_df)

    print(f"\n--- TEST EXTRACTION ---")

    for video_id, session_df in full_df.groupby('video_id'):
        session_df  = session_df.sort_values('start_frame').reset_index(drop=True)
        session_num = get_session_num(video_id)
        has_anomaly = (session_df['label'] != 'correct').any()

        print(f"\n  Session:{video_id} | clips:{len(session_df)} "
              f"| has_anomaly:{has_anomaly}")

        for camera in cameras:
            import os
            video_path = os.path.join(video_dir, f"{session_num}{camera}_rgb.mp4")
            if not os.path.exists(video_path):
                print(f"    Warning: missing {session_num}{camera}_rgb.mp4")
                continue

            seq_wins   = []
            seq_wlbls  = []
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
                    hand_feat = pose_ds.load_hand_poses(
                        video_id, wf60).view(1, -1).to(device)
                    combined_np = torch.cat(
                        (visual_feat, hand_feat), dim=1).cpu().numpy()

                    all_features.append(combined_np)
                    all_window_labels.append(window_label)
                    seq_wins.append(combined_np)
                    seq_wlbls.append(window_label)
                    seq_wtimes.append(ws)

            if not seq_wins:
                continue

            seq_arr  = np.vstack(seq_wins)
            wlbl_arr = np.array(seq_wlbls,  dtype=np.int32)
            time_arr = np.array(seq_wtimes, dtype=np.float32)

            sequences.append(seq_arr)
            sequence_labels.append(1 if has_anomaly else 0)
            sequence_window_labels.append(wlbl_arr)
            sequence_ids.append(f"{video_id}_{camera}")
            sequence_window_times.append(time_arr)

            n_c = (wlbl_arr == 0).sum()
            n_a = (wlbl_arr == 1).sum()
            print(f"    {camera} | total:{len(seq_wins)} "
                  f"correct:{n_c} anomaly:{n_a}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save flat
    fm  = np.concatenate(all_features, axis=0)
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
            np.array(sequence_window_times,  dtype=object), allow_pickle=True)

    lengths = [len(s) for s in sequences]
    print(f"✅ Sequences: test_sequences_tsm.npy | {len(sequences)} seqs | "
          f"min:{min(lengths)} max:{max(lengths)} mean:{np.mean(lengths):.1f}")
    print(f"   Anomalili session: {sum(sequence_labels)}/{len(sequence_labels)}")
