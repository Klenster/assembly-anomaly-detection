# tsm_pipeline/dataset.py
# ---------------------------------------------------------------------------
# Annotation loading and AssemblyHybridDataset.
# ---------------------------------------------------------------------------

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

from .constants import ANNOTATION_FPS, VIDEO_FPS, NUM_FRAMES


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def load_annotations(csv_path):
    """
    Load a single Assembly101 annotation CSV.

    CSV columns (no header):
        start_frame, end_frame, action, object, target, label, note

    Frame space:
        start_frame / ANNOTATION_FPS = start_time (seconds)

    Labels:
        'correct'    → training (normal)
        'mistake'    → testing  (anomaly)
        'correction' → testing  (anomaly)
    """
    import pandas as pd

    df = pd.read_csv(
        csv_path, header=None,
        names=['start_frame', 'end_frame', 'action', 'object', 'target', 'label', 'note']
    )
    for col in ['action', 'object', 'target', 'label']:
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
    df['video_id'] = f"{parts[0]}_{parts[1]}"
    return df


def load_all_annotations(annotation_dir):
    """Load and concatenate all annotation CSVs in a directory."""
    import pandas as pd

    all_dfs = []
    for fname in sorted(os.listdir(annotation_dir)):
        if not fname.endswith('.csv'):
            continue
        try:
            all_dfs.append(load_annotations(os.path.join(annotation_dir, fname)))
        except Exception as e:
            print(f"  Warning: skipping {fname} — {e}")

    if not all_dfs:
        raise RuntimeError(f"No CSVs found in: {annotation_dir}")
    return pd.concat(all_dfs, ignore_index=True)


def get_session_num(video_id):
    """'9033-c13a_9033' → '9033'"""
    return video_id.split('-')[0]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AssemblyHybridDataset(Dataset):
    """
    Provides video transforms and hand pose lookups.

    Hand poses are view-independent (3D world coordinates) — the same
    JSON is reused across all camera views of the same session.
    """

    def __init__(self, video_dir, json_dir, annotations):
        self.video_dir   = video_dir
        self.json_dir    = json_dir
        self.annotations = annotations.reset_index(drop=True)
        self._pose_cache = {}

        self.video_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    ShortSideScale(size=256),             # scale BEFORE subsample
                    UniformTemporalSubsample(NUM_FRAMES), # 16 frames
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                    CenterCropVideo(crop_size=224),
                ]),
            ),
        ])

    # --- pose helpers ---

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
                            flat.extend([0.0] * 63)
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
                            flat.extend([0.0] * 63)
                    lookup[int(frame_str)] = flat

            self._pose_cache[video_id] = lookup
        return self._pose_cache[video_id]

    def load_hand_poses(self, video_id, start_frame_60fps):
        """
        Returns tensor (NUM_FRAMES, 126) for the given 60fps frame index.
        Fallback: zeros for missing frames.
        """
        lookup = self._get_pose_lookup(video_id)
        poses  = [
            lookup.get(start_frame_60fps + i, [0.0] * 126)
            for i in range(NUM_FRAMES)
        ]
        return torch.tensor(poses, dtype=torch.float32)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.annotations.iloc[idx]
