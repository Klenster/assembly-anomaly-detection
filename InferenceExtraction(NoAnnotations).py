import torch
import json
import os
import numpy as np
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
# Video Transform (same as training)
# ---------------------------------------------------------------------------

def build_transform():
    return Compose([
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


# ---------------------------------------------------------------------------
# Hand Pose Loading
# ---------------------------------------------------------------------------

def load_pose_lookup(json_path):
    """
    Parse a hand pose JSON file into a frame_index -> flat 126-float dict.
    Returns empty dict if file is missing or empty.
    """
    if not os.path.exists(json_path):
        print(f"  Warning: JSON not found at {json_path} — all poses will be zero")
        return {}

    if os.path.getsize(json_path) == 0:
        print(f"  Warning: Empty JSON at {json_path} — all poses will be zero")
        return {}

    with open(json_path, 'r') as f:
        full_data = json.load(f)

    lookup = {}
    for entry in full_data:
        flat = []
        for hand_key in ['0', '1']:
            if hand_key in entry['landmarks']:
                for point in entry['landmarks'][hand_key]:  # 21 x [x, y, z]
                    flat.extend(point)
            else:
                flat.extend([0.0] * 63)   # Missing hand → zeros
        lookup[entry['frame_index']] = flat

    return lookup


def get_hand_poses(lookup, start_frame, num_frames=32):
    """
    Load num_frames consecutive pose vectors starting at start_frame.
    Returns tensor of shape [num_frames, 126].
    """
    poses = [
        lookup.get(start_frame + i, [0.0] * 126)
        for i in range(num_frames)
    ]
    return torch.tensor(poses, dtype=torch.float32)   # [32, 126]


# ---------------------------------------------------------------------------
# Main Extraction Function
# ---------------------------------------------------------------------------

def extract_segment(video_path, json_path, start_frame, end_frame,
                    output_file, window_size=2.0):
    """
    Extract features from a specific frame range of a video by sliding
    a window across the segment. No annotation CSV needed.

    Each 2-second window produces one row in the output .npy file:
        visual features (2048) + hand pose features (4032) = 6080 dims

    Args:
        video_path  : path to the .mp4 file
        json_path   : path to the hand pose .json file
        start_frame : first frame of the segment to extract (inclusive)
        end_frame   : last frame of the segment to extract (inclusive)
        output_file : path to save the output .npy file
        window_size : seconds per window (default 2.0)

    Output:
        {output_file}        — float32 array of shape (num_windows, 6080)
        {output_file}_meta.npy — array of window start frames for reference
    """

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Model ---
    print("Loading SlowFast R50...")
    base_model = slowfast_r50(pretrained=True)
    base_model.blocks[-1] = torch.nn.Identity()
    feature_extractor = base_model.to(device)
    feature_extractor.eval()

    # --- Video ---
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    video         = EncodedVideo.from_path(video_path)
    fps           = float(video._container.streams.video[0].average_rate)
    file_duration = float(video.duration)

    print(f"Video FPS     : {fps}")
    print(f"Video duration: {file_duration:.2f}s")

    # Convert frame indices to seconds using real FPS
    s_sec = start_frame / fps
    e_sec = end_frame   / fps

    # Clamp to actual video duration
    s_sec = max(0.0, min(s_sec, file_duration - window_size))
    e_sec = min(e_sec, file_duration)

    segment_duration = e_sec - s_sec
    num_windows      = max(1, int(segment_duration / window_size))

    print(f"Segment       : frame {start_frame} → {end_frame} "
          f"({s_sec:.2f}s → {e_sec:.2f}s)")
    print(f"Windows       : {num_windows} x {window_size}s")

    # --- Hand Poses ---
    pose_lookup = load_pose_lookup(json_path)

    # --- Transform ---
    video_transform = build_transform()

    # --- Slide window across segment ---
    all_features   = []
    all_meta       = []   # window start frames for reference
    window_starts  = [s_sec + i * window_size for i in range(num_windows)]

    for w_idx, ws in enumerate(window_starts):
        we = min(ws + window_size, file_duration)
        ws = max(0.0, ws)

        if we - ws < 0.1:
            continue

        # --- Visual ---
        clip_data = video.get_clip(ws, we)
        if clip_data is None or clip_data.get("video") is None:
            print(f"  Warning: get_clip failed at {ws:.2f}s — skipping window")
            continue

        clip_data   = video_transform(clip_data)
        clip_frames = clip_data["video"]   # [C, 32, H, W]

        if clip_frames.shape[1] != 32:
            print(f"  Warning: Unexpected frame count at {ws:.2f}s — skipping")
            continue

        slow_indices = torch.arange(0, 32, 4)
        slow         = clip_frames[:, slow_indices, :, :]
        fast         = clip_frames

        inp = [slow.unsqueeze(0).to(device), fast.unsqueeze(0).to(device)]

        with torch.no_grad():
            visual_feat = feature_extractor(inp)
            visual_feat = visual_feat.view(1, -1)   # [1, 2048]

        # --- Hand Pose ---
        window_start_frame = int(ws * fps)
        hand_poses = get_hand_poses(pose_lookup, window_start_frame)  # [32, 126]
        hand_feat  = hand_poses.view(1, -1).to(device)                # [1, 4032]

        # --- Fusion ---
        combined = torch.cat((visual_feat, hand_feat), dim=1)         # [1, 6080]
        all_features.append(combined.cpu().numpy())
        all_meta.append(window_start_frame)

        # Progress every 10 windows
        if (w_idx + 1) % 10 == 0 or (w_idx + 1) == num_windows:
            print(f"  Window {w_idx + 1}/{num_windows} — "
                  f"{ws:.1f}s → {we:.1f}s | "
                  f"start frame: {window_start_frame}")

    if not all_features:
        raise RuntimeError("No features extracted — check video path and frame range")

    # --- Save ---
    final_matrix = np.concatenate(all_features, axis=0)   # (num_windows, 6080)
    meta_array   = np.array(all_meta, dtype=np.int64)     # (num_windows,)

    np.save(output_file, final_matrix)
    meta_file = output_file.replace(".npy", "_meta.npy")
    np.save(meta_file, meta_array)

    print(f"\nSaved features : {output_file} | Shape: {final_matrix.shape}")
    print(f"Saved meta     : {meta_file}    | Shape: {meta_array.shape}")
    print(f"Meta contains  : window start frames (use these to map results back to video time)")

    # --- Cleanup ---
    del feature_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Done. GPU memory cleared.")

    return final_matrix, meta_array


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    VIDEO_PATH  = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\videos\9033-c13a_9033.mp4"
    JSON_PATH   = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\HandPoses\9033-c13a_9033.json"
    OUTPUT_FILE = "inference_features.npy"

    # Set your desired frame range here
    START_FRAME = 4781
    END_FRAME   = 28707

    features, meta = extract_segment(
        video_path  = VIDEO_PATH,
        json_path   = JSON_PATH,
        start_frame = START_FRAME,
        end_frame   = END_FRAME,
        output_file = OUTPUT_FILE,
        window_size = 2.0
    )

    print(f"\nExtracted {features.shape[0]} windows")
    print(f"Each window covers {2.0}s of video")
    print(f"Window start frames saved in: {OUTPUT_FILE.replace('.npy', '_meta.npy')}")
    print(f"Use meta array to map anomaly results back to video timestamps")