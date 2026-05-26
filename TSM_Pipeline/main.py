# main.py
# ---------------------------------------------------------------------------
# Entry point for TSM feature extraction pipeline.
#
# Usage:
#   python main.py --mode train
#   python main.py --mode test
#   python main.py --mode both
#
# Colab usage:
#   from main import run
#   run(mode='test', video_dir='.../videos', ...)
# ---------------------------------------------------------------------------

import argparse
import gc
import time
import torch

from tsm_pipeline import (
    TSMExtractor,
    AssemblyHybridDataset,
    load_all_annotations,
    run_train_extractor,
    run_test_extractor,
    run_std_tests_flat,
    run_std_tests_sequences,
    ACTIVE_CAMERAS,
    WINDOW_SIZE,
    FEATURE_DIM,
    VISUAL_DIM,
    POSE_DIM,
    NUM_FRAMES,
)

# ---------------------------------------------------------------------------
# Default paths — update for your environment
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_DIR      = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\videos"
DEFAULT_JSON_DIR       = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\HandPoses"
DEFAULT_ANNOTATION_DIR = r"C:\Users\arapn\Desktop\İşlerGüçler\Assembly101\Assembly-Anomaly-Detection\annots"


# ---------------------------------------------------------------------------
# Core run function (usable from CLI and Colab)
# ---------------------------------------------------------------------------

def run(mode='both',
        video_dir=DEFAULT_VIDEO_DIR,
        json_dir=DEFAULT_JSON_DIR,
        annotation_dir=DEFAULT_ANNOTATION_DIR,
        cameras=ACTIVE_CAMERAS,
        window_size=WINDOW_SIZE):
    """
    Run TSM feature extraction.

    Args:
        mode          : 'train' | 'test' | 'both'
        video_dir     : path to video files
        json_dir      : path to hand pose JSON files
        annotation_dir: path to annotation CSV files
        cameras       : list of camera IDs to use
        window_size   : sliding window size in seconds
    """
    total_start = time.time()

    print(f"TSM Feature Extraction Pipeline")
    print(f"Mode           : {mode}")
    print(f"Frames/window  : {NUM_FRAMES}")
    print(f"Feature dim    : visual({VISUAL_DIM}) + pose({POSE_DIM}) = {FEATURE_DIM}")
    print(f"Cameras        : {cameras}")

    # Load annotations
    full_df  = load_all_annotations(annotation_dir)
    train_df = full_df[full_df['label'] == 'correct'].copy()

    n_correct     = (full_df['label'] == 'correct').sum()
    n_mistakes    = (full_df['label'] == 'mistake').sum()
    n_corrections = (full_df['label'] == 'correction').sum()
    print(f"\nAnnotations — correct:{n_correct} "
          f"mistake:{n_mistakes} correction:{n_corrections}")

    assert len(train_df) > 0,              "train_df boş — annotation CSV kontrol et"
    assert n_mistakes + n_corrections > 0, "anomaly clip bulunamadı"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    feature_extractor = TSMExtractor().to(device)
    feature_extractor.eval()

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    if mode in ('train', 'both'):
        train_start   = time.time()
        train_dataset = AssemblyHybridDataset(video_dir, json_dir, train_df)

        run_train_extractor(
            train_dataset,
            feature_filename  = "train_features_correct_tsm.npy",
            label_filename    = "train_labels_tsm.npy",
            feature_extractor = feature_extractor,
            device            = device,
            window_size       = window_size,
            cameras           = cameras,
        )

        print(f"\n⏱  Train: {time.time() - train_start:.1f}s")

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

    # -----------------------------------------------------------------------
    # Test
    # -----------------------------------------------------------------------
    if mode in ('test', 'both'):
        test_start = time.time()

        run_test_extractor(
            full_df           = full_df,
            video_dir         = video_dir,
            json_dir          = json_dir,
            feature_extractor = feature_extractor,
            device            = device,
            video_transform   = AssemblyHybridDataset(
                                    video_dir, json_dir, full_df).video_transform,
            window_size       = window_size,
            cameras           = cameras,
        )

        print(f"\n⏱  Test: {time.time() - test_start:.1f}s")

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

    print(f"\n⏱  Toplam: {time.time() - total_start:.1f}s")

    del feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSM Feature Extraction Pipeline")
    parser.add_argument(
        "--mode", choices=["train", "test", "both"], default="both",
        help="Extraction mode (default: both)"
    )
    parser.add_argument("--video_dir",      default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--json_dir",       default=DEFAULT_JSON_DIR)
    parser.add_argument("--annotation_dir", default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument(
        "--cameras", nargs="+", default=ACTIVE_CAMERAS,
        help="Camera IDs to use"
    )
    parser.add_argument(
        "--window_size", type=float, default=WINDOW_SIZE,
        help="Sliding window size in seconds (default: 2.0)"
    )
    args = parser.parse_args()

    run(
        mode           = args.mode,
        video_dir      = args.video_dir,
        json_dir       = args.json_dir,
        annotation_dir = args.annotation_dir,
        cameras        = args.cameras,
        window_size    = args.window_size,
    )
