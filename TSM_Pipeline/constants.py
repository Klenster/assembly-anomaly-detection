# tsm_pipeline/constants.py
# ---------------------------------------------------------------------------
# Shared constants across the TSM pipeline.
# ---------------------------------------------------------------------------

# Frame space:
#   Annotation CSV frame numbers → 30fps (ANNOTATION_FPS)
#   Raw video and pose JSON frames → 60fps (VIDEO_FPS)
#   annotation_frame / ANNOTATION_FPS = seconds
#   seconds * VIDEO_FPS = video/pose frame index

ANNOTATION_FPS = 30
VIDEO_FPS      = 60
NUM_FRAMES     = 16    # frames sampled per 2-second window

# Feature dimensions:
#   Visual (TSM ResNet-50)  : 2048
#   Hand pose               : 16 frames × 21 keypoints × 3 values × 2 hands = 2016
#   Total                   : 4064

VISUAL_DIM  = 2048
POSE_DIM    = NUM_FRAMES * 21 * 3 * 2  # 2016
FEATURE_DIM = VISUAL_DIM + POSE_DIM    # 4064

WINDOW_SIZE = 2.0  # seconds

ACTIVE_CAMERAS = [
    'C10095',
    'C10115',
    'C10118',
    'C10119',
    'C10390',
    'C10404',
]
