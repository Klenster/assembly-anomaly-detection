import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Assembly Anomaly Detection Demo",
    layout="wide"
)

st.title("Assembly Video Anomaly Detection Demo")

st.markdown(
    """
This demo visualizes anomaly detection results produced by a TSM + Hand Pose
per-camera autoencoder model. Reconstruction error is computed for each
2-second window and compared with the learned anomaly threshold.
"""
)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Demo Inputs")

video_path_input = st.sidebar.text_input(
    "Video file path",
    value=r"C:\Users\irem\Documents\Bilgisayar Mühendisliği\8. Dönem 2026\Bitirme\assembly-anomaly-detection\FeatureExtraction\FeatureExtractionOutputs\New\16.05\drive-download-20260517T114328Z-3-001\FinalTSMPerCameraAE_Results\single_video_poster_outputs\9064C10095_rgb.mp4"
)

timeline_file = st.sidebar.file_uploader(
    "Upload timeline CSV",
    type=["csv"]
)

fps = st.sidebar.number_input(
    "Video FPS",
    min_value=1,
    max_value=120,
    value=60
)

window_duration = st.sidebar.number_input(
    "Window duration (seconds)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.5
)

st.sidebar.markdown("---")
st.sidebar.markdown("Expected CSV columns:")
st.sidebar.code(
    "window_index, approx_frame, true_label,\n"
    "reconstruction_error, prediction"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.read())
    temp.close()
    return temp.name


def group_anomaly_intervals(df, fps, window_duration):
    anomaly_df = df[df["prediction"] == 1].copy()

    if anomaly_df.empty:
        return pd.DataFrame(columns=[
            "Start Window",
            "End Window",
            "Start Time (s)",
            "End Time (s)",
            "Predicted Anomaly Windows",
            "Ground Truth Anomaly Windows",
            "Correctly Detected Windows",
            "False Alarm Windows",
            "Interval Type",
            "Max Error"
        ])

    windows = anomaly_df["window_index"].tolist()

    groups = []
    start = windows[0]
    prev = windows[0]

    for w in windows[1:]:
        if w == prev + 1:
            prev = w
        else:
            groups.append((start, prev))
            start = w
            prev = w

    groups.append((start, prev))

    rows = []

    for s, e in groups:
        part = df[(df["window_index"] >= s) & (df["window_index"] <= e)].copy()

        start_time = s * window_duration
        end_time = (e + 1) * window_duration

        predicted_count = int((part["prediction"] == 1).sum())

        if "true_label" in part.columns:
            gt_count = int((part["true_label"] == 1).sum())
            correct_detected = int(((part["true_label"] == 1) & (part["prediction"] == 1)).sum())
            false_alarm = int(((part["true_label"] == 0) & (part["prediction"] == 1)).sum())

            if correct_detected > 0 and false_alarm == 0:
                interval_type = "Correct Detection"
            elif correct_detected > 0 and false_alarm > 0:
                interval_type = "Mixed Detection"
            else:
                interval_type = "False Alarm"
        else:
            gt_count = ""
            correct_detected = ""
            false_alarm = ""
            interval_type = "Prediction Only"

        rows.append({
            "Start Window": s,
            "End Window": e,
            "Start Time (s)": round(start_time, 2),
            "End Time (s)": round(end_time, 2),
            "Approx Start Frame": int(start_time * fps),
            "Approx End Frame": int(end_time * fps),
            "Predicted Anomaly Windows": predicted_count,
            "Ground Truth Anomaly Windows": gt_count,
            "Correctly Detected Windows": correct_detected,
            "False Alarm Windows": false_alarm,
            "Interval Type": interval_type,
            "Max Error": round(part["reconstruction_error"].max(), 4)
        })

    return pd.DataFrame(rows)


def plot_timeline(df):
    threshold = df["threshold"].iloc[0] if "threshold" in df.columns else None

    fig, ax = plt.subplots(figsize=(13, 4.5))

    ax.plot(
        df["approx_frame"],
        df["reconstruction_error"],
        linewidth=2,
        label="Reconstruction Error"
    )

    if threshold is not None:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {threshold:.3f}"
        )

    detected = df[df["prediction"] == 1]

    ax.scatter(
        detected["approx_frame"],
        detected["reconstruction_error"],
        marker="x",
        s=70,
        label="Detected Anomaly"
    )

    ax.set_xlabel("Approximate Frame Index")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Detected Anomaly Moments Over Time")
    ax.grid(alpha=0.25)
    ax.legend()

    return fig


def get_frame_at_time(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    success, frame = cap.read()
    cap.release()

    if not success:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame



def extract_video_clip(video_path, start_time, end_time):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 60

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    temp_clip = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    clip_path = temp_clip.name
    temp_clip.close()

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame

    while current_frame <= end_frame:
        success, frame = cap.read()

        if not success:
            break

        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()

    return clip_path


# ============================================================
# MAIN LOGIC
# ============================================================

if timeline_file is None:
    st.info("Please upload a timeline CSV file to start.")
    st.stop()

df = pd.read_csv(timeline_file)

required_cols = {
    "window_index",
    "approx_frame",
    "reconstruction_error",
    "prediction"
}

missing = required_cols - set(df.columns)

if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

if "threshold" not in df.columns:
    # If threshold column is absent, estimate it from prediction boundary
    # or ask user to enter it manually.
    threshold_value = st.sidebar.number_input(
        "Threshold value",
        min_value=0.0,
        value=0.1876,
        step=0.001,
        format="%.4f"
    )
    df["threshold"] = threshold_value
    df["prediction"] = (df["reconstruction_error"] > threshold_value).astype(int)

if "true_label" in df.columns:
    def classify_window(row):
        true = int(row["true_label"])
        pred = int(row["prediction"])

        if true == 1 and pred == 1:
            return "TP - Correct anomaly"
        elif true == 0 and pred == 1:
            return "FP - False alarm"
        elif true == 1 and pred == 0:
            return "FN - Missed anomaly"
        else:
            return "TN - Correct normal"

    df["result"] = df.apply(classify_window, axis=1)

    y_true = df["true_label"].astype(int)
    y_pred = df["prediction"].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

# ============================================================
# WINDOW-LEVEL CORRECTNESS AND METRICS
# ============================================================

if "true_label" in df.columns:
    def classify_window(row):
        true = int(row["true_label"])
        pred = int(row["prediction"])

        if true == 1 and pred == 1:
            return "TP - Correct anomaly"
        elif true == 0 and pred == 1:
            return "FP - False alarm"
        elif true == 1 and pred == 0:
            return "FN - Missed anomaly"
        else:
            return "TN - Correct normal"

    df["result"] = df.apply(classify_window, axis=1)

    y_true = df["true_label"].astype(int)
    y_pred = df["prediction"].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
else:
    tn = fp = fn = tp = None
    acc = prec = rec = f1 = None

# ============================================================
# TOP METRICS
# ============================================================

total_windows = len(df)
detected_count = int((df["prediction"] == 1).sum())

if "true_label" in df.columns:
    true_anomaly_count = int((df["true_label"] == 1).sum())
else:
    true_anomaly_count = None

col1, col2, col3 = st.columns(3)

col1.metric("Total Windows", total_windows)
col2.metric("Detected Anomaly Windows", detected_count)

if true_anomaly_count is not None:
    col3.metric("Ground Truth Anomaly Windows", true_anomaly_count)
else:
    col3.metric("Ground Truth", "Not provided")

if "true_label" in df.columns:
    st.subheader("Single Video Detection Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TP Correct Anomaly", tp)
    m2.metric("FP False Alarm", fp)
    m3.metric("FN Missed Anomaly", fn)
    m4.metric("TN Correct Normal", tn)

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Accuracy", f"{acc * 100:.2f}%")
    m6.metric("Precision", f"{prec * 100:.2f}%")
    m7.metric("Recall", f"{rec * 100:.2f}%")
    m8.metric("F1-score", f"{f1 * 100:.2f}%")


# ============================================================
# VIDEO + PLOT
# ============================================================

left, right = st.columns([1.1, 1.4])

with left:
    st.subheader("Input Video")

    video_path = None

    if video_path_input and Path(video_path_input).exists():
        video_path = video_path_input
        st.video(video_path)
    else:
        st.warning("Please enter a valid video file path.")

with right:
    st.subheader("Reconstruction Error Timeline")
    fig = plot_timeline(df)
    st.pyplot(fig)


# ============================================================
# ANOMALY INTERVAL TABLE
# ============================================================

st.subheader("Detected Anomaly Intervals")

intervals = group_anomaly_intervals(
    df=df,
    fps=fps,
    window_duration=window_duration
)

if intervals.empty:
    st.success("No anomaly interval detected.")
else:
    st.dataframe(intervals, use_container_width=True)


# ============================================================
# VIDEO PREVIEW
# ============================================================

if video_path is not None and not intervals.empty:
    st.subheader("Preview Detected Anomaly Moment")

    selected_idx = st.selectbox(
        "Select anomaly interval",
        options=list(range(len(intervals))),
        format_func=lambda i: (
            f"Interval {i + 1}: "
            f"{intervals.iloc[i]['Start Time (s)']}s - "
            f"{intervals.iloc[i]['End Time (s)']}s | "
            f"TP: {intervals.iloc[i].get('Correctly Detected Windows', 0)} | "
            f"FP: {intervals.iloc[i].get('False Alarm Windows', 0)} | "
            f"{intervals.iloc[i].get('Interval Type', '')}"
        )
    )

    selected_row = intervals.iloc[selected_idx]

    start_time = float(selected_row["Start Time (s)"])
    end_time = float(selected_row["End Time (s)"])

    st.markdown(
        f"""
        **Selected interval:** {start_time:.2f}s - {end_time:.2f}s  
        **Correctly detected anomaly windows:** {selected_row.get("Correctly Detected Windows", 0)}  
        **False alarm windows:** {selected_row.get("False Alarm Windows", 0)}  
        **Interval type:** {selected_row.get("Interval Type", "N/A")}
        """
    )

    clip_path = extract_video_clip(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time
    )

    if clip_path is not None:
        st.video(clip_path)
    else:
        st.warning("Could not extract video clip for the selected interval.")


# ============================================================
# RAW RESULT TABLE
# ============================================================

with st.expander("Show window-level results with correctness"):
    display_cols = [
        "window_index",
        "approx_frame",
        "reconstruction_error",
        "threshold",
        "true_label",
        "prediction",
        "result"
    ]

    existing_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[existing_cols], use_container_width=True)