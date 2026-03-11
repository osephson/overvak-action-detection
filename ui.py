import os
from typing import List

import pandas as pd
import streamlit as st

from src.pipeline import (
    TARGET_ACTIONS,
    ActionPrediction,
    VideoResult,
    download_youtube_video,
    infer_actions_from_video,
)
from src.metrics import PRMetrics, labels_from_row, precision_recall_for_video


DATA_CSV_PATH = os.path.join("data", "labels.csv")


def load_labels_csv() -> pd.DataFrame | None:
    if not os.path.exists(DATA_CSV_PATH):
        return None
    try:
        return pd.read_csv(DATA_CSV_PATH)
    except pd.errors.EmptyDataError:
        # Gracefully handle an empty or placeholder CSV.
        return None


def get_ground_truth_for_video(df: pd.DataFrame, video_id: str) -> List[str] | None:
    if "video_id" not in df.columns:
        return None
    rows = df[df["video_id"] == video_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    label_cols = [c for c in df.columns if c.lower() in TARGET_ACTIONS]
    return labels_from_row({c: row[c] for c in label_cols})


def render_predictions(result: VideoResult):
    st.subheader("Detected actions")
    if not result.mapped_actions:
        st.write("No target actions detected with high confidence.")
    else:
        for pred in result.mapped_actions:
            st.write(f"- **{pred.action.title()}** (score: {pred.score:.3f})")

    with st.expander("Model top‑K Kinetics predictions"):
        for label, score in result.kinetics_topk:
            st.write(f"- {label} ({score:.3f})")


def render_metrics(ground_truth: List[str], preds: List[ActionPrediction]):
    st.subheader("Precision / Recall")
    y_pred = [p.action for p in preds]
    metrics: PRMetrics = precision_recall_for_video(ground_truth, y_pred)

    st.write(f"**Ground truth actions:** {', '.join(ground_truth) or 'None'}")
    st.write(f"**Predicted actions:** {', '.join(y_pred) or 'None'}")

    st.metric("Precision", f"{metrics.precision:.2f}")
    st.metric("Recall", f"{metrics.recall:.2f}")

    with st.expander("Confusion counts"):
        st.write(f"TP: {metrics.tp}, FP: {metrics.fp}, FN: {metrics.fn}")


def main():
    st.title("Overvak – Minimal Action Detection Demo")
    st.write(
        "Paste a YouTube URL for a kids’ activity video, "
        "or provide a path to a local .mp4 file you downloaded yourself. "
        "The pipeline downloads the video, runs a pretrained action model, "
        "maps generic actions to the target label set, and reports results."
    )

    url = st.text_input(
        "YouTube URL or local video path",
        placeholder="https://www.youtube.com/watch?v=...  or  C:\\path\\to\\video.mp4",
    )
    process = st.button("Process")

    if process and url:
        with st.spinner("Downloading and analysing video… this can take a minute."):
            try:
                user_input = url.strip()
                if os.path.exists(user_input):
                    # Treat input as a local file path; skip any network download.
                    video_path = user_input
                    video_id = os.path.splitext(os.path.basename(video_path))[0]
                else:
                    video_path, video_id = download_youtube_video(user_input)
                result = infer_actions_from_video(video_path)
            except Exception as e:
                st.error(f"Failed to process video: {e}")
                return

        st.success(f"Processed video id: {video_id}")
        render_predictions(result)

        # Metrics section (only if label CSV is present).
        df = load_labels_csv()
        if df is None:
            st.info(
                "No `data/labels.csv` found – add it to compute precision/recall "
                "against your ground‑truth annotations."
            )
        else:
            gt = get_ground_truth_for_video(df, video_id)
            if gt is None:
                st.info(
                    "No ground‑truth row found for this video id in `data/labels.csv`."
                )
            else:
                render_metrics(gt, result.mapped_actions)


if __name__ == "__main__":
    main()

