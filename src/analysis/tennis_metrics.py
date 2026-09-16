"""
Tennis serve pose-quality metrics.

There is no trained classifier for tennis serve in this repo (a separate
YOLO+LSTM model is used for that). These metrics describe how reliable the
MediaPipe pose extraction and the rule-based feedback engine
(src/feedback/feedback_generator.py) are on the recorded serve footage.
"""

import numpy as np
import pandas as pd

from src.feedback.feedback_generator import evaluate_tennis_serve_frame

KEY_VIS_COLUMNS = [
    "left_shoulder_vis", "right_shoulder_vis",
    "left_elbow_vis", "right_elbow_vis",
    "left_wrist_vis", "right_wrist_vis",
    "left_hip_vis", "right_hip_vis",
    "left_knee_vis", "right_knee_vis",
]

ANGLE_COLUMNS = [
    "right_shoulder_angle", "left_shoulder_angle",
    "right_elbow_angle", "left_elbow_angle",
    "right_hip_angle", "left_hip_angle",
    "right_knee_angle", "left_knee_angle",
    "trunk_lean_angle",
]


def pose_detection_rate(df: pd.DataFrame, frame_counts: dict) -> pd.DataFrame:
    """
    frame_counts: {video_name: total_frame_count_in_source_video}
    Returns a per-video DataFrame with detected/total/rate.
    """
    detected = df.groupby("video").size().rename("frames_detected")
    rows = []
    for video, total in frame_counts.items():
        n_detected = int(detected.get(video, 0))
        rows.append({
            "video": video,
            "frames_detected": n_detected,
            "frames_total": total,
            "detection_rate": n_detected / total if total else 0.0,
        })
    return pd.DataFrame(rows)


def landmark_confidence_stats(df: pd.DataFrame) -> dict:
    """Mean/min visibility across key joints used in serve angle calculations."""
    cols = [c for c in KEY_VIS_COLUMNS if c in df.columns]
    if not cols:
        return {"mean_visibility": None, "min_visibility": None}
    vis = df[cols]
    return {
        "mean_visibility": float(vis.mean().mean()),
        "min_visibility": float(vis.min().min()),
    }


def angle_jitter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Frame-to-frame standard deviation of the frame-over-frame *difference*
    for each angle column, grouped by video -- a proxy for tracking noise.
    Lower = more stable tracking.
    """
    cols = [c for c in ANGLE_COLUMNS if c in df.columns]
    rows = []
    for video, group in df.sort_values("frame").groupby("video"):
        row = {"video": video}
        for c in cols:
            diffs = group[c].diff().dropna()
            row[f"{c}_jitter"] = float(diffs.std()) if len(diffs) else None
        rows.append(row)
    return pd.DataFrame(rows)


def rule_pass_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run evaluate_tennis_serve_frame on every row and compute the fraction
    of feedback checks that passed, per video and overall.
    """
    def frame_pass_ratio(row):
        angles = {c: row[c] for c in ANGLE_COLUMNS if c in row}
        feedback = evaluate_tennis_serve_frame(angles)
        if not feedback:
            return None
        passed = sum(1 for _, ok in feedback if ok)
        return passed / len(feedback)

    df = df.copy()
    df["_pass_ratio"] = df.apply(frame_pass_ratio, axis=1)

    per_video = (
        df.groupby("video")["_pass_ratio"]
        .mean()
        .rename("rule_pass_rate")
        .reset_index()
    )
    return per_video


def summarize_tennis_metrics(df: pd.DataFrame, frame_counts: dict = None) -> dict:
    """Top-level summary combining all tennis quality metrics."""
    summary = {}
    if frame_counts:
        det = pose_detection_rate(df, frame_counts)
        summary["detection_by_video"] = det
        summary["overall_detection_rate"] = float(
            det["frames_detected"].sum() / det["frames_total"].sum()
        ) if det["frames_total"].sum() else None
    summary.update(landmark_confidence_stats(df))
    summary["jitter_by_video"] = angle_jitter(df)
    pass_rate_df = rule_pass_rate(df)
    summary["pass_rate_by_video"] = pass_rate_df
    summary["overall_pass_rate"] = float(pass_rate_df["rule_pass_rate"].mean())
    return summary
