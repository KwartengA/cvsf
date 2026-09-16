import pandas as pd
import pytest

from src.analysis.tennis_metrics import (
    pose_detection_rate, landmark_confidence_stats, rule_pass_rate,
)


def test_pose_detection_rate_computes_ratio():
    df = pd.DataFrame({"video": ["a.mov"] * 8 + ["b.mov"] * 5})
    frame_counts = {"a.mov": 10, "b.mov": 10}
    result = pose_detection_rate(df, frame_counts)
    result = result.set_index("video")
    assert result.loc["a.mov", "detection_rate"] == 0.8
    assert result.loc["b.mov", "detection_rate"] == 0.5


def test_pose_detection_rate_handles_video_with_zero_detections():
    df = pd.DataFrame({"video": ["a.mov"] * 3})
    frame_counts = {"a.mov": 10, "c.mov": 10}
    result = pose_detection_rate(df, frame_counts).set_index("video")
    assert result.loc["c.mov", "frames_detected"] == 0
    assert result.loc["c.mov", "detection_rate"] == 0.0


def test_landmark_confidence_stats():
    df = pd.DataFrame({
        "left_shoulder_vis": [1.0, 0.8],
        "right_shoulder_vis": [0.9, 0.95],
    })
    stats = landmark_confidence_stats(df)
    assert stats["mean_visibility"] == pytest.approx(0.9125, abs=1e-4)
    assert stats["min_visibility"] == pytest.approx(0.8, abs=1e-4)


def test_landmark_confidence_stats_no_columns_returns_none():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    stats = landmark_confidence_stats(df)
    assert stats["mean_visibility"] is None
    assert stats["min_visibility"] is None


def test_rule_pass_rate_all_good_form_is_one():
    df = pd.DataFrame([
        {"video": "a.mov", "right_elbow_angle": 110, "right_shoulder_angle": 100,
         "right_hip_angle": 150, "left_hip_angle": 150,
         "right_knee_angle": 140, "left_knee_angle": 140}
        for _ in range(4)
    ])
    result = rule_pass_rate(df).set_index("video")
    assert result.loc["a.mov", "rule_pass_rate"] == pytest.approx(1.0, abs=1e-6)


def test_rule_pass_rate_bad_form_is_low():
    df = pd.DataFrame([
        {"video": "b.mov", "right_elbow_angle": 175, "right_shoulder_angle": 150}
        for _ in range(4)
    ])
    result = rule_pass_rate(df).set_index("video")
    assert result.loc["b.mov", "rule_pass_rate"] < 0.5
