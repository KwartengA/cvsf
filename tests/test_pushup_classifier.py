import numpy as np
import pandas as pd
import pytest

from src.analysis.pushup_classifier import (
    detect_rep_bottoms, label_frames_by_threshold, build_feature_matrix,
    train_and_evaluate, FEATURE_COLUMNS,
)


def make_rep_video(video_name, elbow_bottom, hip_bottom=175, n_frames=30):
    """
    Build one synthetic pushup video: elbow angle starts straight (~175),
    dips down to `elbow_bottom` around the midpoint, then returns to
    straight -- i.e. one clean repetition. Hip angle stays at `hip_bottom`
    throughout (straight plank line by default).
    """
    frames = np.arange(n_frames)
    # a single dip: 175 -> elbow_bottom -> 175 (cosine-shaped)
    phase = np.pi * frames / (n_frames - 1)
    elbow = elbow_bottom + (175 - elbow_bottom) * (1 - np.sin(phase))
    rows = []
    for i, e in enumerate(elbow):
        rows.append({
            "video": video_name, "frame": int(frames[i]),
            "left_elbow_angle": e, "right_elbow_angle": e,
            "left_hip_angle": hip_bottom, "right_hip_angle": hip_bottom,
            "left_shoulder_angle": 45, "right_shoulder_angle": 45,
        })
    return pd.DataFrame(rows)


# ── Rep detection ────────────────────────────────────────────────────────

def test_detect_rep_bottoms_finds_one_rep_per_video():
    df = make_rep_video("a.mp4", elbow_bottom=80)
    result = detect_rep_bottoms(df)
    assert result["is_rep_bottom"].any()
    assert result.loc[result["is_rep_bottom"], "rep_id"].nunique() == 1


def test_detect_rep_bottoms_excludes_plank_frames():
    df = make_rep_video("a.mp4", elbow_bottom=80)
    result = detect_rep_bottoms(df)
    # first and last frames are near-straight-arm (top of plank) -- should
    # not be marked as the bottom of the rep
    assert result.iloc[0]["is_rep_bottom"] == False
    assert result.iloc[-1]["is_rep_bottom"] == False


def test_detect_rep_bottoms_no_dip_marks_nothing():
    # flat elbow angle (always at the top) -- no rep occurred
    rows = [
        {"video": "flat.mp4", "frame": i,
         "left_elbow_angle": 175, "right_elbow_angle": 175,
         "left_hip_angle": 175, "right_hip_angle": 175,
         "left_shoulder_angle": 45, "right_shoulder_angle": 45}
        for i in range(20)
    ]
    df = pd.DataFrame(rows)
    result = detect_rep_bottoms(df)
    assert not result["is_rep_bottom"].any()


# ── Threshold labeling on bottom-of-rep frames ───────────────────────────

def test_label_frames_by_threshold_marks_shallow_pushup_incorrect():
    df = make_rep_video("shallow.mp4", elbow_bottom=150)
    labeled = label_frames_by_threshold(df)
    assert len(labeled) > 0
    assert (labeled["form_label"] == "incorrect").all()


def test_label_frames_by_threshold_marks_good_depth_correct():
    # bottom well clear of the elbow_max=100 threshold so the whole bottom
    # window (bottom +/- BOTTOM_WINDOW_DEG) stays on the "correct" side
    df = make_rep_video("good.mp4", elbow_bottom=70)
    labeled = label_frames_by_threshold(df)
    assert len(labeled) > 0
    assert (labeled["form_label"] == "correct").all()


def test_label_frames_by_threshold_marks_hip_sag_incorrect():
    df = make_rep_video("sag.mp4", elbow_bottom=70, hip_bottom=140)
    labeled = label_frames_by_threshold(df)
    assert len(labeled) > 0
    assert (labeled["form_label"] == "incorrect").all()


def test_label_frames_by_threshold_marks_hip_pike_incorrect():
    df = make_rep_video("pike.mp4", elbow_bottom=70, hip_bottom=210)
    labeled = label_frames_by_threshold(df)
    assert len(labeled) > 0
    assert (labeled["form_label"] == "incorrect").all()


def test_label_frames_by_threshold_excludes_non_bottom_frames_by_default():
    df = make_rep_video("good.mp4", elbow_bottom=70, n_frames=30)
    labeled = label_frames_by_threshold(df)
    # only frames near the rep bottom should survive, not all 30
    assert len(labeled) < 30


def test_label_frames_by_threshold_all_frames_when_disabled():
    df = make_rep_video("good.mp4", elbow_bottom=70, n_frames=30)
    labeled = label_frames_by_threshold(df, bottom_frames_only=False)
    assert len(labeled) == 30


# ── Feature matrix ────────────────────────────────────────────────────────

def test_build_feature_matrix_shapes():
    df = make_rep_video("good.mp4", elbow_bottom=70)
    labeled = label_frames_by_threshold(df)
    X, y = build_feature_matrix(labeled)
    assert X.shape == (len(labeled), len(FEATURE_COLUMNS))
    assert set(y) <= {"correct", "incorrect"}


# ── Training smoke test ───────────────────────────────────────────────────

def _synthetic_separable_dataset(n_videos=40, seed=0):
    rng = np.random.default_rng(seed)
    dfs = []
    for i in range(n_videos):
        bottom = rng.normal(70, 8) if i % 2 == 0 else rng.normal(150, 8)
        dfs.append(make_rep_video(f"video_{i}.mp4", elbow_bottom=bottom, n_frames=15))
    df = pd.concat(dfs, ignore_index=True)
    return label_frames_by_threshold(df)


def test_train_and_evaluate_smoke():
    labeled = _synthetic_separable_dataset()
    X, y = build_feature_matrix(labeled)
    results = train_and_evaluate(X, y)

    assert 0.0 <= results["accuracy"] <= 1.0
    # clearly separable synthetic data should be easy to classify well
    assert results["accuracy"] > 0.8
    assert results["confusion_matrix"].shape == (2, 2)
    assert set(results["feature_importances"].keys()) == set(FEATURE_COLUMNS)
