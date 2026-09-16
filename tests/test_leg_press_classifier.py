import numpy as np
import pandas as pd
import pytest

from src.analysis.leg_press_classifier import (
    detect_rep_bottoms, label_frames_by_threshold, build_feature_matrix,
    train_and_evaluate, FEATURE_COLUMNS,
)


def make_rep_video(video_name, knee_bottom, n_frames=30):
    """
    Build one synthetic leg press video: knee angle starts extended (~175,
    lockout), dips down to `knee_bottom` around the midpoint (legs bent,
    sled closest to body), then returns to extended -- i.e. one clean
    repetition.
    """
    frames = np.arange(n_frames)
    phase = np.pi * frames / (n_frames - 1)
    knee = knee_bottom + (175 - knee_bottom) * (1 - np.sin(phase))
    rows = []
    for i, k in enumerate(knee):
        rows.append({
            "video": video_name, "frame": int(frames[i]),
            "left_knee_angle": k, "right_knee_angle": k,
        })
    return pd.DataFrame(rows)


# ── Rep detection ────────────────────────────────────────────────────────

def test_detect_rep_bottoms_finds_one_rep_per_video():
    df = make_rep_video("a.mp4", knee_bottom=85)
    result = detect_rep_bottoms(df)
    assert result["is_rep_bottom"].any()
    assert result.loc[result["is_rep_bottom"], "rep_id"].nunique() == 1


def test_detect_rep_bottoms_excludes_lockout_frames():
    df = make_rep_video("a.mp4", knee_bottom=85)
    result = detect_rep_bottoms(df)
    # first and last frames are near-full-extension (lockout) -- should not
    # be marked as the bottom of the rep
    assert result.iloc[0]["is_rep_bottom"] == False
    assert result.iloc[-1]["is_rep_bottom"] == False


def test_detect_rep_bottoms_no_dip_marks_nothing():
    # flat knee angle (always extended) -- no rep occurred
    rows = [
        {"video": "flat.mp4", "frame": i,
         "left_knee_angle": 175, "right_knee_angle": 175}
        for i in range(20)
    ]
    df = pd.DataFrame(rows)
    result = detect_rep_bottoms(df)
    assert not result["is_rep_bottom"].any()


# ── Threshold labeling on bottom-of-rep frames ───────────────────────────

def test_label_frames_by_threshold_marks_shallow_leg_press_incorrect():
    df = make_rep_video("shallow.mp4", knee_bottom=150)
    labeled = label_frames_by_threshold(df)
    assert len(labeled) > 0
    assert (labeled["form_label"] == "incorrect").all()


def test_label_frames_by_threshold_marks_good_depth_correct():
    # bottom well clear of the knee_max=100 threshold so the whole bottom
    # window (bottom +/- BOTTOM_WINDOW_DEG) stays on the "correct" side
    df = make_rep_video("good.mp4", knee_bottom=70)
    labeled = label_frames_by_threshold(df)
    assert len(labeled) > 0
    assert (labeled["form_label"] == "correct").all()


def test_label_frames_by_threshold_excludes_non_bottom_frames_by_default():
    df = make_rep_video("good.mp4", knee_bottom=70, n_frames=30)
    labeled = label_frames_by_threshold(df)
    # only frames near the rep bottom should survive, not all 30
    assert len(labeled) < 30


def test_label_frames_by_threshold_all_frames_when_disabled():
    df = make_rep_video("good.mp4", knee_bottom=70, n_frames=30)
    labeled = label_frames_by_threshold(df, bottom_frames_only=False)
    assert len(labeled) == 30


# ── Feature matrix ────────────────────────────────────────────────────────

def test_build_feature_matrix_shapes():
    df = make_rep_video("good.mp4", knee_bottom=70)
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
        dfs.append(make_rep_video(f"video_{i}.mp4", knee_bottom=bottom, n_frames=15))
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
