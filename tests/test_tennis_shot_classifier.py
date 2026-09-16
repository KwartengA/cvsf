import numpy as np
import pandas as pd
import pytest

from src.analysis.tennis_shot_classifier import (
    build_hit_feature_rows, build_feature_matrix, train_and_evaluate,
    FEATURE_COLUMNS, ANGLE_FEATURE_COLUMNS,
)


def make_hit_rows(shot_type, n_hits, seed, elbow_bias=100, racket_x_bias=0.0):
    """Synthetic per-hit rows for one shot type -- each shot type gets a
    distinct angle/racket-position profile so the classifier has something
    genuinely separable to learn, mirroring how real shot types differ
    biomechanically (e.g. backhand racket position is behind the body,
    forehand in front)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_hits):
        row = {
            "video": f"{shot_type}_{i}.mov", "frame": i * 20,
            "hit_id": f"{shot_type}_{i}.mov_hit0",
            "shot_type": shot_type,
            "racket_speed": rng.normal(0.12, 0.02),
            "racket_x_relative_to_body": rng.normal(racket_x_bias, 0.05),
        }
        for col in ANGLE_FEATURE_COLUMNS:
            row[col] = rng.normal(elbow_bias, 10)
        rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_dataset(n_per_class=15, seed=0):
    dfs = [
        make_hit_rows("serve", n_per_class, seed, elbow_bias=140, racket_x_bias=0.0),
        make_hit_rows("forehand", n_per_class, seed + 1, elbow_bias=100, racket_x_bias=0.3),
        make_hit_rows("backhand", n_per_class, seed + 2, elbow_bias=90, racket_x_bias=-0.3),
        make_hit_rows("volley", n_per_class, seed + 3, elbow_bias=70, racket_x_bias=0.1),
    ]
    return pd.concat(dfs, ignore_index=True)


def test_build_feature_matrix_shapes():
    df = _synthetic_dataset(n_per_class=5)
    X, y = build_feature_matrix(df)
    assert X.shape == (len(df), len(FEATURE_COLUMNS))
    assert set(y) == {"serve", "forehand", "backhand", "volley"}


def test_train_and_evaluate_multiclass_smoke():
    df = _synthetic_dataset(n_per_class=20)
    X, y = build_feature_matrix(df)
    results = train_and_evaluate(X, y)

    assert 0.0 <= results["accuracy"] <= 1.0
    # clearly separable synthetic data across 4 classes should train well
    assert results["accuracy"] > 0.7
    assert results["confusion_matrix"].shape == (4, 4)
    assert set(results["labels"]) == {"serve", "forehand", "backhand", "volley"}
    assert set(results["feature_importances"].keys()) == set(FEATURE_COLUMNS)


def test_train_and_evaluate_reports_per_class_support():
    df = _synthetic_dataset(n_per_class=20)
    X, y = build_feature_matrix(df)
    results = train_and_evaluate(X, y)

    assert "support" in results
    assert set(results["support"].keys()) == {"serve", "forehand", "backhand", "volley"}
    assert sum(results["support"].values()) == results["n_test"]


def test_train_and_evaluate_handles_uneven_class_sizes():
    # mirrors the real data shape: some classes have far fewer examples
    dfs = [
        make_hit_rows("serve", 30, 0, elbow_bias=140, racket_x_bias=0.0),
        make_hit_rows("forehand", 8, 1, elbow_bias=100, racket_x_bias=0.3),
        make_hit_rows("backhand", 20, 2, elbow_bias=90, racket_x_bias=-0.3),
        make_hit_rows("volley", 8, 3, elbow_bias=70, racket_x_bias=0.1),
    ]
    df = pd.concat(dfs, ignore_index=True)
    X, y = build_feature_matrix(df)
    results = train_and_evaluate(X, y)

    # should not crash with uneven classes, and every class should appear
    # in the test-set support (stratified split keeps at least some of each)
    assert set(results["support"].keys()) == {"serve", "forehand", "backhand", "volley"}


def test_build_hit_feature_rows_picks_peak_speed_frame():
    rows = [
        {"video": "a.mov", "frame": 10, "hit_id": "a.mov_hit0", "is_hit": True,
         "racket_speed": 0.05, "racket_cx": 0.4, "right_hip_x": 0.5, "left_hip_x": 0.5,
         "shot_type": "forehand"},
        {"video": "a.mov", "frame": 11, "hit_id": "a.mov_hit0", "is_hit": True,
         "racket_speed": 0.20, "racket_cx": 0.6, "right_hip_x": 0.5, "left_hip_x": 0.5,
         "shot_type": "forehand"},
        {"video": "a.mov", "frame": 12, "hit_id": "a.mov_hit0", "is_hit": True,
         "racket_speed": 0.08, "racket_cx": 0.55, "right_hip_x": 0.5, "left_hip_x": 0.5,
         "shot_type": "forehand"},
        {"video": "a.mov", "frame": 20, "hit_id": None, "is_hit": False,
         "racket_speed": 0.01, "racket_cx": 0.3, "right_hip_x": 0.5, "left_hip_x": 0.5,
         "shot_type": "forehand"},
    ]
    df = pd.DataFrame(rows)
    result = build_hit_feature_rows(df)

    assert len(result) == 1
    assert result.iloc[0]["frame"] == 11
    assert result.iloc[0]["racket_x_relative_to_body"] == pytest.approx(0.1, abs=1e-6)
