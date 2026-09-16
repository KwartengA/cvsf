"""
Tennis shot-type classifier (multi-class: serve / forehand / backhand /
volley).

Structurally different from squat/pushup/leg_press_classifier.py: those are
binary correct/incorrect classifiers WITHIN one exercise, with labels
derived from biomechanics thresholds because no ground truth exists.
Tennis shot type has real ground truth -- the folder each video was
uploaded into (data/raw/tennis/{serve,forehand,backhand,volley}/correct/)
IS the label, no threshold-derivation needed. This is a stronger footing
than the other three classifiers, worth stating plainly when presenting
results.

CAVEAT: video counts per class are uneven (serve=13, backhand=8,
forehand=3, volley=3 at time of writing) -- forehand/volley results are
lower-confidence given so few source videos. train_and_evaluate reports
per-class support counts precisely so this isn't hidden behind a single
overall accuracy number.

Features: the angle set from tennis_serve_angles() (src/utils/
angle_calculator.py) -- despite the name, it's generic enough for any
overhead/groundstroke swing (shoulder/elbow/wrist/hip/knee/trunk angles),
not serve-specific. Plus racket-speed-at-hit and racket position relative
to the body (distinguishes e.g. a backhand's racket-behind-body moment
from a forehand's racket-in-front moment), from
src/analysis/tennis_rep_utils.py's hit detection.
"""

import numpy as np
import pandas as pd

from src.analysis.rep_classifier_utils import (
    build_feature_matrix as _build_feature_matrix,
    train_and_evaluate as _train_and_evaluate,
    save_model,
)

ANGLE_FEATURE_COLUMNS = [
    "right_shoulder_angle", "left_shoulder_angle",
    "right_elbow_angle", "left_elbow_angle",
    "right_wrist_angle", "left_wrist_angle",
    "right_hip_angle", "left_hip_angle",
    "right_knee_angle", "left_knee_angle",
    "trunk_lean_angle",
]

RACKET_FEATURE_COLUMNS = [
    "racket_speed", "racket_x_relative_to_body",
]

FEATURE_COLUMNS = ANGLE_FEATURE_COLUMNS + RACKET_FEATURE_COLUMNS


def build_hit_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a frame-level DataFrame that already has `is_hit`/`hit_id`
    (see tennis_rep_utils.detect_hits) and racket_cx/racket_speed columns
    alongside the pose angle columns, collapse each hit's window of frames
    down to one row (the frame with peak racket_speed within that hit) and
    add `racket_x_relative_to_body` (racket x position minus torso-center
    x, so it's meaningful regardless of where the swing happens on screen).

    `shot_type` must already be a column on `df` (the label).
    """
    hit_frames = df[df["is_hit"]].copy()
    if hit_frames.empty:
        return hit_frames

    if "right_hip_x" in hit_frames.columns and "left_hip_x" in hit_frames.columns:
        torso_center_x = (hit_frames["right_hip_x"] + hit_frames["left_hip_x"]) / 2
        hit_frames["racket_x_relative_to_body"] = hit_frames["racket_cx"] - torso_center_x
    else:
        hit_frames["racket_x_relative_to_body"] = np.nan

    # one row per hit: the frame with peak speed within that hit's window
    rows = []
    for hit_id, group in hit_frames.groupby("hit_id"):
        peak_row = group.loc[group["racket_speed"].idxmax()]
        rows.append(peak_row)

    return pd.DataFrame(rows).reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame):
    """Return (X, y) ready for sklearn, given a per-hit DataFrame with a
    `shot_type` label column."""
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["shot_type"].to_numpy()
    return X, y


def train_and_evaluate(X, y, random_state: int = 42) -> dict:
    """
    Multi-class train/eval -- pos_label=None so rep_classifier_utils
    macro-averages precision/recall/F1 across all 4 shot types instead of
    scoring one "positive" class, and reports per-class support so class
    imbalance (see module caveat) is visible in the results, not hidden.
    """
    return _train_and_evaluate(X, y, FEATURE_COLUMNS, random_state=random_state, pos_label=None)
