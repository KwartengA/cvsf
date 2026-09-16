"""
Squat correct/incorrect form classifier.

IMPORTANT CAVEAT: There is currently no independently recorded "incorrect
form" footage for squats. Labels are derived by re-applying the biomechanics
thresholds already used for live feedback (src/feedback/feedback_generator.py)
to the per-frame angle data. This gives a genuinely mixed-class dataset to
train and evaluate a real classifier on today, but the resulting accuracy
describes how well the model recovers the threshold rule from raw angles —
not how well it generalizes to real bad-form footage. State this caveat
when presenting these numbers. Swap in real data/raw/gym/squat/incorrect/
videos and rerun this module once that footage exists.

A squat video contains many frames that are not a squat at all (standing,
walking into frame, resting between reps) — those frames have a nearly
straight knee, which is *normal* there, not bad form. Scoring every frame
independently against bottom-of-squat thresholds floods the dataset with
false "incorrect" labels. So form rules are only applied near the bottom of
each detected rep (see `detect_rep_bottoms` in rep_classifier_utils.py),
matching how a human coach would actually judge squat depth/hinge/heel-lift:
at the lowest point of each repetition, not while standing.

NOTE on knee asymmetry: |left_knee_angle - right_knee_angle| is still
computed and included as a classifier *feature*, but it is deliberately
NOT used as a labeling rule. With a single side-view camera, the far leg's
knee angle reads very differently from the near leg's purely from camera
perspective/foreshortening (observed 18-47deg apart on genuinely good-form
reps in the existing footage) -- so a strict asymmetry threshold would
mislabel the deepest, best reps as "incorrect" just because of camera
angle, not actual uneven squatting. Revisit this once multi-angle or
depth-camera footage is available.
"""

import numpy as np
import pandas as pd

from src.feedback.feedback_generator import THRESHOLDS
from src.analysis.rep_classifier_utils import (
    detect_rep_bottoms as _detect_rep_bottoms,
    build_feature_matrix as _build_feature_matrix,
    train_and_evaluate as _train_and_evaluate,
    save_model,
)

FEATURE_COLUMNS = [
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_ankle_angle", "right_ankle_angle",
    "knee_asymmetry",
]

DEPTH_COLUMN = "avg_knee_angle"


def detect_rep_bottoms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify individual squat repetitions per video and mark the frames near
    the bottom of each rep with `is_rep_bottom=True`. See
    rep_classifier_utils.detect_rep_bottoms for the general algorithm; this
    wrapper supplies the squat-specific depth signal (average knee angle).
    """
    df = df.copy()
    df[DEPTH_COLUMN] = (df["left_knee_angle"] + df["right_knee_angle"]) / 2
    return _detect_rep_bottoms(df, depth_col=DEPTH_COLUMN)


def label_frames_by_threshold(df: pd.DataFrame, bottom_frames_only: bool = True) -> pd.DataFrame:
    """
    Add a binary `form_label` column ("correct" / "incorrect") to a squat
    keypoint DataFrame by re-applying the same rules as
    evaluate_squat_frame/evaluate_squat_session in feedback_generator.py.

    A frame is "incorrect" if any of:
      - avg knee angle > THRESHOLDS["knee_max"]   (not deep enough)
      - avg ankle angle < THRESHOLDS["ankle_min"]  (heel rising)

    Knee asymmetry is NOT used as a labeling rule -- see the module
    docstring for why (single side-view camera makes it unreliable). It is
    still computed and included as a classifier feature.

    If bottom_frames_only is True (default), only frames near the bottom of
    a detected rep (see detect_rep_bottoms) are labeled and returned --
    standing/transition frames are excluded since the thresholds don't apply
    to them. Set False to label every frame (not recommended for training).
    """
    df = detect_rep_bottoms(df)

    if bottom_frames_only:
        df = df[df["is_rep_bottom"]].copy()

    avg_knee = (df["left_knee_angle"] + df["right_knee_angle"]) / 2
    avg_ankle = (df["left_ankle_angle"] + df["right_ankle_angle"]) / 2
    knee_asymmetry = (df["left_knee_angle"] - df["right_knee_angle"]).abs()

    is_incorrect = (
        (avg_knee > THRESHOLDS["knee_max"])
        | (avg_ankle < THRESHOLDS["ankle_min"])
    )

    df["knee_asymmetry"] = knee_asymmetry
    df["form_label"] = np.where(is_incorrect, "incorrect", "correct")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Return (X, y) ready for sklearn, given a labeled squat DataFrame."""
    return _build_feature_matrix(df, FEATURE_COLUMNS)


def train_and_evaluate(X, y, random_state: int = 42) -> dict:
    """
    Split, train a LogisticRegression classifier, and a RandomForest (for
    feature importances), and return a dict of metrics + fitted models.
    """
    return _train_and_evaluate(X, y, FEATURE_COLUMNS, random_state=random_state)
