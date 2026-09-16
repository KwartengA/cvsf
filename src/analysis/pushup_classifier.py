"""
Pushup correct/incorrect form classifier.

IMPORTANT CAVEAT: There is currently no real pushup footage at all (correct
or incorrect) -- this module and its tests are validated against synthetic
landmark sequences (see tests/test_pushup_classifier.py) so the pipeline
logic is proven today. Once real videos are processed into
data/processed/pushup_keypoints.csv (via scripts/analyze_pushup.py), rerun
this module for real accuracy numbers. As with squats, "incorrect" labels
are threshold-derived (src/feedback/feedback_generator.py::PUSHUP_THRESHOLDS)
rather than from independently reviewed bad-form footage -- state this
caveat when presenting these numbers.

Same rep-bottom-only labeling rationale as squat_classifier.py: most frames
in a pushup video are the plank/setup/recovery portions, not the actual
bottom of a rep, so only frames near a detected rep bottom (elbow angle
local minimum) are scored against the depth/alignment thresholds.

NOTE on elbow asymmetry: same caveat as squat_classifier.py's knee
asymmetry -- a single side-view camera makes left/right angle comparisons
unreliable (foreshortening), so it is computed as a classifier *feature*
but not used as a labeling rule.
"""

import numpy as np
import pandas as pd

from src.feedback.feedback_generator import PUSHUP_THRESHOLDS
from src.analysis.rep_classifier_utils import (
    detect_rep_bottoms as _detect_rep_bottoms,
    build_feature_matrix as _build_feature_matrix,
    train_and_evaluate as _train_and_evaluate,
    save_model,
)

FEATURE_COLUMNS = [
    "left_elbow_angle", "right_elbow_angle",
    "left_hip_angle", "right_hip_angle",
    "left_shoulder_angle", "right_shoulder_angle",
    "elbow_asymmetry",
]

DEPTH_COLUMN = "avg_elbow_angle"


def detect_rep_bottoms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify individual pushup repetitions per video and mark the frames
    near the bottom of each rep with `is_rep_bottom=True`. See
    rep_classifier_utils.detect_rep_bottoms for the general algorithm; this
    wrapper supplies the pushup-specific depth signal (average elbow angle).
    """
    df = df.copy()
    df[DEPTH_COLUMN] = (df["left_elbow_angle"] + df["right_elbow_angle"]) / 2
    return _detect_rep_bottoms(df, depth_col=DEPTH_COLUMN)


def label_frames_by_threshold(df: pd.DataFrame, bottom_frames_only: bool = True) -> pd.DataFrame:
    """
    Add a binary `form_label` column ("correct" / "incorrect") to a pushup
    keypoint DataFrame by re-applying the same rules as
    evaluate_pushup_frame/evaluate_pushup_session in feedback_generator.py.

    A frame is "incorrect" if any of:
      - avg elbow angle > PUSHUP_THRESHOLDS["elbow_max"]  (not deep enough)
      - avg hip angle outside [hip_sag_min, hip_pike_max]  (sagging/piking)

    Elbow asymmetry is NOT used as a labeling rule -- see the module
    docstring for why (single side-view camera makes it unreliable). It is
    still computed and included as a classifier feature.

    If bottom_frames_only is True (default), only frames near the bottom of
    a detected rep are labeled and returned -- plank/transition frames are
    excluded since the depth thresholds don't apply to them. Set False to
    label every frame (not recommended for training).
    """
    df = detect_rep_bottoms(df)

    if bottom_frames_only:
        df = df[df["is_rep_bottom"]].copy()

    avg_elbow = (df["left_elbow_angle"] + df["right_elbow_angle"]) / 2
    avg_hip = (df["left_hip_angle"] + df["right_hip_angle"]) / 2
    elbow_asymmetry = (df["left_elbow_angle"] - df["right_elbow_angle"]).abs()

    is_incorrect = (
        (avg_elbow > PUSHUP_THRESHOLDS["elbow_max"])
        | (avg_hip < PUSHUP_THRESHOLDS["hip_sag_min"])
        | (avg_hip > PUSHUP_THRESHOLDS["hip_pike_max"])
    )

    df["elbow_asymmetry"] = elbow_asymmetry
    df["form_label"] = np.where(is_incorrect, "incorrect", "correct")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Return (X, y) ready for sklearn, given a labeled pushup DataFrame."""
    return _build_feature_matrix(df, FEATURE_COLUMNS)


def train_and_evaluate(X, y, random_state: int = 42) -> dict:
    """
    Split, train a LogisticRegression classifier, and a RandomForest (for
    feature importances), and return a dict of metrics + fitted models.
    """
    return _train_and_evaluate(X, y, FEATURE_COLUMNS, random_state=random_state)
