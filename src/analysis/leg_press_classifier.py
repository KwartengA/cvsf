"""
Leg press correct/incorrect form classifier.

Same rep-bottom-only labeling rationale as squat/pushup_classifier.py: most
frames in a leg press video are the setup/lockout/recovery portions, not
the actual bottom of a rep, so only frames near a detected rep bottom (knee
angle local minimum) are scored against the depth threshold.

Unlike squats/pushups, the person lies flat -- there is no hip-angle or
ankle-angle signal analogous to "heel rise" or "hip hinge" to check, since
leg_press_angles() only captures knee angle (see its docstring). Knee
asymmetry is a secondary feature; same camera-angle caveat as
squat_classifier.py's knee asymmetry applies here (computed as a feature,
not used as a hard labeling rule) until proven reliable on real footage.

LEG_PRESS_THRESHOLDS (feedback_generator.py) were assumed biomechanics
numbers, since checked against real footage (7 videos, 5283 frames, 62
detected reps in data/processed/leg_press_keypoints.csv): the resulting
label split (1633 correct / 369 incorrect, ~82% correct, no single video
dominating either label) is healthy and non-degenerate, unlike squats' and
pushups' initial thresholds which needed recalibration. knee_max=100 is
being kept as-is. Re-check if new footage shifts the label balance
noticeably.
"""

import numpy as np
import pandas as pd

from src.feedback.feedback_generator import LEG_PRESS_THRESHOLDS
from src.analysis.rep_classifier_utils import (
    detect_rep_bottoms as _detect_rep_bottoms,
    build_feature_matrix as _build_feature_matrix,
    train_and_evaluate as _train_and_evaluate,
    save_model,
)

FEATURE_COLUMNS = [
    "left_knee_angle", "right_knee_angle",
    "knee_asymmetry",
]

DEPTH_COLUMN = "avg_knee_angle"


def detect_rep_bottoms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify individual leg press repetitions per video and mark the frames
    near the bottom of each rep with `is_rep_bottom=True`. See
    rep_classifier_utils.detect_rep_bottoms for the general algorithm; this
    wrapper supplies the leg-press-specific depth signal (average knee angle).
    """
    df = df.copy()
    df[DEPTH_COLUMN] = (df["left_knee_angle"] + df["right_knee_angle"]) / 2
    return _detect_rep_bottoms(df, depth_col=DEPTH_COLUMN)


def label_frames_by_threshold(df: pd.DataFrame, bottom_frames_only: bool = True) -> pd.DataFrame:
    """
    Add a binary `form_label` column ("correct" / "incorrect") to a leg
    press keypoint DataFrame by re-applying the same rules as
    evaluate_leg_press_frame/evaluate_leg_press_session in
    feedback_generator.py.

    A frame is "incorrect" if:
      - avg knee angle > LEG_PRESS_THRESHOLDS["knee_max"]  (not deep enough)

    Knee asymmetry is NOT used as a labeling rule -- same camera-angle
    caveat as squat_classifier.py. It is still computed and included as a
    classifier feature.

    If bottom_frames_only is True (default), only frames near the bottom of
    a detected rep are labeled and returned -- setup/lockout/transition
    frames are excluded since the depth threshold doesn't apply to them.
    Set False to label every frame (not recommended for training).
    """
    df = detect_rep_bottoms(df)

    if bottom_frames_only:
        df = df[df["is_rep_bottom"]].copy()

    avg_knee = (df["left_knee_angle"] + df["right_knee_angle"]) / 2
    knee_asymmetry = (df["left_knee_angle"] - df["right_knee_angle"]).abs()

    is_incorrect = avg_knee > LEG_PRESS_THRESHOLDS["knee_max"]

    df["knee_asymmetry"] = knee_asymmetry
    df["form_label"] = np.where(is_incorrect, "incorrect", "correct")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Return (X, y) ready for sklearn, given a labeled leg press DataFrame."""
    return _build_feature_matrix(df, FEATURE_COLUMNS)


def train_and_evaluate(X, y, random_state: int = 42) -> dict:
    """
    Split, train a LogisticRegression classifier, and a RandomForest (for
    feature importances), and return a dict of metrics + fitted models.
    """
    return _train_and_evaluate(X, y, FEATURE_COLUMNS, random_state=random_state)
