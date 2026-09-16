"""
Shared machinery for rep-based (squat, pushup, ...) form classifiers.

A rep-based activity is one where "correctness" is judged at the bottom of
each individual repetition (squat depth, pushup depth) rather than over a
whole continuous motion (a tennis swing). This module holds the parts that
are identical across activities: detecting rep bottoms from a smoothed angle
curve, and training/evaluating an sklearn classifier on the resulting
per-rep feature rows. Activity-specific modules (squat_classifier.py,
pushup_classifier.py) supply the angle column(s) that define "depth" and the
threshold-based labeling rule for that activity.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

# A frame counts as "near the bottom of a rep" if its depth angle is within
# this many degrees of that rep's local minimum.
BOTTOM_WINDOW_DEG = 15
# Minimum degrees a rep's dip must fall below its surroundings to count as
# a real repetition (filters out camera jitter / noise, not just any wiggle).
MIN_REP_PROMINENCE_DEG = 20
# Minimum width (in frames) a dip must span to count as a real repetition,
# not a single-frame pose-tracking glitch (e.g. one bad landmark reading).
MIN_REP_WIDTH_FRAMES = 5


def detect_rep_bottoms(df: pd.DataFrame, depth_col: str) -> pd.DataFrame:
    """
    Identify individual repetitions per video and mark the frames near the
    bottom of each rep with `is_rep_bottom=True`.

    A "rep" is detected as a local minimum in the (smoothed) `depth_col`
    over time -- the deepest point of one descend/hold/ascend cycle. Videos
    may contain multiple reps; each is detected independently.

    `depth_col` must already exist on `df` (e.g. an avg knee/elbow angle
    computed by the caller).
    """
    df = df.copy()
    df["is_rep_bottom"] = False
    df["rep_id"] = pd.Series(pd.NA, index=df.index, dtype="object")

    for video, group in df.groupby("video"):
        group = group.sort_values("frame")
        smoothed = group[depth_col].rolling(5, center=True, min_periods=1).mean()
        smoothed_vals = smoothed.to_numpy()

        # invert so rep bottoms (angle minima) become peaks for find_peaks
        peaks, _ = find_peaks(
            -smoothed_vals,
            prominence=MIN_REP_PROMINENCE_DEG,
            width=MIN_REP_WIDTH_FRAMES,
        )

        # midpoints between consecutive peaks bound how far one rep's
        # "near bottom" window can reach, so two reps' windows never overlap
        # and a frame is never claimed by a peak it isn't actually next to
        bounds = [0] + [
            (peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)
        ] + [len(smoothed_vals)]

        for rep_num, peak_idx in enumerate(peaks):
            rep_min_angle = smoothed_vals[peak_idx]
            lo, hi = bounds[rep_num], bounds[rep_num + 1]
            window = np.zeros(len(smoothed_vals), dtype=bool)
            window[lo:hi] = smoothed_vals[lo:hi] <= (rep_min_angle + BOTTOM_WINDOW_DEG)

            idx = group.index[window]
            df.loc[idx, "is_rep_bottom"] = True
            df.loc[idx, "rep_id"] = f"{video}_rep{rep_num}"

    return df


def build_feature_matrix(df: pd.DataFrame, feature_columns: list):
    """Return (X, y) ready for sklearn, given a labeled DataFrame."""
    X = df[feature_columns].to_numpy()
    y = df["form_label"].to_numpy()
    return X, y


def train_and_evaluate(X, y, feature_names: list, random_state: int = 42,
                        pos_label: str = "incorrect") -> dict:
    """
    Split, train a LogisticRegression classifier, and a RandomForest (for
    feature importances), and return a dict of metrics + fitted models.

    `pos_label`: for a binary correct/incorrect classifier (squat, pushup,
    leg_press), precision/recall/F1 are reported for this specific class,
    as before. Pass `pos_label=None` for a multi-class classifier (e.g.
    tennis shot type) -- precision/recall/F1 are then macro-averaged across
    all classes instead (unweighted mean per-class score, appropriate here
    since class sizes are uneven, e.g. tennis's 3-video forehand/volley vs
    13-video serve, and macro-averaging doesn't let the largest class hide
    poor performance on the smaller ones the way micro/weighted averaging
    would).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    if pos_label is not None:
        precision = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    else:
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # per-class support (test-set count) -- important context for multi-class
    # results where classes have very different amounts of source footage
    unique, counts = np.unique(y_test, return_counts=True)
    support = dict(zip(unique.tolist(), counts.tolist()))

    results = {
        "model": clf,
        "rf_model": rf,
        "labels": labels,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "support": support,
        "feature_names": feature_names,
        "feature_importances": dict(zip(feature_names, rf.feature_importances_)),
    }
    return results


def save_model(model, output_path: str):
    import joblib
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved model -> {output_path}")
