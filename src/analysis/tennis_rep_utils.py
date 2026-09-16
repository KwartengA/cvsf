"""
Hit-based repetition segmentation for tennis strokes.

Unlike squat/pushup/leg_press -- where a "rep" is the bottom of a
descend/ascend cycle, detected as a local MINIMUM in a joint angle (see
rep_classifier_utils.detect_rep_bottoms) -- a tennis "rep" is a single
swing/contact event ("every hit is a rep", per the user). The signal shape
is opposite: a hit is a local MAXIMUM in racket speed, not a minimum in an
angle, so this is a new segmentation approach, not a reuse of
detect_rep_bottoms.

Racket position (from ObjectDetector, src/pose_estimation/object_detector.py)
is the primary signal: verified directly against real footage
(forehand/backhand/volley test clips) that racket detection is reliable
(10-14/15 sampled frames) while ball detection is not (0-14/15, wildly
inconsistent per video). So hit detection is driven by racket speed alone;
ball position, when available, is only used as supporting/visual
cross-check -- never a requirement, since requiring it would silently drop
real hits on videos where the ball wasn't detected at all.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# A hit's racket-speed peak must exceed the surrounding baseline by at
# least this many standard deviations of THAT VIDEO's own racket-speed
# distribution (relative, not an absolute px/frame cutoff). An absolute
# threshold does not generalize: verified directly against real footage
# that peak racket speed varies a lot by shot type/camera distance/zoom
# (a real serve swing peaked at ~0.13 normalized units on one clip; a real
# forehand swing peaked at only ~0.03 on another -- an absolute threshold
# tuned to the serve clip found zero hits on the forehand clip, even though
# a real swing was clearly present in its speed data as the video's
# dominant peak). Expressing the threshold in standard deviations above
# that video's own mean speed adapts automatically to each video's scale.
MIN_HIT_PROMINENCE_STD = 3.0
# Minimum frames a speed peak must span to count as a real hit, filtering
# out single-frame racket-detection glitches (same rationale as
# MIN_REP_WIDTH_FRAMES in rep_classifier_utils.py, applied to speed here
# instead of angle).
MIN_HIT_WIDTH_FRAMES = 2
# A frame counts as "part of the hit window" if it falls within this many
# frames of the detected speed peak -- the pose/angle snapshot used for
# classification is taken from this window, not just the single peak frame,
# to be robust to the peak landing one frame before/after actual contact.
HIT_WINDOW_FRAMES = 4


def compute_racket_speed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `racket_speed` column: displacement of the racket center
    (racket_cx, racket_cy columns, expected in normalized [0,1] frame
    coordinates) PER FRAME GAP, per video. NaN where racket wasn't detected
    in the current frame, or where there is no earlier detected frame to
    measure from.

    Racket detection has real gaps (verified against real footage: e.g.
    only 65% of frames on a test serve clip). A naive .diff() on a column
    with NaNs silently measures displacement across however many frames
    were skipped, without normalizing by that gap -- so resuming detection
    after a 5-frame gap reads as "5x faster" than the racket actually
    moved, producing false speed spikes exactly at every detection-gap
    boundary (this was observed directly: a 1.7s/100-frame single serve
    produced 6 spurious "hits", most aligned with gap boundaries, before
    this fix). Dividing by the actual frame gap (current frame - last
    detected frame) corrects for this.
    """
    df = df.copy()
    df["racket_speed"] = np.nan

    for video, group in df.groupby("video"):
        group = group.sort_values("frame")
        cx = group["racket_cx"]
        cy = group["racket_cy"]
        frame = group["frame"]

        valid = cx.notna() & cy.notna()
        valid_frame = frame[valid]
        valid_cx = cx[valid]
        valid_cy = cy[valid]

        # displacement + frame-gap between each valid detection and the
        # PREVIOUS valid detection (not the previous row -- rows in
        # between may be NaN)
        dx = valid_cx.diff()
        dy = valid_cy.diff()
        dframe = valid_frame.diff()

        speed = np.sqrt(dx**2 + dy**2) / dframe.replace(0, np.nan)

        df.loc[speed.index, "racket_speed"] = speed.to_numpy()

    return df


def detect_hits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify individual hits (swing/contact events) per video and mark the
    frames in each hit's window with `is_hit=True` / `hit_id`.

    A "hit" is a local maximum in (smoothed) racket speed -- the moment the
    racket is moving fastest, i.e. through contact. Requires `racket_speed`
    (see compute_racket_speed) already on `df`.
    """
    if "racket_speed" not in df.columns:
        df = compute_racket_speed(df)
    else:
        df = df.copy()

    df["is_hit"] = False
    df["hit_id"] = pd.Series(pd.NA, index=df.index, dtype="object")

    for video, group in df.groupby("video"):
        group = group.sort_values("frame")
        # missing-speed frames (racket not detected) treated as zero speed
        # for peak-finding purposes -- a gap shouldn't itself look like a
        # hit, and find_peaks can't operate across NaNs
        speed = group["racket_speed"].fillna(0.0)
        smoothed = speed.rolling(3, center=True, min_periods=1).mean()
        smoothed_vals = smoothed.to_numpy()

        # per-video relative threshold (see MIN_HIT_PROMINENCE_STD docstring
        # for why this can't be a fixed absolute value)
        nonzero = smoothed_vals[smoothed_vals > 0]
        if len(nonzero) < 2:
            continue  # no usable speed signal at all for this video
        prominence = max(nonzero.std() * MIN_HIT_PROMINENCE_STD, 1e-6)

        peaks, _ = find_peaks(
            smoothed_vals,
            prominence=prominence,
            width=MIN_HIT_WIDTH_FRAMES,
        )

        frame_positions = np.arange(len(smoothed_vals))
        for hit_num, peak_idx in enumerate(peaks):
            window = np.abs(frame_positions - peak_idx) <= HIT_WINDOW_FRAMES
            idx = group.index[window]
            df.loc[idx, "is_hit"] = True
            df.loc[idx, "hit_id"] = f"{video}_hit{hit_num}"

    return df


def summarize_hits(df: pd.DataFrame) -> pd.DataFrame:
    """Per-video hit count -- a quick sanity-check table (does the hit
    count look plausible for the video's length?) before trusting a
    classifier trained on the segmented data."""
    hit_df = df[df["is_hit"]]
    return (
        hit_df.groupby("video")["hit_id"]
        .nunique()
        .rename("hit_count")
        .reset_index()
    )
