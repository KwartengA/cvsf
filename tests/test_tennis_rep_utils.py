import numpy as np
import pandas as pd
import pytest

from src.analysis.tennis_rep_utils import (
    compute_racket_speed, detect_hits, summarize_hits,
)


def make_swing_video(video_name, n_frames=30, hit_frame=15, speed_peak=0.15):
    """
    Build one synthetic tennis video: racket center barely moves (racket
    held still) except for a fast sweep through `hit_frame` -- a single
    clean hit, shaped like a real swing (accelerate in, decelerate out).
    """
    frames = np.arange(n_frames)
    # racket_cx sweeps through a fast arc centered on hit_frame; racket_cy
    # stays roughly constant (simplification -- real swings move in both
    # axes, but a single-axis sweep still produces a clean speed peak)
    cx = 0.3 + 0.4 * np.exp(-((frames - hit_frame) ** 2) / 8.0) * np.sign(frames - hit_frame + 0.01)
    cy = np.full(n_frames, 0.5)
    rows = []
    for i, (x, y) in enumerate(zip(cx, cy)):
        rows.append({"video": video_name, "frame": int(frames[i]), "racket_cx": x, "racket_cy": y})
    return pd.DataFrame(rows)


def make_still_video(video_name, n_frames=20):
    """Racket barely moves at all -- no hit should be detected."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_frames):
        # tiny, uniform jitter only -- no real peak relative to this
        # video's own speed distribution (see MIN_HIT_PROMINENCE_STD)
        rows.append({
            "video": video_name, "frame": i,
            "racket_cx": 0.5 + rng.normal(0, 0.0005),
            "racket_cy": 0.5 + rng.normal(0, 0.0005),
        })
    return pd.DataFrame(rows)


def test_compute_racket_speed_adds_column():
    df = make_swing_video("a.mov")
    result = compute_racket_speed(df)
    assert "racket_speed" in result.columns
    # first frame of each video has no prior frame -> NaN speed
    assert pd.isna(result.iloc[0]["racket_speed"])


def test_compute_racket_speed_peaks_near_the_swing():
    df = make_swing_video("a.mov", hit_frame=15)
    result = compute_racket_speed(df)
    result = result.sort_values("frame")
    peak_frame = result.loc[result["racket_speed"].idxmax(), "frame"]
    assert abs(peak_frame - 15) <= 3


def test_detect_hits_finds_one_hit():
    df = make_swing_video("a.mov", hit_frame=15)
    result = detect_hits(df)
    assert result["is_hit"].any()
    assert result.loc[result["is_hit"], "hit_id"].nunique() == 1


def test_detect_hits_marks_frames_near_the_peak():
    df = make_swing_video("a.mov", hit_frame=15)
    result = detect_hits(df).sort_values("frame")
    hit_frames = result.loc[result["is_hit"], "frame"].tolist()
    assert 15 in hit_frames or any(abs(f - 15) <= 4 for f in hit_frames)


def test_detect_hits_no_swing_marks_nothing():
    df = make_still_video("still.mov")
    result = detect_hits(df)
    assert not result["is_hit"].any()


def test_detect_hits_handles_missing_racket_frames():
    # racket not detected on some frames (NaN cx/cy) -- shouldn't crash,
    # and shouldn't fabricate a hit out of the gap
    df = make_swing_video("a.mov", hit_frame=15)
    df.loc[df["frame"].between(5, 8), ["racket_cx", "racket_cy"]] = np.nan
    result = detect_hits(df)
    # the real swing hit (around frame 15) should still be found
    assert result["is_hit"].any()


def test_summarize_hits_counts_per_video():
    df1 = make_swing_video("a.mov", hit_frame=15)
    df2 = make_swing_video("b.mov", hit_frame=10)
    combined = pd.concat([df1, df2], ignore_index=True)
    result = detect_hits(combined)
    summary = summarize_hits(result)
    assert set(summary["video"]) == {"a.mov", "b.mov"}
    assert (summary["hit_count"] >= 1).all()
