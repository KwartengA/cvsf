import pytest

from src.feedback.feedback_generator import (
    THRESHOLDS, SERVE_THRESHOLDS, PUSHUP_THRESHOLDS, LEG_PRESS_THRESHOLDS,
    evaluate_squat_frame, evaluate_squat_session,
    evaluate_tennis_serve_frame, evaluate_tennis_serve_session,
    evaluate_pushup_frame, evaluate_pushup_session,
    evaluate_leg_press_frame, evaluate_leg_press_session,
)


def all_ok(feedback):
    return all(ok for _, ok in feedback)


# ── Squat frame-level ────────────────────────────────────────────────────

def test_squat_frame_good_depth_is_ok():
    angles = {
        "left_knee_angle": 90, "right_knee_angle": 90,
        "left_hip_angle": 80, "right_hip_angle": 80,
        "left_ankle_angle": 80, "right_ankle_angle": 80,
    }
    feedback = evaluate_squat_frame(angles)
    assert all_ok(feedback)


def test_squat_frame_insufficient_depth_flagged():
    angles = {
        "left_knee_angle": 150, "right_knee_angle": 150,
        "left_hip_angle": 120, "right_hip_angle": 120,
        "left_ankle_angle": 80, "right_ankle_angle": 80,
    }
    feedback = evaluate_squat_frame(angles)
    assert not all_ok(feedback)


def test_squat_frame_knee_asymmetry_flagged():
    angles = {
        "left_knee_angle": 90, "right_knee_angle": 120,
        "left_hip_angle": 80, "right_hip_angle": 80,
    }
    feedback = evaluate_squat_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("asymmetry" in m.lower() for m in messages)


def test_squat_frame_heel_rise_flagged():
    angles = {
        "left_knee_angle": 90, "right_knee_angle": 90,
        "left_ankle_angle": 50, "right_ankle_angle": 50,
    }
    feedback = evaluate_squat_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("heel" in m.lower() for m in messages)


def test_squat_frame_missing_angles_returns_empty_list():
    assert evaluate_squat_frame({}) == []


# ── Squat session-level ──────────────────────────────────────────────────

def test_squat_session_passes_with_good_depth():
    rows = [
        {"left_knee_angle": 90, "right_knee_angle": 90,
         "left_hip_angle": 80, "right_hip_angle": 80}
        for _ in range(5)
    ]
    summary = evaluate_squat_session(rows)
    assert summary["depth_reached"] is True
    assert summary["passed"] is True
    assert summary["issues"] == []


def test_squat_session_fails_with_shallow_depth():
    rows = [
        {"left_knee_angle": 150, "right_knee_angle": 150,
         "left_hip_angle": 120, "right_hip_angle": 120}
        for _ in range(5)
    ]
    summary = evaluate_squat_session(rows)
    assert summary["depth_reached"] is False
    assert summary["passed"] is False
    assert len(summary["issues"]) > 0


def test_squat_session_empty_rows_returns_error():
    summary = evaluate_squat_session([])
    assert "error" in summary


# ── Tennis serve frame-level ─────────────────────────────────────────────

def test_tennis_frame_good_form_is_ok():
    angles = {
        "right_elbow_angle": 110,
        "right_shoulder_angle": 100,
        "right_hip_angle": 150, "left_hip_angle": 150,
        "right_knee_angle": 140, "left_knee_angle": 140,
        "trunk_lean_angle": 130,
    }
    feedback = evaluate_tennis_serve_frame(angles)
    assert all_ok(feedback)


def test_tennis_frame_flying_elbow_flagged():
    angles = {"right_shoulder_angle": 150}
    feedback = evaluate_tennis_serve_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("flying elbow" in m.lower() for m in messages)


def test_tennis_frame_straight_arm_flagged():
    angles = {"right_elbow_angle": 175}
    feedback = evaluate_tennis_serve_frame(angles)
    assert not all_ok(feedback)


def test_tennis_frame_missing_angles_returns_empty_list():
    assert evaluate_tennis_serve_frame({}) == []


# ── Tennis serve session-level ───────────────────────────────────────────

def test_tennis_session_passes_with_good_mechanics():
    rows = [
        {"right_elbow_angle": 110, "right_shoulder_angle": 100,
         "right_hip_angle": 150, "left_hip_angle": 150,
         "right_knee_angle": 140, "left_knee_angle": 140}
        for _ in range(5)
    ]
    summary = evaluate_tennis_serve_session(rows)
    assert summary["passed"] is True


def test_tennis_session_flags_straight_legs():
    rows = [
        {"right_elbow_angle": 110, "right_shoulder_angle": 100,
         "right_hip_angle": 150, "left_hip_angle": 150,
         "right_knee_angle": 175, "left_knee_angle": 175}
        for _ in range(5)
    ]
    summary = evaluate_tennis_serve_session(rows)
    assert summary["passed"] is False
    assert any("legs too straight" in i.lower() for i in summary["issues"])


def test_tennis_session_empty_rows_returns_error():
    summary = evaluate_tennis_serve_session([])
    assert "error" in summary


# ── Pushup frame-level ────────────────────────────────────────────────────

def test_pushup_frame_good_form_is_ok():
    angles = {
        "left_elbow_angle": 85, "right_elbow_angle": 85,
        "left_hip_angle": 175, "right_hip_angle": 175,
    }
    feedback = evaluate_pushup_frame(angles)
    assert all_ok(feedback)


def test_pushup_frame_insufficient_depth_flagged():
    angles = {
        "left_elbow_angle": 150, "right_elbow_angle": 150,
        "left_hip_angle": 175, "right_hip_angle": 175,
    }
    feedback = evaluate_pushup_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("lower" in m.lower() for m in messages)


def test_pushup_frame_elbow_asymmetry_flagged():
    angles = {
        "left_elbow_angle": 85, "right_elbow_angle": 120,
        "left_hip_angle": 175, "right_hip_angle": 175,
    }
    feedback = evaluate_pushup_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("asymmetry" in m.lower() for m in messages)


def test_pushup_frame_hip_sag_flagged():
    angles = {
        "left_elbow_angle": 85, "right_elbow_angle": 85,
        "left_hip_angle": 140, "right_hip_angle": 140,
    }
    feedback = evaluate_pushup_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("sagging" in m.lower() for m in messages)


def test_pushup_frame_hip_pike_flagged():
    angles = {
        "left_elbow_angle": 85, "right_elbow_angle": 85,
        "left_hip_angle": 210, "right_hip_angle": 210,
    }
    feedback = evaluate_pushup_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("piking" in m.lower() for m in messages)


def test_pushup_frame_missing_angles_returns_empty_list():
    assert evaluate_pushup_frame({}) == []


# ── Pushup session-level ──────────────────────────────────────────────────

def test_pushup_session_passes_with_good_depth():
    rows = [
        {"left_elbow_angle": 85, "right_elbow_angle": 85,
         "left_hip_angle": 175, "right_hip_angle": 175}
        for _ in range(5)
    ]
    summary = evaluate_pushup_session(rows)
    assert summary["depth_reached"] is True
    assert summary["passed"] is True
    assert summary["issues"] == []


def test_pushup_session_fails_with_shallow_depth():
    rows = [
        {"left_elbow_angle": 150, "right_elbow_angle": 150,
         "left_hip_angle": 175, "right_hip_angle": 175}
        for _ in range(5)
    ]
    summary = evaluate_pushup_session(rows)
    assert summary["depth_reached"] is False
    assert summary["passed"] is False
    assert len(summary["issues"]) > 0


def test_pushup_session_empty_rows_returns_error():
    summary = evaluate_pushup_session([])
    assert "error" in summary


# ── Leg press frame-level ─────────────────────────────────────────────────

def test_leg_press_frame_good_form_is_ok():
    angles = {"left_knee_angle": 90, "right_knee_angle": 90}
    feedback = evaluate_leg_press_frame(angles)
    assert all_ok(feedback)


def test_leg_press_frame_insufficient_depth_flagged():
    angles = {"left_knee_angle": 130, "right_knee_angle": 130}
    feedback = evaluate_leg_press_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("deeper" in m.lower() for m in messages)


def test_leg_press_frame_lockout_flagged():
    angles = {"left_knee_angle": 178, "right_knee_angle": 178}
    feedback = evaluate_leg_press_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("locking" in m.lower() for m in messages)


def test_leg_press_frame_knee_asymmetry_flagged():
    angles = {"left_knee_angle": 90, "right_knee_angle": 130}
    feedback = evaluate_leg_press_frame(angles)
    messages = [msg for msg, ok in feedback if not ok]
    assert any("asymmetry" in m.lower() for m in messages)


def test_leg_press_frame_missing_angles_returns_empty_list():
    assert evaluate_leg_press_frame({}) == []


# ── Leg press session-level ───────────────────────────────────────────────

def test_leg_press_session_passes_with_good_depth():
    rows = [
        {"left_knee_angle": 90, "right_knee_angle": 90}
        for _ in range(5)
    ]
    summary = evaluate_leg_press_session(rows)
    assert summary["depth_reached"] is True
    assert summary["passed"] is True
    assert summary["issues"] == []


def test_leg_press_session_fails_with_shallow_depth():
    rows = [
        {"left_knee_angle": 130, "right_knee_angle": 130}
        for _ in range(5)
    ]
    summary = evaluate_leg_press_session(rows)
    assert summary["depth_reached"] is False
    assert summary["passed"] is False
    assert len(summary["issues"]) > 0


def test_leg_press_session_flags_lockout():
    rows = [
        {"left_knee_angle": 90, "right_knee_angle": 90},
        {"left_knee_angle": 178, "right_knee_angle": 178},
    ]
    summary = evaluate_leg_press_session(rows)
    assert summary["locked_out"] is True
    assert any("locking" in i.lower() for i in summary["issues"])


def test_leg_press_session_empty_rows_returns_error():
    summary = evaluate_leg_press_session([])
    assert "error" in summary
