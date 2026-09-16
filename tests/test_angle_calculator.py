import math
import pytest

from src.utils.angle_calculator import (
    calculate_angle, calculate_angle_3d, landmark_to_point,
    squat_angles, tennis_serve_angles, pushup_angles, leg_press_angles,
)


class FakeLandmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def make_landmarks(count=33, default=(0.5, 0.5, 0.0)):
    return [FakeLandmark(*default) for _ in range(count)]


def test_calculate_angle_straight_line_is_180():
    a, b, c = (0, 0), (1, 0), (2, 0)
    assert calculate_angle(a, b, c) == pytest.approx(180.0, abs=0.05)


def test_calculate_angle_right_angle_is_90():
    a, b, c = (1, 0), (0, 0), (0, 1)
    assert calculate_angle(a, b, c) == pytest.approx(90.0, abs=1e-3)


def test_calculate_angle_same_point_returns_zero():
    a, b, c = (0, 1), (0, 0), (0, 1)
    assert calculate_angle(a, b, c) == pytest.approx(0.0, abs=0.05)


def test_calculate_angle_3d_matches_2d_when_z_zero():
    a, b, c = (1, 0, 0), (0, 0, 0), (0, 1, 0)
    assert calculate_angle_3d(a, b, c) == pytest.approx(90.0, abs=1e-3)


def test_landmark_to_point():
    lm = FakeLandmark(0.1, 0.2, 0.3)
    assert landmark_to_point(lm) == (0.1, 0.2, 0.3)


def test_squat_angles_returns_expected_keys():
    landmarks = make_landmarks()
    angles = squat_angles(landmarks)
    expected = {
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "left_ankle_angle", "right_ankle_angle",
    }
    assert set(angles.keys()) == expected
    for v in angles.values():
        assert isinstance(v, float)


def test_tennis_serve_angles_returns_expected_keys():
    landmarks = make_landmarks()
    angles = tennis_serve_angles(landmarks)
    expected = {
        "right_shoulder_angle", "left_shoulder_angle",
        "right_elbow_angle", "left_elbow_angle",
        "right_wrist_angle", "left_wrist_angle",
        "right_hip_angle", "left_hip_angle",
        "right_knee_angle", "left_knee_angle",
        "trunk_lean_angle",
    }
    assert set(angles.keys()) == expected


def test_pushup_angles_returns_expected_keys():
    landmarks = make_landmarks()
    angles = pushup_angles(landmarks)
    expected = {
        "left_elbow_angle", "right_elbow_angle",
        "left_hip_angle", "right_hip_angle",
        "left_shoulder_angle", "right_shoulder_angle",
    }
    assert set(angles.keys()) == expected
    for v in angles.values():
        assert isinstance(v, float)


def test_leg_press_angles_returns_expected_keys():
    landmarks = make_landmarks()
    angles = leg_press_angles(landmarks)
    expected = {"left_knee_angle", "right_knee_angle"}
    assert set(angles.keys()) == expected
    for v in angles.values():
        assert isinstance(v, float)
