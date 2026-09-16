import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarksConnections

# Colors (BGR)
COLOR_ANGLE_TEXT   = (0, 255, 255)   # yellow
COLOR_FEEDBACK_OK  = (0, 200, 0)     # green
COLOR_FEEDBACK_BAD = (0, 0, 220)     # red
COLOR_OVERLAY_BG   = (30, 30, 30)
COLOR_JOINT        = (255, 255, 0)   # cyan-yellow dot (used by draw_angle)

# Skeleton overlay palette -- bright teal bones with a warm orange joint
# fill, sized to stay legible at a distance (e.g. a Pi camera across a room).
COLOR_BONE        = (200, 180, 20)   # teal/cyan
COLOR_JOINT_FILL  = (30, 130, 255)   # warm orange
COLOR_JOINT_RING  = (255, 255, 255)  # white outline for contrast
BONE_THICKNESS    = 4
JOINT_RADIUS      = 7
JOINT_RING_THICKNESS = 2


# MediaPipe's 33-point pose model includes 10 face landmarks (indices
# 0-10: nose, eyes, ears, mouth corners) -- irrelevant for form analysis
# (squat/pushup/leg press/tennis all judge body/limb angles, never facial
# position) and visually cluttering on the overlay. Skipped in both the
# joint-dot loop and the bone-line loop below.
FACE_LANDMARK_INDICES = set(range(11))


def _draw_landmarks_manual(frame, landmarks):
    """
    Draw skeleton manually using the new Tasks API landmarks.
    landmarks: list of 33 NormalizedLandmark objects. Face landmarks
    (indices 0-10) are intentionally skipped -- see FACE_LANDMARK_INDICES.
    """
    h, w = frame.shape[:2]
    connections = PoseLandmarksConnections.POSE_LANDMARKS

    # Draw connections (bones)
    for conn in connections:
        if conn.start in FACE_LANDMARK_INDICES or conn.end in FACE_LANDMARK_INDICES:
            continue
        start = landmarks[conn.start]
        end   = landmarks[conn.end]
        if start.visibility < 0.3 or end.visibility < 0.3:
            continue
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w),   int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), COLOR_BONE, BONE_THICKNESS, cv2.LINE_AA)

    # Draw joints
    for i, lm in enumerate(landmarks):
        if i in FACE_LANDMARK_INDICES:
            continue
        if lm.visibility < 0.3:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), JOINT_RADIUS, COLOR_JOINT_FILL, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), JOINT_RADIUS, COLOR_JOINT_RING, JOINT_RING_THICKNESS, cv2.LINE_AA)


def draw_pose(frame, result):
    """Draw the MediaPipe skeleton on the frame in-place using Tasks API result."""
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        _draw_landmarks_manual(frame, result.pose_landmarks[0])


def _px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def draw_angle(frame, landmarks, idx_b, angle_deg, label=""):
    """Draw an angle value at joint idx_b."""
    h, w = frame.shape[:2]
    pt = _px(landmarks[idx_b], w, h)

    cv2.circle(frame, pt, 8, COLOR_JOINT, -1)

    # cv2.putText can't render the Unicode degree sign (° -> garbled glyphs
    # with FONT_HERSHEY fonts), so use a plain-ASCII "deg" instead.
    text = f"{label}{angle_deg:.1f}deg"
    cv2.putText(
        frame, text,
        (pt[0] + 10, pt[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        COLOR_ANGLE_TEXT, 2, cv2.LINE_AA,
    )


def draw_squat_angles(frame, landmarks, angles: dict):
    """Draw all squat joint angles on the frame."""
    angle_joints = [
        ("left_knee_angle",   25, "LK:"),
        ("right_knee_angle",  26, "RK:"),
        ("left_hip_angle",    23, "LH:"),
        ("right_hip_angle",   24, "RH:"),
        ("left_ankle_angle",  27, "LA:"),
        ("right_ankle_angle", 28, "RA:"),
    ]
    for key, joint_idx, label in angle_joints:
        if key in angles:
            draw_angle(frame, landmarks, joint_idx, angles[key], label)


def draw_pushup_angles(frame, landmarks, angles: dict):
    """Draw all pushup joint angles on the frame."""
    angle_joints = [
        ("left_elbow_angle",    13, "LE:"),
        ("right_elbow_angle",   14, "RE:"),
        ("left_hip_angle",      23, "LH:"),
        ("right_hip_angle",     24, "RH:"),
        ("left_shoulder_angle", 11, "LS:"),
        ("right_shoulder_angle",12, "RS:"),
    ]
    for key, joint_idx, label in angle_joints:
        if key in angles:
            draw_angle(frame, landmarks, joint_idx, angles[key], label)


def draw_feedback_panel(frame, feedback_lines: list):
    """
    Overlay a semi-transparent panel top-left with feedback text.
    Each item: (text, is_ok: bool).
    """
    if not feedback_lines:
        return

    panel_x, panel_y = 10, 10
    line_h  = 26
    padding = 8
    panel_w = 340
    panel_h = len(feedback_lines) * line_h + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  COLOR_OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (text, is_ok) in enumerate(feedback_lines):
        color = COLOR_FEEDBACK_OK if is_ok else COLOR_FEEDBACK_BAD
        y = panel_y + padding + (i + 1) * line_h - 6
        cv2.putText(frame, text,
                    (panel_x + padding, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2, cv2.LINE_AA)


def draw_frame_info(frame, frame_idx: int, fps: float = None):
    """Draw frame number at the bottom-left."""
    h, w = frame.shape[:2]
    text = f"Frame: {frame_idx}"
    if fps:
        text += f"  |  {fps:.1f} fps"
    cv2.putText(frame, text,
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
