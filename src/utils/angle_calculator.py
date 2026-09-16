import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b formed by vectors b->a and b->c.
    Points are (x, y) or (x, y, z) tuples/arrays.
    Returns angle in degrees [0, 180].
    """
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_angle_3d(a, b, c):
    """Same as calculate_angle but uses x, y, z for more accuracy when available."""
    a = np.array(a[:3], dtype=float)
    b = np.array(b[:3], dtype=float)
    c = np.array(c[:3], dtype=float)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def landmark_to_point(landmark):
    """Convert a MediaPipe NormalizedLandmark to (x, y, z) tuple."""
    return (landmark.x, landmark.y, landmark.z)


def tennis_serve_angles(landmarks):
    """
    Extract key joint angles for a tennis serve from MediaPipe pose landmarks.

    MediaPipe landmark indices used:
      11=left shoulder, 12=right shoulder
      13=left elbow,    14=right elbow
      15=left wrist,    16=right wrist
      23=left hip,      24=right hip
      25=left knee,     26=right knee
      27=left ankle,    28=right ankle

    Assumes a right-handed server (dominant arm = right).
    All bilateral angles are captured so the model can be used for left-handers too.
    """
    lm = landmarks

    def pt(idx):
        return landmark_to_point(lm[idx])

    return {
        # Shoulder abduction / elevation — arm lift during trophy position
        "right_shoulder_angle": calculate_angle(pt(14), pt(12), pt(24)),  # elbow-shoulder-hip
        "left_shoulder_angle":  calculate_angle(pt(13), pt(11), pt(23)),

        # Elbow flexion — bend at trophy position, extension at contact
        "right_elbow_angle":    calculate_angle(pt(12), pt(14), pt(16)),  # shoulder-elbow-wrist
        "left_elbow_angle":     calculate_angle(pt(11), pt(13), pt(15)),

        # Wrist / forearm — extension at contact
        "right_wrist_angle":    calculate_angle(pt(14), pt(16), pt(18)) if len(lm) > 18 else 0.0,
        "left_wrist_angle":     calculate_angle(pt(13), pt(15), pt(17)) if len(lm) > 17 else 0.0,

        # Hip rotation / trunk coil — separation between shoulders and hips
        "right_hip_angle":      calculate_angle(pt(12), pt(24), pt(26)),  # shoulder-hip-knee
        "left_hip_angle":       calculate_angle(pt(11), pt(23), pt(25)),

        # Knee bend — leg drive
        "right_knee_angle":     calculate_angle(pt(24), pt(26), pt(28)),
        "left_knee_angle":      calculate_angle(pt(23), pt(25), pt(27)),

        # Trunk lean — shoulder-hip vertical alignment (right side)
        "trunk_lean_angle":     calculate_angle(pt(12), pt(24), pt(28)),  # shoulder-hip-ankle
    }


def squat_angles(landmarks):
    """
    Extract the key joint angles for a squat from MediaPipe pose landmarks.
    Returns a dict of angle names -> degrees.
    MediaPipe landmark indices used:
      11=left shoulder, 12=right shoulder
      23=left hip,      24=right hip
      25=left knee,     26=right knee
      27=left ankle,    28=right ankle
      31=left foot,     32=right foot
    """
    lm = landmarks

    def pt(idx):
        return landmark_to_point(lm[idx])

    return {
        "left_knee_angle":  calculate_angle(pt(23), pt(25), pt(27)),
        "right_knee_angle": calculate_angle(pt(24), pt(26), pt(28)),
        "left_hip_angle":   calculate_angle(pt(11), pt(23), pt(25)),
        "right_hip_angle":  calculate_angle(pt(12), pt(24), pt(26)),
        "left_ankle_angle": calculate_angle(pt(25), pt(27), pt(31)),
        "right_ankle_angle":calculate_angle(pt(26), pt(28), pt(32)),
    }


def pushup_angles(landmarks):
    """
    Extract the key joint angles for a pushup from MediaPipe pose landmarks.
    Returns a dict of angle names -> degrees.
    MediaPipe landmark indices used:
      11=left shoulder, 12=right shoulder
      13=left elbow,    14=right elbow
      15=left wrist,    16=right wrist
      23=left hip,      24=right hip
      25=left knee,     26=right knee
      27=left ankle,    28=right ankle

    Angles captured:
      - elbow angle (shoulder-elbow-wrist): pushup depth, both sides
      - hip angle (shoulder-hip-knee): plank/back alignment -- flags
        hip sag (angle too high, hips dropped below the line) or piking
        (angle too low, hips raised above the line)
      - shoulder angle (elbow-shoulder-hip): hand placement / elbow flare
        consistency relative to the torso
    """
    lm = landmarks

    def pt(idx):
        return landmark_to_point(lm[idx])

    return {
        "left_elbow_angle":    calculate_angle(pt(11), pt(13), pt(15)),
        "right_elbow_angle":   calculate_angle(pt(12), pt(14), pt(16)),
        "left_hip_angle":      calculate_angle(pt(11), pt(23), pt(25)),
        "right_hip_angle":     calculate_angle(pt(12), pt(24), pt(26)),
        "left_shoulder_angle": calculate_angle(pt(13), pt(11), pt(23)),
        "right_shoulder_angle":calculate_angle(pt(14), pt(12), pt(24)),
    }


def leg_press_angles(landmarks):
    """
    Extract the key leg angles for a leg press from MediaPipe pose
    landmarks. The person lies flat on their back, so only leg motion is
    tracked -- there is no standing torso/trunk-lean context the way there
    is for squat_angles(), and hip/knee/ankle angles mean something
    different lying down than standing. This is deliberately a distinct
    function from squat_angles(), not a reuse of it.

    MediaPipe landmark indices used:
      23=left hip,      24=right hip
      25=left knee,     26=right knee
      27=left ankle,    28=right ankle

    Angle captured:
      - knee angle (hip-knee-ankle): leg extension/retraction depth, the
        primary signal for a leg press rep, both sides.
    """
    lm = landmarks

    def pt(idx):
        return landmark_to_point(lm[idx])

    return {
        "left_knee_angle":  calculate_angle(pt(23), pt(25), pt(27)),
        "right_knee_angle": calculate_angle(pt(24), pt(26), pt(28)),
    }
