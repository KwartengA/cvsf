"""
Rule-based squat feedback.

Thresholds are based on biomechanics guidelines for a standard back squat:
  - Knee angle at bottom: 70–100° is a good parallel-or-below squat
  - Hip angle at bottom: should drop below 100° (hip crease below knee)
  - Knee should not cave inward (tracked via x-coordinate spread)
  - Ankle angle: heel should stay planted (>= 70°)
"""

# Configurable thresholds (degrees)
THRESHOLDS = {
    "knee_min":    70,    # minimum knee flexion for adequate depth
    "knee_max":   100,    # above this at the bottom = not deep enough
    "hip_min":    60,     # hip should flex below this at the bottom
    "hip_max":   100,     # above this at bottom = not reaching depth
    "ankle_min":  70,     # below this = heel raising off the ground
}


def evaluate_squat_frame(angles: dict) -> list:
    """
    Evaluate a single frame's squat angles against thresholds.
    Returns list of (message, is_ok) tuples for the visualizer panel.
    """
    feedback = []

    lk = angles.get("left_knee_angle")
    rk = angles.get("right_knee_angle")
    lh = angles.get("left_hip_angle")
    rh = angles.get("right_hip_angle")
    la = angles.get("left_ankle_angle")
    ra = angles.get("right_ankle_angle")

    if lk is not None and rk is not None:
        avg_knee = (lk + rk) / 2
        if avg_knee < THRESHOLDS["knee_min"]:
            feedback.append((f"Knee: {avg_knee:.0f}° — good depth!", True))
        elif avg_knee <= THRESHOLDS["knee_max"]:
            feedback.append((f"Knee: {avg_knee:.0f}° — parallel depth", True))
        else:
            feedback.append((f"Knee: {avg_knee:.0f}° — go deeper", False))

        symmetry = abs(lk - rk)
        if symmetry > 15:
            feedback.append((f"Knee asymmetry: {symmetry:.0f}°", False))
        else:
            feedback.append(("Knee symmetry: OK", True))

    if lh is not None and rh is not None:
        avg_hip = (lh + rh) / 2
        if avg_hip <= THRESHOLDS["hip_max"]:
            feedback.append((f"Hip: {avg_hip:.0f}° — good hinge", True))
        else:
            feedback.append((f"Hip: {avg_hip:.0f}° — hinge more", False))

    if la is not None and ra is not None:
        avg_ankle = (la + ra) / 2
        if avg_ankle >= THRESHOLDS["ankle_min"]:
            feedback.append(("Heels: planted", True))
        else:
            feedback.append((f"Heels: rising ({avg_ankle:.0f}°)", False))

    return feedback


def evaluate_squat_session(angle_rows: list) -> dict:
    """
    Evaluate all frames of a squat session and return a session summary.
    `angle_rows` is a list of dicts each containing squat angle keys.
    """
    import numpy as np

    knee_angles = [
        (r["left_knee_angle"] + r["right_knee_angle"]) / 2
        for r in angle_rows
        if "left_knee_angle" in r and "right_knee_angle" in r
    ]
    hip_angles = [
        (r["left_hip_angle"] + r["right_hip_angle"]) / 2
        for r in angle_rows
        if "left_hip_angle" in r and "right_hip_angle" in r
    ]

    if not knee_angles:
        return {"error": "No valid frames found"}

    min_knee = min(knee_angles)
    min_hip = min(hip_angles) if hip_angles else None

    summary = {
        "total_frames":       len(angle_rows),
        "min_knee_angle":     round(min_knee, 1),
        "avg_knee_angle":     round(float(np.mean(knee_angles)), 1),
        "min_hip_angle":      round(min_hip, 1) if min_hip else None,
        "depth_reached":      min_knee <= THRESHOLDS["knee_max"],
        "parallel_or_below":  min_knee <= THRESHOLDS["knee_min"],
    }

    issues = []
    if not summary["depth_reached"]:
        issues.append("Squat depth insufficient — knee angle never reached parallel.")
    if min_hip and min_hip > THRESHOLDS["hip_max"]:
        issues.append("Hip hinge limited — try sitting back more.")

    summary["issues"] = issues
    summary["passed"] = len(issues) == 0
    return summary


# ── Tennis serve thresholds ────────────────────────────────────────────────────
# Based on biomechanics of a flat/kick serve:
#   - Elbow should reach ~100–130° at trophy position (loaded, not fully extended)
#   - Shoulder abduction 80–120° at trophy position
#   - Hip angle < 160° = good coil / hip-shoulder separation
#   - Knee bend 120–160° = leg drive loaded
#   - Trunk lean > 20° from vertical indicates good bow-and-arrow posture
SERVE_THRESHOLDS = {
    "elbow_trophy_min":   90,   # minimum elbow bend at trophy
    "elbow_trophy_max":  140,   # above this = arm too straight (no leverage)
    "shoulder_abduct_min": 70,  # minimum shoulder lift
    "shoulder_abduct_max": 130, # above this = over-rotation / flying elbow
    "hip_coil_max":      160,   # below this = good trunk coil
    "knee_drive_min":    120,   # below this = insufficient leg bend
    "knee_drive_max":    165,   # above this = legs too straight (no drive)
    "trunk_lean_min":    140,   # below this = good backward bow
}


def evaluate_tennis_serve_frame(angles: dict) -> list:
    """
    Evaluate a single frame's tennis serve angles.
    Returns list of (message, is_ok) tuples for the feedback panel.
    """
    feedback = []

    re = angles.get("right_elbow_angle")
    rs = angles.get("right_shoulder_angle")
    rh = angles.get("right_hip_angle")
    lh = angles.get("left_hip_angle")
    rk = angles.get("right_knee_angle")
    lk = angles.get("left_knee_angle")
    tr = angles.get("trunk_lean_angle")

    if re is not None:
        if SERVE_THRESHOLDS["elbow_trophy_min"] <= re <= SERVE_THRESHOLDS["elbow_trophy_max"]:
            feedback.append((f"Elbow: {re:.0f}° — good bend", True))
        elif re < SERVE_THRESHOLDS["elbow_trophy_min"]:
            feedback.append((f"Elbow: {re:.0f}° — too bent", False))
        else:
            feedback.append((f"Elbow: {re:.0f}° — straighten less", False))

    if rs is not None:
        if SERVE_THRESHOLDS["shoulder_abduct_min"] <= rs <= SERVE_THRESHOLDS["shoulder_abduct_max"]:
            feedback.append((f"Shoulder: {rs:.0f}° — good lift", True))
        elif rs < SERVE_THRESHOLDS["shoulder_abduct_min"]:
            feedback.append((f"Shoulder: {rs:.0f}° — lift arm higher", False))
        else:
            feedback.append((f"Shoulder: {rs:.0f}° — flying elbow", False))

    if rh is not None and lh is not None:
        avg_hip = (rh + lh) / 2
        if avg_hip <= SERVE_THRESHOLDS["hip_coil_max"]:
            feedback.append((f"Hip coil: {avg_hip:.0f}° — good rotation", True))
        else:
            feedback.append((f"Hip coil: {avg_hip:.0f}° — rotate hips more", False))

    if rk is not None and lk is not None:
        avg_knee = (rk + lk) / 2
        if SERVE_THRESHOLDS["knee_drive_min"] <= avg_knee <= SERVE_THRESHOLDS["knee_drive_max"]:
            feedback.append((f"Legs: {avg_knee:.0f}° — good drive", True))
        elif avg_knee < SERVE_THRESHOLDS["knee_drive_min"]:
            feedback.append((f"Legs: {avg_knee:.0f}° — too deep", False))
        else:
            feedback.append((f"Legs: {avg_knee:.0f}° — bend knees more", False))

    if tr is not None:
        if tr <= SERVE_THRESHOLDS["trunk_lean_min"]:
            feedback.append((f"Trunk: {tr:.0f}° — good bow", True))
        else:
            feedback.append((f"Trunk: {tr:.0f}° — lean back more", False))

    return feedback


def evaluate_tennis_serve_session(angle_rows: list) -> dict:
    """
    Evaluate all frames of a tennis serve session and return a summary.
    """
    import numpy as np

    elbow_angles  = [r["right_elbow_angle"]    for r in angle_rows if "right_elbow_angle"    in r]
    shoulder_angles = [r["right_shoulder_angle"] for r in angle_rows if "right_shoulder_angle" in r]
    knee_angles   = [
        (r["right_knee_angle"] + r["left_knee_angle"]) / 2
        for r in angle_rows if "right_knee_angle" in r and "left_knee_angle" in r
    ]
    hip_angles    = [
        (r["right_hip_angle"] + r["left_hip_angle"]) / 2
        for r in angle_rows if "right_hip_angle" in r and "left_hip_angle" in r
    ]

    if not elbow_angles:
        return {"error": "No valid frames found"}

    min_elbow = min(elbow_angles)
    avg_elbow = float(np.mean(elbow_angles))
    avg_shoulder = float(np.mean(shoulder_angles)) if shoulder_angles else None
    min_knee = min(knee_angles) if knee_angles else None

    issues = []
    if min_elbow > SERVE_THRESHOLDS["elbow_trophy_max"]:
        issues.append("Elbow never bent enough — load the arm at trophy position.")
    if avg_shoulder and avg_shoulder < SERVE_THRESHOLDS["shoulder_abduct_min"]:
        issues.append("Shoulder lift too low on average — raise the hitting arm higher.")
    if min_knee and min_knee > SERVE_THRESHOLDS["knee_drive_max"]:
        issues.append("Legs too straight — bend knees for more power from leg drive.")

    summary = {
        "total_frames":      len(angle_rows),
        "min_elbow_angle":   round(min_elbow, 1),
        "avg_elbow_angle":   round(avg_elbow, 1),
        "avg_shoulder_angle": round(avg_shoulder, 1) if avg_shoulder else None,
        "min_knee_angle":    round(min_knee, 1) if min_knee else None,
        "avg_hip_angle":     round(float(np.mean(hip_angles)), 1) if hip_angles else None,
        "issues":            issues,
        "passed":            len(issues) == 0,
    }
    return summary


def print_tennis_serve_report(summary: dict, video_name: str = ""):
    print(f"\n{'='*50}")
    print(f"  TENNIS SERVE REPORT  {video_name}")
    print(f"{'='*50}")
    print(f"  Frames analysed   : {summary.get('total_frames')}")
    print(f"  Min elbow angle   : {summary.get('min_elbow_angle')}°")
    print(f"  Avg elbow angle   : {summary.get('avg_elbow_angle')}°")
    print(f"  Avg shoulder angle: {summary.get('avg_shoulder_angle')}°")
    print(f"  Min knee angle    : {summary.get('min_knee_angle')}°")
    print(f"  Avg hip angle     : {summary.get('avg_hip_angle')}°")

    if summary.get("issues"):
        print("\n  Issues detected:")
        for issue in summary["issues"]:
            print(f"    ! {issue}")
    else:
        print("\n  Serve mechanics look good!")
    print(f"{'='*50}\n")


def print_session_report(summary: dict, video_name: str = ""):
    print(f"\n{'='*50}")
    print(f"  SQUAT REPORT  {video_name}")
    print(f"{'='*50}")
    print(f"  Frames analysed : {summary.get('total_frames')}")
    print(f"  Min knee angle  : {summary.get('min_knee_angle')}°")
    print(f"  Avg knee angle  : {summary.get('avg_knee_angle')}°")
    print(f"  Min hip angle   : {summary.get('min_hip_angle')}°")
    print(f"  Depth reached   : {'YES' if summary.get('depth_reached') else 'NO'}")
    print(f"  Parallel/below  : {'YES' if summary.get('parallel_or_below') else 'NO'}")

    if summary.get("issues"):
        print("\n  Issues detected:")
        for issue in summary["issues"]:
            print(f"    ! {issue}")
    else:
        print("\n  Form looks good!")
    print(f"{'='*50}\n")
