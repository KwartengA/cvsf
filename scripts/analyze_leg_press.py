"""
analyze_leg_press.py
---------------------
End-to-end leg press analysis script.

Usage (from the project root with venv active):
    python scripts/analyze_leg_press.py               # full run, window + saves video + CSV
    python scripts/analyze_leg_press.py --no-save      # no window, no annotated video -- just CSV
    python scripts/analyze_leg_press.py --no-display   # no window, but still saves annotated
                                                        # video + CSV -- use this in a headless/
                                                        # non-interactive shell (no display server):
                                                        # cv2.imshow/waitKey misbehave there and can
                                                        # cause frames to be silently skipped.

What it does:
  1. Processes all leg press videos in data/raw/gym/leg_presses/correct/
  2. Per frame: detects the main person (YOLOv8n, largest bounding box --
     handles gym-background footage with other people in frame), crops to
     that person with a margin, runs MediaPipe pose on the crop, remaps
     landmarks back to full-frame coordinates
  3. Displays each video with skeleton overlay + live knee angles + feedback panel
  4. Saves annotated video to data/processed/
  5. Exports a keypoint CSV to data/processed/leg_press_keypoints.csv
  6. Prints a per-video form report to the terminal

Press  Q  while the video window is open to skip to the next video.

NOTE: sled/footplate tracking (src.pose_estimation.object_detector.SledTracker)
is not wired into this script yet -- it needs a per-video starting bounding
box, which isn't something that can be auto-determined generically. This
script currently uses knee-angle-only depth, same signal shape as squats.
"""

import sys
import os
import cv2
from pathlib import Path

NO_SAVE = "--no-save" in sys.argv
NO_DISPLAY = "--no-display" in sys.argv or NO_SAVE

# Allow imports from the project root regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.video_utils import open_video, video_info
from src.pose_estimation.extractor import PoseExtractor
from src.pose_estimation.object_detector import (
    PersonDetector, crop_with_margin, remap_landmark_to_full_frame,
    score_landscape_subject,
)
from src.pose_estimation.visualizer import draw_feedback_panel, draw_frame_info
from src.utils.angle_calculator import leg_press_angles, calculate_angle
from src.feedback.feedback_generator import (
    evaluate_leg_press_frame, evaluate_leg_press_session, print_leg_press_report,
)
from src.analysis.dataset_creator import rows_to_dataframe, save_csv, summarize_dataset

# ── Paths ────────────────────────────────────────────────────────────────────
CORRECT_DIR   = Path("data/raw/gym/leg_presses/correct")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PROCESSED_DIR / "leg_press_keypoints.csv"

# Landmark indices needed for leg_press_angles() + drawing (hip, knee, ankle)
LEG_LANDMARK_INDICES = [23, 24, 25, 26, 27, 28]


class _RemappedLandmark:
    """Lightweight stand-in for a MediaPipe landmark with full-frame
    normalized coordinates, built from a crop-normalized landmark."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def remap_landmarks(landmarks, crop_origin, crop_shape, full_frame_shape):
    """Remap every landmark from crop-normalized to full-frame-normalized
    coordinates. z and visibility pass through unchanged (see
    object_detector.remap_landmark_to_full_frame docstring)."""
    remapped = []
    for lm in landmarks:
        full_x, full_y = remap_landmark_to_full_frame(
            lm, crop_origin, crop_shape, full_frame_shape
        )
        remapped.append(_RemappedLandmark(full_x, full_y, lm.z, getattr(lm, "visibility", 0.0)))
    return remapped


def draw_leg_press_overlay(frame, landmarks, angles, person_bbox):
    """Minimal overlay: person bbox, hip-knee-ankle lines/points for both
    legs, and the knee angle labels. Doesn't reuse draw_pose/draw_squat_angles
    since those assume a standing full-body skeleton; leg press only needs
    the leg landmarks drawn on top of a lying-down body."""
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in person_bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2, cv2.LINE_AA)

    for hip_idx, knee_idx, ankle_idx, label in [(23, 25, 27, "L"), (24, 26, 28, "R")]:
        hip, knee, ankle = landmarks[hip_idx], landmarks[knee_idx], landmarks[ankle_idx]
        pts = [(int(p.x * w), int(p.y * h)) for p in (hip, knee, ankle)]
        cv2.line(frame, pts[0], pts[1], (200, 180, 20), 4, cv2.LINE_AA)
        cv2.line(frame, pts[1], pts[2], (200, 180, 20), 4, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 7, (30, 130, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 7, (255, 255, 255), 2, cv2.LINE_AA)

        angle_key = f"{'left' if label == 'L' else 'right'}_knee_angle"
        if angle_key in angles:
            cv2.putText(frame, f"{label}K:{angles[angle_key]:.0f}deg",
                        (pts[1][0] + 10, pts[1][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)


def analyze_video(video_path: Path, label: str = "correct") -> list:
    info = video_info(str(video_path))
    fps    = info["fps"]
    width  = info["width"]
    height = info["height"]

    # .mp4 extension, not video_path's original (matches the mp4v codec
    # actually written below -- an .mov container tagged as mp4v content
    # can cause some players, e.g. QuickTime, to misjudge playback speed)
    out_path = PROCESSED_DIR / f"annotated_{video_path.stem}.mp4"
    if not NO_SAVE:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    else:
        writer = None

    print(f"\n{'─'*55}")
    print(f"  Video : {video_path.name}")
    print(f"  Size  : {width}x{height}  FPS: {fps:.1f}  Frames: {info['frame_count']}")
    if not NO_DISPLAY:
        print(f"  Press Q in the window to skip to next video.")
    print(f"{'─'*55}")

    all_rows      = []
    angle_history = []

    cap = open_video(str(video_path))
    # Leg press: the subject lies flat, so their box is wide/short -- use
    # that to avoid locking onto a standing background mural/gym-goer with
    # similar confidence/area (see object_detector.score_landscape_subject).
    person_detector = PersonDetector(score_fn=score_landscape_subject)

    with PoseExtractor() as extractor:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            bbox = person_detector.detect_main_person(frame)

            if bbox is not None:
                crop, origin = crop_with_margin(frame, bbox)
                results, landmarks = extractor.extract_frame(crop, frame_idx=frame_idx, fps=fps)
            else:
                landmarks = None

            if landmarks:
                full_frame_landmarks = remap_landmarks(landmarks, origin, crop.shape, frame.shape)

                angles = leg_press_angles(full_frame_landmarks)

                feedback = evaluate_leg_press_frame(angles)
                draw_leg_press_overlay(frame, full_frame_landmarks, angles, bbox)
                draw_feedback_panel(frame, feedback)

                row = extractor.landmarks_to_dict(full_frame_landmarks, frame_idx=frame_idx, label=label)
                row["video"] = video_path.name
                row["sport"] = "leg_press"
                row.update(angles)
                all_rows.append(row)
                angle_history.append(angles)
            else:
                cv2.putText(frame, "No pose detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            draw_frame_info(frame, frame_idx, fps)

            if writer:
                writer.write(frame)

            if not NO_DISPLAY:
                cv2.imshow(f"Leg Press Analysis — {video_path.name}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"  Annotated video saved -> {out_path}")
    cv2.destroyAllWindows()

    if angle_history:
        summary = evaluate_leg_press_session(angle_history)
        print_leg_press_report(summary, video_path.name)

    return all_rows


def main():
    video_files = (
        sorted(CORRECT_DIR.glob("*.mp4")) + sorted(CORRECT_DIR.glob("*.MP4")) +
        sorted(CORRECT_DIR.glob("*.mov")) + sorted(CORRECT_DIR.glob("*.MOV"))
    )

    if not video_files:
        print(f"No videos found in {CORRECT_DIR}")
        sys.exit(1)

    print(f"\nFound {len(video_files)} video(s) to analyse.")

    all_rows = []
    for vf in video_files:
        rows = analyze_video(vf, label="correct")
        all_rows.extend(rows)

    if all_rows:
        df = rows_to_dataframe(all_rows)
        save_csv(df, str(OUTPUT_CSV))
        summarize_dataset(df)
    else:
        print("No pose data extracted from any video.")


if __name__ == "__main__":
    main()
