"""
analyze_pushup.py
------------------
End-to-end pushup analysis script.

Usage (from the project root with venv active):
    python scripts/analyze_pushup.py               # full run, window + saves video + CSV
    python scripts/analyze_pushup.py --no-save      # no window, no annotated video -- just CSV
    python scripts/analyze_pushup.py --no-display   # no window, but still saves annotated
                                                     # video + CSV -- use this in a headless/
                                                     # non-interactive shell (no display server):
                                                     # cv2.imshow/waitKey misbehave there and can
                                                     # cause frames to be silently skipped.

What it does:
  1. Processes all pushup videos in data/raw/gym/pushup/correct/
  2. Displays each video with skeleton overlay + live joint angles + feedback panel
  3. Saves annotated video to data/processed/
  4. Exports a keypoint CSV to data/processed/pushup_keypoints.csv
  5. Prints a per-video form report to the terminal

Press  Q  while the video window is open to skip to the next video.
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
from src.pose_estimation.visualizer import (
    draw_pose, draw_pushup_angles, draw_feedback_panel, draw_frame_info
)
from src.utils.angle_calculator import pushup_angles
from src.feedback.feedback_generator import (
    evaluate_pushup_frame, evaluate_pushup_session, print_pushup_report
)
from src.analysis.dataset_creator import rows_to_dataframe, save_csv, summarize_dataset

# ── Paths ────────────────────────────────────────────────────────────────────
CORRECT_DIR   = Path("data/raw/gym/pushup/correct")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PROCESSED_DIR / "pushup_keypoints.csv"


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

    with PoseExtractor() as extractor:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results, landmarks = extractor.extract_frame(frame, frame_idx=frame_idx, fps=fps)

            if landmarks:
                # Draw skeleton
                draw_pose(frame, results)

                # Calculate and draw joint angles
                angles = pushup_angles(landmarks)
                draw_pushup_angles(frame, landmarks, angles)

                # Per-frame feedback panel
                feedback = evaluate_pushup_frame(angles)
                draw_feedback_panel(frame, feedback)

                # Collect data for CSV
                row = extractor.landmarks_to_dict(landmarks, frame_idx=frame_idx, label=label)
                row["video"] = video_path.name
                row["sport"] = "pushup"
                row.update(angles)
                all_rows.append(row)
                angle_history.append(angles)
            else:
                # No pose detected — show a small notice
                cv2.putText(frame, "No pose detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            draw_frame_info(frame, frame_idx, fps)

            if writer:
                writer.write(frame)

            if not NO_DISPLAY:
                cv2.imshow(f"Pushup Analysis — {video_path.name}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"  Annotated video saved -> {out_path}")
    cv2.destroyAllWindows()

    # Session-level report
    if angle_history:
        summary = evaluate_pushup_session(angle_history)
        print_pushup_report(summary, video_path.name)

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
