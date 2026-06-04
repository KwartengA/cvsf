"""
analyze_squat.py
----------------
End-to-end squat analysis script.

Usage (from the project root with venv active):
    python scripts/analyze_squat.py

What it does:
  1. Processes both squat videos in data/raw/gym/squat/correct/
  2. Displays each video with skeleton overlay + live joint angles + feedback panel
  3. Saves annotated video to data/processed/
  4. Exports a keypoint CSV to data/processed/squat_keypoints.csv
  5. Prints a per-video form report to the terminal

Press  Q  while the video window is open to skip to the next video.
"""

import sys
import os
import cv2
from pathlib import Path

# Allow imports from the project root regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.video_utils import open_video, video_info
from src.pose_estimation.extractor import PoseExtractor
from src.pose_estimation.visualizer import (
    draw_pose, draw_squat_angles, draw_feedback_panel, draw_frame_info
)
from src.utils.angle_calculator import squat_angles
from src.feedback.feedback_generator import (
    evaluate_squat_frame, evaluate_squat_session, print_session_report
)
from src.analysis.dataset_creator import rows_to_dataframe, save_csv, summarize_dataset

# ── Paths ────────────────────────────────────────────────────────────────────
CORRECT_DIR   = Path("data/raw/gym/squat/correct")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PROCESSED_DIR / "squat_keypoints.csv"


def analyze_video(video_path: Path, label: str = "correct") -> list:
    info = video_info(str(video_path))
    fps    = info["fps"]
    width  = info["width"]
    height = info["height"]

    out_path = PROCESSED_DIR / f"annotated_{video_path.name}"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"\n{'─'*55}")
    print(f"  Video : {video_path.name}")
    print(f"  Size  : {width}x{height}  FPS: {fps:.1f}  Frames: {info['frame_count']}")
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
                angles = squat_angles(landmarks)
                draw_squat_angles(frame, landmarks, angles)

                # Per-frame feedback panel
                feedback = evaluate_squat_frame(angles)
                draw_feedback_panel(frame, feedback)

                # Collect data for CSV
                row = extractor.landmarks_to_dict(landmarks, frame_idx=frame_idx, label=label)
                row["video"] = video_path.name
                row["sport"] = "squat"
                row.update(angles)
                all_rows.append(row)
                angle_history.append(angles)
            else:
                # No pose detected — show a small notice
                cv2.putText(frame, "No pose detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            draw_frame_info(frame, frame_idx, fps)

            # Write to output video
            writer.write(frame)

            # Show live
            cv2.imshow(f"Squat Analysis — {video_path.name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"  Annotated video saved -> {out_path}")

    # Session-level report
    if angle_history:
        summary = evaluate_squat_session(angle_history)
        print_session_report(summary, video_path.name)

    return all_rows


def main():
    video_files = sorted(CORRECT_DIR.glob("*.mp4")) + sorted(CORRECT_DIR.glob("*.MP4"))

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
