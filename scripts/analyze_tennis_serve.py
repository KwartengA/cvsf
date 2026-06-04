"""
analyze_tennis_serve.py
-----------------------
End-to-end tennis serve analysis script.

Usage (from the project root with venv active):
    python scripts/analyze_tennis_serve.py            # full run, saves video + CSV
    python scripts/analyze_tennis_serve.py --no-save  # preview only, nothing written to disk

Press  Q  while the video window is open to skip to the next video.
"""

import sys
import os
import cv2
from pathlib import Path

NO_SAVE = "--no-save" in sys.argv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.video_utils import open_video, video_info
from src.pose_estimation.extractor import PoseExtractor
from src.pose_estimation.visualizer import (
    draw_pose, draw_angle, draw_feedback_panel, draw_frame_info
)
from src.utils.angle_calculator import tennis_serve_angles
from src.feedback.feedback_generator import (
    evaluate_tennis_serve_frame, evaluate_tennis_serve_session,
    print_tennis_serve_report,
)
from src.analysis.dataset_creator import rows_to_dataframe, save_csv, summarize_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
CORRECT_DIR   = Path("data/raw/tennis/serve/corrrect")   # note triple-r in folder name
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PROCESSED_DIR / "tennis_serve_keypoints.csv"

# MediaPipe joint indices for the angles we want to show on-screen
SERVE_ANGLE_JOINTS = [
    ("right_shoulder_angle", 12, "RS:"),
    ("left_shoulder_angle",  11, "LS:"),
    ("right_elbow_angle",    14, "RE:"),
    ("left_elbow_angle",     13, "LE:"),
    ("right_hip_angle",      24, "RH:"),
    ("left_hip_angle",       23, "LH:"),
    ("right_knee_angle",     26, "RK:"),
    ("left_knee_angle",      25, "LK:"),
]


def draw_serve_angles(frame, landmarks, angles: dict):
    for key, joint_idx, label in SERVE_ANGLE_JOINTS:
        if key in angles and angles[key]:
            draw_angle(frame, landmarks, joint_idx, angles[key], label)


def analyze_video(video_path: Path, label: str = "correct") -> list:
    info   = video_info(str(video_path))
    fps    = info["fps"]
    width  = info["width"]
    height = info["height"]

    out_path = PROCESSED_DIR / f"annotated_{video_path.stem}.mp4"
    if not NO_SAVE:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    else:
        writer = None

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
                draw_pose(frame, results)

                angles = tennis_serve_angles(landmarks)
                draw_serve_angles(frame, landmarks, angles)

                feedback = evaluate_tennis_serve_frame(angles)
                draw_feedback_panel(frame, feedback)

                row = extractor.landmarks_to_dict(landmarks, frame_idx=frame_idx, label=label)
                row["video"] = video_path.name
                row["sport"] = "tennis_serve"
                row.update(angles)
                all_rows.append(row)
                angle_history.append(angles)
            else:
                cv2.putText(frame, "No pose detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            draw_frame_info(frame, frame_idx, fps)
            if writer:
                writer.write(frame)

            cv2.imshow(f"Tennis Serve — {video_path.name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"  Annotated video saved -> {out_path}")
    cv2.destroyAllWindows()

    if angle_history:
        summary = evaluate_tennis_serve_session(angle_history)
        print_tennis_serve_report(summary, video_path.name)

    return all_rows


def main():
    video_files = (
        sorted(CORRECT_DIR.glob("*.mp4")) +
        sorted(CORRECT_DIR.glob("*.MP4")) +
        sorted(CORRECT_DIR.glob("*.mov")) +
        sorted(CORRECT_DIR.glob("*.MOV"))
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
        if NO_SAVE:
            df = rows_to_dataframe(all_rows)
            summarize_dataset(df)
        else:
            df = rows_to_dataframe(all_rows)
            save_csv(df, str(OUTPUT_CSV))
            summarize_dataset(df)
    else:
        print("No pose data extracted from any video.")


if __name__ == "__main__":
    main()
