"""
analyze_tennis_shots.py
------------------------
Multi-shot-type tennis analysis script: processes serve, forehand,
backhand, and volley footage into one labeled dataset for the shot-type
classifier (src/analysis/tennis_shot_classifier.py).

Not a replacement for analyze_tennis_serve.py, which stays serve-only
rule-based feedback (already working, unrelated concern). This script is
about WHICH shot was played, not whether serve form was good.

Usage (from the project root with venv active):
    python scripts/analyze_tennis_shots.py               # full run, window + saves video + CSV
    python scripts/analyze_tennis_shots.py --no-save      # no window, no annotated video -- just CSV
    python scripts/analyze_tennis_shots.py --no-display   # no window, but still saves annotated
                                                           # video + CSV -- use this in a headless/
                                                           # non-interactive shell (no display server):
                                                           # cv2.imshow/waitKey misbehave there and
                                                           # can cause frames to be silently skipped.

What it does, per video:
  1. Per frame: detects the person (YOLOv8n, largest bounding box) and,
     separately, the racket (and opportunistically the ball) -- both COCO
     classes, no custom training needed.
  2. Runs MediaPipe pose on the full frame (tennis subjects are standing
     and typically fill most of the frame -- unlike leg press, no crop is
     needed here).
  3. Tracks racket position over time; after all frames are collected,
     detects hits (racket-speed peaks -- see tennis_rep_utils.py) and
     collapses each hit down to one feature row (angles + racket features
     at the peak-speed frame).
  4. Displays skeleton + racket/ball boxes + live info.
  5. Saves annotated video to data/processed/.
  6. Exports data/processed/tennis_shots_keypoints.csv (per-frame, with a
     shot_type column) -- the shot classifier itself does hit-detection and
     feature-row collapsing on this CSV (see generate_report.py), not this
     script, so the raw per-frame data stays available for inspection.
  7. Prints a per-video hit count so hit detection can be sanity-checked
     against what's plausible for the video's length before trusting a
     classifier trained on it.

Press  Q  while the video window is open to skip to the next video.
"""

import sys
import os
import cv2
from pathlib import Path

NO_SAVE = "--no-save" in sys.argv
NO_DISPLAY = "--no-display" in sys.argv or NO_SAVE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.video_utils import open_video, video_info
from src.pose_estimation.extractor import PoseExtractor
from src.pose_estimation.object_detector import (
    PersonDetector, ObjectDetector, load_yolo_model,
    RACKET_CLASS_ID, BALL_CLASS_ID,
)
from src.pose_estimation.visualizer import draw_pose, draw_frame_info
from src.utils.angle_calculator import tennis_serve_angles
from src.analysis.dataset_creator import rows_to_dataframe, save_csv
from src.analysis.tennis_rep_utils import detect_hits, summarize_hits

# ── Paths ────────────────────────────────────────────────────────────────────
TENNIS_ROOT = Path("data/raw/tennis")
SHOT_TYPES = ["serve", "forehand", "backhand", "volley"]
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PROCESSED_DIR / "tennis_shots_keypoints.csv"


def draw_racket_ball(frame, racket_bbox, ball_bbox):
    if racket_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in racket_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "racket", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 100, 0), 2, cv2.LINE_AA)
    if ball_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in ball_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "ball", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 0), 2, cv2.LINE_AA)


def analyze_video(video_path: Path, shot_type: str, yolo_model,
                   person_detector, racket_detector, ball_detector) -> list:
    info = video_info(str(video_path))
    fps    = info["fps"]
    width  = info["width"]
    height = info["height"]

    out_path = PROCESSED_DIR / f"annotated_{shot_type}_{video_path.stem}.mp4"
    if not NO_SAVE:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    else:
        writer = None

    print(f"\n{'─'*55}")
    print(f"  Video : {video_path.name}  [{shot_type}]")
    print(f"  Size  : {width}x{height}  FPS: {fps:.1f}  Frames: {info['frame_count']}")
    if not NO_DISPLAY:
        print(f"  Press Q in the window to skip to next video.")
    print(f"{'─'*55}")

    all_rows = []
    cap = open_video(str(video_path))

    with PoseExtractor() as extractor:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results, landmarks = extractor.extract_frame(frame, frame_idx=frame_idx, fps=fps)
            racket_bbox = racket_detector.detect(frame)
            ball_bbox = ball_detector.detect(frame)

            if landmarks:
                draw_pose(frame, results)
                angles = tennis_serve_angles(landmarks)

                row = extractor.landmarks_to_dict(landmarks, frame_idx=frame_idx, label=shot_type)
                row["video"] = video_path.name
                row["shot_type"] = shot_type
                row.update(angles)

                if racket_bbox is not None:
                    rcx, rcy = ObjectDetector.center(racket_bbox)
                    row["racket_cx"] = rcx / width
                    row["racket_cy"] = rcy / height
                else:
                    row["racket_cx"] = None
                    row["racket_cy"] = None

                if ball_bbox is not None:
                    bcx, bcy = ObjectDetector.center(ball_bbox)
                    row["ball_cx"] = bcx / width
                    row["ball_cy"] = bcy / height
                else:
                    row["ball_cx"] = None
                    row["ball_cy"] = None

                all_rows.append(row)
            else:
                cv2.putText(frame, "No pose detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            draw_racket_ball(frame, racket_bbox, ball_bbox)
            draw_frame_info(frame, frame_idx, fps)

            if writer:
                writer.write(frame)

            if not NO_DISPLAY:
                cv2.imshow(f"Tennis Shots — {shot_type} — {video_path.name}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"  Annotated video saved -> {out_path}")
    cv2.destroyAllWindows()

    return all_rows


def main():
    yolo_model = load_yolo_model()
    person_detector = PersonDetector(model=yolo_model)
    racket_detector = ObjectDetector(RACKET_CLASS_ID, model=yolo_model)
    ball_detector = ObjectDetector(BALL_CLASS_ID, model=yolo_model)

    all_rows = []
    videos_found = 0

    for shot_type in SHOT_TYPES:
        correct_dir = TENNIS_ROOT / shot_type / "correct"
        video_files = (
            sorted(correct_dir.glob("*.mp4")) + sorted(correct_dir.glob("*.MP4")) +
            sorted(correct_dir.glob("*.mov")) + sorted(correct_dir.glob("*.MOV"))
        )
        if not video_files:
            print(f"No videos found in {correct_dir} -- skipping {shot_type}.")
            continue

        print(f"\nFound {len(video_files)} {shot_type} video(s) to analyse.")
        videos_found += len(video_files)

        for vf in video_files:
            rows = analyze_video(
                vf, shot_type, yolo_model,
                person_detector, racket_detector, ball_detector,
            )
            all_rows.extend(rows)

    if not all_rows:
        print(f"\nNo pose data extracted from any video (checked {videos_found} videos).")
        sys.exit(1)

    df = rows_to_dataframe(all_rows)
    save_csv(df, str(OUTPUT_CSV))

    print(f"\nDataset summary")
    print(f"  Total frames : {len(df)}")
    print(f"  Shot type counts:")
    print(df["shot_type"].value_counts().to_string())
    print(f"  Videos       : {df['video'].nunique()}")

    hit_df = detect_hits(df)
    hit_summary = summarize_hits(hit_df)
    total_hits = hit_df.loc[hit_df["is_hit"], "hit_id"].nunique() if hit_df["is_hit"].any() else 0
    print(f"\n  Hits detected per video (sanity-check these look plausible "
          f"for each clip's length before trusting the classifier):")
    print(hit_summary.to_string(index=False))
    print(f"  Total hits across all videos: {total_hits}")


if __name__ == "__main__":
    main()
