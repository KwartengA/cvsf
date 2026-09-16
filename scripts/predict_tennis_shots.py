"""
predict_tennis_shots.py
------------------------
Runs the trained tennis shot-type classifier (models/tennis_shot_classifier.joblib)
against ONE unlabeled video and prints which shot type (serve / forehand /
backhand / volley) was played at each detected hit.

Unlike analyze_tennis_shots.py -- which processes data/raw/tennis/<shot_type>/
correct/ folders where the folder name IS the label, for building the
training dataset -- this script takes an arbitrary video with an unknown
shot type and predicts it, using the same pose/racket-tracking/hit-detection
pipeline (src/analysis/tennis_rep_utils.py) and feature set
(src/analysis/tennis_shot_classifier.py) the model was trained on.

Usage (from the project root with venv active):
    python scripts/predict_tennis_shots.py --source path/to/clip.mp4
    python scripts/predict_tennis_shots.py --source path/to/clip.mp4 --no-display
"""

import argparse
import sys
from pathlib import Path

import cv2
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.video_utils import open_video, video_info
from src.pose_estimation.extractor import PoseExtractor
from src.pose_estimation.object_detector import (
    PersonDetector, ObjectDetector, load_yolo_model,
    RACKET_CLASS_ID, BALL_CLASS_ID,
)
from src.pose_estimation.visualizer import draw_pose, draw_frame_info
from src.utils.angle_calculator import tennis_serve_angles
from src.analysis.dataset_creator import rows_to_dataframe
from src.analysis.tennis_rep_utils import detect_hits
from src.analysis.tennis_shot_classifier import build_hit_feature_rows, FEATURE_COLUMNS

MODEL_PATH = Path("models/tennis_shot_classifier.joblib")


def draw_racket_ball(frame, racket_bbox, ball_bbox):
    if racket_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in racket_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "racket", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if ball_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in ball_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)


def extract_rows(video_path: Path, no_display: bool) -> list:
    yolo_model = load_yolo_model()
    racket_detector = ObjectDetector(RACKET_CLASS_ID, model=yolo_model)
    ball_detector = ObjectDetector(BALL_CLASS_ID, model=yolo_model)

    info = video_info(str(video_path))
    fps, width, height = info["fps"], info["width"], info["height"]

    print(f"\n{'─'*55}")
    print(f"  Video : {video_path.name}")
    print(f"  Size  : {width}x{height}  FPS: {fps:.1f}  Frames: {info['frame_count']}")
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

                row = extractor.landmarks_to_dict(landmarks, frame_idx=frame_idx)
                row["video"] = video_path.name
                row["frame"] = frame_idx
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

            if not no_display:
                cv2.imshow(f"Predict Tennis Shot — {video_path.name}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    cap.release()
    if not no_display:
        cv2.destroyAllWindows()

    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Path to a tennis video file")
    parser.add_argument("--no-display", action="store_true",
                         help="Don't open a preview window (for headless use)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        print("Train it first via scripts/analyze_tennis_shots.py + scripts/generate_report.py.")
        sys.exit(1)

    video_path = Path(args.source)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    rows = extract_rows(video_path, args.no_display)
    if not rows:
        print("\nNo pose data extracted -- can't predict shot type.")
        sys.exit(1)

    df = rows_to_dataframe(rows)
    hit_df = detect_hits(df)

    if not hit_df["is_hit"].any():
        print("\nNo hits detected in this video -- can't predict shot type.")
        sys.exit(1)

    feature_rows = build_hit_feature_rows(hit_df)
    if feature_rows.empty:
        print("\nNo usable hit windows -- can't predict shot type.")
        sys.exit(1)

    X = feature_rows[FEATURE_COLUMNS].to_numpy()
    predictions = model.predict(X)

    print(f"\n{len(predictions)} hit(s) detected in {video_path.name}:")
    for i, (hit_id, pred) in enumerate(zip(feature_rows["hit_id"], predictions), start=1):
        frame_num = feature_rows.iloc[i - 1]["frame"]
        print(f"  Hit {i} (frame {frame_num}): {pred}")


if __name__ == "__main__":
    main()
