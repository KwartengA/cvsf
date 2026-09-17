"""
pi_live_inference.py
---------------------
Live camera inference for squats/pushups, meant to run on the Raspberry Pi 5
(or locally against a video file for testing beforehand).

For each completed repetition, scores the bottom-of-rep frame with the
trained classifier (models/squat_form_classifier.joblib or
models/pushup_form_classifier.joblib) and POSTs a session result to the
CVSF backend API once the session ends (window closed / Q pressed / video
ends).

Usage:
    # local testing against a video file, with the display window
    python scripts/pi_live_inference.py --activity squat \\
        --source data/raw/gym/squat/correct/squat_correct_01.mp4

    # on the Pi, live camera, headless (no display), pointed at the server
    # NOTE: the CSI camera (e.g. Camera Module 3) needs the libcamerify
    # wrapper -- cv2.VideoCapture(0) opens fine without it but silently
    # reads 0 frames forever, so it MUST be run as:
    libcamerify python scripts/pi_live_inference.py --activity squat \\
        --source 0 --no-display \\
        --api-url http://192.168.1.50:3000/api/sessions \\
        --user-id <uuid> --stream-port 5001

Press Q to end the session early (ignored with --no-display; use Ctrl+C
instead, or pass --max-seconds to auto-stop).

KNOWN SIMPLIFICATION: the offline classifier modules
(squat_classifier.py / pushup_classifier.py) detect rep bottoms using
scipy.signal.find_peaks over an entire pre-recorded angle series -- that
can't work on a live stream since it needs the whole curve up front. This
script instead uses a simple streaming state machine (see
StreamingRepDetector below): a rep is "in progress" once the depth angle
drops MIN_DIP_DEG below a rolling standing baseline, the rep's bottom is
the minimum angle seen while in progress, and the rep completes once the
angle rises back near the baseline. This is intentionally simpler than the
offline peak-finding approach -- exact prominence/width tuning matters less
live since the user can watch the skeleton overlay to sanity-check rep
counts in person.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.video_utils import open_video, video_info
from src.pose_estimation.extractor import PoseExtractor
from src.pose_estimation.visualizer import (
    draw_pose, draw_squat_angles, draw_pushup_angles,
    draw_feedback_panel, draw_frame_info,
)
from src.utils.angle_calculator import squat_angles, pushup_angles
from src.feedback.feedback_generator import (
    evaluate_squat_frame, evaluate_pushup_frame, THRESHOLDS, PUSHUP_THRESHOLDS,
)
from src.streaming.mjpeg_server import MjpegServer

MODELS_DIR = Path("models")

# Website's cv-mapping.ts (FEATURE_TO_JOINT) expects feature_importances keys
# shaped "{joint}_angle_{side}" -- see deviation_importances() below, which
# scores each angle by how far it broke its threshold at the rep bottom.
ACTIVITY_CONFIG = {
    "squat": {
        "angles_fn": squat_angles,
        "draw_fn": draw_squat_angles,
        "feedback_fn": evaluate_squat_frame,
        "depth_keys": ("left_knee_angle", "right_knee_angle"),
        "feature_columns": [
            "left_knee_angle", "right_knee_angle",
            "left_hip_angle", "right_hip_angle",
            "left_ankle_angle", "right_ankle_angle",
            "knee_asymmetry",
        ],
        "asymmetry_keys": ("left_knee_angle", "right_knee_angle"),
        "asymmetry_feature": "knee_asymmetry",
        "model_path": MODELS_DIR / "squat_form_classifier.joblib",
        "standing_angle": 175,
        "min_dip_deg": 25,
        # (angle key, website feature key, deviation direction) for deviation_importances()
        "deviation_specs": [
            ("left_knee_angle", "knee_angle_left", "max", THRESHOLDS["knee_max"]),
            ("right_knee_angle", "knee_angle_right", "max", THRESHOLDS["knee_max"]),
            ("left_hip_angle", "hip_angle_left", "max", THRESHOLDS["hip_max"]),
            ("right_hip_angle", "hip_angle_right", "max", THRESHOLDS["hip_max"]),
            ("left_ankle_angle", "ankle_angle_left", "min", THRESHOLDS["ankle_min"]),
            ("right_ankle_angle", "ankle_angle_right", "min", THRESHOLDS["ankle_min"]),
        ],
    },
    "pushup": {
        "angles_fn": pushup_angles,
        "draw_fn": draw_pushup_angles,
        "feedback_fn": evaluate_pushup_frame,
        "depth_keys": ("left_elbow_angle", "right_elbow_angle"),
        "feature_columns": [
            "left_elbow_angle", "right_elbow_angle",
            "left_hip_angle", "right_hip_angle",
            "left_shoulder_angle", "right_shoulder_angle",
            "elbow_asymmetry",
        ],
        "asymmetry_keys": ("left_elbow_angle", "right_elbow_angle"),
        "asymmetry_feature": "elbow_asymmetry",
        "model_path": MODELS_DIR / "pushup_form_classifier.joblib",
        "standing_angle": 175,
        "min_dip_deg": 25,
        "deviation_specs": [
            ("left_elbow_angle", "elbow_angle_left", "max", PUSHUP_THRESHOLDS["elbow_max"]),
            ("right_elbow_angle", "elbow_angle_right", "max", PUSHUP_THRESHOLDS["elbow_max"]),
            ("left_hip_angle", "hip_angle_left", "min", PUSHUP_THRESHOLDS["hip_sag_min"]),
            ("right_hip_angle", "hip_angle_right", "min", PUSHUP_THRESHOLDS["hip_sag_min"]),
        ],
    },
}


def deviation_importances(deviation_specs, angles: dict) -> dict:
    """
    Score each configured angle by how far past its threshold it is at a
    rep's bottom (0 if within range). Keyed to match the website's
    FEATURE_TO_JOINT table (lib/cv-mapping.ts) so the joint map can
    highlight whichever joint actually broke form on that rep.
    """
    scores = {}
    for angle_key, feature_key, direction, threshold in deviation_specs:
        value = angles.get(angle_key)
        if value is None:
            continue
        if direction == "max":
            scores[feature_key] = max(0.0, value - threshold)
        else:
            scores[feature_key] = max(0.0, threshold - value)
    return scores


class StreamingRepDetector:
    """
    Minimal online rep-bottom detector. See module docstring for why this
    differs from the offline find_peaks-based approach used for training.
    """

    def __init__(self, standing_angle: float, min_dip_deg: float):
        self.standing_angle = standing_angle
        self.min_dip_deg = min_dip_deg
        self.in_rep = False
        self.rep_min_angle = None
        self.rep_bottom_angles = None  # full angles dict at the rep's bottom
        self.rep_count = 0

    def update(self, depth_angle: float, angles: dict):
        """
        Feed one frame's depth angle + full angle dict. Returns the
        completed rep's bottom-frame angles dict if a rep just completed
        this frame, else None.
        """
        dip = self.standing_angle - depth_angle

        if not self.in_rep and dip >= self.min_dip_deg:
            self.in_rep = True
            self.rep_min_angle = depth_angle
            self.rep_bottom_angles = angles

        elif self.in_rep:
            if depth_angle < self.rep_min_angle:
                self.rep_min_angle = depth_angle
                self.rep_bottom_angles = angles

            # rep completes once the angle rises back near standing
            if (self.standing_angle - depth_angle) < (self.min_dip_deg * 0.3):
                self.in_rep = False
                self.rep_count += 1
                completed = self.rep_bottom_angles
                self.rep_min_angle = None
                self.rep_bottom_angles = None
                return completed

        return None


def score_rep(model, feature_columns, asymmetry_keys, asymmetry_feature, angles: dict):
    """Run the trained classifier on one bottom-of-rep angles dict."""
    left_key, right_key = asymmetry_keys
    row = dict(angles)
    row[asymmetry_feature] = abs(angles[left_key] - angles[right_key])
    X = [[row[c] for c in feature_columns]]
    pred = model.predict(X)[0]
    return pred


def _send(api_url, method, payload):
    """POST/PATCH payload to api_url, returning the parsed JSON response or None on failure."""
    if not api_url:
        print(f"\n[no --api-url given, printing {method} instead of sending]")
        print(json.dumps(payload, indent=2))
        return None

    try:
        import urllib.request
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            print(f"\n{method}ed -> {api_url} (status {resp.status})")
            return body
    except Exception as e:
        print(f"\nFailed to {method} to {api_url}: {e}")
        print("Payload (not sent):")
        print(json.dumps(payload, indent=2))
        return None


def start_session(api_url, user_id, sport):
    """POST a new live session at the start of a run so reps can be PATCHed onto it."""
    payload = {"userId": user_id, "sport": sport, "status": "live"}
    result = _send(api_url, "POST", payload)
    return result["id"] if result else None


def patch_rep(api_url, session_id, rep_index, rep_score, form_score,
              errors_detected, rep_count, duration_sec, feature_importances):
    """PATCH the live session after a single rep completes."""
    if not session_id:
        return
    payload = {
        "formScore": form_score,
        "repCount": rep_count,
        "durationSec": round(duration_sec),
        "errorsDetected": errors_detected,
        "featureImportances": feature_importances,
        "newRep": {"repIndex": rep_index, "score": rep_score},
    }
    _send(f"{api_url}/{session_id}", "PATCH", payload)


def finish_session(api_url, session_id, form_score, reps, duration_sec):
    """PATCH the session to mark it complete once the run ends."""
    if not session_id:
        return
    payload = {
        "status": "complete",
        "formScore": form_score,
        "repCount": reps,
        "durationSec": round(duration_sec),
    }
    _send(f"{api_url}/{session_id}", "PATCH", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activity", choices=list(ACTIVITY_CONFIG.keys()), required=True)
    parser.add_argument("--source", default="0",
                         help="Camera index (e.g. 0) or path to a video file")
    parser.add_argument("--no-display", action="store_true",
                         help="Don't open a preview window (for headless Pi use)")
    parser.add_argument("--max-seconds", type=float, default=None,
                         help="Auto-stop the session after this many seconds")
    parser.add_argument("--max-reps", type=int, default=None,
                         help="Auto-stop the session after this many completed reps")
    parser.add_argument("--api-url", default=os.environ.get("CVSF_API_URL"),
                         help="POST endpoint, e.g. http://<server>:3000/api/sessions "
                              "(or set CVSF_API_URL env var). If omitted, prints the "
                              "result instead of sending it.")
    parser.add_argument("--user-id", default=os.environ.get("CVSF_USER_ID"),
                         help="User UUID to attribute the workout to (or CVSF_USER_ID env var)")
    parser.add_argument("--stream-port", type=int, default=None,
                         help="Serve the annotated feed as MJPEG at "
                              "http://<pi-ip>:<port>/stream.mjpg (e.g. 5001). "
                              "Omit to disable streaming.")
    args = parser.parse_args()

    config = ACTIVITY_CONFIG[args.activity]

    if not config["model_path"].exists():
        print(f"ERROR: {config['model_path']} not found.")
        print(f"Train it first (e.g. scripts/generate_report.py for squat).")
        sys.exit(1)

    import joblib
    model = joblib.load(config["model_path"])

    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open source {args.source}")
        sys.exit(1)

    # Lower capture resolution -- MediaPipe inference cost and MJPEG bandwidth
    # both scale with frame size, so a smaller capture keeps fps up and the
    # stream smooth over wifi rather than fighting a needlessly large frame.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture resolution: {actual_w}x{actual_h}")

    detector = StreamingRepDetector(config["standing_angle"], config["min_dip_deg"])
    left_key, right_key = config["depth_keys"]

    rep_results = []  # list of "correct"/"incorrect" per completed rep
    errors_seen = set()
    importance_totals = {}  # website feature key -> summed deviation across all reps

    session_id = start_session(args.api_url, args.user_id, args.activity)

    stream_server = None
    if args.stream_port:
        stream_server = MjpegServer(port=args.stream_port)
        stream_server.start()
        print(f"Streaming annotated feed at http://0.0.0.0:{args.stream_port}/stream.mjpg")

    print(f"\nStarting live {args.activity} session on source={args.source} "
          f"(fps~{fps:.1f}). Press Q to stop." if not args.no_display else
          f"\nStarting live {args.activity} session on source={args.source} (headless).")

    start_time = time.time()
    frame_idx = 0
    consecutive_read_failures = 0
    # CSI cameras via libcamerify take a few frames to start streaming after
    # VideoCapture opens, so tolerate a warm-up window of failed reads rather
    # than bailing on the very first one; a real end-of-stream (video file
    # EOF, camera unplugged) will keep failing past this threshold.
    MAX_CONSECUTIVE_READ_FAILURES = 60

    # Require a sustained run of detected pose landmarks before counting reps
    # -- MediaPipe occasionally reports a landmark set for an isolated frame
    # even when nobody is actually in view yet, and those isolated readings
    # are noisy enough to look like a rep to the dip/rise detector. Waiting
    # for a stable run filters that out and lets the person get set before
    # anything starts counting.
    POSE_WARMUP_FRAMES = 20
    stable_pose_frames = 0

    with PoseExtractor() as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_read_failures += 1
                if consecutive_read_failures > MAX_CONSECUTIVE_READ_FAILURES:
                    break
                time.sleep(0.05)
                continue
            consecutive_read_failures = 0

            elapsed = time.time() - start_time
            if args.max_seconds and elapsed >= args.max_seconds:
                break

            results, landmarks = extractor.extract_frame(frame, frame_idx=frame_idx, fps=fps)

            reached_max_reps = False

            if landmarks:
                stable_pose_frames += 1
                angles = config["angles_fn"](landmarks)
                depth_angle = (angles[left_key] + angles[right_key]) / 2

                if stable_pose_frames >= POSE_WARMUP_FRAMES:
                    completed_rep_angles = detector.update(depth_angle, angles)
                    if completed_rep_angles is not None:
                        pred = score_rep(
                            model, config["feature_columns"],
                            config["asymmetry_keys"], config["asymmetry_feature"],
                            completed_rep_angles,
                        )
                        rep_results.append(pred)
                        print(f"\n  Rep {detector.rep_count}: {pred}")

                        for feature_key, score in deviation_importances(
                            config["deviation_specs"], completed_rep_angles
                        ).items():
                            importance_totals[feature_key] = importance_totals.get(feature_key, 0.0) + score

                        rep_score = 100 if pred == "correct" else 40
                        running_form_score = round(
                            100 * sum(1 for r in rep_results if r == "correct") / len(rep_results)
                        )
                        patch_rep(
                            args.api_url, session_id,
                            rep_index=detector.rep_count - 1,
                            rep_score=rep_score,
                            form_score=running_form_score,
                            errors_detected=sorted(errors_seen),
                            rep_count=len(rep_results),
                            duration_sec=time.time() - start_time,
                            feature_importances=importance_totals,
                        )

                        if args.max_reps and detector.rep_count >= args.max_reps:
                            reached_max_reps = True

                need_overlay = not args.no_display or stream_server is not None
                if need_overlay:
                    draw_pose(frame, results)
                    config["draw_fn"](frame, landmarks, angles)
                    feedback = config["feedback_fn"](angles)
                    draw_feedback_panel(frame, feedback)
                    for msg, ok in feedback:
                        if not ok:
                            errors_seen.add(msg)
            else:
                stable_pose_frames = 0

            if not args.no_display or stream_server is not None:
                draw_frame_info(frame, frame_idx, fps)
                status = (
                    f"Reps: {detector.rep_count}"
                    if stable_pose_frames >= POSE_WARMUP_FRAMES
                    else "Reading pose..."
                )
                cv2.putText(frame, status, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if stream_server is not None:
                stream_server.update_frame(frame)

            if reached_max_reps:
                print(f"\nReached {args.max_reps} reps -- stopping session.")
                break

            if not args.no_display:
                cv2.imshow(f"CVSF Live — {args.activity}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            running_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            pose_status = "YES" if landmarks else "no "
            sys.stdout.write(
                f"\r  frame {frame_idx:>6}  |  {running_fps:5.1f} fps  |  "
                f"pose: {pose_status}  |  reps: {detector.rep_count}   "
            )
            sys.stdout.flush()

            frame_idx += 1

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    duration_sec = time.time() - start_time
    n_reps = len(rep_results)
    n_correct = sum(1 for r in rep_results if r == "correct")
    form_score = round(100 * n_correct / n_reps) if n_reps else None

    print(f"\nSession complete: {n_reps} reps, form_score={form_score}, "
          f"duration={duration_sec:.1f}s")

    finish_session(
        api_url=args.api_url,
        session_id=session_id,
        form_score=form_score,
        reps=n_reps,
        duration_sec=duration_sec,
    )


if __name__ == "__main__":
    main()
