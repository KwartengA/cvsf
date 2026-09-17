"""
pi_live_tennis.py
-------------------
Live camera tennis shot detection for the Raspberry Pi 5 -- the real-time
analog of predict_tennis_shots.py, mirroring how pi_live_inference.py is the
real-time analog of the offline squat/pushup analyzers.

For each detected swing, scores it with the trained classifier
(models/tennis_shot_classifier.joblib) and PATCHes the hit (shot type) onto
the CVSF backend session as it happens, and optionally streams the
annotated feed as MJPEG for the web dashboard's live camera panel.

Usage:
    # local testing against a video file, with the display window
    python scripts/pi_live_tennis.py --source path/to/clip.mp4

    # on the Pi, live camera, headless, pointed at the server
    # NOTE: the CSI camera (e.g. Camera Module 3) needs the libcamerify
    # wrapper -- cv2.VideoCapture(0) opens fine without it but silently
    # reads 0 frames forever (see pi_live_inference.py for the same gotcha).
    libcamerify python scripts/pi_live_tennis.py --source 0 --no-display \\
        --stream-port 5002 --api-url http://192.168.1.50:3000/api/sessions \\
        --user-id <uuid>

Press Q to end the session early (ignored with --no-display; use Ctrl+C
instead, or pass --max-seconds to auto-stop).

CAVEAT -- UNVALIDATED AGAINST REAL SWINGS: tennis_rep_utils.detect_hits()
(used by predict_tennis_shots.py) finds swings by looking for racket-speed
peaks that exceed MIN_HIT_PROMINENCE_STD standard deviations above THAT
WHOLE CLIP's own mean speed -- not available yet on a live feed with no end.
StreamingHitDetector below re-derives the same relative threshold from a
rolling window of recent speed samples instead, the same kind of adaptation
pi_live_inference.py's StreamingRepDetector already made for squat/pushup
rep-bottom detection. Unlike that one, this hasn't been checked against a
real racket swing (built without one on hand) -- only smoke-tested that it
runs without crashing. Expect to retune --hit-sensitivity and
ROLLING_WINDOW_FRAMES against the first real swings tomorrow; if real hits
aren't registering, lower --hit-sensitivity (e.g. 1.5) rather than editing
the threshold logic under time pressure.

PERFORMANCE NOTE: unlike predict_tennis_shots.py (offline, fps doesn't
matter), this runs YOLO racket detection *and* MediaPipe pose *every frame*
live. Ball detection is dropped entirely here (predict_tennis_shots.py's own
comment notes ball detection is unreliable and only ever used as a visual
cross-check, never required for classification) to save a second YOLO pass
per frame -- racket detection alone is the real signal. Still expect lower
fps than the squat/pushup script; test this live before the actual demo.
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pose_estimation.extractor import PoseExtractor
from src.pose_estimation.object_detector import ObjectDetector, load_yolo_model, RACKET_CLASS_ID
from src.pose_estimation.visualizer import draw_pose, draw_frame_info
from src.utils.angle_calculator import tennis_serve_angles
from src.analysis.tennis_shot_classifier import FEATURE_COLUMNS
from src.streaming.mjpeg_server import MjpegServer

MODEL_PATH = Path("models/tennis_shot_classifier.joblib")

# Same relative-threshold reasoning as tennis_rep_utils.MIN_HIT_PROMINENCE_STD
# (see that module's docstring for why an absolute px/frame cutoff doesn't
# generalize across camera distance/zoom) -- reapplied here over a rolling
# window instead of the whole clip, since a live feed has no "whole clip".
DEFAULT_MIN_HIT_PROMINENCE_STD = 3.0
ROLLING_WINDOW_FRAMES = 60    # ~2s at 30fps of recent nonzero speed samples for the baseline
MIN_HIT_WIDTH_FRAMES = 2      # frames speed must stay elevated to count as a swing, not a glitch
HIT_COOLDOWN_FRAMES = 10      # ignore new swings this soon after the last one (follow-through)


class StreamingHitDetector:
    """
    Live re-implementation of tennis_rep_utils.detect_hits() -- see module
    docstring for why it can't reuse that function directly. Tracks a
    rolling baseline of recent racket speed and flags a swing once speed
    rises `min_hit_std` standard deviations above it, capturing the
    peak-speed frame's pose/racket snapshot as the hit's feature row, then
    closes the hit once speed drops back down.
    """

    def __init__(self, min_hit_std: float = DEFAULT_MIN_HIT_PROMINENCE_STD):
        self.min_hit_std = min_hit_std
        self.recent_speeds = deque(maxlen=ROLLING_WINDOW_FRAMES)
        self.in_swing = False
        self.frames_above = 0
        self.frames_since_hit = HIT_COOLDOWN_FRAMES
        self.peak_speed = 0.0
        self.peak_snapshot = None
        self.hit_count = 0

    def _threshold(self):
        if len(self.recent_speeds) < 5:
            return None
        std = float(np.std(self.recent_speeds))
        return max(std * self.min_hit_std, 1e-4)

    def update(self, speed: float, snapshot: dict | None):
        """
        Feed one frame's racket speed (0.0 if racket not detected this
        frame) plus that frame's feature snapshot (None if pose wasn't
        detected this frame). Returns the completed hit's snapshot dict once
        a swing finishes, else None.
        """
        self.frames_since_hit += 1
        threshold = self._threshold()
        if speed > 0:
            self.recent_speeds.append(speed)

        if threshold is None:
            return None

        if not self.in_swing:
            if speed > threshold and self.frames_since_hit > HIT_COOLDOWN_FRAMES:
                self.in_swing = True
                self.frames_above = 1
                self.peak_speed = speed
                self.peak_snapshot = snapshot
            return None

        # already tracking a swing
        if speed > self.peak_speed:
            self.peak_speed = speed
            if snapshot is not None:
                self.peak_snapshot = snapshot
        if speed > threshold:
            self.frames_above += 1
            return None

        # speed dropped back under threshold -- swing is ending
        self.in_swing = False
        too_short = self.frames_above < MIN_HIT_WIDTH_FRAMES
        completed = None if too_short or self.peak_snapshot is None else self.peak_snapshot
        self.peak_snapshot = None
        self.peak_speed = 0.0
        if completed is not None:
            self.hit_count += 1
            self.frames_since_hit = 0
        return completed


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


def start_session(api_url, user_id):
    payload = {"userId": user_id, "sport": "tennis_shot", "status": "live"}
    result = _send(api_url, "POST", payload)
    return result["id"] if result else None


def patch_hit(api_url, session_id, hit_index, shot_type, hit_count):
    if not session_id:
        return
    payload = {
        "repCount": hit_count,
        "newRep": {"repIndex": hit_index, "score": 0, "shotType": shot_type},
    }
    _send(f"{api_url}/{session_id}", "PATCH", payload)


def finish_session(api_url, session_id, hit_count):
    if not session_id:
        return
    payload = {"status": "complete", "repCount": hit_count}
    _send(f"{api_url}/{session_id}", "PATCH", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or path to a video file")
    parser.add_argument("--no-display", action="store_true", help="Don't open a preview window (for headless Pi use)")
    parser.add_argument("--max-seconds", type=float, default=None, help="Auto-stop the session after this many seconds")
    parser.add_argument("--api-url", default=os.environ.get("CVSF_API_URL"),
                         help="POST endpoint, e.g. http://<server>:3000/api/sessions (or CVSF_API_URL env var)")
    parser.add_argument("--user-id", default=os.environ.get("CVSF_USER_ID"),
                         help="Tester id to attribute the session to (or CVSF_USER_ID env var)")
    parser.add_argument("--stream-port", type=int, default=None,
                         help="Serve the annotated feed as MJPEG at http://<pi-ip>:<port>/stream.mjpg")
    parser.add_argument("--hit-sensitivity", type=float, default=DEFAULT_MIN_HIT_PROMINENCE_STD,
                         help=f"Standard deviations above the rolling speed baseline a swing must "
                              f"reach to count as a hit (default {DEFAULT_MIN_HIT_PROMINENCE_STD}). "
                              f"Lower this if real swings aren't being detected.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        print("Train it first via scripts/analyze_tennis_shots.py + scripts/generate_report.py.")
        sys.exit(1)

    import joblib
    model = joblib.load(MODEL_PATH)

    yolo_model = load_yolo_model()
    racket_detector = ObjectDetector(RACKET_CLASS_ID, model=yolo_model)

    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open source {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0

    detector = StreamingHitDetector(min_hit_std=args.hit_sensitivity)
    last_racket = None  # (cx_norm, cy_norm, frame_idx) of the last detected racket position
    last_shot_label = ""

    session_id = start_session(args.api_url, args.user_id)

    stream_server = None
    if args.stream_port:
        stream_server = MjpegServer(port=args.stream_port)
        stream_server.start()
        print(f"Streaming annotated feed at http://0.0.0.0:{args.stream_port}/stream.mjpg")

    print(f"\nStarting live tennis session on source={args.source} "
          f"(fps~{fps:.1f}, hit-sensitivity={args.hit_sensitivity}). Press Q to stop."
          if not args.no_display else
          f"\nStarting live tennis session on source={args.source} (headless).")

    start_time = time.time()
    frame_idx = 0
    consecutive_read_failures = 0
    # Same CSI-camera warm-up tolerance as pi_live_inference.py.
    MAX_CONSECUTIVE_READ_FAILURES = 60

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
            racket_bbox = racket_detector.detect(frame)

            speed = 0.0
            racket_cx_norm = None
            if racket_bbox is not None:
                rcx, rcy = ObjectDetector.center(racket_bbox)
                racket_cx_norm = rcx / width
                racket_cy_norm = rcy / height
                if last_racket is not None:
                    last_cx, last_cy, last_frame = last_racket
                    dframe = frame_idx - last_frame
                    if dframe > 0:
                        speed = ((racket_cx_norm - last_cx) ** 2 + (racket_cy_norm - last_cy) ** 2) ** 0.5 / dframe
                last_racket = (racket_cx_norm, racket_cy_norm, frame_idx)

            snapshot = None
            if landmarks is not None and racket_cx_norm is not None:
                angles = tennis_serve_angles(landmarks)
                torso_center_x = (landmarks[23].x + landmarks[24].x) / 2
                snapshot = dict(angles)
                snapshot["racket_speed"] = speed
                snapshot["racket_x_relative_to_body"] = racket_cx_norm - torso_center_x

            completed_hit = detector.update(speed, snapshot)
            if completed_hit is not None:
                X = [[completed_hit[c] for c in FEATURE_COLUMNS]]
                pred = str(model.predict(X)[0])
                last_shot_label = pred
                print(f"\n  Hit {detector.hit_count}: {pred}")
                patch_hit(
                    args.api_url, session_id,
                    hit_index=detector.hit_count - 1,
                    shot_type=pred,
                    hit_count=detector.hit_count,
                )

            need_overlay = not args.no_display or stream_server is not None
            if need_overlay:
                if landmarks is not None:
                    draw_pose(frame, results)
                if racket_bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in racket_bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                draw_frame_info(frame, frame_idx, fps)
                cv2.putText(frame, f"Hits: {detector.hit_count}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                if last_shot_label:
                    cv2.putText(frame, f"Last: {last_shot_label}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            if stream_server is not None:
                stream_server.update_frame(frame)

            if not args.no_display:
                cv2.imshow("CVSF Live — tennis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            running_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            pose_status = "YES" if landmarks else "no "
            racket_status = "YES" if racket_bbox is not None else "no "
            sys.stdout.write(
                f"\r  frame {frame_idx:>6}  |  {running_fps:5.1f} fps  |  "
                f"pose: {pose_status}  |  racket: {racket_status}  |  hits: {detector.hit_count}   "
            )
            sys.stdout.flush()

            frame_idx += 1

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    duration_sec = time.time() - start_time
    print(f"\nSession complete: {detector.hit_count} hits, duration={duration_sec:.1f}s")

    finish_session(args.api_url, session_id, hit_count=detector.hit_count)


if __name__ == "__main__":
    main()
