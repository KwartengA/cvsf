"""
Person/object detection (for picking the right subject out of a
multi-person frame, and for tracking task-relevant equipment) and classical
object tracking for equipment with no COCO class of its own:
  - other people may be in frame (gym background, tennis court neighbors),
    but only one subject matters
  - for leg press: the moving sled/footplate is a clean, independent depth
    signal alongside joint-angle-based pose analysis (no COCO class fits,
    so a classical tracker is used -- see SledTracker)
  - for tennis: a racket and (opportunistically) a ball ARE COCO classes,
    so ObjectDetector below reuses the same YOLO model with no custom
    training needed

PersonDetector/ObjectDetector both wrap a COCO-pretrained YOLOv8n model
(~6MB) -- "person" (0), "sports ball" (32), and "tennis racket" (38) are
all native COCO classes. SledTracker wraps OpenCV's classical CSRT tracker
for gym equipment, which has no COCO class; it needs a starting bounding
box (e.g. marked on the first frame) rather than auto-detecting the sled
itself.
"""

from pathlib import Path

import cv2

MODEL_PATH = Path("models/yolov8n.pt")
PERSON_CLASS_ID = 0   # COCO class index for "person"
BALL_CLASS_ID   = 32  # COCO class index for "sports ball"
RACKET_CLASS_ID = 38  # COCO class index for "tennis racket"


def _ensure_model():
    """Download the YOLOv8n weights if not already present (ultralytics
    handles the actual download; this just ensures the target directory
    exists and passes the right destination path)."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_yolo_model():
    """
    Load (downloading first if needed) the shared YOLOv8n model. Pass the
    result to PersonDetector/ObjectDetector's `model=` param to reuse one
    loaded model across several detectors instead of loading yolov8n.pt
    repeatedly per frame.
    """
    _ensure_model()
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


class PersonDetector:
    """
    Finds the main person in a frame when multiple people may be present
    (e.g. gym background, or a wall mural/poster of a person -- COCO-trained
    YOLO happily detects painted figures too).

    "Main person" is picked by a scoring function over (confidence, aspect
    ratio) rather than raw area alone: on a real leg-press video, a
    background mural of a standing figure can score a *larger* bounding box
    area, at similar confidence, than the actual subject -- area alone
    flips between them almost randomly frame to frame (observed directly:
    0.60 conf/real-person vs 0.59 conf/mural, similar areas, in
    data/raw/gym/leg_presses/correct/IMG_6856.MOV around frame 800). See
    `score_landscape_subject` for the leg-press-specific fix (the subject
    lies flat, so their box is wide/short -- the opposite shape of a
    standing mural or background gym-goer, which reads tall/narrow).
    """

    def __init__(self, confidence: float = 0.5, score_fn=None, model=None):
        """
        score_fn(confidence, x1, y1, x2, y2) -> float, higher = more likely
        the real subject. Defaults to plain confidence*area (the original
        behavior) if not given -- pass `score_landscape_subject` for
        activities where the subject lies flat (e.g. leg press).

        `model`: an already-loaded ultralytics YOLO instance to reuse
        (e.g. shared with an ObjectDetector for racket/ball) instead of
        loading yolov8n.pt again. Pass `load_yolo_model()`'s result in to
        share one model across several detectors on the same video.
        """
        self._model = model if model is not None else load_yolo_model()
        self._confidence = confidence
        self._score_fn = score_fn or _score_by_area

    def detect_main_person(self, frame_bgr):
        """
        Returns (x1, y1, x2, y2) in pixel coordinates for the
        highest-scoring detected person (see `score_fn`), or None if no
        person is detected.
        """
        results = self._model(
            frame_bgr, classes=[PERSON_CLASS_ID],
            conf=self._confidence, verbose=False,
        )
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None

        best_box = None
        best_score = -1.0
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            score = self._score_fn(conf, x1, y1, x2, y2)
            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)
        return best_box


class ObjectDetector:
    """
    Detects a single COCO object class (e.g. tennis racket, sports ball)
    and returns the highest-confidence box, or None. Deliberately much
    simpler than PersonDetector -- there's normally only one racket/ball in
    frame, so "highest confidence" (not an aspect-ratio heuristic) is
    sufficient; no evidence of a mural/poster-style false-positive problem
    for these classes the way there was for PersonDetector on leg press.

    Confidence threshold defaults lower than PersonDetector's (0.5): rackets
    and balls are smaller, faster-moving, and more prone to motion blur, so
    YOLO reports lower confidence on genuine detections. Verified directly
    against real tennis footage (data/raw/tennis/{forehand,backhand,volley}/
    correct/): a 0.5 threshold missed rackets that were clearly, correctly
    detected at 0.15-0.25 confidence.
    """

    def __init__(self, class_id: int, confidence: float = 0.2, model=None):
        self._model = model if model is not None else load_yolo_model()
        self._class_id = class_id
        self._confidence = confidence

    def detect(self, frame_bgr):
        """Returns (x1, y1, x2, y2) in pixel coordinates for the
        highest-confidence detection of this class, or None."""
        results = self._model(
            frame_bgr, classes=[self._class_id],
            conf=self._confidence, verbose=False,
        )
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None

        best_box = None
        best_conf = -1.0
        for box in boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = tuple(box.xyxy[0].tolist())
        return best_box

    @staticmethod
    def center(bbox):
        """(x1, y1, x2, y2) -> (cx, cy) center point."""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


def _score_by_area(confidence, x1, y1, x2, y2):
    """Default scoring: confidence * area (the original 'largest box' rule)."""
    return confidence * (x2 - x1) * (y2 - y1)


def score_landscape_subject(confidence, x1, y1, x2, y2):
    """
    Scoring for activities where the real subject lies flat (e.g. leg
    press): favors wide/short boxes (width > height) over tall/narrow ones,
    on top of confidence and area. A standing figure -- a real background
    person, or a painted mural -- reads tall/narrow and gets penalized here
    even if it has comparable confidence/area to the real subject.
    """
    w, h = x2 - x1, y2 - y1
    aspect_ratio = w / max(h, 1e-6)
    return confidence * w * h * aspect_ratio


def crop_with_margin(frame_bgr, bbox, margin_ratio: float = 0.15):
    """
    Crop `frame_bgr` to `bbox` (x1, y1, x2, y2) plus a margin on each side
    (helps MediaPipe pose detection, which expects some context around the
    person rather than a tight crop). Returns (cropped_frame, crop_origin)
    where crop_origin = (x1, y1) of the crop in the original frame, needed
    to remap landmark coordinates back to full-frame space.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin_ratio, bh * margin_ratio

    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(w, int(x2 + mx))
    cy2 = min(h, int(y2 + my))

    return frame_bgr[cy1:cy2, cx1:cx2], (cx1, cy1)


def remap_landmark_to_full_frame(landmark, crop_origin, crop_shape, full_frame_shape):
    """
    Convert a MediaPipe landmark's normalized (x, y) -- normalized to the
    *crop* -- into coordinates normalized to the *full frame*. Returns
    (x_norm, y_norm) in [0, 1] full-frame space. z/visibility are unaffected
    (z is relative depth, not spatial position; visibility is a confidence
    score) and should be copied through unchanged by the caller.
    """
    crop_h, crop_w = crop_shape[:2]
    full_h, full_w = full_frame_shape[:2]
    cx1, cy1 = crop_origin

    px = landmark.x * crop_w + cx1
    py = landmark.y * crop_h + cy1

    return px / full_w, py / full_h


class SledTracker:
    """
    Tracks the leg press sled/footplate position over time using OpenCV's
    CSRT tracker. Needs an initial bounding box (e.g. marked on the first
    frame) -- this class does not auto-detect the sled's starting position.
    """

    def __init__(self):
        self._tracker = None

    def init(self, frame_bgr, bbox):
        """bbox: (x, y, w, h) in pixel coordinates (OpenCV tracker convention,
        width/height rather than x2/y2)."""
        self._tracker = cv2.TrackerCSRT_create()
        self._tracker.init(frame_bgr, bbox)

    def update(self, frame_bgr):
        """Returns (x, y, w, h) for the tracked sled in this frame, or None
        if tracking was lost."""
        if self._tracker is None:
            raise RuntimeError("SledTracker.init() must be called before update()")
        ok, bbox = self._tracker.update(frame_bgr)
        if not ok:
            return None
        return bbox

    @staticmethod
    def vertical_center(bbox):
        """Return the vertical (y) center of a (x, y, w, h) bbox -- the
        depth signal for rep detection (sled moves along one axis)."""
        x, y, w, h = bbox
        return y + h / 2
