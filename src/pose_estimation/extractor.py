import cv2
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Model file — downloaded automatically on first run
MODEL_PATH = Path("models/pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

# MediaPipe landmark index -> human-readable name (33 landmarks)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def _ensure_model():
    """Download the pose landmarker model if not already present."""
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model (~29 MB) -> {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


class PoseExtractor:
    """
    Wraps the MediaPipe Tasks PoseLandmarker for frame-by-frame extraction.

    Each detected landmark has:
        .x, .y  — normalized [0,1] relative to frame width/height
        .z      — depth (negative = closer to camera)
        .visibility, .presence — confidence scores
    """

    def __init__(self):
        _ensure_model()
        base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._frame_ms = 0

    def extract_frame(self, frame_bgr, frame_idx: int = 0, fps: float = 30.0):
        """
        Process a single BGR frame.
        Returns (result, landmarks) where landmarks is a list of 33 landmark
        objects (NormalizedLandmark), or None if no pose detected.
        frame_idx and fps are used to compute the video timestamp.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((frame_idx / max(fps, 1)) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result, result.pose_landmarks[0]   # first detected person
        return result, None

    def landmarks_to_dict(self, landmarks, frame_idx=None, label=None):
        """
        Flatten 33 landmarks into a flat dict ready for a CSV row.
        Keys: nose_x, nose_y, nose_z, nose_vis, left_eye_inner_x, ...
        """
        row = {}
        if frame_idx is not None:
            row["frame"] = frame_idx
        if label is not None:
            row["label"] = label

        for i, name in enumerate(LANDMARK_NAMES):
            lm = landmarks[i]
            row[f"{name}_x"]   = round(lm.x, 6)
            row[f"{name}_y"]   = round(lm.y, 6)
            row[f"{name}_z"]   = round(lm.z, 6)
            row[f"{name}_vis"] = round(getattr(lm, "visibility", 0.0), 4)

        return row

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
