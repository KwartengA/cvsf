import cv2


# Phones (notably iPhones) often record video with a 90/180/270deg rotation
# stored as container metadata rather than physically rotating the pixel
# data -- most players apply it automatically, but OpenCV's VideoCapture
# does not, so frames come back sideways unless corrected here. Confirmed
# via cv2.CAP_PROP_ORIENTATION_META on real leg press footage
# (data/raw/gym/leg_presses/correct/IMG_68*.MOV, all 90deg).
_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class _RotationCorrectedCapture:
    """
    Thin wrapper around cv2.VideoCapture that applies the container's
    orientation metadata to every frame read, so callers get frames in the
    orientation a human would see when playing the file normally. Exposes
    the same .read()/.get()/.release() interface as cv2.VideoCapture, so it
    is a drop-in replacement wherever `cap = open_video(...)` is used.
    """

    def __init__(self, cap, rotate_code):
        self._cap = cap
        self._rotate_code = rotate_code

    def read(self):
        ret, frame = self._cap.read()
        if ret and self._rotate_code is not None:
            frame = cv2.rotate(frame, self._rotate_code)
        return ret, frame

    def get(self, prop_id):
        value = self._cap.get(prop_id)
        # width/height need swapping if we're rotating 90/270 -- the
        # underlying capture reports pre-rotation dimensions, but every
        # caller of video_info()/open_video() expects post-rotation ones
        # (annotated-video writers, angle calculations against frame.shape).
        if self._rotate_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        return value

    def release(self):
        self._cap.release()

    def isOpened(self):
        return self._cap.isOpened()


def open_video(path: str):
    """
    Open a video file and return a cv2.VideoCapture-compatible object.
    Raises if not found. Automatically corrects for rotation metadata
    (common in phone-recorded video) so frames/dimensions come back in
    the orientation a human would see when playing the file normally.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    rotate_code = _ROTATE_CODES.get(orientation)

    if rotate_code is None:
        return cap
    return _RotationCorrectedCapture(cap, rotate_code)


def video_info(path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = open_video(path)
    info = {
        "path": str(path),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def iter_frames(path: str):
    """
    Generator that yields (frame_index, frame_bgr) for every frame in the video.
    Releases the capture automatically when done.
    """
    cap = open_video(path)
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def save_video(frames, output_path: str, fps: float, width: int, height: int):
    """Write a list of BGR frames to an mp4 file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
