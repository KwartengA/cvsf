import cv2



def open_video(path: str):
    """Open a video file and return the cv2.VideoCapture object. Raises if not found."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


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
