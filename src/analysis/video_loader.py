from pathlib import Path
from tqdm import tqdm

from src.utils.video_utils import open_video, video_info, iter_frames
from src.pose_estimation.extractor import PoseExtractor
from src.utils.angle_calculator import squat_angles, tennis_serve_angles


def process_video(video_path: str, label: str = "correct", sport: str = "squat") -> list:
    """
    Run pose extraction on every frame of a video.
    Returns a list of dicts — one per frame that has a detected pose.
    Each dict contains flattened landmark coords, joint angles, metadata.
    """
    info = video_info(video_path)
    print(f"\nProcessing: {Path(video_path).name}")
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.1f}fps  |  "
          f"{info['frame_count']} frames  |  {info['duration_sec']:.1f}s")

    rows = []
    with PoseExtractor() as extractor:
        for frame_idx, frame in tqdm(iter_frames(video_path),
                                     total=info["frame_count"],
                                     desc="  Extracting"):
            _, landmarks = extractor.extract_frame(frame)
            if landmarks is None:
                continue

            row = extractor.landmarks_to_dict(landmarks, frame_idx=frame_idx, label=label)
            row["video"] = Path(video_path).name
            row["sport"] = sport

            if sport == "squat":
                angles = squat_angles(landmarks)
                row.update(angles)
            elif sport == "tennis_serve":
                angles = tennis_serve_angles(landmarks)
                row.update(angles)

            rows.append(row)

    print(f"  Detected pose in {len(rows)}/{info['frame_count']} frames")
    return rows


def process_video_folder(folder_path: str, label: str, sport: str = "squat") -> list:
    """Process all .mp4 files in a folder with the same label."""
    folder = Path(folder_path)
    all_rows = []
    video_files = list(folder.glob("*.mp4")) + list(folder.glob("*.MP4"))

    video_files += list(folder.glob("*.mov")) + list(folder.glob("*.MOV"))

    if not video_files:
        print(f"No video files found in {folder_path}")
        return all_rows

    for vf in sorted(video_files):
        rows = process_video(str(vf), label=label, sport=sport)
        all_rows.extend(rows)

    return all_rows
