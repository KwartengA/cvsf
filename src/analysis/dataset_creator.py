import pandas as pd
from pathlib import Path


def rows_to_dataframe(rows: list) -> pd.DataFrame:
    """Convert a list of landmark row dicts to a tidy DataFrame."""
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, output_path: str):
    """Save the DataFrame to CSV, creating parent directories as needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows -> {output_path}")


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load a previously saved keypoint CSV."""
    return pd.read_csv(csv_path)


def summarize_dataset(df: pd.DataFrame):
    """Print a quick summary of the dataset."""
    print(f"\nDataset summary")
    print(f"  Total frames : {len(df)}")
    if "label" in df.columns:
        print(f"  Label counts :")
        print(df["label"].value_counts().to_string(header=False))
    if "video" in df.columns:
        print(f"  Videos       : {df['video'].nunique()}")
    angle_cols = [c for c in df.columns if c.endswith("_angle")]
    if angle_cols:
        print(f"\n  Mean joint angles (degrees):")
        print(df[angle_cols].mean().round(1).to_string())


def build_tennis_serve_dataset(
    correct_folder: str,
    incorrect_folder: str = None,
    output_csv: str = "data/processed/tennis_serve_keypoints.csv",
):
    """
    High-level helper: process correct (and optionally incorrect) tennis serve
    folders, combine into one CSV dataset.
    """
    from src.analysis.video_loader import process_video_folder

    all_rows = process_video_folder(correct_folder, label="correct", sport="tennis_serve")

    if incorrect_folder and Path(incorrect_folder).exists():
        bad_rows = process_video_folder(incorrect_folder, label="incorrect", sport="tennis_serve")
        all_rows.extend(bad_rows)

    if not all_rows:
        print("No data extracted — check your video files.")
        return None

    df = rows_to_dataframe(all_rows)
    save_csv(df, output_csv)
    summarize_dataset(df)
    return df


def build_squat_dataset(correct_folder: str, incorrect_folder: str = None,
                        output_csv: str = "data/processed/squat_keypoints.csv"):
    """
    High-level helper: process correct (and optionally incorrect) squat folders,
    combine into one CSV dataset.
    """
    from src.analysis.video_loader import process_video_folder

    all_rows = process_video_folder(correct_folder, label="correct", sport="squat")

    if incorrect_folder and Path(incorrect_folder).exists():
        bad_rows = process_video_folder(incorrect_folder, label="incorrect", sport="squat")
        all_rows.extend(bad_rows)

    if not all_rows:
        print("No data extracted — check your video files.")
        return None

    df = rows_to_dataframe(all_rows)
    save_csv(df, output_csv)
    summarize_dataset(df)
    return df
