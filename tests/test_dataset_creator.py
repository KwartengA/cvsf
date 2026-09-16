import pandas as pd

from src.analysis.dataset_creator import (
    rows_to_dataframe, save_csv, load_csv, summarize_dataset,
)


def sample_rows():
    return [
        {"frame": 0, "label": "correct", "video": "a.mp4", "left_knee_angle": 90.0},
        {"frame": 1, "label": "correct", "video": "a.mp4", "left_knee_angle": 95.0},
        {"frame": 0, "label": "incorrect", "video": "b.mp4", "left_knee_angle": 150.0},
    ]


def test_rows_to_dataframe():
    df = rows_to_dataframe(sample_rows())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["frame", "label", "video", "left_knee_angle"]


def test_save_and_load_csv_roundtrip(tmp_path):
    df = rows_to_dataframe(sample_rows())
    out_path = tmp_path / "nested" / "keypoints.csv"
    save_csv(df, str(out_path))

    assert out_path.exists()
    loaded = load_csv(str(out_path))
    assert len(loaded) == len(df)
    assert list(loaded["label"]) == list(df["label"])


def test_summarize_dataset_runs_without_error(capsys):
    df = rows_to_dataframe(sample_rows())
    summarize_dataset(df)
    captured = capsys.readouterr()
    assert "Total frames" in captured.out
    assert "Label counts" in captured.out
