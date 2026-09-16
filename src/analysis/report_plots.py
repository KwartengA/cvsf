"""
Matplotlib figures for the capstone report/presentation.
No seaborn dependency (not installed) -- plain matplotlib only.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure -> {path}")
    return path


def plot_confusion_matrix(cm, labels, output_dir: Path, filename="squat_confusion_matrix.png",
                           title="Squat Form Classifier — Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


def plot_metric_bars(metrics: dict, output_dir: Path, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    bars = ax.bar(names, values, color="#4C72B0")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


def plot_feature_importance(importances: dict, output_dir: Path,
                             filename="squat_feature_importance.png",
                             title="Squat Classifier — Feature Importance (Random Forest)"):
    items = sorted(importances.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names, values, color="#55A868")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


def plot_angle_boxplot_by_label(df, angle_col: str, label_col: str, output_dir: Path,
                                 filename="squat_knee_angle_by_label.png",
                                 title="Knee Angle Distribution: Correct vs Incorrect"):
    labels = sorted(df[label_col].unique())
    data = [df.loc[df[label_col] == lbl, angle_col].dropna().to_numpy() for lbl in labels]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel(f"{angle_col} (degrees)")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, output_dir, filename)


def plot_bar_by_video(df, value_col: str, output_dir: Path, filename: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    videos = df["video"].astype(str)
    values = df[value_col]
    ax.bar(videos, values, color="#C44E52")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(videos)))
    ax.set_xticklabels(videos, rotation=45, ha="right")
    fig.tight_layout()
    return _save(fig, output_dir, filename)


def plot_angle_boxplot_multi(df, angle_cols: list, output_dir: Path,
                              filename="tennis_angle_distributions.png",
                              title="Tennis Serve — Joint Angle Distributions"):
    data = [df[c].dropna().to_numpy() for c in angle_cols]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(data, tick_labels=angle_cols)
    ax.set_ylabel("degrees")
    ax.set_title(title)
    ax.set_xticklabels(angle_cols, rotation=45, ha="right")
    fig.tight_layout()
    return _save(fig, output_dir, filename)


def plot_angle_over_frames(df, video_name: str, angle_col: str, output_dir: Path,
                            filename="tennis_elbow_angle_timeline.png",
                            title=None):
    subset = df[df["video"] == video_name].sort_values("frame")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(subset["frame"], subset[angle_col], color="#4C72B0")
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"{angle_col} (degrees)")
    ax.set_title(title or f"{angle_col} over time — {video_name}")
    fig.tight_layout()
    return _save(fig, output_dir, filename)
