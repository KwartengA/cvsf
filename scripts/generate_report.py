"""
generate_report.py
-------------------
One-shot script that trains/evaluates the squat form classifier, computes
tennis serve pose-quality metrics, saves all report figures, and prints a
terminal summary to read from while presenting.

Usage (from the project root with venv active):
    python scripts/generate_report.py

Prerequisite: run scripts/analyze_squat.py and scripts/analyze_tennis_serve.py
at least once so data/processed/*.csv exist.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.dataset_creator import load_csv
from src.analysis import squat_classifier
from src.analysis import pushup_classifier
from src.analysis import leg_press_classifier
from src.analysis.squat_classifier import save_model
from src.analysis.tennis_metrics import summarize_tennis_metrics
from src.analysis.tennis_rep_utils import detect_hits
from src.analysis.tennis_shot_classifier import (
    build_hit_feature_rows, build_feature_matrix as build_shot_feature_matrix,
    train_and_evaluate as train_shot_classifier,
)
from src.analysis import report_plots as plots
from src.utils.video_utils import video_info

SQUAT_CSV = Path("data/processed/squat_keypoints.csv")
PUSHUP_CSV = Path("data/processed/pushup_keypoints.csv")
LEG_PRESS_CSV = Path("data/processed/leg_press_keypoints.csv")
TENNIS_CSV = Path("data/processed/tennis_serve_keypoints.csv")
TENNIS_SHOTS_CSV = Path("data/processed/tennis_shots_keypoints.csv")
TENNIS_RAW_DIR = Path("data/raw/tennis/serve/correct")
FIGURES_DIR = Path("docs/figures")
SQUAT_MODEL_PATH = Path("models/squat_form_classifier.joblib")
PUSHUP_MODEL_PATH = Path("models/pushup_form_classifier.joblib")
LEG_PRESS_MODEL_PATH = Path("models/leg_press_form_classifier.joblib")
TENNIS_SHOT_MODEL_PATH = Path("models/tennis_shot_classifier.joblib")


def require_csv(path: Path, generator_script: str):
    if not path.exists():
        print(f"\nERROR: {path} not found.")
        print(f"Run `python {generator_script}` first to generate it.\n")
        sys.exit(1)


def run_rep_activity_report(activity_name, csv_path, analyze_script, classifier_module,
                             depth_angle_col, model_path):
    display_name = activity_name.replace("_", " ")
    print(f"\n{'='*60}")
    print(f"  {display_name.upper()} FORM CLASSIFIER")
    print(f"{'='*60}")

    require_csv(csv_path, analyze_script)
    df = load_csv(str(csv_path))
    labeled = classifier_module.label_frames_by_threshold(df)

    n_reps = labeled["rep_id"].nunique()
    counts = labeled["form_label"].value_counts()
    print(f"  Total frames in CSV     : {len(df)}")
    print(f"  Reps detected           : {n_reps} (across {df['video'].nunique()} videos)")
    print(f"  Bottom-of-rep frames used for training: {len(labeled)}")
    print(f"  Label distribution      : {counts.to_dict()}")

    X, y = classifier_module.build_feature_matrix(labeled)
    results = classifier_module.train_and_evaluate(X, y)

    print(f"\n  Train/test split  : {results['n_train']} train / {results['n_test']} test")
    print(f"  Accuracy          : {results['accuracy']:.3f}")
    print(f"  Precision (incorrect): {results['precision']:.3f}")
    print(f"  Recall (incorrect)   : {results['recall']:.3f}")
    print(f"  F1 (incorrect)       : {results['f1']:.3f}")
    print(f"  Confusion matrix ({results['labels']}):")
    print(f"    {results['confusion_matrix']}")

    print("\n  Feature importances (Random Forest):")
    for name, imp in sorted(results["feature_importances"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:20s} {imp:.3f}")

    print("\n  CAVEAT: 'incorrect' labels are threshold-derived from the same rules")
    print("  used for live feedback, not independently recorded bad-form footage.")
    print("  State this when presenting these numbers.")

    save_model(results["model"], str(model_path))

    plots.plot_confusion_matrix(
        results["confusion_matrix"], results["labels"], FIGURES_DIR,
        filename=f"{activity_name}_confusion_matrix.png",
        title=f"{display_name.capitalize()} Form Classifier — Confusion Matrix",
    )
    plots.plot_metric_bars(
        {"accuracy": results["accuracy"], "precision": results["precision"],
         "recall": results["recall"], "f1": results["f1"]},
        FIGURES_DIR, f"{display_name.capitalize()} Form Classifier — Metrics",
        f"{activity_name}_metrics_bar.png",
    )
    plots.plot_feature_importance(
        results["feature_importances"], FIGURES_DIR,
        filename=f"{activity_name}_feature_importance.png",
        title=f"{display_name.capitalize()} Classifier — Feature Importance (Random Forest)",
    )
    plots.plot_angle_boxplot_by_label(
        labeled, depth_angle_col, "form_label", FIGURES_DIR,
        filename=f"{activity_name}_{depth_angle_col}_by_label.png",
        title=f"{display_name.capitalize()}: {depth_angle_col} — Correct vs Incorrect",
    )

    return results


def run_squat_report():
    return run_rep_activity_report(
        "squat", SQUAT_CSV, "scripts/analyze_squat.py", squat_classifier,
        "left_knee_angle", SQUAT_MODEL_PATH,
    )


def run_pushup_report():
    return run_rep_activity_report(
        "pushup", PUSHUP_CSV, "scripts/analyze_pushup.py", pushup_classifier,
        "left_elbow_angle", PUSHUP_MODEL_PATH,
    )


def run_leg_press_report():
    return run_rep_activity_report(
        "leg_press", LEG_PRESS_CSV, "scripts/analyze_leg_press.py", leg_press_classifier,
        "left_knee_angle", LEG_PRESS_MODEL_PATH,
    )


def run_tennis_report():
    print(f"\n{'='*60}")
    print("  TENNIS SERVE — POSE QUALITY METRICS (rule-based, no trained classifier)")
    print(f"{'='*60}")

    require_csv(TENNIS_CSV, "scripts/analyze_tennis_serve.py")
    df = load_csv(str(TENNIS_CSV))

    frame_counts = {}
    if TENNIS_RAW_DIR.exists():
        for vf in sorted(TENNIS_RAW_DIR.glob("*.mov")) + sorted(TENNIS_RAW_DIR.glob("*.mp4")):
            try:
                frame_counts[vf.name] = video_info(str(vf))["frame_count"]
            except Exception:
                pass

    summary = summarize_tennis_metrics(df, frame_counts=frame_counts or None)

    print(f"  Videos analysed     : {df['video'].nunique()}")
    print(f"  Frames with pose    : {len(df)}")
    if summary.get("overall_detection_rate") is not None:
        print(f"  Pose detection rate : {summary['overall_detection_rate']*100:.1f}%")
    print(f"  Avg landmark visibility: {summary['mean_visibility']:.3f}")
    print(f"  Min landmark visibility: {summary['min_visibility']:.3f}")
    print(f"  Overall rule pass rate : {summary['overall_pass_rate']*100:.1f}%")

    print("\n  Per-video rule pass rate:")
    print(summary["pass_rate_by_video"].to_string(index=False))

    print("\n  CAVEAT: rule_pass_rate scores every frame (including wind-up,")
    print("  toss, and follow-through), not just the trophy/contact position")
    print("  the thresholds are designed for -- so this number is expected")
    print("  to look low. It's a pose-quality/consistency signal, not a")
    print("  'form accuracy' figure. State this when presenting it.")

    if "detection_by_video" in summary:
        plots.plot_bar_by_video(
            summary["detection_by_video"], "detection_rate", FIGURES_DIR,
            "tennis_detection_rate.png", "Tennis Serve — Pose Detection Rate per Video",
            "Detection rate",
        )
    plots.plot_bar_by_video(
        summary["pass_rate_by_video"], "rule_pass_rate", FIGURES_DIR,
        "tennis_rule_pass_rate.png", "Tennis Serve — Rule-Based Pass Rate per Video",
        "Pass rate",
    )
    plots.plot_angle_boxplot_multi(
        df, ["right_elbow_angle", "right_shoulder_angle", "right_hip_angle",
             "right_knee_angle", "trunk_lean_angle"],
        FIGURES_DIR,
    )

    first_video = sorted(df["video"].unique())[0]
    plots.plot_angle_over_frames(df, first_video, "right_elbow_angle", FIGURES_DIR)

    return summary


def run_tennis_shot_report():
    print(f"\n{'='*60}")
    print("  TENNIS SHOT-TYPE CLASSIFIER (serve / forehand / backhand / volley)")
    print(f"{'='*60}")

    require_csv(TENNIS_SHOTS_CSV, "scripts/analyze_tennis_shots.py")
    df = load_csv(str(TENNIS_SHOTS_CSV))

    print(f"  Total frames in CSV : {len(df)}")
    print(f"  Videos              : {df['video'].nunique()}")
    print(f"  Source video counts per shot type (uneven -- see caveat below):")
    video_counts = df.groupby("shot_type")["video"].nunique()
    print(video_counts.to_string())

    hit_df = detect_hits(df)
    hit_rows = build_hit_feature_rows(hit_df)
    print(f"\n  Hits detected (= training rows): {len(hit_rows)}")
    print(f"  Hit counts per shot type:")
    print(hit_rows["shot_type"].value_counts().to_string())

    X, y = build_shot_feature_matrix(hit_rows)
    results = train_shot_classifier(X, y)

    print(f"\n  Train/test split  : {results['n_train']} train / {results['n_test']} test")
    print(f"  Accuracy          : {results['accuracy']:.3f}")
    print(f"  Precision (macro) : {results['precision']:.3f}")
    print(f"  Recall (macro)    : {results['recall']:.3f}")
    print(f"  F1 (macro)        : {results['f1']:.3f}")
    print(f"  Per-class test support: {results['support']}")
    print(f"  Confusion matrix ({results['labels']}):")
    print(f"    {results['confusion_matrix']}")

    print("\n  Feature importances (Random Forest):")
    for name, imp in sorted(results["feature_importances"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:28s} {imp:.3f}")

    print("\n  CAVEAT (read before presenting these numbers):")
    print("  1. Labels ARE real ground truth (the folder each video was")
    print("     uploaded into) -- stronger footing than squat/pushup/leg_press's")
    print("     threshold-derived labels, worth stating as a positive.")
    print("  2. Source video counts are UNEVEN across classes (see table above)")
    print("     -- forehand/volley have far fewer source videos than")
    print("     backhand/serve, so their per-class accuracy is lower-confidence.")
    print("     Macro-averaged precision/recall/F1 (not accuracy alone) reflect")
    print("     this -- a class the model rarely predicts correctly pulls the")
    print("     macro average down even if overall accuracy looks fine.")
    print("  3. 'Hits' (rep boundaries) are detected from racket-speed peaks,")
    print("     calibrated per-video (relative to that video's own speed")
    print("     distribution) rather than an absolute threshold -- verified")
    print("     against real clips from all 4 shot types, but not frame-by-frame")
    print("     validated against ground truth (i.e. someone manually marking")
    print("     every real swing) due to time constraints.")

    save_model(results["model"], str(TENNIS_SHOT_MODEL_PATH))

    plots.plot_confusion_matrix(
        results["confusion_matrix"], results["labels"], FIGURES_DIR,
        filename="tennis_shot_confusion_matrix.png",
        title="Tennis Shot-Type Classifier — Confusion Matrix",
    )
    plots.plot_metric_bars(
        {"accuracy": results["accuracy"], "precision (macro)": results["precision"],
         "recall (macro)": results["recall"], "f1 (macro)": results["f1"]},
        FIGURES_DIR, "Tennis Shot-Type Classifier — Metrics",
        "tennis_shot_metrics_bar.png",
    )
    plots.plot_feature_importance(
        results["feature_importances"], FIGURES_DIR,
        filename="tennis_shot_feature_importance.png",
        title="Tennis Shot-Type Classifier — Feature Importance (Random Forest)",
    )

    return results


def main():
    squat_results = run_squat_report()

    pushup_results = None
    if PUSHUP_CSV.exists():
        pushup_results = run_pushup_report()
    else:
        print(f"\n  Skipping pushup report -- {PUSHUP_CSV} not found yet.")
        print(f"  Run `python scripts/analyze_pushup.py` once footage is in "
              f"data/raw/gym/pushup/correct/.")

    leg_press_results = None
    if LEG_PRESS_CSV.exists():
        leg_press_results = run_leg_press_report()
    else:
        print(f"\n  Skipping leg press report -- {LEG_PRESS_CSV} not found yet.")
        print(f"  Run `python scripts/analyze_leg_press.py` once footage is in "
              f"data/raw/gym/leg_presses/correct/.")

    tennis_summary = run_tennis_report()

    tennis_shot_results = None
    if TENNIS_SHOTS_CSV.exists():
        tennis_shot_results = run_tennis_shot_report()
    else:
        print(f"\n  Skipping tennis shot-type report -- {TENNIS_SHOTS_CSV} not found yet.")
        print(f"  Run `python scripts/analyze_tennis_shots.py` once footage is in "
              f"data/raw/tennis/{{serve,forehand,backhand,volley}}/correct/.")

    print(f"\n{'='*60}")
    print("  TALKING POINTS SUMMARY")
    print(f"{'='*60}")
    print(f"  SQUAT MODEL (Logistic Regression, threshold-derived labels)")
    print(f"    Accuracy: {squat_results['accuracy']*100:.1f}%  |  "
          f"F1: {squat_results['f1']:.3f}  |  "
          f"n_test={squat_results['n_test']}")
    top_feature = max(squat_results["feature_importances"].items(), key=lambda kv: kv[1])
    print(f"    Most important feature: {top_feature[0]} ({top_feature[1]:.2f})")
    print(f"    Caveat: labels derived from biomechanics thresholds, not reviewed footage")

    if pushup_results:
        print()
        print(f"  PUSHUP MODEL (Logistic Regression, threshold-derived labels)")
        print(f"    Accuracy: {pushup_results['accuracy']*100:.1f}%  |  "
              f"F1: {pushup_results['f1']:.3f}  |  "
              f"n_test={pushup_results['n_test']}")
        top_feature = max(pushup_results["feature_importances"].items(), key=lambda kv: kv[1])
        print(f"    Most important feature: {top_feature[0]} ({top_feature[1]:.2f})")
        print(f"    Caveat: labels derived from biomechanics thresholds, not reviewed footage")

    if leg_press_results:
        print()
        print(f"  LEG PRESS MODEL (Logistic Regression, threshold-derived labels)")
        print(f"    Accuracy: {leg_press_results['accuracy']*100:.1f}%  |  "
              f"F1: {leg_press_results['f1']:.3f}  |  "
              f"n_test={leg_press_results['n_test']}")
        top_feature = max(leg_press_results["feature_importances"].items(), key=lambda kv: kv[1])
        print(f"    Most important feature: {top_feature[0]} ({top_feature[1]:.2f})")
        print(f"    Caveat: labels derived from biomechanics thresholds, not reviewed footage")

    print()
    print(f"  TENNIS SERVE (rule-based only — separate YOLO+LSTM model used elsewhere)")
    if tennis_summary.get("overall_detection_rate") is not None:
        print(f"    Pose detection rate: {tennis_summary['overall_detection_rate']*100:.1f}%")
    print(f"    Avg landmark confidence: {tennis_summary['mean_visibility']:.3f}")
    print(f"    Rule-based form pass rate: {tennis_summary['overall_pass_rate']*100:.1f}%")

    if tennis_shot_results:
        print()
        print(f"  TENNIS SHOT-TYPE CLASSIFIER (serve/forehand/backhand/volley, real labels)")
        print(f"    Accuracy: {tennis_shot_results['accuracy']*100:.1f}%  |  "
              f"F1 (macro): {tennis_shot_results['f1']:.3f}  |  "
              f"n_test={tennis_shot_results['n_test']}")
        print(f"    Per-class test support: {tennis_shot_results['support']}")
        top_feature = max(tennis_shot_results["feature_importances"].items(), key=lambda kv: kv[1])
        print(f"    Most important feature: {top_feature[0]} ({top_feature[1]:.2f})")
        print(f"    Caveat: uneven source video counts per class (see full report above) --")
        print(f"    macro F1 reflects weaker classes (e.g. forehand), not just overall accuracy")

    print(f"\n  Figures saved to: {FIGURES_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
