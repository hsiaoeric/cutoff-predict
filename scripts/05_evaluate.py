#!/usr/bin/env python3
"""
Phase 5: Model Evaluation & Diagnostic Plots
==============================================
Evaluates trained models against baseline, performs segment analysis,
and generates comprehensive diagnostic visualisations.

Usage:
    uv run python scripts/05_evaluate.py
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from math import sqrt
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

matplotlib.use("Agg")

# Use Heiti TC for CJK on macOS
try:
    matplotlib.font_manager.fontManager.addfont(
        "/System/Library/Fonts/STHeiti Light.ttc"
    )
    plt.rcParams["font.family"] = ["Heiti TC", "sans-serif"]
except Exception:
    pass

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ─── Constants ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "notebooks" / "eda_plots"

TARGET = "cutoff_weight"

# Which model columns to evaluate (from 04_train_model predictions CSV)
MODEL_COLS = {
    "baseline_pred": "Baseline",
    "lgb_v1_pred": "LGB v1",
    "lgb_v2_pred": "LGB v2 (tuned)",
    "twostage_pred": "Two-Stage",
    "ensemble_pred": "Ensemble",
}

BEST_MODEL_COL = "ensemble_pred"  # will be updated after comparison


# ─── Helpers ─────────────────────────────────────────────────────────────────

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
    }


def load_predictions(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and metadata for a split ('val' or 'test')."""
    pred = pd.read_csv(MODEL_DIR / f"{split}_predictions.csv")
    meta = pd.read_csv(MODEL_DIR / f"{split}_metadata.csv")
    return pred, meta


# ─── 1. Overall Metrics ─────────────────────────────────────────────────────

def overall_metrics_table(pred: pd.DataFrame, split_name: str) -> str:
    """Build markdown-style table of overall metrics for each model."""
    y = pred[TARGET].values
    lines = [f"\n{'─'*70}", f"Overall Metrics — {split_name} Set", f"{'─'*70}"]
    lines.append(f"{'Model':<20s} {'MAE':>7s} {'RMSE':>7s} {'R²':>7s} {'MedAE':>7s}")
    lines.append("-" * 52)

    for col, label in MODEL_COLS.items():
        if col not in pred.columns:
            continue
        m = eval_metrics(y, pred[col].values)
        lines.append(f"{label:<20s} {m['MAE']:7.2f} {m['RMSE']:7.2f} {m['R²']:7.3f} {m['MedianAE']:7.2f}")

    return "\n".join(lines)


# ─── 2. Segment Analysis ────────────────────────────────────────────────────

def segment_analysis(
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    model_col: str,
    model_label: str,
) -> str:
    """Break down MAE by department, history length, popularity, weight range."""
    merged = pred.merge(meta, on=["course_key", TARGET], how="left", suffixes=("", "_m"))
    y = merged[TARGET].values
    yhat = merged[model_col].values
    errors = np.abs(y - yhat)

    lines = [f"\n{'─'*70}", f"Segment Analysis — {model_label}", f"{'─'*70}"]

    # --- By dept_cluster ---
    lines.append(f"\n{'By Department Cluster':}")
    lines.append(f"  {'Cluster':<15s} {'n':>5s} {'MAE':>7s} {'MedAE':>7s}")
    lines.append("  " + "-" * 38)
    for dept in sorted(merged["dept_cluster"].dropna().unique()):
        mask = merged["dept_cluster"] == dept
        n = mask.sum()
        mae = np.mean(errors[mask])
        medae = np.median(errors[mask])
        lines.append(f"  {dept:<15s} {n:5d} {mae:7.2f} {medae:7.2f}")

    # --- By semesters_offered bucket ---
    lines.append(f"\n{'By Course History Length':}")
    bins = [(1, 1, "1 (new)"), (2, 3, "2–3"), (4, 6, "4–6"), (7, 10, "7–10"), (11, 999, "11+")]
    lines.append(f"  {'History':<12s} {'n':>5s} {'MAE':>7s} {'MedAE':>7s}")
    lines.append("  " + "-" * 36)
    for lo, hi, label in bins:
        mask = (merged["semesters_offered"] >= lo) & (merged["semesters_offered"] <= hi)
        n = mask.sum()
        if n == 0:
            continue
        mae = np.mean(errors[mask])
        medae = np.median(errors[mask])
        lines.append(f"  {label:<12s} {n:5d} {mae:7.2f} {medae:7.2f}")

    # --- By popularity_tier ---
    lines.append(f"\n{'By Popularity Tier':}")
    tiers = ["low", "medium", "high", "very_high"]
    lines.append(f"  {'Tier':<12s} {'n':>5s} {'MAE':>7s} {'MedAE':>7s}")
    lines.append("  " + "-" * 36)
    for tier in tiers:
        mask = merged["popularity_tier"] == tier
        n = mask.sum()
        if n == 0:
            continue
        mae = np.mean(errors[mask])
        medae = np.median(errors[mask])
        lines.append(f"  {tier:<12s} {n:5d} {mae:7.2f} {medae:7.2f}")

    # --- By cutoff_weight range ---
    lines.append(f"\n{'By Weight Range':}")
    wbins = [(0, 0, "Zero (0)"), (1, 20, "1–20"), (21, 50, "21–50"), (51, 130, "50+")]
    lines.append(f"  {'Range':<12s} {'n':>5s} {'MAE':>7s} {'MedAE':>7s}")
    lines.append("  " + "-" * 36)
    for lo, hi, label in wbins:
        mask = (y >= lo) & (y <= hi)
        n = mask.sum()
        if n == 0:
            continue
        mae = np.mean(errors[mask])
        medae = np.median(errors[mask])
        lines.append(f"  {label:<12s} {n:5d} {mae:7.2f} {medae:7.2f}")

    # --- Zero vs Non-zero classification accuracy ---
    lines.append(f"\n{'Zero vs Non-zero Classification':}")
    actual_zero = (y == 0).astype(int)
    pred_zero = (yhat < 0.5).astype(int)  # predictions ≈0 treated as zero
    acc = accuracy_score(actual_zero, pred_zero)
    if actual_zero.sum() > 0 and (1 - actual_zero).sum() > 0:
        prec = precision_score(actual_zero, pred_zero, zero_division=0)
        rec = recall_score(actual_zero, pred_zero, zero_division=0)
        cm = confusion_matrix(actual_zero, pred_zero)
        lines.append(f"  Accuracy:  {acc:.3f}")
        lines.append(f"  Precision (zero class): {prec:.3f}")
        lines.append(f"  Recall    (zero class): {rec:.3f}")
        lines.append(f"  Confusion matrix:")
        lines.append(f"    Predicted:  NonZero  Zero")
        lines.append(f"    Actual NZ:  {cm[0,0]:>5d}  {cm[0,1]:>5d}")
        lines.append(f"    Actual  0:  {cm[1,0]:>5d}  {cm[1,1]:>5d}")
    else:
        lines.append(f"  Accuracy: {acc:.3f}")

    return "\n".join(lines)


# ─── 3. Diagnostic Plots ────────────────────────────────────────────────────

def plot_predicted_vs_actual(pred: pd.DataFrame, model_col: str, label: str, path: Path):
    """Scatter plot: predicted vs actual with diagonal reference line."""
    y = pred[TARGET].values
    yhat = pred[model_col].values
    mae = mean_absolute_error(y, yhat)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, yhat, alpha=0.4, s=18, c="#5B8DEE", edgecolors="none")
    lims = [0, max(y.max(), yhat.max()) * 1.05]
    ax.plot(lims, lims, "--", color="#E74C3C", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Cutoff Weight")
    ax.set_ylabel("Predicted Cutoff Weight")
    ax.set_title(f"Predicted vs Actual — {label}\nTest MAE = {mae:.2f}")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    fig.savefig(path)
    plt.close(fig)


def plot_residual_distribution(pred: pd.DataFrame, model_col: str, label: str, path: Path):
    """Histogram of prediction residuals (error = predicted − actual)."""
    y = pred[TARGET].values
    yhat = pred[model_col].values
    residuals = yhat - y

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, color="#5B8DEE", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#E74C3C", linewidth=1.5, linestyle="--")
    ax.axvline(np.mean(residuals), color="#2ECC71", linewidth=1.5, linestyle="-",
               label=f"Mean error = {np.mean(residuals):.1f}")
    ax.set_xlabel("Prediction Error (predicted − actual)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution — {label}")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_error_by_department(
    pred: pd.DataFrame, meta: pd.DataFrame, model_col: str, label: str, path: Path
):
    """Horizontal bar chart: MAE per department cluster."""
    merged = pred.merge(meta, on=["course_key", TARGET], how="left", suffixes=("", "_m"))
    y = merged[TARGET].values
    yhat = merged[model_col].values
    errors = np.abs(y - yhat)

    dept_mae = {}
    for dept in merged["dept_cluster"].dropna().unique():
        mask = merged["dept_cluster"] == dept
        dept_mae[dept] = np.mean(errors[mask])

    # Sort by MAE
    sorted_depts = sorted(dept_mae, key=dept_mae.get, reverse=True)
    vals = [dept_mae[d] for d in sorted_depts]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_depts)))
    bars = ax.barh(range(len(sorted_depts)), vals, color=colors)
    ax.set_yticks(range(len(sorted_depts)))
    ax.set_yticklabels(sorted_depts)
    ax.set_xlabel("Mean Absolute Error")
    ax.set_title(f"MAE by Department — {label}")
    # Add value labels
    for bar, v in zip(bars, vals):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_error_by_history(
    pred: pd.DataFrame, meta: pd.DataFrame, model_col: str, label: str, path: Path
):
    """Line chart: MAE vs semesters_offered."""
    merged = pred.merge(meta, on=["course_key", TARGET], how="left", suffixes=("", "_m"))
    y = merged[TARGET].values
    yhat = merged[model_col].values
    errors = np.abs(y - yhat)

    history_groups = {}
    for sem_count in sorted(merged["semesters_offered"].unique()):
        mask = merged["semesters_offered"] == sem_count
        n = mask.sum()
        if n >= 3:  # only plot groups with enough samples
            history_groups[sem_count] = (np.mean(errors[mask]), n)

    xs = list(history_groups.keys())
    ys = [history_groups[x][0] for x in xs]
    ns = [history_groups[x][1] for x in xs]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(xs, ys, "o-", color="#5B8DEE", linewidth=2, markersize=6, label="MAE")
    ax1.set_xlabel("Semesters Offered (course history length)")
    ax1.set_ylabel("Mean Absolute Error", color="#5B8DEE")
    ax1.set_title(f"MAE by Course History Length — {label}")

    ax2 = ax1.twinx()
    ax2.bar(xs, ns, alpha=0.15, color="#999999", label="Sample count")
    ax2.set_ylabel("Sample count", color="#999999")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.savefig(path)
    plt.close(fig)


def plot_model_comparison_bar(pred: pd.DataFrame, path: Path):
    """Side-by-side bar chart comparing all models' MAE on the test set."""
    y = pred[TARGET].values
    models = []
    maes = []
    for col, label in MODEL_COLS.items():
        if col in pred.columns:
            mae = mean_absolute_error(y, pred[col].values)
            models.append(label)
            maes.append(mae)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#95a5a6" if m == "Baseline" else "#5B8DEE" for m in models]
    # Highlight the best
    best_idx = np.argmin(maes)
    colors[best_idx] = "#2ECC71"

    bars = ax.bar(range(len(models)), maes, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("Model Comparison — Test Set MAE")

    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.savefig(path)
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 5: Model Evaluation & Diagnostic Plots")
    print("=" * 70)

    # Load predictions
    print("\n📂 Loading predictions…")
    test_pred, test_meta = load_predictions("test")
    val_pred, val_meta = load_predictions("val")

    print(f"  Test: {len(test_pred)} rows")
    print(f"  Val:  {len(val_pred)} rows")

    # Determine best model
    y_test = test_pred[TARGET].values
    best_mae = float("inf")
    best_col = "ensemble_pred"
    for col in MODEL_COLS:
        if col in test_pred.columns and col != "baseline_pred":
            mae = mean_absolute_error(y_test, test_pred[col].values)
            if mae < best_mae:
                best_mae = mae
                best_col = col
    best_label = MODEL_COLS[best_col]
    print(f"  Best model: {best_label} (MAE={best_mae:.2f})")

    # ── Build report text ─────────────────────────────────────────────────
    report_parts = []

    header = textwrap.dedent(f"""\
    ======================================================================
    Evaluation Report — Cutoff Weight Prediction
    ======================================================================
    Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Best model: {best_label}
    """)
    report_parts.append(header)

    # Overall metrics
    print("\n📊 Computing overall metrics…")
    report_parts.append(overall_metrics_table(val_pred, "Validation"))
    report_parts.append(overall_metrics_table(test_pred, "Test"))

    # Baseline comparison
    bl_mae = mean_absolute_error(y_test, test_pred["baseline_pred"].values)
    improvement = (1 - best_mae / bl_mae) * 100
    report_parts.append(f"\n{'─'*70}")
    report_parts.append(f"Improvement over baseline: {improvement:+.1f}%")
    report_parts.append(f"  Baseline MAE: {bl_mae:.2f}")
    report_parts.append(f"  Best MAE:     {best_mae:.2f}")

    # Segment analysis (best model, test set)
    print("📊 Segment analysis…")
    report_parts.append(segment_analysis(test_pred, test_meta, best_col, f"{best_label} — Test"))

    # Also do segment for validation
    if best_col in val_pred.columns:
        y_val = val_pred[TARGET].values
        best_val_mae = mean_absolute_error(y_val, val_pred[best_col].values)
        report_parts.append(segment_analysis(val_pred, val_meta, best_col, f"{best_label} — Validation"))

    report_parts.append(f"\n{'='*70}")
    report_parts.append("End of Evaluation Report")
    report_parts.append(f"{'='*70}\n")

    report_text = "\n".join(report_parts)
    report_path = MODEL_DIR / "evaluation_report.txt"
    report_path.write_text(report_text)
    print(f"  Saved → {report_path}")

    # ── Generate plots ────────────────────────────────────────────────────
    print("\n🎨 Generating evaluation plots…")

    # 1. Predicted vs Actual (best model)
    plot_predicted_vs_actual(
        test_pred, best_col, best_label,
        PLOT_DIR / "eval_predicted_vs_actual.png",
    )
    print(f"  ✓ Predicted vs Actual scatter")

    # 2. Residual distribution (best model)
    plot_residual_distribution(
        test_pred, best_col, best_label,
        PLOT_DIR / "eval_residual_distribution.png",
    )
    print(f"  ✓ Residual distribution histogram")

    # 3. Error by department (best model)
    plot_error_by_department(
        test_pred, test_meta, best_col, best_label,
        PLOT_DIR / "eval_error_by_department.png",
    )
    print(f"  ✓ Error by department")

    # 4. Error by course history length (best model)
    plot_error_by_history(
        test_pred, test_meta, best_col, best_label,
        PLOT_DIR / "eval_error_by_history.png",
    )
    print(f"  ✓ Error by history length")

    # 5. Model comparison bar chart
    plot_model_comparison_bar(test_pred, PLOT_DIR / "eval_model_comparison.png")
    print(f"  ✓ Model comparison bar chart")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🏁 Evaluation complete!")
    print(f"\n📈 Results summary:\n")
    print(overall_metrics_table(test_pred, "Test"))
    print(f"\n  Best model:  {best_label}")
    print(f"  Best MAE:    {best_mae:.2f}")
    print(f"  Baseline MAE: {bl_mae:.2f}")
    print(f"  Improvement: {improvement:+.1f}%")

    # Success level
    if best_mae > bl_mae:
        level = "🔴 Fail — model is worse than baseline"
    elif abs(best_mae - bl_mae) <= 2:
        level = "🟡 OK — roughly matches naive approach"
    elif bl_mae - best_mae >= 3:
        level = "🟢 Good — meaningful improvement"
    if best_mae < 8:
        level = "🌟 Great — trustworthy within ~8 weight points"
    if best_mae < 5:
        level = "🏆 Excellent — production-ready quality!"

    print(f"  Level:       {level}")
    print("=" * 70)


if __name__ == "__main__":
    main()
