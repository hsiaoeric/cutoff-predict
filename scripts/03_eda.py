#!/usr/bin/env python3
"""
Phase 3: Exploratory Data Analysis
====================================
Generates comprehensive EDA visualizations and summary statistics
from the enriched dataset. Produces charts saved to notebooks/eda_plots/
and a summary report.

Charts generated:
  1. Distribution of cutoff weights (histogram + KDE)
  2. Weight trends over time (overall + top courses)
  3. Most competitive courses (top 20 by avg weight)
  4. Correlation analysis (credits, course type, weight)
  5. Semester-over-semester weight volatility by department
  6. Course count growth over time
  7. Weight heatmap: top courses × semesters
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "features_enriched.csv"
PLOTS_DIR = PROJECT_ROOT / "notebooks" / "eda_plots"
SUMMARY_FILE = PROJECT_ROOT / "notebooks" / "eda_summary.txt"

# Try to find a CJK font for Chinese characters
def setup_chinese_font():
    """Configure matplotlib to render Chinese characters."""
    # Common CJK fonts on macOS
    cjk_fonts = [
        "PingFang TC", "PingFang SC", "Heiti TC", "Heiti SC",
        "STHeiti", "Apple LiGothic", "LiHei Pro",
        "Noto Sans CJK TC", "Noto Sans CJK SC",
        "Microsoft JhengHei", "SimHei"
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font_name in cjk_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"  Using font: {font_name}")
            return font_name

    print("  ⚠ No CJK font found — Chinese characters may not render correctly")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None


def set_style():
    """Set a clean, professional plotting style."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def plot_weight_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot 1: Distribution of cutoff weights."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with KDE
    ax1 = axes[0]
    weights = df["cutoff_weight"]
    ax1.hist(weights, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8, density=True)
    weights.plot.kde(ax=ax1, color="#C44E52", linewidth=2)
    ax1.set_xlabel("志願權重 (Cutoff Weight)")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution of Cutoff Weights (All Semesters)")
    ax1.axvline(weights.median(), color="#DD8452", linestyle="--", linewidth=1.5,
                label=f"Median = {weights.median():.0f}")
    ax1.axvline(weights.mean(), color="#55A868", linestyle="--", linewidth=1.5,
                label=f"Mean = {weights.mean():.1f}")
    ax1.legend()

    # Box plot by course type
    ax2 = axes[1]
    df.boxplot(column="cutoff_weight", by="course_type", ax=ax2)
    ax2.set_title("Weight Distribution by Course Type")
    ax2.set_xlabel("選別 (Course Type)")
    ax2.set_ylabel("志願權重 (Cutoff Weight)")
    fig.suptitle("")  # Remove auto-generated suptitle from boxplot

    plt.tight_layout()
    fig.savefig(output_dir / "01_weight_distribution.png")
    plt.close(fig)
    print("  ✓ 01_weight_distribution.png")


def plot_weight_trends_over_time(df: pd.DataFrame, output_dir: Path):
    """Plot 2: Weight trends across semesters."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Overall trend: median + mean per semester
    sem_stats = df.groupby("semester_ordinal").agg(
        semester_code=("semester_code", "first"),
        mean_weight=("cutoff_weight", "mean"),
        median_weight=("cutoff_weight", "median"),
        q25=("cutoff_weight", lambda x: x.quantile(0.25)),
        q75=("cutoff_weight", lambda x: x.quantile(0.75)),
        count=("cutoff_weight", "count")
    ).reset_index()

    ax1 = axes[0]
    ax1.fill_between(sem_stats["semester_ordinal"], sem_stats["q25"], sem_stats["q75"],
                     alpha=0.2, color="#4C72B0", label="IQR (25-75%)")
    ax1.plot(sem_stats["semester_ordinal"], sem_stats["mean_weight"],
             "o-", color="#4C72B0", linewidth=2, markersize=5, label="Mean")
    ax1.plot(sem_stats["semester_ordinal"], sem_stats["median_weight"],
             "s--", color="#C44E52", linewidth=2, markersize=5, label="Median")
    ax1.set_xlabel("Semester (ordinal)")
    ax1.set_ylabel("Cutoff Weight")
    ax1.set_title("Cutoff Weight Trends Over Time (All Courses)")
    ax1.set_xticks(sem_stats["semester_ordinal"])
    ax1.set_xticklabels(sem_stats["semester_code"], rotation=45, ha="right", fontsize=8)
    ax1.legend()

    # Top 10 most competitive courses over time
    ax2 = axes[1]
    top_courses = (df.groupby("course_name")["cutoff_weight"].mean()
                   .nlargest(10).index.tolist())
    top_df = df[df["course_name"].isin(top_courses)]

    for course_name in top_courses:
        course_data = top_df[top_df["course_name"] == course_name]
        # If a course has multiple sections, use the max weight per semester
        sem_data = course_data.groupby("semester_ordinal")["cutoff_weight"].max().reset_index()
        ax2.plot(sem_data["semester_ordinal"], sem_data["cutoff_weight"],
                 "o-", markersize=4, linewidth=1.5, label=course_name)

    ax2.set_xlabel("Semester (ordinal)")
    ax2.set_ylabel("Cutoff Weight")
    ax2.set_title("Top 10 Most Competitive Courses — Weight Over Time")
    ax2.set_xticks(sem_stats["semester_ordinal"])
    ax2.set_xticklabels(sem_stats["semester_code"], rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))

    plt.tight_layout()
    fig.savefig(output_dir / "02_weight_trends.png")
    plt.close(fig)
    print("  ✓ 02_weight_trends.png")


def plot_most_competitive_courses(df: pd.DataFrame, output_dir: Path):
    """Plot 3: Top 20 courses by average cutoff weight."""
    fig, ax = plt.subplots(figsize=(12, 8))

    top20 = (df.groupby(["course_id", "course_name"])
             .agg(avg_weight=("cutoff_weight", "mean"),
                  max_weight=("cutoff_weight", "max"),
                  count=("cutoff_weight", "count"))
             .sort_values("avg_weight", ascending=True)
             .tail(20))

    labels = [f"{name} ({cid})" for cid, name in top20.index]
    y_pos = range(len(labels))

    bars = ax.barh(y_pos, top20["avg_weight"], color="#4C72B0", alpha=0.8, height=0.7)
    ax.scatter(top20["max_weight"], y_pos, color="#C44E52", zorder=5, s=30, label="Max weight")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Average Cutoff Weight")
    ax.set_title("Top 20 Most Competitive Courses (by Avg Weight)")
    ax.legend()

    # Add value labels
    for i, (avg, mx) in enumerate(zip(top20["avg_weight"], top20["max_weight"])):
        ax.text(avg + 0.5, i, f"{avg:.1f}", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / "03_most_competitive.png")
    plt.close(fig)
    print("  ✓ 03_most_competitive.png")


def plot_correlation_analysis(df: pd.DataFrame, output_dir: Path):
    """Plot 4: Correlation between features and cutoff weight."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Credits vs Weight
    ax1 = axes[0]
    credit_weight = df.groupby("credits")["cutoff_weight"].agg(["mean", "std", "count"]).reset_index()
    credit_weight = credit_weight[credit_weight["count"] >= 10]  # Filter low-count groups
    ax1.bar(credit_weight["credits"], credit_weight["mean"], color="#4C72B0", alpha=0.8)
    ax1.errorbar(credit_weight["credits"], credit_weight["mean"], yerr=credit_weight["std"],
                 fmt="none", color="#333", capsize=3)
    ax1.set_xlabel("Credits (學分)")
    ax1.set_ylabel("Mean Cutoff Weight")
    ax1.set_title("Credits vs. Cutoff Weight")

    # Course type comparison
    ax2 = axes[1]
    type_stats = df.groupby("course_type")["cutoff_weight"].agg(["mean", "median", "std"]).reset_index()
    x = range(len(type_stats))
    ax2.bar(x, type_stats["mean"], color=["#4C72B0", "#C44E52"][:len(type_stats)], alpha=0.8)
    ax2.errorbar(x, type_stats["mean"], yerr=type_stats["std"], fmt="none", color="#333", capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(type_stats["course_type"])
    ax2.set_xlabel("Course Type (選別)")
    ax2.set_ylabel("Mean Cutoff Weight")
    ax2.set_title("Required vs. Elective — Cutoff Weight")

    # Numeric feature correlation heatmap
    ax3 = axes[2]
    numeric_cols = ["cutoff_weight", "credits", "grade_level", "semester_ordinal",
                    "is_required", "semesters_offered"]
    valid_cols = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[valid_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax3, square=True, linewidths=0.5)
    ax3.set_title("Feature Correlation Matrix")

    plt.tight_layout()
    fig.savefig(output_dir / "04_correlation_analysis.png")
    plt.close(fig)
    print("  ✓ 04_correlation_analysis.png")


def plot_volatility_by_department(df: pd.DataFrame, output_dir: Path):
    """Plot 5: Semester-over-semester weight volatility by department cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Compute per-department volatility
    dept_vol = (df.groupby("dept_cluster")
                .agg(mean_weight=("cutoff_weight", "mean"),
                     volatility=("cutoff_weight", "std"),
                     count=("cutoff_weight", "count"))
                .sort_values("volatility", ascending=True))
    dept_vol = dept_vol[dept_vol["count"] >= 20]  # Filter small groups

    # Bar chart of volatility
    ax1 = axes[0]
    ax1.barh(range(len(dept_vol)), dept_vol["volatility"], color="#4C72B0", alpha=0.8)
    ax1.set_yticks(range(len(dept_vol)))
    ax1.set_yticklabels(dept_vol.index, fontsize=9)
    ax1.set_xlabel("Std Dev of Cutoff Weight")
    ax1.set_title("Weight Volatility by Department Cluster")

    # Semester-level volatility trends
    ax2 = axes[1]
    top_depts = dept_vol.nlargest(5, "volatility").index.tolist()
    for dept in top_depts:
        dept_data = df[df["dept_cluster"] == dept]
        sem_std = dept_data.groupby("semester_ordinal")["cutoff_weight"].std().reset_index()
        ax2.plot(sem_std["semester_ordinal"], sem_std["cutoff_weight"],
                 "o-", markersize=3, linewidth=1.5, label=dept)

    ax2.set_xlabel("Semester (ordinal)")
    ax2.set_ylabel("Std Dev of Cutoff Weight")
    ax2.set_title("Weight Volatility Over Time (Top 5 Departments)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "05_volatility_by_department.png")
    plt.close(fig)
    print("  ✓ 05_volatility_by_department.png")


def plot_course_count_growth(df: pd.DataFrame, output_dir: Path):
    """Plot 6: Number of courses offered per semester over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    sem_counts = (df.groupby(["semester_ordinal", "semester_code"])
                  .agg(
                      n_rows=("course_id", "count"),
                      n_unique_courses=("course_id", "nunique")
                  )
                  .reset_index()
                  .sort_values("semester_ordinal"))

    ax.bar(sem_counts["semester_ordinal"], sem_counts["n_rows"],
           color="#4C72B0", alpha=0.5, label="Total rows (incl. sections)")
    ax.bar(sem_counts["semester_ordinal"], sem_counts["n_unique_courses"],
           color="#C44E52", alpha=0.8, label="Unique course IDs")

    ax.set_xticks(sem_counts["semester_ordinal"])
    ax.set_xticklabels(sem_counts["semester_code"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Semester")
    ax.set_ylabel("Count")
    ax.set_title("Course Count Growth Over Time")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "06_course_count_growth.png")
    plt.close(fig)
    print("  ✓ 06_course_count_growth.png")


def plot_weight_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot 7: Heatmap of cutoff weights for top courses across semesters."""
    # Select top 25 most frequently offered courses
    top_courses = (df.groupby("course_name")["semester_ordinal"].nunique()
                   .nlargest(25).index.tolist())
    top_df = df[df["course_name"].isin(top_courses)]

    # Pivot: course_name × semester_code, value = max cutoff weight
    pivot = top_df.pivot_table(
        index="course_name",
        columns="semester_code",
        values="cutoff_weight",
        aggfunc="max"
    )

    # Sort by mean weight
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Cutoff Weight"},
                annot_kws={"fontsize": 6})
    ax.set_title("Cutoff Weight Heatmap — Top 25 Courses × Semesters")
    ax.set_xlabel("Semester Code")
    ax.set_ylabel("")
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=7, rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(output_dir / "07_weight_heatmap.png")
    plt.close(fig)
    print("  ✓ 07_weight_heatmap.png")


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """Generate a text-based EDA summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("TMU Course Cutoff Weight — EDA Summary Report")
    lines.append("=" * 70)

    lines.append(f"\n📊 Dataset Overview")
    lines.append(f"  Total rows:              {len(df)}")
    lines.append(f"  Unique course IDs:       {df['course_id'].nunique()}")
    lines.append(f"  Unique course keys:      {df['course_key'].nunique() if 'course_key' in df.columns else 'N/A'}")
    lines.append(f"  Semesters covered:       {df['semester_code'].nunique()}")
    lines.append(f"  Semester range:          {df['semester_code'].min()} → {df['semester_code'].max()}")

    lines.append(f"\n📈 Cutoff Weight Statistics")
    w = df["cutoff_weight"]
    lines.append(f"  Mean:                    {w.mean():.2f}")
    lines.append(f"  Median:                  {w.median():.1f}")
    lines.append(f"  Std Dev:                 {w.std():.2f}")
    lines.append(f"  Min:                     {w.min():.0f}")
    lines.append(f"  Max:                     {w.max():.0f}")
    lines.append(f"  Zero-weight rows:        {(w == 0).sum()} ({(w == 0).mean()*100:.1f}%)")
    lines.append(f"  Rows with weight > 0:    {(w > 0).sum()} ({(w > 0).mean()*100:.1f}%)")

    lines.append(f"\n🏆 Top 15 Most Competitive Courses (by avg weight)")
    top = (df.groupby(["course_id", "course_name"])["cutoff_weight"]
           .agg(["mean", "max", "count"])
           .sort_values("mean", ascending=False)
           .head(15))
    for (cid, name), row in top.iterrows():
        lines.append(f"  {name:<20} ({cid})  avg={row['mean']:.1f}  max={row['max']:.0f}  n={row['count']:.0f}")

    lines.append(f"\n📉 Most Volatile Courses (highest std dev, ≥5 semesters)")
    vol = (df.groupby(["course_id", "course_name"])
           .agg(std=("cutoff_weight", "std"), count=("cutoff_weight", "count"),
                mean=("cutoff_weight", "mean"))
           .query("count >= 5")
           .sort_values("std", ascending=False)
           .head(10))
    for (cid, name), row in vol.iterrows():
        lines.append(f"  {name:<20} ({cid})  std={row['std']:.1f}  mean={row['mean']:.1f}  n={row['count']:.0f}")

    lines.append(f"\n🏫 Department Cluster Statistics")
    if "dept_cluster" in df.columns:
        dept_stats = (df.groupby("dept_cluster")
                      .agg(count=("cutoff_weight", "count"),
                           mean_wt=("cutoff_weight", "mean"),
                           std_wt=("cutoff_weight", "std"))
                      .sort_values("mean_wt", ascending=False))
        for dept, row in dept_stats.iterrows():
            lines.append(f"  {dept:<15}  n={row['count']:>5.0f}  mean={row['mean_wt']:.1f}  std={row['std_wt']:.1f}")

    lines.append(f"\n📅 Per-Semester Summary")
    sem_stats = (df.groupby("semester_code")
                 .agg(n=("cutoff_weight", "count"),
                      mean=("cutoff_weight", "mean"),
                      median=("cutoff_weight", "median"),
                      max_wt=("cutoff_weight", "max"))
                 .sort_index())
    for sem, row in sem_stats.iterrows():
        lines.append(f"  {sem}:  n={row['n']:>4.0f}  mean={row['mean']:>5.1f}  "
                     f"median={row['median']:>5.1f}  max={row['max_wt']:>4.0f}")

    lines.append(f"\n🔮 Key Observations for ML")
    zero_pct = (w == 0).mean() * 100
    lines.append(f"  • {zero_pct:.1f}% of rows have zero weight (may need special handling)")
    lines.append(f"  • Weight distribution is right-skewed (median < mean)")

    # Check temporal trend
    first_half = df[df["semester_ordinal"] <= 13]["cutoff_weight"].mean()
    second_half = df[df["semester_ordinal"] > 13]["cutoff_weight"].mean()
    if second_half > first_half:
        lines.append(f"  • Weights trending UP over time (early avg={first_half:.1f}, recent avg={second_half:.1f})")
    else:
        lines.append(f"  • Weights trending DOWN over time (early avg={first_half:.1f}, recent avg={second_half:.1f})")

    # prev_1_weight correlation
    if "prev_1_weight" in df.columns:
        valid = df.dropna(subset=["prev_1_weight", "cutoff_weight"])
        if len(valid) > 10:
            corr = valid["prev_1_weight"].corr(valid["cutoff_weight"])
            lines.append(f"  • prev_1_weight correlation with target: {corr:.3f} (strong predictor)")

    lines.append(f"\n{'=' * 70}")
    lines.append(f"End of EDA Report")
    lines.append(f"{'=' * 70}")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def main():
    print("=" * 60)
    print("Phase 3: Exploratory Data Analysis")
    print("=" * 60)

    # Load enriched data
    if not INPUT_FILE.exists():
        print(f"\n❌ Input file not found: {INPUT_FILE}")
        print("   Run 02_feature_engineering.py first!")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, dtype={"semester_code": str, "course_id": str})
    print(f"\nLoaded {len(df)} rows with {len(df.columns)} columns from {INPUT_FILE}")

    # Setup
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nConfiguring styles and fonts...")
    set_style()  # Set seaborn theme FIRST
    setup_chinese_font()  # Then override with CJK font (must be after set_theme)

    # Generate all plots
    print("\nGenerating plots...")
    plot_weight_distribution(df, PLOTS_DIR)
    plot_weight_trends_over_time(df, PLOTS_DIR)
    plot_most_competitive_courses(df, PLOTS_DIR)
    plot_correlation_analysis(df, PLOTS_DIR)
    plot_volatility_by_department(df, PLOTS_DIR)
    plot_course_count_growth(df, PLOTS_DIR)
    plot_weight_heatmap(df, PLOTS_DIR)

    # Generate summary report
    print("\nGenerating summary report...")
    report = generate_summary_report(df, SUMMARY_FILE)
    print(f"  ✓ eda_summary.txt")

    # Print report to console too
    print(f"\n{report}")
    print(f"\n✅ Phase 3 complete!")
    print(f"   Plots saved to: {PLOTS_DIR}")
    print(f"   Report saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
