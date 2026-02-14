#!/usr/bin/env python3
"""
Phase 2: Data Enrichment & Feature Engineering
================================================
Takes the raw parsed data from data/all_cutoff_weights.csv and creates
enriched features for ML modeling. Outputs data/features_enriched.csv.

Features created:
  - Temporal: year_roc, year_ad, semester, semester_ordinal
  - Lag features: prev_1_weight, prev_2_weight
  - Rolling averages: avg_weight_3sem, avg_weight_all
  - Trend: weight_trend (linear slope over last 5 semesters)
  - History: semesters_offered, weight_volatility
  - Categorical: is_required, dept_cluster, popularity_tier
  - Enrollment demand (lagged): prev_1_oversub_ratio, prev_2_oversub_ratio,
    avg_oversub_ratio_3sem, prev_1_remaining_spots, demand_trend
  - Instructor: instructor_avg_cutoff, instructor_course_count
  - Time slot: is_prime_time, num_time_slots
  - Domain: domain_category
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "all_cutoff_weights.csv"
ENROLLMENT_FILE = DATA_DIR / "enrollment_data.csv"
OUTPUT_FILE = DATA_DIR / "features_enriched.csv"

# Ordered list of all known semesters for ordinal encoding
SEMESTER_ORDER = [
    "1012", "1021", "1022", "1031", "1032", "1041", "1042",
    "1051", "1052", "1061", "1062", "1071", "1072", "1081", "1082",
    "1091", "1092", "1101", "1102", "1111", "1112", "1121", "1122",
    "1131", "1132", "1141", "1142"
]

# Prime-time slots: weekdays (Mon-Fri = 1-5), periods 01-04 (morning)
PRIME_DAYS = {1, 2, 3, 4, 5}  # Mon–Fri
PRIME_PERIODS = {"01", "02", "03", "04"}  # Morning periods


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features derived from semester_code."""
    df = df.copy()

    # Extract ROC year and semester number
    df["year_roc"] = df["semester_code"].str[:3].astype(int)
    df["semester"] = df["semester_code"].str[3:].astype(int)

    # Convert ROC year to AD year
    df["year_ad"] = df["year_roc"] + 1911

    # Create ordinal index for chronological ordering
    sem_to_ord = {s: i + 1 for i, s in enumerate(SEMESTER_ORDER)}
    df["semester_ordinal"] = df["semester_code"].map(sem_to_ord)

    return df


def compute_course_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite key for identifying unique course offerings.
    
    Uses course_id + section + target_unit to uniquely identify a
    course offering across semesters. This handles cases where the
    same course_id has different sections or targets.
    """
    df = df.copy()
    df["course_key"] = (
        df["course_id"].str.strip() + "|" +
        df["section"].fillna("").str.strip() + "|" +
        df["target_unit"].fillna("").str.strip()
    )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged weight features per course_key.
    
    For each course, look back 1 and 2 semesters to get historical
    cutoff weights. These are powerful predictors (recent history).
    """
    df = df.copy()

    # Sort by course_key and semester_ordinal for correct temporal ordering
    df = df.sort_values(["course_key", "semester_ordinal"])

    # Group by course_key (unique course offering)
    grouped = df.groupby("course_key")

    # Lag features: previous 1 and 2 semesters
    df["prev_1_weight"] = grouped["cutoff_weight"].shift(1)
    df["prev_2_weight"] = grouped["cutoff_weight"].shift(2)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling average and cumulative statistics per course_key.
    
    Includes:
    - avg_weight_3sem: rolling mean of last 3 semesters
    - avg_weight_all: expanding (cumulative) mean up to current row
    - weight_volatility: expanding std dev (how volatile this course is)
    - semesters_offered: cumulative count of how many times offered
    """
    df = df.copy()
    df = df.sort_values(["course_key", "semester_ordinal"])

    grouped = df.groupby("course_key")

    # Rolling 3-semester average (min_periods=1 to handle courses with < 3 history)
    # shift(1) so we only use *past* data, not the current row
    df["avg_weight_3sem"] = grouped["cutoff_weight"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # Expanding (cumulative) mean — all past semesters
    df["avg_weight_all"] = grouped["cutoff_weight"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Weight volatility (std dev of all past weights)
    df["weight_volatility"] = grouped["cutoff_weight"].transform(
        lambda x: x.shift(1).expanding(min_periods=2).std()
    )

    # Number of semesters this course has been offered (cumulative count)
    df["semesters_offered"] = grouped.cumcount() + 1

    return df


def compute_weight_trend(group: pd.Series, window: int = 5) -> pd.Series:
    """
    Compute the linear trend (slope) of cutoff weights over the last `window` semesters.
    
    Uses OLS slope: how many weight points per semester is the course trending?
    Positive = getting more competitive, negative = getting easier.
    """
    result = pd.Series(np.nan, index=group.index)

    for i in range(len(group)):
        # Use shifted data (only past values, not current)
        past_values = group.iloc[max(0, i - window):i]

        if len(past_values) < 2:
            continue

        x = np.arange(len(past_values))
        y = past_values.values

        # Skip if any NaN
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            continue

        slope, _, _, _, _ = scipy_stats.linregress(x[mask], y[mask])
        result.iloc[i] = slope

    return result


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weight trend (slope) feature per course_key."""
    df = df.copy()
    df = df.sort_values(["course_key", "semester_ordinal"])

    print("  Computing weight trends (this may take a moment)...")
    df["weight_trend"] = df.groupby("course_key")["cutoff_weight"].transform(
        compute_weight_trend
    )

    return df


def add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add categorical / derived features.
    
    - is_required: binary (必修=1, 選修=0)
    - dept_cluster: simplified department grouping
    - popularity_tier: quantile-based tier from historical avg weight
    """
    df = df.copy()

    # Binary: is this a required course?
    df["is_required"] = (df["course_type"] == "必修").astype(int)

    # Department clustering: group similar departments
    def cluster_department(dept: str) -> str:
        dept = str(dept).strip()
        if "通識" in dept:
            return "通識"
        elif "體育" in dept:
            return "體育"
        elif "醫學" in dept or "醫學院" in dept:
            return "醫學相關"
        elif "護理" in dept:
            return "護理"
        elif "藥學" in dept:
            return "藥學"
        elif "公衛" in dept or "公共衛生" in dept:
            return "公衛"
        elif "口腔" in dept or "牙" in dept:
            return "口腔醫學"
        elif "管理" in dept:
            return "管理"
        elif "人文" in dept or "人社" in dept:
            return "人文社會"
        elif "資訊" in dept or "電機" in dept or "工程" in dept:
            return "資訊工程"
        elif "營養" in dept or "保健" in dept:
            return "營養保健"
        elif "軍訓" in dept:
            return "軍訓"
        else:
            return "其他"

    df["dept_cluster"] = df["department"].apply(cluster_department)

    # Popularity tier: based on per-course average weight (across all semesters)
    course_avg = df.groupby("course_key")["cutoff_weight"].transform("mean")
    df["popularity_tier"] = pd.qcut(
        course_avg, q=4, labels=["low", "medium", "high", "very_high"],
        duplicates="drop"
    )

    return df


# ── Enrollment / Demand Features ────────────────────────────────────────────


def merge_enrollment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge enrollment data from enrollment_data.csv into the main dataframe.
    Left-join so rows without enrollment data get NaN (semesters 1012–1071).
    """
    if not ENROLLMENT_FILE.exists():
        print("  ⚠️  enrollment_data.csv not found — skipping enrollment features")
        return df

    enroll = pd.read_csv(
        ENROLLMENT_FILE,
        dtype={"semester_code": str, "course_id": str},
    )
    print(f"  Loaded {len(enroll)} enrollment rows")

    # Strip whitespace from join keys
    enroll["target_unit"] = enroll["target_unit"].fillna("").str.strip()
    df["target_unit"] = df["target_unit"].fillna("").str.strip()

    # Select enrollment columns to merge
    enroll_cols = [
        "semester_code", "course_id", "target_unit",
        "total_enrollment", "lottery_capacity", "remaining_spots",
        "oversubscription_ratio", "fill_rate", "is_oversubscribed",
        "instructor", "time_slots", "domain_category",
    ]
    enroll_subset = enroll[enroll_cols].copy()

    # Merge
    df = df.merge(
        enroll_subset,
        on=["semester_code", "course_id", "target_unit"],
        how="left",
    )

    matched = df["total_enrollment"].notna().sum()
    print(f"  Merged enrollment data: {matched}/{len(df)} rows matched")

    return df


def parse_time_slot(slot: str) -> tuple[int, str]:
    """
    Parse a single time slot in DPP format.
    D = day of week (1-7), PP = period (01-0A).
    Returns (day, period_str).
    """
    slot = slot.strip()
    if len(slot) < 2:
        return (0, "00")
    day = int(slot[0])
    period = slot[1:]
    return (day, period)


def add_enrollment_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged demand features per course_key from enrollment data.
    
    Features:
    - prev_1_oversub_ratio: oversubscription ratio from previous semester
    - prev_2_oversub_ratio: from 2 semesters ago
    - avg_oversub_ratio_3sem: rolling 3-semester average  
    - prev_1_remaining_spots: remaining spots from last semester
    - demand_trend: linear slope of oversubscription ratio over last 5 sems
    """
    df = df.copy()
    df = df.sort_values(["course_key", "semester_ordinal"])
    grouped = df.groupby("course_key")

    # Lag features for oversubscription_ratio
    df["prev_1_oversub_ratio"] = grouped["oversubscription_ratio"].shift(1)
    df["prev_2_oversub_ratio"] = grouped["oversubscription_ratio"].shift(2)

    # Rolling 3-semester average of oversubscription ratio
    df["avg_oversub_ratio_3sem"] = grouped["oversubscription_ratio"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # Lag remaining spots
    df["prev_1_remaining_spots"] = grouped["remaining_spots"].shift(1)

    # Demand trend (slope of oversubscription ratio)
    print("  Computing demand trends...")
    df["demand_trend"] = grouped["oversubscription_ratio"].transform(
        lambda group: _compute_trend(group, window=5)
    )

    return df


def _compute_trend(group: pd.Series, window: int = 5) -> pd.Series:
    """Compute linear slope over a rolling window of past values."""
    result = pd.Series(np.nan, index=group.index)
    for i in range(len(group)):
        past_values = group.iloc[max(0, i - window):i]
        if len(past_values) < 2:
            continue
        x = np.arange(len(past_values))
        y = past_values.values
        mask = ~np.isnan(y.astype(float))
        if mask.sum() < 2:
            continue
        slope, _, _, _, _ = scipy_stats.linregress(x[mask], y[mask].astype(float))
        result.iloc[i] = slope
    return result


def add_instructor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add instructor-based features.
    
    - instructor_avg_cutoff: historical average cutoff weight for this instructor
      (computed using only data BEFORE the current row to avoid leakage)
    - instructor_course_count: number of unique courses this instructor has taught
    """
    df = df.copy()
    if "instructor" not in df.columns or df["instructor"].isna().all():
        print("  ⚠️  No instructor data — skipping instructor features")
        df["instructor_avg_cutoff"] = np.nan
        df["instructor_course_count"] = np.nan
        return df

    df = df.sort_values("semester_ordinal")

    # Compute expanding mean of cutoff_weight per instructor
    # Use only past data (shift to avoid leakage)
    instructor_avg = []
    instructor_count = []
    
    # Build lookup: for each semester_ordinal, compute instructor stats from prior data
    semesters = sorted(df["semester_ordinal"].dropna().unique())
    
    # Pre-compute instructor stats for each semester boundary
    inst_stats_cache = {}
    for sem_ord in semesters:
        past = df[df["semester_ordinal"] < sem_ord]
        if len(past) == 0 or "instructor" not in past.columns:
            inst_stats_cache[sem_ord] = ({}, {})
            continue
        
        past_valid = past[past["instructor"].notna() & (past["instructor"] != "")]
        if len(past_valid) == 0:
            inst_stats_cache[sem_ord] = ({}, {})
            continue
        
        avg_by_inst = past_valid.groupby("instructor")["cutoff_weight"].mean().to_dict()
        count_by_inst = past_valid.groupby("instructor")["course_key"].nunique().to_dict()
        inst_stats_cache[sem_ord] = (avg_by_inst, count_by_inst)

    # Apply to each row
    avg_vals = []
    count_vals = []
    for _, row in df.iterrows():
        inst = row.get("instructor", "")
        sem_ord = row.get("semester_ordinal", np.nan)
        
        if pd.isna(sem_ord) or not inst or inst == "":
            avg_vals.append(np.nan)
            count_vals.append(np.nan)
            continue
        
        avg_dict, count_dict = inst_stats_cache.get(sem_ord, ({}, {}))
        avg_vals.append(avg_dict.get(inst, np.nan))
        count_vals.append(count_dict.get(inst, np.nan))

    df["instructor_avg_cutoff"] = avg_vals
    df["instructor_course_count"] = count_vals

    return df


def add_time_slot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-slot-based features.
    
    - is_prime_time: 1 if class is on popular slots (weekday morning)
    - num_time_slots: number of time slots the class meets per week
    """
    df = df.copy()
    if "time_slots" not in df.columns or df["time_slots"].isna().all():
        print("  ⚠️  No time slot data — skipping time slot features")
        df["is_prime_time"] = np.nan
        df["num_time_slots"] = np.nan
        return df

    is_prime = []
    num_slots = []

    for ts in df["time_slots"]:
        if pd.isna(ts) or str(ts).strip() == "" or str(ts).strip() == "nan":
            is_prime.append(np.nan)
            num_slots.append(np.nan)
            continue

        slots = str(ts).split(",")
        n = len(slots)
        num_slots.append(n)

        # Check if ANY slot is prime time
        prime = False
        for slot in slots:
            slot = slot.strip()
            if len(slot) >= 3:
                day, period = parse_time_slot(slot)
                if day in PRIME_DAYS and period in PRIME_PERIODS:
                    prime = True
                    break
        is_prime.append(1 if prime else 0)

    df["is_prime_time"] = is_prime
    df["num_time_slots"] = num_slots

    return df


def add_domain_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain_category feature (領域別).
    Meaningful mainly for 通識 courses; others get 'N/A'.
    """
    df = df.copy()
    if "domain_category" not in df.columns:
        df["domain_category"] = "N/A"
    else:
        df["domain_category"] = df["domain_category"].fillna("N/A").astype(str).str.strip()
        # Clean up empty strings
        df.loc[df["domain_category"] == "", "domain_category"] = "N/A"
        df.loc[df["domain_category"] == "nan", "domain_category"] = "N/A"

    return df


def main():
    print("=" * 60)
    print("Phase 2: Data Enrichment & Feature Engineering")
    print("=" * 60)

    # Load raw parsed data
    if not INPUT_FILE.exists():
        print(f"\n❌ Input file not found: {INPUT_FILE}")
        print("   Run 01_parse_html.py first!")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, dtype={"semester_code": str, "course_id": str})
    print(f"\nLoaded {len(df)} rows from {INPUT_FILE}")

    # Step 1: Temporal features
    print("\n[1/10] Adding temporal features...")
    df = add_temporal_features(df)

    # Step 2: Course key
    print("[2/10] Creating course keys...")
    df = compute_course_key(df)
    print(f"  Unique course keys: {df['course_key'].nunique()}")

    # Step 3: Lag features
    print("[3/10] Computing lag features...")
    df = add_lag_features(df)

    # Step 4: Rolling features
    print("[4/10] Computing rolling statistics...")
    df = add_rolling_features(df)

    # Step 5: Trend features
    print("[5/10] Computing trend features...")
    df = add_trend_features(df)

    # Step 6: Categorical features
    print("[6/10] Adding categorical features...")
    df = add_categorical_features(df)

    # Step 7: Merge enrollment data
    print("[7/10] Merging enrollment data...")
    df = merge_enrollment_data(df)

    # Step 8: Enrollment lag features
    print("[8/10] Computing enrollment lag features...")
    df = add_enrollment_lag_features(df)

    # Step 9: Instructor features
    print("[9/10] Computing instructor features...")
    df = add_instructor_features(df)

    # Step 10: Time slot & domain features
    print("[10/10] Adding time slot & domain features...")
    df = add_time_slot_features(df)
    df = add_domain_feature(df)

    # ── Drop intermediate enrollment columns ──────────────────────────────
    # Keep only the lagged/derived features, not same-semester raw data
    # (to avoid data leakage when training)
    cols_to_drop = [
        "total_enrollment", "lottery_capacity", "remaining_spots",
        "fill_rate", "is_oversubscribed",
        "instructor", "time_slots",
    ]
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Save enriched data
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Print summary
    print(f"\n{'─' * 60}")
    print(f"Feature Summary")
    print(f"{'─' * 60}")
    print(f"  Total rows:           {len(df)}")
    print(f"  Total columns:        {len(df.columns)}")
    print(f"  Unique course keys:   {df['course_key'].nunique()}")
    print(f"  Semester range:       {df['semester_code'].min()} → {df['semester_code'].max()}")

    print(f"\n  Column list:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        print(f"    {col:<30} {non_null:>5} non-null ({pct:.1f}%)")

    print(f"\n  Feature statistics (cutoff_weight):")
    print(f"    Mean:   {df['cutoff_weight'].mean():.2f}")
    print(f"    Median: {df['cutoff_weight'].median():.1f}")
    print(f"    Std:    {df['cutoff_weight'].std():.2f}")
    print(f"    Min:    {df['cutoff_weight'].min():.0f}")
    print(f"    Max:    {df['cutoff_weight'].max():.0f}")

    print(f"\n  Department clusters:")
    for cluster, count in df["dept_cluster"].value_counts().items():
        print(f"    {cluster:<15} {count:>5} rows")

    # New features summary
    print(f"\n  Enrollment features coverage:")
    for col in ["prev_1_oversub_ratio", "prev_2_oversub_ratio",
                 "avg_oversub_ratio_3sem", "prev_1_remaining_spots",
                 "demand_trend", "instructor_avg_cutoff",
                 "instructor_course_count", "is_prime_time",
                 "num_time_slots", "domain_category"]:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"    {col:<30} {non_null:>5} non-null ({pct:.1f}%)")

    print(f"\n  Output saved: {OUTPUT_FILE}")
    print(f"\n✅ Phase 2 complete!")

    return df


if __name__ == "__main__":
    main()
