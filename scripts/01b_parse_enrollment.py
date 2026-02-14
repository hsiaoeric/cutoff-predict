#!/usr/bin/env python3
"""
Phase 1b: Parse Enrollment / Demand Data
==========================================
Reads the enrollment XLS file (TKE3100) and produces a clean CSV with
demand-side data (enrollment counts, capacity, oversubscription ratio, etc.).

This data will be merged with the cutoff weight data in 02_feature_engineering.py.

Input:  TKE3100 (6).xls  (1,996 rows, 15 semesters: 1072–1142)
Output: data/enrollment_data.csv
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
XLS_FILE = PROJECT_ROOT / "TKE3100 (6).xls"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "enrollment_data.csv"


def main():
    print("=" * 60)
    print("Phase 1b: Parse Enrollment Data (TKE3100)")
    print("=" * 60)

    if not XLS_FILE.exists():
        print(f"\n❌ XLS file not found: {XLS_FILE}")
        sys.exit(1)

    # ── Read XLS ──────────────────────────────────────────────────────────
    df = pd.read_excel(XLS_FILE, header=0)
    print(f"\nLoaded {len(df)} rows, {len(df.columns)} columns from XLS")

    # ── Rename columns to English ─────────────────────────────────────────
    rename_map = {
        "學年期": "semester_code",
        "開課單位": "enroll_department",
        "課號": "course_id",
        "課程班別": "section",
        "課程名稱": "enroll_course_name",
        "修課單位": "target_unit",
        "領域別": "domain_category",
        "年級": "enroll_grade_level",
        "學分": "enroll_credits",
        "學期別": "semester_type",
        "選別": "enroll_course_type",
        "主授教師": "instructor",
        "上課時間": "time_slots",
        "上課地點": "classroom",
        "校內選課人數": "internal_enrollment",
        "外加名額選課人數": "extra_quota_enrollment",
        "總選課人數": "total_enrollment",
        "人數上限": "capacity_limit",
        "抽籤上限": "lottery_capacity",
        "開課標準": "min_enrollment",
        "餘額": "remaining_spots",
    }
    df = df.rename(columns=rename_map)

    # ── Clean key columns ─────────────────────────────────────────────────
    df["semester_code"] = df["semester_code"].astype(str).str.strip()
    df["course_id"] = df["course_id"].astype(str).str.strip()
    df["section"] = df["section"].fillna("").astype(str).str.strip()
    df["target_unit"] = df["target_unit"].fillna("").astype(str).str.strip()
    df["instructor"] = df["instructor"].fillna("").astype(str).str.strip()
    df["time_slots"] = df["time_slots"].fillna("").astype(str).str.strip()
    df["domain_category"] = df["domain_category"].fillna("N/A").astype(str).str.strip()

    # ── Filter out rows where capacity_limit = 0 ─────────────────────────
    zero_cap = (df["capacity_limit"] == 0).sum()
    print(f"  Filtering out {zero_cap} rows with capacity_limit=0")
    df = df[df["capacity_limit"] > 0].copy()

    # ── Compute oversubscription ratio ────────────────────────────────────
    # oversubscription_ratio = total_enrollment / lottery_capacity
    # When lottery_capacity = 0, use capacity_limit instead
    effective_capacity = df["lottery_capacity"].where(
        df["lottery_capacity"] > 0, df["capacity_limit"]
    )
    df["oversubscription_ratio"] = df["total_enrollment"] / effective_capacity

    # ── Compute fill rate ─────────────────────────────────────────────────
    # fill_rate = total_enrollment / capacity_limit
    df["fill_rate"] = df["total_enrollment"] / df["capacity_limit"]

    # ── Compute is_oversubscribed flag ────────────────────────────────────
    df["is_oversubscribed"] = (df["remaining_spots"] == 0).astype(int)

    # ── Deduplicate on (semester_code, course_id, target_unit) ────────────
    # Some courses appear multiple times for different sections of PE, etc.
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["semester_code", "course_id", "target_unit"],
        keep="first",
    )
    print(f"  Deduplicated: {before_dedup} → {len(df)} rows (removed {before_dedup - len(df)} dupes)")

    # ── Select columns to output ──────────────────────────────────────────
    output_cols = [
        "semester_code", "course_id", "section", "target_unit",
        "domain_category", "instructor", "time_slots",
        "internal_enrollment", "total_enrollment",
        "capacity_limit", "lottery_capacity",
        "remaining_spots", "min_enrollment",
        "oversubscription_ratio", "fill_rate", "is_oversubscribed",
    ]
    df_out = df[output_cols].copy()

    # ── Save ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Summary")
    print(f"{'─' * 60}")
    print(f"  Total rows:             {len(df_out)}")
    print(f"  Unique course_ids:      {df_out['course_id'].nunique()}")
    print(f"  Unique instructors:     {df_out['instructor'].nunique()}")
    print(f"  Semester range:         {df_out['semester_code'].min()} → {df_out['semester_code'].max()}")
    print(f"  Domain categories:      {df_out['domain_category'].nunique()}")
    print(f"  Oversubscription ratio: mean={df_out['oversubscription_ratio'].mean():.2f}, "
          f"max={df_out['oversubscription_ratio'].max():.2f}")
    print(f"  Oversubscribed courses: {df_out['is_oversubscribed'].sum()} "
          f"({df_out['is_oversubscribed'].mean()*100:.1f}%)")

    print(f"\n  Per-semester row counts:")
    for sem, count in df_out.groupby("semester_code").size().items():
        print(f"    {sem}: {count:>4} rows")

    print(f"\n  Output saved: {OUTPUT_FILE}")
    print(f"\n✅ Phase 1b complete!")

    return df_out


if __name__ == "__main__":
    main()
