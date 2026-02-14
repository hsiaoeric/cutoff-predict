#!/usr/bin/env python3
"""
Phase 1: Parse & Extract — TMU Cutoff Weight HTML Parser
=========================================================
Parses all cached HTML files from availability_results/raw/ and extracts
structured course cutoff weight data into data/all_cutoff_weights.csv.

Each HTML file is an ASP.NET AJAX partial response containing a data table
with 10 columns of course registration lottery cutoff weights.
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "availability_results" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "all_cutoff_weights.csv"

# Column definitions (Chinese → English mapping)
COLUMNS_ZH = [
    "開課學期", "開課單位", "課號", "課程班別", "課程名稱",
    "修課單位", "開課年級", "選別", "學分", "志願權重"
]
COLUMNS_EN = [
    "semester_code", "department", "course_id", "section", "course_name",
    "target_unit", "grade_level", "course_type", "credits", "cutoff_weight"
]


def parse_html_file(filepath: Path) -> list[dict]:
    """
    Parse a single HTML file and extract all course rows.
    
    The HTML is an ASP.NET AJAX partial update response. The data table
    uses alternating row classes: tdGrayLight and tdWhite.
    Files with no data contain '查無符合資料!!' instead.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for "no data" marker
    if "查無符合資料" in content:
        return []

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(content, "lxml")

    # Find all data rows (alternating gray/white classes)
    data_rows = soup.find_all("tr", class_=re.compile(r"tdGrayLight|tdWhite"))

    rows = []
    for tr in data_rows:
        cells = tr.find_all("td")
        if len(cells) != 10:
            continue  # Skip malformed rows

        values = [cell.get_text(strip=True) for cell in cells]
        row = dict(zip(COLUMNS_EN, values))
        rows.append(row)

    return rows


def main():
    print("=" * 60)
    print("Phase 1: TMU Cutoff Weight HTML Parser")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all HTML files (exclude _init.html)
    html_files = sorted(RAW_DIR.glob("*.html"))
    html_files = [f for f in html_files if f.stem != "_init"]

    print(f"\nFound {len(html_files)} HTML files in {RAW_DIR}")

    all_rows = []
    stats = {"parsed": 0, "no_data": 0, "total_rows": 0}

    for filepath in html_files:
        semester_code = filepath.stem
        rows = parse_html_file(filepath)

        if rows:
            all_rows.extend(rows)
            stats["parsed"] += 1
            stats["total_rows"] += len(rows)
            print(f"  ✓ {semester_code}: {len(rows):>4} courses")
        else:
            stats["no_data"] += 1
            print(f"  ✗ {semester_code}: no data")

    # Create DataFrame
    df = pd.DataFrame(all_rows, columns=COLUMNS_EN)

    # Data type conversions
    df["semester_code"] = df["semester_code"].astype(str)
    df["course_id"] = df["course_id"].astype(str)
    df["section"] = df["section"].replace("", pd.NA).fillna("")
    df["grade_level"] = pd.to_numeric(df["grade_level"], errors="coerce")
    df["credits"] = pd.to_numeric(df["credits"], errors="coerce")
    df["cutoff_weight"] = pd.to_numeric(df["cutoff_weight"], errors="coerce")

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Print summary
    print(f"\n{'─' * 60}")
    print(f"Summary")
    print(f"{'─' * 60}")
    print(f"  Files with data:    {stats['parsed']}")
    print(f"  Files without data: {stats['no_data']}")
    print(f"  Total rows:         {stats['total_rows']}")
    print(f"  Unique courses:     {df['course_id'].nunique()}")
    print(f"  Semester range:     {df['semester_code'].min()} → {df['semester_code'].max()}")
    print(f"  Weight range:       {df['cutoff_weight'].min():.0f} → {df['cutoff_weight'].max():.0f}")
    print(f"  Output saved:       {OUTPUT_FILE}")

    # Validation: check semester code consistency
    print(f"\n{'─' * 60}")
    print(f"Per-semester row counts")
    print(f"{'─' * 60}")
    sem_counts = df.groupby("semester_code").size()
    for sem, count in sem_counts.items():
        print(f"  {sem}: {count:>4} rows")

    print(f"\n✅ Phase 1 complete!")
    return df


if __name__ == "__main__":
    main()
