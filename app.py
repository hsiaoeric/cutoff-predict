#!/usr/bin/env python3
"""
TMU 選課志願權重預測 — Streamlit Web App
==========================================
Predicts course registration lottery cutoff weights for TMU students.

Usage:
    export DYLD_LIBRARY_PATH="$HOME/.nix-profile/lib:$DYLD_LIBRARY_PATH"
    uv run streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats

# ─── Constants ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

SEMESTER_ORDER = [
    "1012", "1021", "1022", "1031", "1032", "1041", "1042",
    "1051", "1052", "1061", "1062", "1071", "1072", "1081", "1082",
    "1091", "1092", "1101", "1102", "1111", "1112", "1121", "1122",
    "1131", "1132", "1141", "1142",
]

# Semesters available for prediction: backtest (have actuals) + future
BACKTEST_SEMESTERS = ["1142", "1141"]  # test set — we have actual results
FUTURE_SEMESTERS = ["1151", "1152", "1161", "1162"]
ALL_PRED_SEMESTERS = BACKTEST_SEMESTERS + FUTURE_SEMESTERS

NUMERIC_FEATURES = [
    "prev_1_weight", "prev_2_weight",
    "avg_weight_3sem", "avg_weight_all",
    "weight_trend", "weight_volatility",
    "semesters_offered", "credits", "grade_level",
    "semester", "semester_ordinal",
    # Enrollment demand features
    "oversubscription_ratio",
    "prev_1_oversub_ratio", "prev_2_oversub_ratio",
    "avg_oversub_ratio_3sem", "prev_1_remaining_spots",
    "demand_trend",
    # Instructor features
    "instructor_avg_cutoff", "instructor_course_count",
    # Time slot features
    "is_prime_time", "num_time_slots",
]

CATEGORICAL_FEATURES = ["dept_cluster", "is_required", "popularity_tier", "domain_category"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TIER_MAP = {"low": 0, "medium": 1, "high": 2, "very_high": 3}

DEPT_CLUSTERS = [
    "通識", "體育", "其他", "醫學相關", "藥學",
    "管理", "營養保健", "口腔醫學", "護理", "公衛",
]

# Prime-time slots for time slot feature
PRIME_DAYS = {1, 2, 3, 4, 5}
PRIME_PERIODS = {"01", "02", "03", "04"}

MODEL_MAE = 7.50  # Two-stage model test MAE


# ─── Semester helpers ─────────────────────────────────────────────────────────

def semester_label(code: str) -> str:
    """Convert semester code like '1142' to human-readable label."""
    year_roc = int(code[:3])
    sem = code[3]
    year_ad = year_roc + 1911
    sem_name = "上學期" if sem == "1" else "下學期"
    base = f"{year_roc} 學年度 {sem_name} ({year_ad}-{year_ad + 1})"
    if code in BACKTEST_SEMESTERS:
        return f"🔍 {base} [驗證]"
    return base


def is_backtest(code: str) -> bool:
    """Check if a semester code is a backtest (has actual data)."""
    return code in BACKTEST_SEMESTERS


def semester_ordinal(code: str) -> int:
    """Get ordinal index for a semester code."""
    if code in SEMESTER_ORDER:
        return SEMESTER_ORDER.index(code) + 1
    # For future semesters, calculate offset after last known
    last_ordinal = len(SEMESTER_ORDER)
    # Parse year and semester
    year = int(code[:3])
    sem = int(code[3])
    last_year = int(SEMESTER_ORDER[-1][:3])
    last_sem = int(SEMESTER_ORDER[-1][3])
    # Each year has 2 semesters
    diff = (year - last_year) * 2 + (sem - last_sem)
    return last_ordinal + diff


def semester_number(code: str) -> int:
    """Extract semester number (1 or 2) from code."""
    return int(code[3])


# ─── Data & Model Loading ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the enriched feature dataset."""
    df = pd.read_csv(
        DATA_DIR / "features_enriched.csv",
        dtype={"semester_code": str},
    )
    return df


@st.cache_data(show_spinner=False)
def load_enrollment_data() -> pd.DataFrame | None:
    """Load enrollment data for demand features."""
    path = DATA_DIR / "enrollment_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"semester_code": str, "course_id": str})


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the two-stage model (classifier + regressor)."""
    model_data = joblib.load(MODEL_DIR / "two_stage.joblib")
    return model_data["classifier"], model_data["regressor"]


# ─── Prediction Engine ───────────────────────────────────────────────────────

def _parse_time_slot(slot: str) -> tuple[int, str]:
    """Parse a single DPP time slot. Returns (day, period)."""
    slot = slot.strip()
    if len(slot) < 2:
        return (0, "00")
    return (int(slot[0]), slot[1:])


def build_prediction_features(df: pd.DataFrame, target_semester: str) -> pd.DataFrame:
    """
    Build feature rows for ALL courses for a target semester.
    For backtest semesters, only uses data BEFORE the target to avoid leakage.
    For future semesters, uses all available data.
    """
    backtest = is_backtest(target_semester)
    target_ord = semester_ordinal(target_semester)

    # For backtest: filter data strictly before target semester
    if backtest:
        train_df = df[df["semester_ordinal"] < target_ord].copy()
    else:
        train_df = df.copy()

    # Get the most recent record for each course (from allowed data)
    latest_records = (
        train_df.sort_values("semester_ordinal")
        .groupby("course_key")
        .last()
        .reset_index()
    )

    # For backtest, also prepare actual weights lookup
    actual_weights = {}
    if backtest:
        actual_rows = df[df["semester_code"] == target_semester]
        actual_weights = dict(
            zip(actual_rows["course_key"], actual_rows["cutoff_weight"])
        )

    # Load enrollment data for demand features
    enroll_df = load_enrollment_data()
    enroll_lookup = {}
    if enroll_df is not None:
        enroll_df["target_unit"] = enroll_df["target_unit"].fillna("").str.strip()
        for _, er in enroll_df.iterrows():
            key = (er["semester_code"], er["course_id"], er["target_unit"])
            enroll_lookup[key] = er

    # Build instructor/time stats from past data (for instructor_avg_cutoff etc.)
    inst_avg_map = {}
    inst_count_map = {}
    if enroll_df is not None:
        # Merge enrollment with training cutoff data for instructor stats
        train_with_enroll = train_df.merge(
            enroll_df[["semester_code", "course_id", "target_unit", "instructor"]],
            on=["semester_code", "course_id", "target_unit"],
            how="left",
        )
        valid = train_with_enroll[
            train_with_enroll["instructor"].notna()
            & (train_with_enroll["instructor"] != "")
        ]
        if len(valid) > 0:
            inst_avg_map = valid.groupby("instructor")["cutoff_weight"].mean().to_dict()
            inst_count_map = valid.groupby("instructor")["course_key"].nunique().to_dict()

    rows = []
    for _, course in latest_records.iterrows():
        ck = course["course_key"]
        history = (
            train_df[train_df["course_key"] == ck]
            .sort_values("semester_ordinal")
        )

        weights = history["cutoff_weight"].values
        n_semesters = len(weights)

        # Lag features
        prev_1 = weights[-1] if n_semesters >= 1 else np.nan
        prev_2 = weights[-2] if n_semesters >= 2 else np.nan

        # Rolling averages
        last_3 = weights[-3:] if n_semesters >= 3 else weights
        avg_3sem = float(np.mean(last_3)) if len(last_3) > 0 else np.nan
        avg_all = float(np.mean(weights)) if n_semesters > 0 else np.nan

        # Trend (slope of last 5 or fewer)
        last_5 = weights[-5:] if n_semesters >= 5 else weights
        if len(last_5) >= 2:
            x = np.arange(len(last_5))
            slope = float(np.polyfit(x, last_5, 1)[0])
        else:
            slope = np.nan

        # Volatility
        volatility = float(np.std(weights)) if n_semesters >= 2 else np.nan

        # ── Enrollment demand features (lagged) ──
        course_id = course["course_id"]
        target_unit = str(course.get("target_unit", "")).strip()

        # Get enrollment history for this course from past semesters
        enroll_history = []
        for _, h_row in history.iterrows():
            ekey = (h_row["semester_code"], h_row["course_id"], target_unit)
            if ekey in enroll_lookup:
                er = enroll_lookup[ekey]
                enroll_history.append(er)

        # Oversubscription ratio history
        oversub_vals = [
            float(e["oversubscription_ratio"])
            for e in enroll_history
            if pd.notna(e.get("oversubscription_ratio"))
        ]
        n_oversub = len(oversub_vals)
        prev_1_oversub = oversub_vals[-1] if n_oversub >= 1 else np.nan
        prev_2_oversub = oversub_vals[-2] if n_oversub >= 2 else np.nan
        avg_oversub_3 = float(np.mean(oversub_vals[-3:])) if n_oversub >= 1 else np.nan

        # Remaining spots
        remaining_vals = [
            float(e["remaining_spots"])
            for e in enroll_history
            if pd.notna(e.get("remaining_spots"))
        ]
        prev_1_remaining = remaining_vals[-1] if len(remaining_vals) >= 1 else np.nan

        # Demand trend
        demand_trend_val = np.nan
        if n_oversub >= 2:
            recent = oversub_vals[-5:] if n_oversub >= 5 else oversub_vals
            if len(recent) >= 2:
                x = np.arange(len(recent))
                demand_trend_val = float(np.polyfit(x, recent, 1)[0])

        # ── Instructor features ──
        instructor = ""
        if enroll_history:
            instructor = str(enroll_history[-1].get("instructor", ""))
        inst_avg = inst_avg_map.get(instructor, np.nan) if instructor else np.nan
        inst_count = inst_count_map.get(instructor, np.nan) if instructor else np.nan

        # ── Time slot features ──
        ts = ""
        if enroll_history:
            ts = str(enroll_history[-1].get("time_slots", ""))

        if ts and ts != "nan" and ts.strip():
            slots = ts.split(",")
            n_slots = len(slots)
            prime = 0
            for s in slots:
                s = s.strip()
                if len(s) >= 3:
                    day, period = _parse_time_slot(s)
                    if day in PRIME_DAYS and period in PRIME_PERIODS:
                        prime = 1
                        break
        else:
            n_slots = np.nan
            prime = np.nan

        # ── Domain category ──
        domain = "N/A"
        if enroll_history:
            d = str(enroll_history[-1].get("domain_category", "N/A"))
            domain = d if d and d != "nan" else "N/A"

        row = {
            # Identification
            "course_key": ck,
            "course_name": course["course_name"],
            "course_id": course_id,
            "section": course.get("section", ""),
            "department": course.get("department", ""),
            "target_unit": target_unit,
            "course_type": course.get("course_type", ""),
            # Model features (original)
            "prev_1_weight": prev_1,
            "prev_2_weight": prev_2,
            "avg_weight_3sem": avg_3sem,
            "avg_weight_all": avg_all,
            "weight_trend": slope,
            "weight_volatility": volatility,
            "semesters_offered": n_semesters + 1,
            "credits": course["credits"],
            "grade_level": course["grade_level"],
            "semester": semester_number(target_semester),
            "semester_ordinal": semester_ordinal(target_semester),
            "dept_cluster": course["dept_cluster"],
            "is_required": course["is_required"],
            "popularity_tier": course["popularity_tier"],
            # Enrollment demand features
            "oversubscription_ratio": prev_1_oversub,  # best estimate for upcoming semester
            "prev_1_oversub_ratio": prev_1_oversub,
            "prev_2_oversub_ratio": prev_2_oversub,
            "avg_oversub_ratio_3sem": avg_oversub_3,
            "prev_1_remaining_spots": prev_1_remaining,
            "demand_trend": demand_trend_val,
            # Instructor features
            "instructor_avg_cutoff": inst_avg,
            "instructor_course_count": inst_count,
            # Time slot features
            "is_prime_time": prime,
            "num_time_slots": n_slots,
            # Domain category
            "domain_category": domain,
        }

        # Attach actual weight for backtest
        if backtest:
            row["actual_weight"] = actual_weights.get(ck, np.nan)

        rows.append(row)

    result = pd.DataFrame(rows)

    # For backtest: keep only courses that actually existed in the target semester
    if backtest:
        result = result[result["actual_weight"].notna()].copy()

    return result


def prepare_model_features(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Encode features to match training format.

    Must set the exact same category dtype values that the trained model
    expects, otherwise LightGBM raises
    'train and valid dataset categorical_feature do not match'.
    """
    # Retrieve the category lists the model was trained with
    clf, _reg = load_model()
    trained_cats = clf.booster_.pandas_categorical
    # trained_cats is a list of lists, one per categorical feature
    # Order matches the order categoricals appear in the feature columns:
    #   [dept_cluster cats, is_required cats, domain_category cats]

    X = pred_df[ALL_FEATURES].copy()
    X["popularity_tier"] = X["popularity_tier"].map(TIER_MAP)

    # Set exact trained categories — values not in the list become NaN
    # (LightGBM handles NaN natively as missing)
    X["dept_cluster"] = pd.Categorical(X["dept_cluster"], categories=trained_cats[0])
    X["is_required"] = pd.Categorical(X["is_required"], categories=trained_cats[1])
    X["domain_category"] = pd.Categorical(X["domain_category"], categories=trained_cats[2])
    return X


def predict_weights(pred_df: pd.DataFrame) -> np.ndarray:
    """Run two-stage prediction."""
    clf, reg = load_model()
    X = prepare_model_features(pred_df)

    p_nonzero = clf.predict_proba(X)[:, 1]
    pred_weight = reg.predict(X)
    final = np.where(p_nonzero > 0.5, pred_weight, 0.0)
    final = np.clip(final, 0, None)
    return np.round(final).astype(int)


# ─── Styling & UI Helpers ────────────────────────────────────────────────────

def inject_custom_css():
    """Inject premium custom CSS."""
    st.markdown("""
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

    .stApp {
        font-family: 'Noto Sans TC', 'Inter', sans-serif;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.4rem;
    }
    .metric-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
    }
    .metric-card.orange {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
        box-shadow: 0 8px 32px rgba(242, 153, 74, 0.3);
    }
    .metric-card.red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        box-shadow: 0 8px 32px rgba(235, 51, 73, 0.3);
    }
    .metric-card.blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
    }

    /* ── Weight badge ── */
    .weight-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
    }
    .weight-low { background: #d4edda; color: #155724; }
    .weight-med { background: #fff3cd; color: #856404; }
    .weight-high { background: #f8d7da; color: #721c24; }
    .weight-vhigh { background: #721c24; color: white; }

    /* ── Course detail card ── */
    .detail-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }

    /* ── Recommendation box ── */
    .reco-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        font-size: 1rem;
        color: #5a3e28;
        border-left: 4px solid #F2994A;
    }

    /* ── Trend arrows ── */
    .trend-up { color: #e74c3c; font-weight: 700; }
    .trend-down { color: #27ae60; font-weight: 700; }
    .trend-flat { color: #7f8c8d; font-weight: 700; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }

    /* ── Section dividers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .section-header h3 {
        margin: 0;
        color: #333;
    }

    /* ── Info banner ── */
    .info-banner {
        background: linear-gradient(90deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Hide default streamlit footer ── */
    footer { visibility: hidden; }

    /* ── About page cards ── */
    .about-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-left: 4px solid #667eea;
    }
    .about-card h4 { margin-top: 0; color: #333; }
    .about-card p { color: #555; line-height: 1.7; }

    /* ── Backtest banner ── */
    .backtest-banner {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
        color: white;
        font-weight: 500;
    }
    .backtest-banner strong { color: white; }

    /* ── Accuracy pill ── */
    .accuracy-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .accuracy-good { background: #d4edda; color: #155724; }
    .accuracy-ok { background: #fff3cd; color: #856404; }
    .accuracy-bad { background: #f8d7da; color: #721c24; }
    </style>
    """, unsafe_allow_html=True)


def weight_color_class(w: float) -> str:
    """Return CSS class based on weight level."""
    if w <= 0:
        return "weight-low"
    elif w <= 15:
        return "weight-low"
    elif w <= 35:
        return "weight-med"
    elif w <= 60:
        return "weight-high"
    else:
        return "weight-vhigh"


def weight_emoji(w: float) -> str:
    """Return emoji indicator for weight level."""
    if w <= 0:
        return "✅"
    elif w <= 15:
        return "✅"
    elif w <= 35:
        return "⚠️"
    elif w <= 60:
        return "🔥"
    else:
        return "🔥🔥"


def trend_indicator(trend: float) -> str:
    """Return trend arrow HTML."""
    if pd.isna(trend):
        return '<span class="trend-flat">—</span>'
    elif trend > 2:
        return '<span class="trend-up">↑ 上升</span>'
    elif trend < -2:
        return '<span class="trend-down">↓ 下降</span>'
    else:
        return '<span class="trend-flat">→ 持平</span>'


def trend_text(trend: float) -> str:
    """Return plain text trend indicator."""
    if pd.isna(trend):
        return "—"
    elif trend > 2:
        return "↑ 上升"
    elif trend < -2:
        return "↓ 下降"
    else:
        return "→ 持平"


def recommendation_text(pred_weight: int, trend: float, volatility: float) -> str:
    """Generate Chinese recommendation text."""
    parts = []

    if pred_weight == 0:
        return "✅ 此課程預計不需抽籤，無需配置權重即可選上。"

    if pred_weight >= 60:
        parts.append(f"🔥 此課程競爭非常激烈，建議至少配 **{pred_weight + 10}** 點權重。")
    elif pred_weight >= 35:
        parts.append(f"⚠️ 此課程有中高度競爭，建議配 **{pred_weight + 5}~{pred_weight + 15}** 點權重。")
    elif pred_weight >= 15:
        parts.append(f"📊 此課程有一定競爭，建議配 **{pred_weight}~{pred_weight + 10}** 點權重。")
    else:
        parts.append(f"✅ 此課程競爭較低，配 **{pred_weight}~{pred_weight + 5}** 點權重應足夠。")

    if not pd.isna(trend) and trend > 3:
        parts.append("📈 近期權重持續上升，可能需要額外多配點數。")
    elif not pd.isna(trend) and trend < -3:
        parts.append("📉 近期權重呈下降趨勢，競爭有所緩和。")

    if not pd.isna(volatility) and volatility > 15:
        parts.append("⚡ 此課程權重波動較大，預測不確定性較高，請預留更多彈性空間。")

    return " ".join(parts)


# ─── Page 1: Weight Prediction ───────────────────────────────────────────────

def page_prediction():
    """Main prediction page."""
    df = load_data()

    # Header
    st.markdown("""
    <div class="info-banner">
        <h2 style="margin:0; color: #333;">🔮 志願權重預測</h2>
        <p style="margin:0.3rem 0 0; color: #555;">
            輸入課程名稱或課號，查看下學期預測的選課權重門檻
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: Filters ──
    with st.sidebar:
        st.markdown("### 🎯 篩選條件")

        target_sem = st.selectbox(
            "📅 預測學期",
            options=ALL_PRED_SEMESTERS,
            format_func=semester_label,
            index=0,
        )

        search_query = st.text_input(
            "🔍 搜尋課程（名稱或課號）",
            placeholder="例：情緒管理、00000069",
        )

        dept_filter = st.multiselect(
            "🏢 科系類別篩選",
            options=DEPT_CLUSTERS,
            default=[],
            placeholder="選擇科系類別（可多選）",
        )

        weight_range = st.slider(
            "⚖️ 預測權重範圍",
            min_value=0,
            max_value=130,
            value=(0, 130),
        )

        only_competitive = st.checkbox("🔥 只顯示競爭課程（權重 > 0）", value=False)

    backtest = is_backtest(target_sem)

    # ── Backtest banner ──
    if backtest:
        st.markdown(f"""
        <div class="backtest-banner">
            🔍 <strong>驗證模式</strong>：你正在查看 {semester_label(target_sem).replace('🔍 ', '').replace(' [驗證]', '')} 的預測結果。
            此學期已有實際資料，可以對比預測 vs. 實際的準確度。
        </div>
        """, unsafe_allow_html=True)

    # ── Build predictions ──
    with st.spinner("🧠 正在計算預測結果..."):
        pred_df = build_prediction_features(df, target_sem)
        pred_df["predicted_weight"] = predict_weights(pred_df)

    # ── Apply filters ──
    filtered = pred_df.copy()

    if search_query:
        q = search_query.strip().lower()
        filtered = filtered[
            filtered["course_name"].str.lower().str.contains(q, na=False)
            | filtered["course_id"].str.lower().str.contains(q, na=False)
        ]

    if dept_filter:
        filtered = filtered[filtered["dept_cluster"].isin(dept_filter)]

    filtered = filtered[
        (filtered["predicted_weight"] >= weight_range[0])
        & (filtered["predicted_weight"] <= weight_range[1])
    ]

    if only_competitive:
        filtered = filtered[filtered["predicted_weight"] > 0]

    # ── Summary metrics ──
    if backtest and len(filtered) > 0:
        # Backtest accuracy metrics
        mae = (filtered["predicted_weight"] - filtered["actual_weight"]).abs().mean()
        within_5 = ((filtered["predicted_weight"] - filtered["actual_weight"]).abs() <= 5).mean() * 100
        within_10 = ((filtered["predicted_weight"] - filtered["actual_weight"]).abs() <= 10).mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-value">{len(filtered)}</div>
                <div class="metric-label">驗證課程數</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card {'green' if mae < 8 else 'orange'}">
                <div class="metric-value">{mae:.1f}</div>
                <div class="metric-label">MAE（平均誤差）</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card {'green' if within_5 > 50 else 'orange'}">
                <div class="metric-value">{within_5:.0f}%</div>
                <div class="metric-label">誤差 ≤5 點</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card {'green' if within_10 > 70 else 'orange'}">
                <div class="metric-value">{within_10:.0f}%</div>
                <div class="metric-label">誤差 ≤10 點</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-value">{len(filtered)}</div>
                <div class="metric-label">符合條件課程</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            avg_w = filtered["predicted_weight"].mean() if len(filtered) > 0 else 0
            st.markdown(f"""
            <div class="metric-card orange">
                <div class="metric-value">{avg_w:.0f}</div>
                <div class="metric-label">平均預測權重</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            competitive = (filtered["predicted_weight"] > 30).sum() if len(filtered) > 0 else 0
            st.markdown(f"""
            <div class="metric-card red">
                <div class="metric-value">{competitive}</div>
                <div class="metric-label">高競爭課程 (&gt;30)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            no_lottery = (filtered["predicted_weight"] == 0).sum() if len(filtered) > 0 else 0
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-value">{no_lottery}</div>
                <div class="metric-label">免抽籤課程</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results table ──
    if len(filtered) == 0:
        st.info("📭 沒有符合條件的課程，請嘗試調整篩選條件。")
        return

    # Prepare display df — base columns
    base_cols = [
        "course_name", "course_id", "section", "dept_cluster",
        "credits", "predicted_weight", "prev_1_weight",
        "weight_trend", "course_key",
    ]
    if backtest:
        base_cols.append("actual_weight")

    display_df = filtered[base_cols].copy()

    display_df["信心區間"] = display_df["predicted_weight"].apply(
        lambda w: f"±{MODEL_MAE:.0f}" if w > 0 else "—"
    )
    display_df["趨勢"] = display_df["weight_trend"].apply(trend_text)
    display_df["競爭度"] = display_df["predicted_weight"].apply(weight_emoji)

    rename_map = {
        "course_name": "課程名稱",
        "course_id": "課號",
        "section": "班別",
        "dept_cluster": "科系類別",
        "credits": "學分",
        "predicted_weight": "預測權重",
        "prev_1_weight": "上學期權重",
    }
    if backtest:
        rename_map["actual_weight"] = "實際權重"
        # Compute prediction error
        display_df["誤差"] = (
            display_df["predicted_weight"] - display_df["actual_weight"]
        ).round(0).astype(int)

    display_df = display_df.rename(columns=rename_map)

    # Sort by predicted weight descending
    display_df = display_df.sort_values("預測權重", ascending=False).reset_index(drop=True)

    # Format for display
    if backtest:
        show_cols = [
            "競爭度", "課程名稱", "課號", "班別", "科系類別",
            "學分", "預測權重", "實際權重", "誤差", "趨勢",
        ]
    else:
        show_cols = [
            "競爭度", "課程名稱", "課號", "班別", "科系類別",
            "學分", "預測權重", "上學期權重", "信心區間", "趨勢",
        ]

    col_config = {
        "預測權重": st.column_config.NumberColumn(
            "預測權重 🎯",
            help="模型預測的選課門檻權重",
            format="%d",
        ),
        "學分": st.column_config.NumberColumn(
            "學分",
            format="%.1f",
        ),
    }
    if backtest:
        col_config["實際權重"] = st.column_config.NumberColumn(
            "實際權重 ✅",
            help="該學期實際的權重門檻",
            format="%.0f",
        )
        col_config["誤差"] = st.column_config.NumberColumn(
            "誤差",
            help="預測 − 實際（正=高估，負=低估）",
            format="%+d",
        )
    else:
        col_config["上學期權重"] = st.column_config.NumberColumn(
            "上學期權重",
            help="最近一個學期的實際權重",
            format="%.0f",
        )

    st.dataframe(
        display_df[show_cols],
        width="stretch",
        height=500,
        column_config=col_config,
    )

    # ── Course detail section ──
    st.markdown("---")
    st.markdown("### 📋 課程詳細分析")

    # Build a lookup for course selection
    course_options = display_df["課程名稱"] + " (" + display_df["課號"] + ")"
    selected_course = st.selectbox(
        "選擇課程查看詳細資訊",
        options=course_options.values,
        index=0 if len(course_options) > 0 else None,
        placeholder="選擇一門課程...",
    )

    if selected_course:
        # Extract course_key
        idx = course_options[course_options == selected_course].index[0]
        course_key = display_df.loc[idx, "course_key"]
        pred_w = int(display_df.loc[idx, "預測權重"])
        actual_w = int(display_df.loc[idx, "實際權重"]) if backtest else None

        # Get historical data (full history for display)
        history = df[df["course_key"] == course_key].sort_values("semester_ordinal")
        course_meta = display_df.loc[idx]

        # Detail card
        col_left, col_right = st.columns([2, 1])

        with col_left:
            # Build badge line
            badge_html = f'<span class="weight-badge {weight_color_class(pred_w)}">預測 {pred_w} 點</span>'
            if backtest and actual_w is not None:
                badge_html += f' &nbsp;→&nbsp; <span class="weight-badge {weight_color_class(actual_w)}">實際 {actual_w} 點</span>'
                err = pred_w - actual_w
                abs_err = abs(err)
                if abs_err <= 5:
                    acc_cls = "accuracy-good"
                elif abs_err <= 10:
                    acc_cls = "accuracy-ok"
                else:
                    acc_cls = "accuracy-bad"
                badge_html += f' &nbsp;<span class="accuracy-pill {acc_cls}">誤差 {err:+d}</span>'

            st.markdown(f"""
            <div class="detail-card">
                <h3 style="margin-top:0; color:#333;">
                    {course_meta['課程名稱']}
                </h3>
                <p>{badge_html}</p>
                <p>
                    <strong>課號：</strong>{course_meta['課號']} &nbsp;|&nbsp;
                    <strong>班別：</strong>{course_meta['班別'] if pd.notna(course_meta['班別']) else '—'} &nbsp;|&nbsp;
                    <strong>類別：</strong>{course_meta['科系類別']} &nbsp;|&nbsp;
                    <strong>學分：</strong>{course_meta['學分']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Historical chart
            if len(history) > 1:
                st.markdown("#### 📈 歷史權重趨勢")

                chart_data = history[["semester_code", "cutoff_weight"]].copy()
                chart_data["學期"] = chart_data["semester_code"].apply(
                    lambda c: f"{c[:3]}-{c[3]}"
                )
                chart_data = chart_data.rename(columns={"cutoff_weight": "權重"})

                st.line_chart(
                    chart_data.set_index("學期")["權重"],
                    height=300,
                )
            else:
                st.info("📊 此課程歷史資料不足，無法繪製趨勢圖。")

        with col_right:
            # Stats
            st.markdown("#### 📊 統計資訊")

            if len(history) > 0:
                avg = history["cutoff_weight"].mean()
                std = history["cutoff_weight"].std() if len(history) > 1 else 0
                max_w = history["cutoff_weight"].max()
                min_w = history["cutoff_weight"].min()
                n_sem = len(history)

                st.metric("平均權重", f"{avg:.1f}")
                st.metric("最高紀錄", f"{max_w:.0f}")
                st.metric("最低紀錄", f"{min_w:.0f}")
                st.metric("波動度 (標準差)", f"{std:.1f}")
                st.metric("開設學期數", f"{n_sem}")

                if backtest and actual_w is not None:
                    st.markdown("---")
                    st.metric(
                        "預測 vs 實際",
                        f"{actual_w}",
                        delta=f"{pred_w - actual_w:+d} 預測誤差",
                        delta_color="inverse",
                    )
            else:
                st.write("無歷史資料")

        # Recommendation
        if backtest:
            if actual_w is not None:
                err = pred_w - actual_w
                abs_err = abs(err)
                if abs_err <= 3:
                    reco = f"✅ 預測非常準確！預測 {pred_w}，實際 {actual_w}，誤差僅 {abs_err} 點。"
                elif abs_err <= 8:
                    direction = "高估" if err > 0 else "低估"
                    reco = f"📊 預測表現不錯。預測 {pred_w}，實際 {actual_w}，{direction} {abs_err} 點，在信心區間 (±{MODEL_MAE:.0f}) 之內。"
                else:
                    direction = "高估" if err > 0 else "低估"
                    reco = f"⚠️ 預測偏差較大。預測 {pred_w}，實際 {actual_w}，{direction} {abs_err} 點。此類課程的權重波動較難預測。"
            else:
                reco = "—"
        else:
            trend_val = float(display_df.loc[idx, "weight_trend"]) if pd.notna(display_df.loc[idx, "weight_trend"]) else np.nan
            vol_val = float(history["cutoff_weight"].std()) if len(history) > 1 else np.nan
            reco = recommendation_text(pred_w, trend_val, vol_val)
        st.markdown(f'<div class="reco-box">{reco}</div>', unsafe_allow_html=True)


# ─── Page 2: Statistics & Insights ───────────────────────────────────────────

def page_statistics():
    """Statistics and insights page."""
    df = load_data()

    st.markdown("""
    <div class="info-banner">
        <h2 style="margin:0; color: #333;">📊 統計與洞察</h2>
        <p style="margin:0.3rem 0 0; color: #555;">
            探索歷史趨勢、模型效能，以及各科系的選課競爭概況
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Performance ──
    st.markdown("### 🏆 模型效能")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    with perf_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">7.62</div>
            <div class="metric-label">Two-Stage MAE</div>
        </div>
        """, unsafe_allow_html=True)
    with perf_col2:
        st.markdown("""
        <div class="metric-card green">
            <div class="metric-value">24.7%</div>
            <div class="metric-label">優於基準模型</div>
        </div>
        """, unsafe_allow_html=True)
    with perf_col3:
        st.markdown("""
        <div class="metric-card orange">
            <div class="metric-value">10.12</div>
            <div class="metric-label">基準 MAE (上學期)</div>
        </div>
        """, unsafe_allow_html=True)
    with perf_col4:
        st.markdown("""
        <div class="metric-card blue">
            <div class="metric-value">3,531</div>
            <div class="metric-label">訓練資料筆數</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    | 模型 | 測試 MAE | 測試 RMSE | 測試 R² | 比基準進步 |
    |------|---------|-----------|---------|-----------|
    | 基準（上學期權重） | 10.12 | 16.50 | 0.533 | — |
    | LightGBM v1 | 8.60 | 14.45 | 0.642 | +15.0% |
    | LightGBM v2 (調參) | 7.95 | 13.03 | 0.709 | +21.4% |
    | **Two-Stage (最佳)** | **7.62** | **14.81** | **0.624** | **+24.7%** |
    | Ensemble | 7.70 | 13.16 | 0.703 | +23.9% |
    """)

    st.markdown("---")

    # ── Department Heatmap ──
    st.markdown("### 🗺️ 各科系平均權重 (近 6 學期)")

    recent_sems = SEMESTER_ORDER[-6:]
    recent_data = df[df["semester_code"].isin(recent_sems)]

    heatmap_data = (
        recent_data.groupby(["dept_cluster", "semester_code"])["cutoff_weight"]
        .mean()
        .unstack(fill_value=0)
    )

    if not heatmap_data.empty:
        # Re-label columns
        heatmap_data.columns = [f"{c[:3]}-{c[3]}" for c in heatmap_data.columns]
        st.dataframe(
            heatmap_data.style.background_gradient(cmap="YlOrRd", axis=None).format("{:.1f}"),
            width="stretch",
        )
    else:
        st.info("無近期資料可顯示。")

    st.markdown("---")

    # ── Top Competitive Courses ──
    st.markdown("### 🔥 近期最高競爭課程 (1142 學期)")

    latest = df[df["semester_code"] == SEMESTER_ORDER[-1]].copy()
    top_courses = (
        latest.nlargest(20, "cutoff_weight")[
            ["course_name", "course_id", "dept_cluster", "credits", "cutoff_weight"]
        ]
        .rename(columns={
            "course_name": "課程名稱",
            "course_id": "課號",
            "dept_cluster": "科系類別",
            "credits": "學分",
            "cutoff_weight": "權重",
        })
        .reset_index(drop=True)
    )
    top_courses.index = top_courses.index + 1
    st.dataframe(top_courses, width="stretch")

    st.markdown("---")

    # ── Weight Distribution ──
    st.markdown("### 📉 權重分佈 (1142 學期)")

    if len(latest) > 0:
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.bar_chart(
                latest["cutoff_weight"].value_counts().sort_index(),
                height=300,
            )
        with col_dist2:
            dist_stats = latest["cutoff_weight"].describe()
            st.markdown(f"""
            **統計摘要：**
            - 課程數：**{len(latest)}**
            - 平均權重：**{dist_stats['mean']:.1f}**
            - 中位數：**{dist_stats['50%']:.1f}**
            - 標準差：**{dist_stats['std']:.1f}**
            - 最大值：**{dist_stats['max']:.0f}**
            - 零權重比例：**{(latest['cutoff_weight'] == 0).mean()*100:.1f}%**
            """)

    st.markdown("---")

    # ── Trend Analysis ──
    st.markdown("### 📈 權重趨勢分析 (近期上升 / 下降最多的課程)")

    latest_with_trend = latest[latest["weight_trend"].notna()].copy()

    if len(latest_with_trend) > 0:
        col_up, col_down = st.columns(2)

        with col_up:
            st.markdown("**🔺 上升趨勢 Top 10**")
            rising = (
                latest_with_trend.nlargest(10, "weight_trend")[
                    ["course_name", "dept_cluster", "cutoff_weight", "weight_trend"]
                ]
                .rename(columns={
                    "course_name": "課程名稱",
                    "dept_cluster": "類別",
                    "cutoff_weight": "權重",
                    "weight_trend": "趨勢斜率",
                })
                .reset_index(drop=True)
            )
            rising.index = rising.index + 1
            st.dataframe(rising, width="stretch")

        with col_down:
            st.markdown("**🔻 下降趨勢 Top 10**")
            declining = (
                latest_with_trend.nsmallest(10, "weight_trend")[
                    ["course_name", "dept_cluster", "cutoff_weight", "weight_trend"]
                ]
                .rename(columns={
                    "course_name": "課程名稱",
                    "dept_cluster": "類別",
                    "cutoff_weight": "權重",
                    "weight_trend": "趨勢斜率",
                })
                .reset_index(drop=True)
            )
            declining.index = declining.index + 1
            st.dataframe(declining, width="stretch")


# ─── Page 3: About ───────────────────────────────────────────────────────────

def page_about():
    """About page with usage instructions."""
    st.markdown("""
    <div class="info-banner">
        <h2 style="margin:0; color: #333;">📖 使用說明</h2>
        <p style="margin:0.3rem 0 0; color: #555;">
            了解志願權重系統的運作方式與本預測工具的使用方法
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <h4>🎯 什麼是志願權重？</h4>
        <p>
            臺北醫學大學的選課系統中，當某門課程的選課人數超過名額時，會進行「志願權重抽籤」。
            每位同學擁有一定的權重點數，可以將點數分配到不同的課程上。<br><br>
            <strong>權重門檻（cutoff_weight）</strong>是指該課程最終被錄取的最低權重分數。
            例如某門課的權重門檻是 40，代表配了 40 點以上的同學才選得上。<br><br>
            權重門檻為 <strong>0</strong> 代表該課程不需要抽籤，所有報名的同學都能選上。
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <h4>🤖 模型如何預測？</h4>
        <p>
            本系統使用 <strong>Two-Stage LightGBM</strong> 機器學習模型，分兩階段進行預測：
        </p>
        <ol style="color:#555; line-height: 1.8;">
            <li><strong>第一階段（分類器）：</strong>判斷課程是否需要抽籤（權重 > 0 或 = 0）</li>
            <li><strong>第二階段（回歸器）：</strong>如果需要抽籤，預測具體的門檻權重</li>
        </ol>
        <p>
            模型使用了 <strong>14 個特徵</strong>，包括：
        </p>
        <ul style="color:#555; line-height: 1.8;">
            <li>📊 歷史權重：上學期、前兩學期的權重記錄</li>
            <li>📈 統計指標：平均值、趨勢斜率、波動度</li>
            <li>🏫 課程屬性：學分、年級、科系類別、必選修</li>
            <li>📅 時間資訊：學期編號、累積開課次數</li>
        </ul>
        <p>
            模型使用了 <strong>27 個學期、3,531 筆</strong>歷史資料進行訓練。
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <h4>🎯 預測準確度</h4>
        <p>
            模型的精確度以 <strong>MAE（平均絕對誤差）</strong>來衡量：
        </p>
        <ul style="color:#555; line-height: 1.8;">
            <li><strong>Two-Stage 模型 MAE = 7.62</strong>：平均預測誤差約 7-8 點</li>
            <li>比「直接用上學期權重」的方法準確 <strong>24.7%</strong></li>
            <li>信心區間大約為 <strong>±8 點</strong></li>
        </ul>
        <p>
            <em>舉例：如果模型預測某門課的權重門檻為 30，實際值大約落在 22~38 之間。</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <h4>💡 使用建議</h4>
        <p>
        </p>
        <ol style="color:#555; line-height: 1.8;">
            <li><strong>參考趨勢：</strong>如果一門課的權重持續上升，最好多配一些點數</li>
            <li><strong>關注波動度：</strong>波動大的課程較難預測，建議多留彈性空間</li>
            <li><strong>注意信心區間：</strong>預測值都有 ±8 的誤差範圍，別剛好配預測值</li>
            <li><strong>分散風險：</strong>不要把所有權重押在一門課上，分散投注更穩妥</li>
            <li><strong>查看歷史：</strong>點選課程後查看歷史趨勢圖，了解該課程的選課模式</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="border-left-color: #e74c3c;">
        <h4>⚠️ 免責聲明</h4>
        <p style="color:#666;">
            本預測系統僅供參考，預測結果並非保證。實際權重門檻會受到多種無法預測的因素影響，
            包括但不限於：新生人數變化、課程內容調整、授課教師更換、社群口碑效應等。<br><br>
            請將本工具作為選課策略的<strong>輔助參考</strong>，而非唯一依據。<br><br>
            <em>資料來源：TMU 選課系統歷史資料（101 至 114 學年度）</em>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="TMU 選課志願權重預測",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <h1 style="color:white; margin:0; font-size:1.8rem;">🎓</h1>
            <h2 style="color:white; margin:0; font-size:1.2rem;">TMU 選課權重預測</h2>
            <p style="color:#aaa; font-size:0.85rem; margin-top:0.3rem;">
                臺北醫學大學 · AI 預測系統
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "📑 功能選單",
            options=["🔮 權重預測", "📊 統計與洞察", "📖 使用說明"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Footer info
        st.markdown("""
        <div style="text-align:center; margin-top:2rem;">
            <p style="color:#888; font-size:0.75rem;">
                模型：Two-Stage LightGBM<br>
                MAE：7.62 · 資料：3,531 筆<br>
                學期範圍：101-2 ~ 114-2
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Page routing
    if page == "🔮 權重預測":
        page_prediction()
    elif page == "📊 統計與洞察":
        page_statistics()
    elif page == "📖 使用說明":
        page_about()


if __name__ == "__main__":
    main()
