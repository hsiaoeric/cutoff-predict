#!/usr/bin/env python3
"""
Phase 4: Train Cutoff Weight Prediction Models
================================================
Trains LightGBM models to predict course cutoff weights.

Models produced:
  1. Baseline  — "same as last semester" (prev_1_weight, NaN → global median)
  2. LGB v1   — LightGBM with default hyperparameters
  3. LGB v2   — LightGBM with Optuna-tuned hyperparameters
  4. Two-stage — binary classifier (lottery?) + regressor (weight if lottery)
  5. Ensemble  — weighted blend of LGB v2 + two-stage + baseline

Usage:
    uv run python scripts/04_train_model.py
"""

from __future__ import annotations

import argparse
import json
import textwrap
import warnings
from datetime import datetime
from math import sqrt
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Constants ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "notebooks" / "eda_plots"

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

TEST_SEMESTERS = ["1141", "1142"]
VAL_SEMESTERS = ["1131", "1132"]

# Features the model is allowed to use
NUMERIC_FEATURES = [
    "prev_1_weight",
    "prev_2_weight",
    "avg_weight_3sem",
    "avg_weight_all",
    "weight_trend",
    "weight_volatility",
    "semesters_offered",
    "credits",
    "grade_level",
    "semester",
    "semester_ordinal",
    # Enrollment demand features
    "oversubscription_ratio",
    "prev_1_oversub_ratio",
    "prev_2_oversub_ratio",
    "avg_oversub_ratio_3sem",
    "prev_1_remaining_spots",
    "demand_trend",
    # Instructor features
    "instructor_avg_cutoff",
    "instructor_course_count",
    # Time slot features
    "is_prime_time",
    "num_time_slots",
]

CATEGORICAL_FEATURES = [
    "dept_cluster",
    "is_required",
    "popularity_tier",
    "domain_category",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "cutoff_weight"

# Popularity tier → ordinal encoding
TIER_MAP = {"low": 0, "medium": 1, "high": 2, "very_high": 3}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features_enriched.csv and perform temporal train/val/test split."""
    df = pd.read_csv(DATA_DIR / "features_enriched.csv", dtype={"semester_code": str})

    # Drop rows where target is NaN
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    test = df[df["semester_code"].isin(TEST_SEMESTERS)].copy()
    val = df[df["semester_code"].isin(VAL_SEMESTERS)].copy()
    train = df[~df["semester_code"].isin(TEST_SEMESTERS + VAL_SEMESTERS)].copy()

    print(f"  Train: {len(train):,} rows  (semesters < 1131)")
    print(f"  Val:   {len(val):,} rows  ({', '.join(VAL_SEMESTERS)})")
    print(f"  Test:  {len(test):,} rows  ({', '.join(TEST_SEMESTERS)})")
    return train, val, test


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and encode features for modelling."""
    X = df[ALL_FEATURES].copy()

    # Ordinal encode popularity_tier
    X["popularity_tier"] = X["popularity_tier"].map(TIER_MAP)

    # LightGBM handles categoricals natively — convert to 'category' dtype
    X["dept_cluster"] = X["dept_cluster"].astype("category")
    X["is_required"] = X["is_required"].astype("category")
    X["domain_category"] = X["domain_category"].astype("category")

    return X


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
        "MedianAE": float(np.median(np.abs(y_true - y_pred))),
    }


# ─── 1. Baseline ─────────────────────────────────────────────────────────────

def compute_baseline(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """'Same as last semester' baseline: predict prev_1_weight, else global median."""
    global_median = train[TARGET].median()
    print(f"\n  Global median (train): {global_median:.1f}")

    results = {}
    for name, df in [("val", val), ("test", test)]:
        preds = df["prev_1_weight"].fillna(global_median).values
        y = df[TARGET].values
        m = eval_metrics(y, preds)
        results[name] = m
        print(f"  Baseline {name}: MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R²={m['R²']:.3f}")

    return results, global_median


# ─── 2. LightGBM v1 (default params) ────────────────────────────────────────

def train_lgb_v1(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
) -> lgb.LGBMRegressor:
    """Train LightGBM with reasonable default hyperparameters."""
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=500,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


# ─── 3. LightGBM v2 (Optuna-tuned) ──────────────────────────────────────────

def tune_lgb(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    n_trials: int = 100,
) -> dict:
    """Run Optuna hyperparameter search, return best params."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbose": -1,
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": 1000,
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = m.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best trial MAE: {study.best_value:.3f}")
    print(f"  Best params: {study.best_params}")

    best = study.best_params
    best.update({
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_estimators": 1000,
    })
    return best


def train_lgb_v2(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    params: dict,
) -> lgb.LGBMRegressor:
    """Train final model with tuned hyperparameters."""
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


# ─── 4. Two-Stage Model ─────────────────────────────────────────────────────

def train_two_stage(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    reg_params: dict | None = None,
) -> tuple[lgb.LGBMClassifier, lgb.LGBMRegressor]:
    """
    Stage 1: Binary classifier — will there be a lottery (weight > 0)?
    Stage 2: Regression — if lottery, what's the cutoff weight?
    """
    # Stage 1: classifier
    y_cls = (y_train > 0).astype(int)
    y_cls_val = (y_val > 0).astype(int)

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        verbose=-1,
    )
    clf.fit(
        X_train, y_cls,
        eval_set=[(X_val, y_cls_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Stage 2: regression on non-zero rows only
    nonzero_mask = y_train > 0
    X_train_nz = X_train[nonzero_mask]
    y_train_nz = y_train[nonzero_mask]

    nonzero_mask_val = y_val > 0
    X_val_nz = X_val[nonzero_mask_val]
    y_val_nz = y_val[nonzero_mask_val]

    r_params = reg_params or {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }
    reg = lgb.LGBMRegressor(**r_params)
    reg.fit(
        X_train_nz, y_train_nz,
        eval_set=[(X_val_nz, y_val_nz)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    return clf, reg


def predict_two_stage(
    clf: lgb.LGBMClassifier, reg: lgb.LGBMRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Combine binary classifier + regressor predictions."""
    p_nonzero = clf.predict_proba(X)[:, 1]
    pred_weight = reg.predict(X)
    # Use hard threshold: if P(lottery) > 0.5, use regression prediction, else 0
    final = np.where(p_nonzero > 0.5, pred_weight, 0.0)
    # Clip negative predictions
    return np.clip(final, 0, None)


# ─── 5. Ensemble ─────────────────────────────────────────────────────────────

def ensemble_predict(
    lgb_pred: np.ndarray,
    twostage_pred: np.ndarray,
    baseline_pred: np.ndarray,
    w_lgb: float = 0.5,
    w_ts: float = 0.3,
    w_bl: float = 0.2,
) -> np.ndarray:
    """Weighted average ensemble of three model predictions."""
    final = w_lgb * lgb_pred + w_ts * twostage_pred + w_bl * baseline_pred
    return np.clip(final, 0, None)


# ─── 6. Feature Importance Plot ─────────────────────────────────────────────

def plot_feature_importance(model: lgb.LGBMRegressor, path: Path):
    """Save top-15 feature importance bar chart."""
    imp = model.feature_importances_
    names = model.feature_name_
    idx = np.argsort(imp)[::-1][:15]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))
    ax.barh(range(len(idx)), imp[idx][::-1], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1])
    ax.set_xlabel("Feature Importance (split count)")
    ax.set_title("Top 15 Feature Importances — LightGBM")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Feature importance plot → {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train cutoff weight prediction models")
    parser.add_argument("--optuna-trials", type=int, default=100,
                        help="Number of Optuna HPO trials (default: 100)")
    parser.add_argument("--skip-tune", action="store_true",
                        help="Skip Optuna tuning, use only default params")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4: Train Cutoff Weight Prediction Models")
    print("=" * 70)

    # ── Load & split ──────────────────────────────────────────────────────
    print("\n📂 Loading data…")
    train, val, test = load_and_split()

    X_train = prepare_features(train)
    X_val = prepare_features(val)
    X_test = prepare_features(test)
    y_train = train[TARGET].values
    y_val = val[TARGET].values
    y_test = test[TARGET].values

    # ── 1. Baseline ───────────────────────────────────────────────────────
    print("\n📏 Baseline: 'same as last semester'")
    baseline_results, global_median = compute_baseline(train, val, test)
    baseline_test_mae = baseline_results["test"]["MAE"]

    # Baseline predictions (needed for ensemble)
    baseline_val_pred = val["prev_1_weight"].fillna(global_median).values
    baseline_test_pred = test["prev_1_weight"].fillna(global_median).values

    # ── 2. LightGBM v1 ───────────────────────────────────────────────────
    print("\n🌲 LightGBM v1 (default hyperparameters)…")
    lgb_v1 = train_lgb_v1(X_train, y_train, X_val, y_val)

    v1_val_pred = lgb_v1.predict(X_val)
    v1_test_pred = lgb_v1.predict(X_test)
    v1_val = eval_metrics(y_val, v1_val_pred)
    v1_test = eval_metrics(y_test, v1_test_pred)
    print(f"  Val:  MAE={v1_val['MAE']:.2f}  RMSE={v1_val['RMSE']:.2f}  R²={v1_val['R²']:.3f}")
    print(f"  Test: MAE={v1_test['MAE']:.2f}  RMSE={v1_test['RMSE']:.2f}  R²={v1_test['R²']:.3f}")
    print(f"  Improvement over baseline (test): {(1 - v1_test['MAE']/baseline_test_mae)*100:.1f}%")

    joblib.dump(lgb_v1, MODEL_DIR / "lgb_v1_default.joblib")
    print(f"  Saved → models/lgb_v1_default.joblib")

    # ── 3. LightGBM v2 (Optuna tuned) ────────────────────────────────────
    if not args.skip_tune:
        print(f"\n🔍 Optuna hyperparameter tuning ({args.optuna_trials} trials)…")
        best_params = tune_lgb(X_train, y_train, X_val, y_val, n_trials=args.optuna_trials)

        print("\n🌲 LightGBM v2 (tuned)…")
        lgb_v2 = train_lgb_v2(X_train, y_train, X_val, y_val, best_params)
    else:
        print("\n⏭️  Skipping Optuna tuning (--skip-tune)")
        lgb_v2 = lgb_v1
        best_params = lgb_v1.get_params()

    v2_val_pred = lgb_v2.predict(X_val)
    v2_test_pred = lgb_v2.predict(X_test)
    v2_val = eval_metrics(y_val, v2_val_pred)
    v2_test = eval_metrics(y_test, v2_test_pred)
    print(f"  Val:  MAE={v2_val['MAE']:.2f}  RMSE={v2_val['RMSE']:.2f}  R²={v2_val['R²']:.3f}")
    print(f"  Test: MAE={v2_test['MAE']:.2f}  RMSE={v2_test['RMSE']:.2f}  R²={v2_test['R²']:.3f}")
    print(f"  Improvement over baseline (test): {(1 - v2_test['MAE']/baseline_test_mae)*100:.1f}%")

    joblib.dump(lgb_v2, MODEL_DIR / "lgb_v2_tuned.joblib")
    print(f"  Saved → models/lgb_v2_tuned.joblib")

    # ── 4. Two-stage model ────────────────────────────────────────────────
    print("\n🎯 Two-stage model (classifier + regressor)…")
    clf, reg = train_two_stage(X_train, y_train, X_val, y_val)

    ts_val_pred = predict_two_stage(clf, reg, X_val)
    ts_test_pred = predict_two_stage(clf, reg, X_test)
    ts_val = eval_metrics(y_val, ts_val_pred)
    ts_test = eval_metrics(y_test, ts_test_pred)
    print(f"  Val:  MAE={ts_val['MAE']:.2f}  RMSE={ts_val['RMSE']:.2f}  R²={ts_val['R²']:.3f}")
    print(f"  Test: MAE={ts_test['MAE']:.2f}  RMSE={ts_test['RMSE']:.2f}  R²={ts_test['R²']:.3f}")
    print(f"  Improvement over baseline (test): {(1 - ts_test['MAE']/baseline_test_mae)*100:.1f}%")

    joblib.dump({"classifier": clf, "regressor": reg}, MODEL_DIR / "two_stage.joblib")
    print(f"  Saved → models/two_stage.joblib")

    # ── 5. Ensemble ───────────────────────────────────────────────────────
    print("\n🔗 Ensemble (LGB v2 50% + Two-stage 30% + Baseline 20%)…")
    ens_val_pred = ensemble_predict(v2_val_pred, ts_val_pred, baseline_val_pred)
    ens_test_pred = ensemble_predict(v2_test_pred, ts_test_pred, baseline_test_pred)
    ens_val = eval_metrics(y_val, ens_val_pred)
    ens_test = eval_metrics(y_test, ens_test_pred)
    print(f"  Val:  MAE={ens_val['MAE']:.2f}  RMSE={ens_val['RMSE']:.2f}  R²={ens_val['R²']:.3f}")
    print(f"  Test: MAE={ens_test['MAE']:.2f}  RMSE={ens_test['RMSE']:.2f}  R²={ens_test['R²']:.3f}")
    print(f"  Improvement over baseline (test): {(1 - ens_test['MAE']/baseline_test_mae)*100:.1f}%")

    # ── Feature importance plot ───────────────────────────────────────────
    print("\n📊 Feature importance…")
    best_model = lgb_v2
    plot_feature_importance(best_model, PLOT_DIR / "model_feature_importance.png")

    # ── Save training log ─────────────────────────────────────────────────
    print("\n📝 Saving training log…")

    log = textwrap.dedent(f"""\
    ======================================================================
    Training Log — Cutoff Weight Prediction
    ======================================================================
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Python features: {len(ALL_FEATURES)} features
    Feature list: {ALL_FEATURES}

    Data splits:
      Train: {len(train)} rows
      Val:   {len(val)} rows  ({', '.join(VAL_SEMESTERS)})
      Test:  {len(test)} rows ({', '.join(TEST_SEMESTERS)})

    === Results Summary (Test Set) ===

    | Model          |   MAE |  RMSE |    R² | vs Baseline |
    |----------------|------:|------:|------:|------------:|
    | Baseline       | {baseline_results["test"]["MAE"]:5.2f} | {baseline_results["test"]["RMSE"]:5.2f} | {baseline_results["test"]["R²"]:5.3f} |           — |
    | LGB v1 default | {v1_test["MAE"]:5.2f} | {v1_test["RMSE"]:5.2f} | {v1_test["R²"]:5.3f} | {(1 - v1_test["MAE"]/baseline_test_mae)*100:+5.1f}%    |
    | LGB v2 tuned   | {v2_test["MAE"]:5.2f} | {v2_test["RMSE"]:5.2f} | {v2_test["R²"]:5.3f} | {(1 - v2_test["MAE"]/baseline_test_mae)*100:+5.1f}%    |
    | Two-stage      | {ts_test["MAE"]:5.2f} | {ts_test["RMSE"]:5.2f} | {ts_test["R²"]:5.3f} | {(1 - ts_test["MAE"]/baseline_test_mae)*100:+5.1f}%    |
    | Ensemble       | {ens_test["MAE"]:5.2f} | {ens_test["RMSE"]:5.2f} | {ens_test["R²"]:5.3f} | {(1 - ens_test["MAE"]/baseline_test_mae)*100:+5.1f}%    |

    === LGB v2 Tuned Hyperparameters ===
    {json.dumps(best_params, indent=2)}

    ======================================================================
    """)
    (MODEL_DIR / "training_log.txt").write_text(log)
    print(f"  Saved → models/training_log.txt")

    # ── Save predictions for evaluation script ────────────────────────────
    pred_df = test[["semester_code", "course_key", "course_name", TARGET]].copy()
    pred_df["baseline_pred"] = baseline_test_pred
    pred_df["lgb_v1_pred"] = v1_test_pred
    pred_df["lgb_v2_pred"] = v2_test_pred
    pred_df["twostage_pred"] = ts_test_pred
    pred_df["ensemble_pred"] = ens_test_pred
    pred_df.to_csv(MODEL_DIR / "test_predictions.csv", index=False)

    pred_val = val[["semester_code", "course_key", "course_name", TARGET]].copy()
    pred_val["baseline_pred"] = baseline_val_pred
    pred_val["lgb_v1_pred"] = v1_val_pred
    pred_val["lgb_v2_pred"] = v2_val_pred
    pred_val["twostage_pred"] = ts_val_pred
    pred_val["ensemble_pred"] = ens_val_pred
    pred_val.to_csv(MODEL_DIR / "val_predictions.csv", index=False)

    # Also save the full val/test metadata for segment analysis
    val_meta = val[["semester_code", "course_key", TARGET, "dept_cluster",
                     "semesters_offered", "popularity_tier", "is_required"]].copy()
    test_meta = test[["semester_code", "course_key", TARGET, "dept_cluster",
                       "semesters_offered", "popularity_tier", "is_required"]].copy()
    val_meta.to_csv(MODEL_DIR / "val_metadata.csv", index=False)
    test_meta.to_csv(MODEL_DIR / "test_metadata.csv", index=False)

    print(f"  Saved prediction CSVs → models/")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🏁 Training complete!")
    print(f"   Best single model MAE (test): {min(v1_test['MAE'], v2_test['MAE'], ts_test['MAE']):.2f}")
    print(f"   Ensemble MAE (test):          {ens_test['MAE']:.2f}")
    print(f"   Baseline MAE (test):          {baseline_test_mae:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
