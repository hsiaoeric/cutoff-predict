# TMU 選課志願權重預測 (Cutoff Weight Predictor)

Predicts course registration lottery cutoff weights for Taipei Medical University (TMU) students using historical data and machine learning.

When a course is oversubscribed, TMU uses a weighted lottery system. This tool helps students estimate the minimum weight needed to enroll in a course, so they can allocate their preference points more strategically.

## Model Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Baseline | 10.12 | 16.50 | 0.533 |
| LGB v2 (tuned) | 7.86 | 12.81 | 0.719 |
| Two-Stage | 7.50 | 12.78 | 0.720 |
| Ensemble | 7.63 | 12.55 | **0.730** |

Best model achieves **~26% improvement** over baseline on test set.

## Features

- **Streamlit web app** — interactive predictions by semester and course
- **Two-stage model** — first classifies zero vs non-zero cutoff, then predicts weight
- **20+ engineered features** — historical weights, enrollment demand, instructor stats, time slots, department clusters

## Project Structure

```
scripts/
  01_parse_html.py          # Parse raw HTML lottery results
  01b_parse_enrollment.py   # Parse enrollment/capacity data
  02_feature_engineering.py # Build ML features
  03_eda.py                 # Exploratory data analysis + plots
  04_train_model.py         # Train LightGBM models
  05_evaluate.py            # Evaluate and generate reports
app.py                      # Streamlit web app
models/                     # Trained models + evaluation outputs
notebooks/eda_plots/        # Visualization outputs
```

## Setup

Requires Python 3.14+.

```bash
# Install dependencies
uv sync

# Run the web app
uv run streamlit run app.py
```

## ML Pipeline

```bash
uv run python scripts/01_parse_html.py
uv run python scripts/01b_parse_enrollment.py
uv run python scripts/02_feature_engineering.py
uv run python scripts/03_eda.py
uv run python scripts/04_train_model.py
uv run python scripts/05_evaluate.py
```
