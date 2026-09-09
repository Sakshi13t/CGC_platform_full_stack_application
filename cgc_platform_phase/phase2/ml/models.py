"""
backend/app/ml/models.py
==========================
Phase 2 (Classical ML) — trains and compares three approaches to
forecasting sec_true_t_plus_24h (see dataset.py for the target definition
and leakage-safety rules):

  1. Naive persistence baseline (no fitting): predicts the current measured
     sec_kwh_per_t unchanged.
  2. Random Forest (scikit-learn), with a median imputer FIT ON THE TRAIN
     SPLIT ONLY (RandomForestRegressor cannot accept NaN natively).
  3. XGBoost, trained directly on the raw (NaN-containing) features — it
     handles missing values internally via its own learned default
     direction per split, so no imputation is applied for this model.

All three are evaluated on the SAME validation/test rows so metrics are
directly comparable. No metric in this file or in
MODEL_EVALUATION_REPORT.md is invented — every number printed/saved here
comes from actually running these models against dataset.load_ml_dataset().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from ..preprocessing import config as preprocessing_config
from . import dataset as ds_module

MODELS_DIR = preprocessing_config.OUTPUTS_DIR / "models"
REPORT_PATH = preprocessing_config.REPO_ROOT / "MODEL_EVALUATION_REPORT.md"


@dataclass
class SplitMetrics:
    mae: float
    rmse: float
    r2: float
    n: int


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> SplitMetrics:
    return SplitMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        n=int(len(y_true)),
    )


def evaluate_naive(ds: ds_module.MLDataset) -> dict[str, SplitMetrics]:
    return {
        "validation": _metrics(ds.y_val, ds.naive_proxy_val.to_numpy()),
        "test": _metrics(ds.y_test, ds.naive_proxy_test.to_numpy()),
    }


def train_random_forest(ds: ds_module.MLDataset, random_state: int = 42) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # .fit() below sees TRAIN split only
        ("rf", RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=5,
            n_jobs=1, random_state=random_state,
        )),
    ])
    pipe.fit(ds.X_train, ds.y_train)
    return pipe


def train_xgboost(ds: ds_module.MLDataset, random_state: int = 42) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=random_state,
        n_jobs=1, tree_method="hist",
        eval_metric="mae",
    )
    model.fit(
        ds.X_train, ds.y_train,
        eval_set=[(ds.X_val, ds.y_val)],
        verbose=False,
    )
    return model


def evaluate_model(model, ds: ds_module.MLDataset) -> dict[str, SplitMetrics]:
    pred_val = model.predict(ds.X_val)
    pred_test = model.predict(ds.X_test)
    return {
        "validation": _metrics(ds.y_val, pred_val),
        "test": _metrics(ds.y_test, pred_test),
    }


def feature_importance(model, feature_columns: list[str], top_n: int = 20) -> list[tuple[str, float]]:
    if hasattr(model, "named_steps"):
        importances = model.named_steps["rf"].feature_importances_
    else:
        importances = model.feature_importances_
    pairs = sorted(zip(feature_columns, importances), key=lambda p: -p[1])
    return pairs[:top_n]


def _fmt_metrics(m: dict[str, SplitMetrics]) -> str:
    lines = []
    for split_name, sm in m.items():
        lines.append(f"| {split_name} | {sm.mae:.3f} | {sm.rmse:.3f} | {sm.r2:.4f} | {sm.n} |")
    return "\n".join(lines)


def run_all(save: bool = True) -> dict:
    ds = ds_module.load_ml_dataset()

    naive_metrics = evaluate_naive(ds)
    rf_model = train_random_forest(ds)
    rf_metrics = evaluate_model(rf_model, ds)
    xgb_model = train_xgboost(ds)
    xgb_metrics = evaluate_model(xgb_model, ds)

    rf_importance = feature_importance(rf_model, ds.feature_columns)
    xgb_importance = feature_importance(xgb_model, ds.feature_columns)

    results = {
        "metadata": ds.metadata,
        "naive": {k: asdict(v) for k, v in naive_metrics.items()},
        "random_forest": {k: asdict(v) for k, v in rf_metrics.items()},
        "xgboost": {k: asdict(v) for k, v in xgb_metrics.items()},
        "random_forest_top_features": rf_importance,
        "xgboost_top_features": xgb_importance,
    }

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_model, MODELS_DIR / "random_forest_sec_forecast.joblib")
        xgb_model.save_model(str(MODELS_DIR / "xgboost_sec_forecast.json"))
        with open(MODELS_DIR / "feature_metadata.json", "w") as f:
            json.dump(ds.metadata, f, indent=2)
        with open(MODELS_DIR / "evaluation_results.json", "w") as f:
            json.dump(
                {k: v for k, v in results.items() if k not in ("random_forest_top_features", "xgboost_top_features")}
                | {
                    "random_forest_top_features": [[n, float(v)] for n, v in rf_importance],
                    "xgboost_top_features": [[n, float(v)] for n, v in xgb_importance],
                },
                f, indent=2,
            )
        _write_report(results)

    return results


def _write_report(results: dict) -> None:
    meta = results["metadata"]
    lines = [
        "# MODEL_EVALUATION_REPORT.md — Phase 2: Classical ML Baseline",
        "",
        "Generated by `backend/app/ml/models.py:run_all()`. Every number below comes",
        "from an actual training/evaluation run against the real dataset — nothing",
        "here is hand-typed or estimated.",
        "",
        "## Target",
        "",
        f"- Target: `{meta['target']}` — `{meta['target_source_column']}` "
        f"(ground_truth_reference.csv) shifted {meta['forecast_horizon_hours']}h into the future, per train.",
        "- Ground truth is used ONLY as the label; it never appears in the feature set.",
        f"- {meta['n_features']} features, built from raw telemetry + physics_features.py + features.py "
        "(see dataset.py for the full, explicit list).",
        "",
        "## Data",
        "",
        f"- {meta['rows_total_telemetry']} total telemetry rows -> {meta['rows_after_shutdown_drop']} "
        "after dropping current-timestep shutdown rows -> "
        f"{meta['rows_after_target_dropna']} after dropping rows with no valid target "
        "(shutdown-window ground truth NaNs + the last horizon hours of each train's series).",
        f"- Chronological split (config.SPLIT_BOUNDARIES): train={meta['n_train']}, "
        f"validation={meta['n_validation']}, test={meta['n_test']}. No shuffling; split assigned by "
        "the CURRENT timestep's calendar date, not the label's.",
        "",
        "## Naive baseline",
        "",
        f"> {meta['naive_baseline_definition']}",
        "",
        "| split | MAE (kWh/t) | RMSE (kWh/t) | R² | n |",
        "|---|---|---|---|---|",
        _fmt_metrics_dict(results["naive"]),
        "",
        "## Random Forest",
        "",
        "150 trees, max_depth=10, min_samples_leaf=5. NaN features imputed with a "
        "median imputer fit on the TRAIN split only (leakage-safe).",
        "",
        "| split | MAE (kWh/t) | RMSE (kWh/t) | R² | n |",
        "|---|---|---|---|---|",
        _fmt_metrics_dict(results["random_forest"]),
        "",
        "Top features by importance:",
        "",
        "\n".join(f"{i+1}. `{name}` — {imp:.4f}" for i, (name, imp) in enumerate(results["random_forest_top_features"][:10])),
        "",
        "## XGBoost",
        "",
        "400 trees, max_depth=6, lr=0.05, subsample/colsample=0.8. Trained directly on "
        "raw (NaN-containing) features — XGBoost's `hist` tree method learns its own "
        "default split direction for missing values; no imputation applied.",
        "",
        "| split | MAE (kWh/t) | RMSE (kWh/t) | R² | n |",
        "|---|---|---|---|---|",
        _fmt_metrics_dict(results["xgboost"]),
        "",
        "Top features by importance:",
        "",
        "\n".join(f"{i+1}. `{name}` — {imp:.4f}" for i, (name, imp) in enumerate(results["xgboost_top_features"][:10])),
        "",
        "## Honest comparison",
        "",
        _comparison_paragraph(results),
        "",
        "## Limitations",
        "",
        "- Single 24h horizon; a multi-horizon (e.g. 24h/72h/168h) comparison was out of "
        "scope for this pass and is a natural extension.",
        "- The naive baseline uses the noisy *measured* sec_kwh_per_t as its persistence "
        "input (the true current value is never observable — using ground truth here "
        "would itself be a form of leakage). This makes the naive baseline slightly "
        "harder to beat than a textbook persistence forecast would be, which is the "
        "honest, available comparison.",
        "- Hyperparameters were set to reasonable defaults for this data size, not tuned "
        "via a formal search — documented as a limitation rather than reporting a tuned "
        "number that wasn't actually searched for.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def _fmt_metrics_dict(d: dict) -> str:
    rows = []
    for split_name, m in d.items():
        rows.append(f"| {split_name} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.4f} | {m['n']} |")
    return "\n".join(rows)


def _comparison_paragraph(results: dict) -> str:
    naive_test = results["naive"]["test"]["mae"]
    rf_test = results["random_forest"]["test"]["mae"]
    xgb_test = results["xgboost"]["test"]["mae"]
    best_name, best_mae = min(
        [("naive", naive_test), ("Random Forest", rf_test), ("XGBoost", xgb_test)], key=lambda p: p[1]
    )
    note = ""
    if best_name == "naive":
        note = (
            " A plausible reason: fouling-driven SEC drift is slow relative to a 24h "
            "horizon and highly autocorrelated hour-to-hour, so 'tomorrow looks like "
            "today' is already close to the achievable floor — the tree models, "
            "regularized against overfitting (max_depth/min_samples_leaf), have limited "
            "room to add value from the other engineered features at this horizon. This "
            "is a hypothesis consistent with the result, not a separately tested claim; "
            "a shorter/longer horizon sweep would be the natural next check."
        )
    return (
        f"On the held-out test split, MAE is {naive_test:.3f} kWh/t (naive), "
        f"{rf_test:.3f} kWh/t (Random Forest), {xgb_test:.3f} kWh/t (XGBoost). "
        f"**{best_name}** has the lowest test MAE of the three. This is reported as measured — "
        "no claim is made that the more complex model must win." + note
    )


if __name__ == "__main__":
    r = run_all()
    print(json.dumps(
        {k: v for k, v in r.items() if k in ("naive", "random_forest", "xgboost")},
        indent=2,
    ))
