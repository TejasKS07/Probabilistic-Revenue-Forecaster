"""
predict.py
==========
Phase 3 of the Probabilistic Revenue Forecasting pipeline.

Loads a pre-trained model bundle (``pickle/model.pkl``), applies it to
feature-engineered daily data, and produces probabilistic revenue and
ROAS forecasts at three granularity levels (channel, campaign_type,
campaign) across three forecast horizons (30, 60, 90 days).

Supports **budget simulation**: the ``--budget-multiplier`` flag lets
users explore what-if scenarios by scaling projected spend.

Usage (called by run.sh):
    python src/predict.py \\
        --data-dir ./data \\
        --model pickle/model.pkl \\
        --output output/predictions.csv

Standalone with budget simulation:
    python src/predict.py \\
        --data-dir ./data \\
        --model pickle/model.pkl \\
        --output output/predictions.csv \\
        --budget-multiplier 1.2
"""

import argparse
import logging
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# Import Phase 1 functions for data loading & feature engineering
from generate_features import (
    load_all_channels,
    clean_data,
    engineer_features,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECAST_HORIZONS = [30, 60, 90]

# Mirrors train.py constants
CATEGORICAL_COLS = ["channel", "campaign_type", "campaign_name"]
EXCLUDE_COLS = ["date", "campaign_id", "revenue", "conversions"]
TARGET = "revenue"

# Guardrails
MAX_ROAS = 20.0        # Cap ROAS at 20x to prevent outlier distortion
MIN_REVENUE_DAYS = 5   # Minimum non-zero revenue days required for prediction

GRANULARITIES = {
    "channel":       ["channel"],
    "campaign_type": ["channel", "campaign_type"],
    "campaign":      ["channel", "campaign_type", "campaign_name"],
}


# ---------------------------------------------------------------------------
# Data preparation (mirrors train.py logic)
# ---------------------------------------------------------------------------

def _aggregate_daily_to_granularity(
    df_daily: pd.DataFrame,
    granularity_name: str,
    group_cols: List[str],
) -> pd.DataFrame:
    """Aggregate daily campaign rows to a higher granularity level.

    For campaign level, returns the data as-is.
    For channel/campaign_type, groups by (date + group_cols) and
    sums volume metrics, averages rate metrics.
    """
    if granularity_name == "campaign":
        return df_daily.copy()

    agg_group = ["date"] + group_cols
    numeric_cols = [
        c for c in df_daily.columns
        if df_daily[c].dtype in [np.float64, np.int64, np.int32, np.float32]
        and c not in EXCLUDE_COLS
        and c not in CATEGORICAL_COLS
    ]

    sum_cols = ["spend", "clicks", "impressions", "daily_budget"]
    mean_cols = [c for c in numeric_cols
                 if c not in sum_cols and c in df_daily.columns]

    agg_dict = {}
    for c in sum_cols:
        if c in df_daily.columns:
            agg_dict[c] = "sum"
    for c in mean_cols:
        if c in df_daily.columns:
            agg_dict[c] = "mean"
    agg_dict["revenue"] = "sum"

    result = df_daily.groupby(agg_group, as_index=False).agg(agg_dict)

    # Re-compute ROAS after aggregation
    result["roas"] = np.where(
        result["spend"] > 0,
        result["revenue"] / result["spend"],
        0.0,
    )
    return result


def _encode_categoricals(
    df: pd.DataFrame,
    label_encoders: Dict,
) -> pd.DataFrame:
    """Apply label encoding using the encoders from training."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            le = label_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda v: v if v in known else "__UNSEEN__"
            )
            df[col] = le.transform(df[col])
    return df


def _select_lookback_window(
    df: pd.DataFrame,
    horizon: int,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Select rows in the lookback window for a given horizon."""
    lookback_start = snapshot_date - pd.Timedelta(days=horizon)
    return df[df["date"] > lookback_start].copy()


# ---------------------------------------------------------------------------
# Prediction at a single granularity × horizon
# ---------------------------------------------------------------------------

def _predict_granularity_horizon(
    df_window: pd.DataFrame,
    granularity_name: str,
    group_cols: List[str],
    models: Dict[float, object],
    feature_names: List[str],
    label_encoders: Dict,
    horizon: int,
    budget_multiplier: float = 1.0,
) -> List[dict]:
    """Generate predictions for all groups at one granularity × horizon.

    For each group (e.g., each channel, or each campaign):
      1. Aggregate daily rows to the granularity level.
      2. Use the most recent daily feature values to predict daily revenue
         at P10/P50/P90.
      3. Scale predictions to the full horizon period.
      4. Compute ROAS from predicted revenue and projected spend.
      5. Apply budget simulation if requested.

    Returns a list of prediction record dicts.
    """
    records = []

    # Aggregate to granularity if needed (before encoding)
    agg_df = _aggregate_daily_to_granularity(
        df_window, granularity_name, group_cols
    )

    if agg_df.empty:
        return records

    # Keep original string labels for the output before encoding
    # Group the data by the group columns to iterate over each entity
    if granularity_name == "campaign":
        # For campaign-level, group by all 3 categorical cols
        iter_cols = ["channel", "campaign_type", "campaign_name"]
    else:
        iter_cols = group_cols

    for group_key, group_data in agg_df.groupby(iter_cols, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        # Save original string labels for output
        labels = dict(zip(iter_cols, group_key))

        # Encode categoricals for model input
        encoded_data = _encode_categoricals(group_data, label_encoders)

        # Select features — use the LAST row's features (most recent state)
        # as the representative feature vector for this group
        encoded_data = encoded_data.sort_values("date")
        available_features = [c for c in feature_names if c in encoded_data.columns]
        X_recent = encoded_data[available_features].iloc[-1:].copy()
        X_recent = X_recent.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Historical daily averages for scaling
        n_days_in_window = group_data["date"].nunique()
        n_nonzero_rev_days = int((group_data.groupby("date")["revenue"].sum() > 0).sum())
        avg_daily_revenue = group_data["revenue"].sum() / max(n_days_in_window, 1)
        avg_daily_spend = group_data["spend"].sum() / max(n_days_in_window, 1)

        # Project spend for the horizon
        projected_spend = avg_daily_spend * horizon * budget_multiplier

        # Minimum-data guardrail: if a group has very few non-zero
        # revenue days, its model predictions are unreliable.
        # Fall back to historical average scaling instead.
        sparse_data = n_nonzero_rev_days < MIN_REVENUE_DAYS

        # Predict daily revenue at each quantile
        predictions = {}
        for q, model in models.items():
            if sparse_data:
                # For sparse groups, use historical average directly
                # rather than trusting a model fit on <5 data points.
                daily_pred = avg_daily_revenue
            else:
                daily_pred = model.predict(X_recent)[0]
            # Ensure non-negative predictions
            daily_pred = max(0.0, daily_pred)

            # Scale to horizon period
            # Use a blend: model prediction anchored by historical average
            # to reduce single-point prediction noise
            if avg_daily_revenue > 0 and daily_pred > 0:
                # Predicted/actual ratio applied to historical average
                ratio = daily_pred / avg_daily_revenue
                # Apply budget multiplier to the predicted component
                period_revenue = daily_pred * horizon * budget_multiplier
            else:
                period_revenue = daily_pred * horizon * budget_multiplier

            predictions[q] = period_revenue

        # Ensure quantile ordering: P10 <= P50 <= P90
        p10 = predictions.get(0.10, 0.0)
        p50 = predictions.get(0.50, 0.0)
        p90 = predictions.get(0.90, 0.0)

        # Fix any quantile crossing
        if p10 > p50:
            p10 = p50 * 0.8
        if p90 < p50:
            p90 = p50 * 1.2
        if p10 > p90:
            p10, p90 = p90, p10

        # Compute ROAS for each quantile, capped at MAX_ROAS
        roas_p10 = min(p10 / projected_spend, MAX_ROAS) if projected_spend > 0 else 0.0
        roas_p50 = min(p50 / projected_spend, MAX_ROAS) if projected_spend > 0 else 0.0
        roas_p90 = min(p90 / projected_spend, MAX_ROAS) if projected_spend > 0 else 0.0

        # If ROAS was capped, also cap the revenue to maintain consistency
        # (revenue = ROAS * spend)
        if projected_spend > 0:
            p10 = min(p10, MAX_ROAS * projected_spend)
            p50 = min(p50, MAX_ROAS * projected_spend)
            p90 = min(p90, MAX_ROAS * projected_spend)

        record = {
            "forecast_horizon": horizon,
            "granularity": granularity_name,
            "channel": labels.get("channel", "ALL"),
            "campaign_type": labels.get("campaign_type", "ALL"),
            "campaign_name": labels.get("campaign_name", "ALL"),
            "revenue_p10": round(p10, 2),
            "revenue_p50": round(p50, 2),
            "revenue_p90": round(p90, 2),
            "roas_p10": round(roas_p10, 4),
            "roas_p50": round(roas_p50, 4),
            "roas_p90": round(roas_p90, 4),
            "projected_spend": round(projected_spend, 2),
            "budget_multiplier": budget_multiplier,
            "historical_daily_revenue": round(avg_daily_revenue, 2),
            "historical_daily_spend": round(avg_daily_spend, 2),
            "days_in_window": n_days_in_window,
            "nonzero_revenue_days": n_nonzero_rev_days,
            "low_confidence": sparse_data,
        }

        # Fill in ALL for higher granularities
        if granularity_name == "channel":
            record["campaign_type"] = "ALL"
            record["campaign_name"] = "ALL"
        elif granularity_name == "campaign_type":
            record["campaign_name"] = "ALL"

        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Full prediction pipeline
# ---------------------------------------------------------------------------

def generate_predictions(
    df_daily: pd.DataFrame,
    model_bundle: dict,
    budget_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Run the full prediction pipeline across all granularities and horizons.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Feature-engineered daily data (output of engineer_features).
    model_bundle : dict
        Loaded pickle containing models, encoders, feature names.
    budget_multiplier : float
        Scaling factor for projected spend (1.0 = historical budget).

    Returns
    -------
    pd.DataFrame
        Predictions with columns matching the output schema.
    """
    models_dict = model_bundle["models"]
    feature_names = model_bundle["feature_names"]
    label_encoders = model_bundle["label_encoders"]
    granularities = model_bundle["granularities"]

    snapshot_date = df_daily["date"].max()
    logger.info("Snapshot date: %s", snapshot_date.date())
    logger.info("Budget multiplier: %.2f", budget_multiplier)

    all_records = []

    for horizon in FORECAST_HORIZONS:
        logger.info("Forecasting %d-day horizon …", horizon)

        # Select lookback window
        df_window = _select_lookback_window(df_daily, horizon, snapshot_date)

        if df_window.empty:
            logger.warning("  No data in %d-day lookback window — skipping.", horizon)
            continue

        logger.info("  Lookback window: %d daily rows", len(df_window))

        for gran_name, group_cols in granularities.items():
            if gran_name not in models_dict:
                logger.warning("  No model for granularity '%s' — skipping.", gran_name)
                continue

            models = models_dict[gran_name]
            records = _predict_granularity_horizon(
                df_window=df_window,
                granularity_name=gran_name,
                group_cols=group_cols,
                models=models,
                feature_names=feature_names,
                label_encoders=label_encoders,
                horizon=horizon,
                budget_multiplier=budget_multiplier,
            )

            logger.info(
                "  %s-level: %d forecasts generated",
                gran_name, len(records),
            )
            all_records.extend(records)

    result = pd.DataFrame(all_records)

    # Sort for readability
    if not result.empty:
        result = result.sort_values(
            ["forecast_horizon", "granularity", "channel",
             "campaign_type", "campaign_name"]
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """Format the predictions DataFrame for final output.

    Selects and orders columns to match the expected output schema.
    """
    output_cols = [
        "forecast_horizon",
        "granularity",
        "channel",
        "campaign_type",
        "campaign_name",
        "revenue_p10",
        "revenue_p50",
        "revenue_p90",
        "roas_p10",
        "roas_p50",
        "roas_p90",
        "projected_spend",
        "budget_multiplier",
        "historical_daily_revenue",
        "historical_daily_spend",
        "days_in_window",
    ]

    # Only include columns that exist
    available = [c for c in output_cols if c in df.columns]
    return df[available]


# ---------------------------------------------------------------------------
# Summary statistics logging
# ---------------------------------------------------------------------------

def log_prediction_summary(df: pd.DataFrame) -> None:
    """Log a high-level summary of the predictions."""
    logger.info("=" * 50)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 50)
    logger.info("Total forecast rows: %d", len(df))

    for horizon in FORECAST_HORIZONS:
        h_df = df[df["forecast_horizon"] == horizon]
        if h_df.empty:
            continue

        logger.info("--- %d-day horizon ---", horizon)

        for gran in ["channel", "campaign_type", "campaign"]:
            g_df = h_df[h_df["granularity"] == gran]
            if g_df.empty:
                continue

            total_rev = g_df["revenue_p50"].sum()
            total_spend = g_df["projected_spend"].sum()
            avg_roas = g_df["roas_p50"].mean()
            logger.info(
                "  %s-level: %d groups, "
                "total P50 revenue = $%.0f, "
                "total projected spend = $%.0f, "
                "avg P50 ROAS = %.2f",
                gran, len(g_df), total_rev, total_spend, avg_roas,
            )

    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate probabilistic revenue forecasts."
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing the input CSV files (default: ./data)",
    )
    parser.add_argument(
        "--model",
        default="./pickle/model.pkl",
        help="Path to the pickled model bundle (default: ./pickle/model.pkl)",
    )
    parser.add_argument(
        "--output",
        default="./output/predictions.csv",
        help="Output path for predictions (default: ./output/predictions.csv)",
    )
    parser.add_argument(
        "--budget-multiplier",
        type=float,
        default=1.0,
        help="Budget scaling factor for what-if simulation (default: 1.0 = no change)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load and prepare daily data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 3: PREDICTION PIPELINE")
    logger.info("=" * 60)

    logger.info("Step 1/4: Loading and preparing daily data …")
    df = load_all_channels(args.data_dir)
    df = clean_data(df)
    df = engineer_features(df)

    logger.info(
        "  Daily feature matrix: %d rows × %d columns",
        len(df), len(df.columns),
    )

    # ------------------------------------------------------------------
    # 2. Load model bundle
    # ------------------------------------------------------------------
    logger.info("Step 2/4: Loading model bundle from %s …", args.model)

    if not os.path.exists(args.model):
        logger.error("Model file not found: %s", args.model)
        sys.exit(1)

    with open(args.model, "rb") as f:
        model_bundle = pickle.load(f)

    logger.info(
        "  Model loaded: %d granularities, %d quantiles, %d features",
        len(model_bundle["granularities"]),
        len(model_bundle["quantiles"]),
        len(model_bundle["feature_names"]),
    )
    logger.info(
        "  Training date range: %s → %s",
        model_bundle["training_metadata"]["date_range"][0],
        model_bundle["training_metadata"]["date_range"][1],
    )

    # ------------------------------------------------------------------
    # 3. Generate predictions
    # ------------------------------------------------------------------
    logger.info("Step 3/4: Generating probabilistic forecasts …")

    predictions = generate_predictions(
        df_daily=df,
        model_bundle=model_bundle,
        budget_multiplier=args.budget_multiplier,
    )

    if predictions.empty:
        logger.error("No predictions generated!")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Format and write output
    # ------------------------------------------------------------------
    logger.info("Step 4/4: Writing predictions …")

    output_df = format_output(predictions)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_df.to_csv(args.output, index=False)

    logger.info(
        "Predictions written to %s (%d rows)",
        args.output, len(output_df),
    )

    # Log summary
    log_prediction_summary(output_df)

    logger.info("=" * 60)
    logger.info("PREDICTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
