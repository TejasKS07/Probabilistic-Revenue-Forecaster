"""
train.py
========
Phase 2 of the Probabilistic Revenue Forecasting pipeline.

Trains LightGBM quantile regression models on daily-level campaign data
to predict revenue.  Produces three quantile models (P10, P50, P90) at
each of three granularity levels (channel, campaign_type, campaign),
yielding 9 models total.

The trained model bundle is pickled to ``pickle/model.pkl`` and committed
to the repository.  The automated evaluation pipeline does NOT retrain —
it only loads the pickle and runs predictions.

Usage:
    python src/train.py --data-dir ./data --out pickle/model.pkl

This script is for development only; it is NOT called by run.sh.
"""

import argparse
import logging
import os
import pickle
import sys
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# Import Phase 1 functions so we don't duplicate logic
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

# Quantiles for probabilistic forecasting
QUANTILES = [0.10, 0.50, 0.90]

# Target column
TARGET = "revenue"

# Granularity definitions — mirrors generate_features._GRANULARITIES but
# adds encoding info for each level.
GRANULARITIES = {
    "channel":       ["channel"],
    "campaign_type": ["channel", "campaign_type"],
    "campaign":      ["channel", "campaign_type", "campaign_name"],
}

# Categorical columns that need label-encoding for LightGBM
CATEGORICAL_COLS = ["channel", "campaign_type", "campaign_name"]

# Feature columns to exclude from training (identifiers / target / date)
EXCLUDE_COLS = [
    "date",
    "campaign_id",
    "revenue",
    "conversions",   # leakage: derived from revenue for Meta
]

# LightGBM hyperparameters — tuned for our small-to-medium dataset
# with regularisation to prevent overfitting.
LGB_BASE_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,       # L1 regularisation
    "reg_lambda": 1.0,      # L2 regularisation
    "random_state": 42,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    label_encoders: Dict[str, LabelEncoder] = None,
    fit_encoders: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, LabelEncoder]]:
    """Prepare the daily DataFrame for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned & feature-engineered daily data from Phase 1.
    label_encoders : dict, optional
        Pre-fitted label encoders (used at prediction time).
    fit_encoders : bool
        If True, fit new encoders. If False, transform only.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric only, ready for LightGBM).
    feature_names : list[str]
        Ordered feature column names.
    label_encoders : dict
        Fitted LabelEncoder instances keyed by column name.
    """
    if label_encoders is None:
        label_encoders = {}

    df = df.copy()

    # Label-encode categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if fit_encoders:
                le = LabelEncoder()
                # Add an "UNSEEN" class so the encoder can handle new values
                # at prediction time.
                all_values = list(df[col].astype(str).unique()) + ["__UNSEEN__"]
                le.fit(all_values)
                label_encoders[col] = le
            else:
                le = label_encoders[col]

            # Transform — map unseen values to "__UNSEEN__"
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda v: v if v in known else "__UNSEEN__"
            )
            df[col] = le.transform(df[col])

    # Select feature columns: everything except excluded cols
    feature_names = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and df[c].dtype in [np.float64, np.int64, np.int32, np.float32]
    ]

    X = df[feature_names].copy()

    # Replace remaining NaN/inf with 0
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return X, feature_names, label_encoders


# ---------------------------------------------------------------------------
# Time-series cross-validation evaluation
# ---------------------------------------------------------------------------

def evaluate_with_tscv(
    X: pd.DataFrame,
    y: pd.Series,
    quantile: float,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate a quantile model using time-series cross-validation.

    Returns
    -------
    dict
        Metrics averaged across folds: pinball_loss, coverage (for P10/P90).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pinball_losses = []
    coverages = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        params = {**LGB_BASE_PARAMS, "objective": "quantile", "alpha": quantile}
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_pred = model.predict(X_val)

        # Pinball loss (quantile loss)
        errors = y_val.values - y_pred
        pinball = np.mean(
            np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
        )
        pinball_losses.append(pinball)

        # Coverage: fraction of actuals below the prediction (should ≈ quantile)
        coverage = np.mean(y_val.values <= y_pred)
        coverages.append(coverage)

    return {
        "mean_pinball_loss": np.mean(pinball_losses),
        "mean_coverage": np.mean(coverages),
        "target_coverage": quantile,
    }


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_quantile_model(
    X: pd.DataFrame,
    y: pd.Series,
    quantile: float,
) -> lgb.LGBMRegressor:
    """Train a single LightGBM quantile regression model on the full data.

    Uses early stopping on the last 20% of the data (time-ordered) to
    prevent overfitting while still training on as much data as possible.
    """
    # Use the last 20% as a holdout for early stopping
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    params = {**LGB_BASE_PARAMS, "objective": "quantile", "alpha": quantile}
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    logger.info(
        "    Q%.0f model: best_iteration=%d, n_features=%d",
        quantile * 100,
        model.best_iteration_,
        X_train.shape[1],
    )
    return model


def train_granularity_level(
    df_daily: pd.DataFrame,
    granularity_name: str,
    group_cols: List[str],
    feature_names: List[str],
    label_encoders: Dict[str, LabelEncoder] = None,
) -> Dict[float, lgb.LGBMRegressor]:
    """Train quantile models for one granularity level.

    For higher-level granularities (channel, campaign_type), we aggregate
    daily data to the group level before training.  For campaign-level, we
    train on per-campaign daily rows directly.

    Returns a dict mapping quantile -> trained model.
    """
    logger.info("  Training %s-level models …", granularity_name)

    if granularity_name == "campaign":
        # Campaign level: train on raw daily rows — each row is already
        # at the campaign × day level.
        train_df = df_daily.copy()
    else:
        # Aggregate daily data to the granularity level
        # e.g. for channel level: group by (date, channel) and sum/mean
        agg_group = ["date"] + group_cols
        numeric_cols = [c for c in df_daily.columns
                        if df_daily[c].dtype in [np.float64, np.int64, np.int32, np.float32]
                        and c not in EXCLUDE_COLS
                        and c not in CATEGORICAL_COLS]

        # Aggregate: sum for volume metrics, mean for rate metrics
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
        # Revenue is our target — sum it at the group level
        agg_dict["revenue"] = "sum"

        train_df = df_daily.groupby(agg_group, as_index=False).agg(agg_dict)

        # Re-compute ROAS after aggregation
        train_df["roas"] = np.where(
            train_df["spend"] > 0,
            train_df["revenue"] / train_df["spend"],
            0.0,
        )

    # Sort chronologically for time-series splits
    train_df = train_df.sort_values("date").reset_index(drop=True)

    # Label-encode any categorical columns that remain in the training data
    if label_encoders:
        for col in CATEGORICAL_COLS:
            if col in train_df.columns and pd.api.types.is_string_dtype(train_df[col]):
                le = label_encoders[col]
                known = set(le.classes_)
                train_df[col] = train_df[col].astype(str).apply(
                    lambda v: v if v in known else "__UNSEEN__"
                )
                train_df[col] = le.transform(train_df[col])

    # Extract features and target
    X = train_df[[c for c in feature_names if c in train_df.columns]].copy()
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = train_df[TARGET].copy()

    logger.info(
        "    Training data: %d rows, %d features, revenue range [%.2f, %.2f]",
        len(X), len(X.columns), y.min(), y.max(),
    )

    # Evaluate with time-series CV first
    models = {}
    for q in QUANTILES:
        cv_metrics = evaluate_with_tscv(X, y, q, n_splits=5)
        logger.info(
            "    Q%.0f CV — pinball_loss=%.4f, coverage=%.3f (target=%.2f)",
            q * 100,
            cv_metrics["mean_pinball_loss"],
            cv_metrics["mean_coverage"],
            cv_metrics["target_coverage"],
        )

        # Train final model on full data
        model = train_quantile_model(X, y, q)
        models[q] = model

    return models


# ---------------------------------------------------------------------------
# Feature importance extraction (for LLM summaries)
# ---------------------------------------------------------------------------

def extract_feature_importances(
    models: Dict[str, Dict[float, lgb.LGBMRegressor]],
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract and aggregate feature importances across all models.

    Returns a DataFrame with columns:
        granularity, quantile, feature, importance
    """
    records = []
    for gran_name, quantile_models in models.items():
        for q, model in quantile_models.items():
            importances = model.feature_importances_
            model_features = model.feature_name_
            for feat, imp in zip(model_features, importances):
                records.append({
                    "granularity": gran_name,
                    "quantile": q,
                    "feature": feat,
                    "importance": imp,
                })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train quantile regression models for revenue forecasting."
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing the input CSV files (default: ./data)",
    )
    parser.add_argument(
        "--out",
        default="./pickle/model.pkl",
        help="Output path for the pickled model bundle (default: ./pickle/model.pkl)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load and prepare daily data (reuse Phase 1 pipeline)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL TRAINING")
    logger.info("=" * 60)

    logger.info("Step 1/4: Loading and preparing daily data …")
    df = load_all_channels(args.data_dir)
    df = clean_data(df)
    df = engineer_features(df)

    logger.info(
        "Daily feature matrix: %d rows × %d columns",
        len(df), len(df.columns),
    )

    # ------------------------------------------------------------------
    # 2. Prepare feature matrix (label-encode categoricals)
    # ------------------------------------------------------------------
    logger.info("Step 2/4: Preparing feature matrix …")
    X_full, feature_names, label_encoders = prepare_features(
        df, fit_encoders=True
    )
    y_full = df[TARGET].copy()

    logger.info(
        "  Feature matrix: %d rows × %d features",
        X_full.shape[0], X_full.shape[1],
    )
    logger.info("  Features: %s", feature_names[:10])
    if len(feature_names) > 10:
        logger.info("    … and %d more", len(feature_names) - 10)

    # ------------------------------------------------------------------
    # 3. Train models at each granularity level
    # ------------------------------------------------------------------
    logger.info("Step 3/4: Training quantile models …")

    all_models = {}
    for gran_name, group_cols in GRANULARITIES.items():
        models = train_granularity_level(
            df_daily=df,
            granularity_name=gran_name,
            group_cols=group_cols,
            feature_names=feature_names,
            label_encoders=label_encoders,
        )
        all_models[gran_name] = models

    # ------------------------------------------------------------------
    # 4. Extract feature importances & pickle everything
    # ------------------------------------------------------------------
    logger.info("Step 4/4: Packaging and saving model bundle …")

    importances_df = extract_feature_importances(all_models, feature_names)

    # Log top features for the median (P50) model at each granularity
    for gran_name in GRANULARITIES:
        gran_imp = importances_df[
            (importances_df["granularity"] == gran_name) &
            (importances_df["quantile"] == 0.5)
        ].sort_values("importance", ascending=False).head(10)

        logger.info("  Top 10 features (%s, P50):", gran_name)
        for _, row in gran_imp.iterrows():
            logger.info("    %-30s  %8.0f", row["feature"], row["importance"])

    # Build the model bundle
    model_bundle = {
        "models": all_models,                  # {gran: {quantile: model}}
        "feature_names": feature_names,        # ordered feature list
        "label_encoders": label_encoders,      # {col: LabelEncoder}
        "feature_importances": importances_df, # DataFrame for LLM summaries
        "quantiles": QUANTILES,                # [0.1, 0.5, 0.9]
        "granularities": GRANULARITIES,        # {name: group_cols}
        "target": TARGET,
        "lgb_params": LGB_BASE_PARAMS,
        "training_metadata": {
            "n_rows": len(df),
            "n_features": len(feature_names),
            "date_range": [
                str(df["date"].min().date()),
                str(df["date"].max().date()),
            ],
            "channels": sorted(df["channel"].unique().tolist()),
            "campaign_types": sorted(df["campaign_type"].unique().tolist()),
        },
    }

    # Create output directory and save
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(args.out)
    logger.info(
        "Model bundle saved to %s (%.2f MB)",
        args.out, file_size / 1_048_576,
    )

    # Also save feature importances as CSV for easy inspection
    imp_path = os.path.join(out_dir or ".", "feature_importances.csv")
    importances_df.to_csv(imp_path, index=False)
    logger.info("Feature importances saved to %s", imp_path)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(
        "  Models: %d granularities × %d quantiles = %d models",
        len(GRANULARITIES), len(QUANTILES),
        len(GRANULARITIES) * len(QUANTILES),
    )
    logger.info("  Pickle: %s", args.out)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
