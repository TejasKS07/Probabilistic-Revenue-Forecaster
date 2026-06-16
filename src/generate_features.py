"""
generate_features.py
====================
Phase 1 of the Probabilistic Revenue Forecasting pipeline.

Reads raw channel-level CSVs (Google Ads, Bing/MS Ads, Meta Ads) from the
data/ directory, normalizes them into a unified schema, cleans the data,
engineers features, and writes the result to a Parquet file for downstream
model consumption.

Usage (standalone):
    python src/generate_features.py --data-dir ./data --out features.parquet

Called by run.sh:
    python src/generate_features.py \
        --data-dir "$DATA_DIR" \
        --out features.parquet
"""

import argparse
import glob
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# STEP 1 — UNIFIED SCHEMA NORMALIZATION

UNIFIED_COLUMNS = [
    "date",
    "channel",
    "campaign_type",
    "campaign_name",
    "campaign_id",
    "revenue",
    "spend",
    "clicks",
    "impressions",
    "conversions",
    "daily_budget",
]

# -- Campaign-type standardization map --

CAMPAIGN_TYPE_MAP = {
    # Google Ads — already upper-case, but ensure consistency
    "SEARCH": "SEARCH",
    "PERFORMANCE_MAX": "PERFORMANCE_MAX",
    "DISPLAY": "DISPLAY",
    "VIDEO": "VIDEO",
    "DEMAND_GEN": "DEMAND_GEN",
    "SHOPPING": "SHOPPING",
    # Bing/MS Ads — mixed case
    "Search": "SEARCH",
    "PerformanceMax": "PERFORMANCE_MAX",
    "Audience": "AUDIENCE",
    "Shopping": "SHOPPING",
}

# -- Meta Ads: infer campaign type from campaign name prefix --
# Meta doesn't provide an explicit campaign_type column.  We derive it
# from the naming convention observed in the dataset:
#   "Generic_Campaign_02"           → GENERIC
#   "Prospecting_DPA_Campaign_04"   → PROSPECTING
#   "Remarketing_DPA_Campaign_03"   → REMARKETING
#   "Prospecting_Brand_Campaign_02" → PROSPECTING
#   "Remarketing_Brand_Campaign_01" → REMARKETING
#   "Prospecting_Adv_Plus_Campaign" → PROSPECTING
#   "Generic_Brand_Campaign_01"     → GENERIC
META_CAMPAIGN_TYPE_PREFIXES = {
    "Generic": "GENERIC",
    "Prospecting": "PROSPECTING",
    "Remarketing": "REMARKETING",
}


def _infer_meta_campaign_type(campaign_name: str) -> str:
    """Derive a campaign type from a Meta Ads campaign name.

    Splits on underscore and maps the first token to a standard type.
    Falls back to 'OTHER' if the prefix is unrecognized.
    """
    if not isinstance(campaign_name, str):
        return "OTHER"
    prefix = campaign_name.split("_")[0]
    return META_CAMPAIGN_TYPE_PREFIXES.get(prefix, "OTHER")


# ---------------------------------------------------------------------------
# Per-channel loaders
# ---------------------------------------------------------------------------

def load_google_ads(filepath: str) -> pd.DataFrame:
    """Load and normalize a Google Ads campaign stats CSV.

    Key transformations:
      - ``metrics_cost_micros`` is divided by 1 000 000 to get spend in dollars.
      - ``campaign_advertising_channel_type`` is mapped to the standard
        campaign-type vocabulary via ``CAMPAIGN_TYPE_MAP``.
      - ``segments_date`` becomes the unified ``date`` column.
    """
    logger.info("Loading Google Ads data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("  Raw shape: %s", df.shape)

    unified = pd.DataFrame()
    unified["date"] = pd.to_datetime(df["segments_date"])
    unified["channel"] = "google"
    unified["campaign_type"] = (
        df["campaign_advertising_channel_type"]
        .map(CAMPAIGN_TYPE_MAP)
        .fillna("OTHER"))
    unified["campaign_name"] = df["campaign_name"].astype(str)
    unified["campaign_id"] = df["campaign_id"].astype(str)
    unified["revenue"] = df["metrics_conversions_value"].astype(float)
    unified["spend"] = df["metrics_cost_micros"].astype(float) / 1_000_000
    unified["clicks"] = df["metrics_clicks"].astype(int)
    unified["impressions"] = df["metrics_impressions"].astype(int)
    unified["conversions"] = df["metrics_conversions"].astype(float)
    unified["daily_budget"] = df["campaign_budget_amount"].astype(float)

    logger.info(
        "  Google Ads: %d rows, %d campaigns, date range %s → %s",
        len(unified),
        unified["campaign_name"].nunique(),
        unified["date"].min().date(),
        unified["date"].max().date(),
    )
    return unified


def load_bing_ads(filepath: str) -> pd.DataFrame:
    """Load and normalize a Bing / Microsoft Ads campaign stats CSV.

    Key transformations:
      - Column names use PascalCase in the raw file
        (``TimePeriod``, ``Revenue``, ``Spend``, etc.) and are remapped.
      - ``CampaignType`` is mapped via ``CAMPAIGN_TYPE_MAP``.
    """
    logger.info("Loading Bing/MS Ads data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("  Raw shape: %s", df.shape)

    unified = pd.DataFrame()
    unified["date"] = pd.to_datetime(df["TimePeriod"])
    unified["channel"] = "bing"
    unified["campaign_type"] = (
        df["CampaignType"]
        .map(CAMPAIGN_TYPE_MAP)
        .fillna("OTHER"))
    unified["campaign_name"] = df["CampaignName"].astype(str)
    unified["campaign_id"] = df["CampaignId"].astype(str)
    unified["revenue"] = df["Revenue"].astype(float)
    unified["spend"] = df["Spend"].astype(float)
    unified["clicks"] = df["Clicks"].astype(int)
    unified["impressions"] = df["Impressions"].astype(int)
    unified["conversions"] = df["Conversions"].astype(float)
    unified["daily_budget"] = df["DailyBudget"].astype(float)

    logger.info(
        "  Bing Ads: %d rows, %d campaigns, date range %s → %s",
        len(unified),
        unified["campaign_name"].nunique(),
        unified["date"].min().date(),
        unified["date"].max().date(),)
    return unified


def load_meta_ads(filepath: str) -> pd.DataFrame:
    """Load and normalize a Meta Ads campaign stats CSV.

    Key transformations:
      - ``conversion`` column is treated as **conversion value (revenue)**
        based on its magnitude (values up to ~$26 000).
      - Campaign type is inferred from the campaign name prefix
        (Generic / Prospecting / Remarketing).
      - ``conversions`` count is not directly available; we leave it as NaN
        and will derive it downstream if needed.
    """
    logger.info("Loading Meta Ads data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("  Raw shape: %s", df.shape)

    unified = pd.DataFrame()
    unified["date"] = pd.to_datetime(df["date_start"])
    unified["channel"] = "meta"
    unified["campaign_type"] = df["campaign_name"].apply(
        _infer_meta_campaign_type)
    unified["campaign_name"] = df["campaign_name"].astype(str)
    unified["campaign_id"] = df["campaign_id"].astype(str)
    # `conversion` in Meta data holds monetary conversion value (revenue)
    unified["revenue"] = df["conversion"].astype(float)
    unified["spend"] = df["spend"].astype(float)
    unified["clicks"] = df["clicks"].astype(float).astype(int)
    unified["impressions"] = df["impressions"].astype(float).astype(int)
    # Meta does not provide a separate conversions count
    unified["conversions"] = np.nan
    unified["daily_budget"] = df["daily_budget"].astype(float)

    logger.info(
        "  Meta Ads: %d rows, %d campaigns, date range %s → %s",
        len(unified),
        unified["campaign_name"].nunique(),
        unified["date"].min().date(),
        unified["date"].max().date(),)
    return unified


# ---------------------------------------------------------------------------
# Channel file discovery
# ---------------------------------------------------------------------------

# Map recognizable filename substrings to their loader function.
_CHANNEL_LOADERS = {
    "google": load_google_ads,
    "bing": load_bing_ads,
    "meta": load_meta_ads,
}


def _detect_channel(filename: str) -> Optional[str]:
    """Return a channel key if the filename matches a known pattern."""
    lower = filename.lower()
    for key in _CHANNEL_LOADERS:
        if key in lower:
            return key
    return None


def load_all_channels(data_dir: str) -> pd.DataFrame:
    """Discover CSV files in *data_dir*, normalize each, and concatenate.

    Files are matched to a channel loader by checking if the filename
    contains ``"google"``, ``"bing"``, or ``"meta"`` (case-insensitive).
    Unrecognized files are skipped with a warning.

    Returns
    -------
    pd.DataFrame
        A single DataFrame with the unified schema (see ``UNIFIED_COLUMNS``).
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        logger.error("No CSV files found in %s", data_dir)
        sys.exit(1)

    logger.info("Found %d CSV file(s) in %s", len(csv_files), data_dir)

    frames: List[pd.DataFrame] = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        channel = _detect_channel(fname)
        if channel is None:
            logger.warning("Skipping unrecognized file: %s", fname)
            continue
        loader = _CHANNEL_LOADERS[channel]
        df = loader(fpath)
        frames.append(df)

    if not frames:
        logger.error(
            "No recognized channel files found.  Expected filenames "
            "containing 'google', 'bing', or 'meta'."
        )
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Enforce column order
    combined = combined[UNIFIED_COLUMNS]

    logger.info(
        "Combined dataset: %d rows, %d columns, channels=%s",
        len(combined),
        len(combined.columns),
        sorted(combined["channel"].unique()),
    )
    return combined


# =========================================================================
# STEP 2 — DATA CLEANING
# =========================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules to the unified dataset.

    1. Drop the ``Unnamed: 0`` residual index column (already handled
       during per-channel loading — this is a safety net).
    2. Forward-fill missing ``daily_budget`` within each campaign group.
    3. Remove rows where **both** revenue and spend are zero (no activity).
    4. Sort by date for time-series consistency.
    """
    logger.info("Cleaning data …")
    initial_rows = len(df)

    # 1. Safety: drop any leftover unnamed columns
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # 2. Forward-fill missing daily_budget within each campaign
    budget_nulls = df["daily_budget"].isna().sum()
    if budget_nulls > 0:
        logger.info("  Filling %d missing daily_budget values (ffill per campaign)", budget_nulls)
        df["daily_budget"] = (
            df.groupby(["channel", "campaign_name"])["daily_budget"]
            .transform(lambda s: s.ffill().bfill())
        )
        # If still NaN (entire campaign has no budget), fill with 0
        remaining_nulls = df["daily_budget"].isna().sum()
        if remaining_nulls > 0:
            logger.info("  %d budget values still null — filling with 0", remaining_nulls)
            df["daily_budget"] = df["daily_budget"].fillna(0.0)

    # 3. Remove zero-activity rows (both revenue AND spend are 0)
    zero_mask = (df["revenue"] == 0) & (df["spend"] == 0)
    zero_count = zero_mask.sum()
    if zero_count > 0:
        logger.info("  Dropping %d zero-activity rows (revenue=0, spend=0)", zero_count)
        df = df[~zero_mask].reset_index(drop=True)

    # 4. Sort chronologically
    df = df.sort_values(["date", "channel", "campaign_name"]).reset_index(drop=True)

    logger.info(
        "  Cleaning complete: %d → %d rows (%d removed)",
        initial_rows,
        len(df),
        initial_rows - len(df),
    )
    return df


# =========================================================================
# STEP 3 — FEATURE ENGINEERING
# =========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer model-ready features from the cleaned unified data.

    Adds the following feature groups to the daily data:

    1. **Seasonality** — month, day_of_week, week_of_year, quarter, is_weekend
    2. **Efficiency ratios** — ROAS, CPC, CVR, budget utilization (safe div)
    3. **Campaign maturity** — days since the campaign's first appearance
    4. **Rolling window stats** — 7d / 14d / 30d rolling mean and std for
       revenue, spend, and ROAS, computed per campaign
    5. **Momentum indicators** — 7d-vs-30d ratios for revenue and spend
    """
    logger.info("Engineering features …")
    initial_cols = len(df.columns)

    # Ensure chronological order within each campaign for correct rolling
    df = df.sort_values(
        ["channel", "campaign_name", "date"]
    ).reset_index(drop=True)

    # ----- 1. Seasonality features -----
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ----- 2. Efficiency ratios (guarded against division by zero) -----
    df["roas"] = np.where(
        df["spend"] > 0, df["revenue"] / df["spend"], 0.0
    )
    df["cpc"] = np.where(
        df["clicks"] > 0, df["spend"] / df["clicks"], 0.0
    )
    df["cvr"] = np.where(
        df["clicks"] > 0, df["conversions"] / df["clicks"], 0.0
    )
    # Fill NaN in cvr that can arise when conversions is NaN (Meta)
    df["cvr"] = df["cvr"].fillna(0.0)
    df["budget_utilization"] = np.where(
        df["daily_budget"] > 0, df["spend"] / df["daily_budget"], 0.0
    )

    # ----- 3. Campaign maturity -----
    campaign_start = (
        df.groupby(["channel", "campaign_name"])["date"]
        .transform("min")
    )
    df["days_since_campaign_start"] = (df["date"] - campaign_start).dt.days

    # ----- 4. Rolling window stats (per campaign) -----
    group_keys = ["channel", "campaign_name"]
    for window in [7, 14, 30]:
        logger.info("  Computing %d-day rolling features …", window)
        grp = df.groupby(group_keys)

        # Revenue
        df[f"revenue_roll_{window}d_mean"] = grp["revenue"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"revenue_roll_{window}d_std"] = (
            grp["revenue"]
            .transform(lambda s: s.rolling(window, min_periods=1).std())
            .fillna(0.0)
        )

        # Spend
        df[f"spend_roll_{window}d_mean"] = grp["spend"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"spend_roll_{window}d_std"] = (
            grp["spend"]
            .transform(lambda s: s.rolling(window, min_periods=1).std())
            .fillna(0.0)
        )

        # ROAS
        df[f"roas_roll_{window}d_mean"] = grp["roas"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

        # Clicks
        df[f"clicks_roll_{window}d_mean"] = grp["clicks"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    # ----- 5. Momentum indicators (short-term vs long-term trend) -----
    df["revenue_momentum"] = np.where(
        df["revenue_roll_30d_mean"] > 0,
        df["revenue_roll_7d_mean"] / df["revenue_roll_30d_mean"],
        1.0,
    )
    df["spend_momentum"] = np.where(
        df["spend_roll_30d_mean"] > 0,
        df["spend_roll_7d_mean"] / df["spend_roll_30d_mean"],
        1.0,
    )

    new_cols = len(df.columns) - initial_cols
    logger.info(
        "  Feature engineering complete: +%d columns (total %d)",
        new_cols,
        len(df.columns),
    )
    return df


# =========================================================================
# STEP 4 — AGGREGATION TO PLANNING PERIODS
# =========================================================================

# The three granularity levels and their grouping columns.
_GRANULARITIES = [
    ("channel",       ["channel"]),
    ("campaign_type", ["channel", "campaign_type"]),
    ("campaign",      ["channel", "campaign_type", "campaign_name"]),
]

# Horizons (in days) for which we build feature rows.
_HORIZONS = [30, 60, 90]

# Rolling feature columns whose *most recent* value we carry forward
# into the period-level row.
_CARRY_FORWARD_COLS = [
    "revenue_roll_7d_mean",
    "revenue_roll_14d_mean",
    "revenue_roll_30d_mean",
    "revenue_roll_7d_std",
    "revenue_roll_14d_std",
    "revenue_roll_30d_std",
    "spend_roll_7d_mean",
    "spend_roll_14d_mean",
    "spend_roll_30d_mean",
    "spend_roll_7d_std",
    "spend_roll_14d_std",
    "spend_roll_30d_std",
    "roas_roll_7d_mean",
    "roas_roll_14d_mean",
    "roas_roll_30d_mean",
    "clicks_roll_7d_mean",
    "clicks_roll_14d_mean",
    "clicks_roll_30d_mean",
    "revenue_momentum",
    "spend_momentum",
]


def _build_period_record(
    group_data: pd.DataFrame,
    group_cols: List[str],
    group_key: tuple,
    horizon: int,
    gran_name: str,
    snapshot_date: pd.Timestamp,
    lookback_start: pd.Timestamp,
) -> dict:
    """Compute one period-level feature row from a slice of daily data."""

    record: dict = {
        "forecast_horizon": horizon,
        "granularity": gran_name,
        "snapshot_date": snapshot_date,
    }

    # Identification columns
    for col, val in zip(group_cols, group_key):
        record[col] = val
    for col in ["channel", "campaign_type", "campaign_name"]:
        if col not in record:
            record[col] = "ALL"

    # ---------- aggregate statistics ----------
    record["total_revenue"]       = group_data["revenue"].sum()
    record["total_spend"]         = group_data["spend"].sum()
    record["total_clicks"]        = group_data["clicks"].sum()
    record["total_impressions"]   = group_data["impressions"].sum()
    record["total_conversions"]   = group_data["conversions"].sum()

    record["mean_daily_revenue"]  = group_data["revenue"].mean()
    record["mean_daily_spend"]    = group_data["spend"].mean()

    rev_std = group_data["revenue"].std()
    record["std_daily_revenue"]   = rev_std if pd.notna(rev_std) else 0.0
    spend_std = group_data["spend"].std()
    record["std_daily_spend"]     = spend_std if pd.notna(spend_std) else 0.0

    # ---------- efficiency ----------
    record["mean_roas"]    = group_data["roas"].mean()
    record["overall_roas"] = (
        record["total_revenue"] / record["total_spend"]
        if record["total_spend"] > 0
        else 0.0
    )
    record["mean_cpc"]         = group_data["cpc"].mean()
    record["mean_cvr"]         = group_data["cvr"].mean()
    record["mean_budget_util"] = group_data["budget_utilization"].mean()

    # ---------- budget ----------
    record["mean_daily_budget"] = group_data["daily_budget"].mean()
    record["total_budget"]      = group_data["daily_budget"].sum()

    # ---------- activity ----------
    record["active_days"]   = group_data["date"].nunique()
    record["num_campaigns"] = group_data["campaign_name"].nunique()

    # ---------- trend: first half vs second half ----------
    mid = lookback_start + pd.Timedelta(days=horizon // 2)
    first_half  = group_data[group_data["date"] <= mid]
    second_half = group_data[group_data["date"] >  mid]

    fh_rev = first_half["revenue"].sum()  if len(first_half)  else 0.0
    sh_rev = second_half["revenue"].sum() if len(second_half) else 0.0
    record["revenue_trend"] = (
        (sh_rev - fh_rev) / fh_rev if fh_rev > 0 else 0.0
    )

    fh_spend = first_half["spend"].sum()  if len(first_half)  else 0.0
    sh_spend = second_half["spend"].sum() if len(second_half) else 0.0
    record["spend_trend"] = (
        (sh_spend - fh_spend) / fh_spend if fh_spend > 0 else 0.0
    )

    # ---------- seasonality summary ----------
    record["avg_month"]      = group_data["month"].mean()
    record["avg_quarter"]    = group_data["quarter"].mean()
    record["weekend_ratio"]  = group_data["is_weekend"].mean()

    # ---------- carry forward most-recent rolling values ----------
    # For campaign-level groups we take the last row; for higher-level
    # groups we average the per-campaign last values.
    for feat in _CARRY_FORWARD_COLS:
        if feat in group_data.columns:
            # last non-NaN value across the group
            record[feat] = group_data[feat].iloc[-1]
        else:
            record[feat] = 0.0

    return record


def aggregate_to_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate enriched daily data into planning-period feature rows.

    For each combination of *granularity level*
    (channel / campaign-type / campaign) and *forecast horizon*
    (30 / 60 / 90 days), this function:

    1. Selects a lookback window of ``horizon`` days ending at the
       most recent date in the data (the *snapshot date*).
    2. Groups the windowed data by the granularity's key columns.
    3. Computes ~40 summary features per group (totals, means, efficiency
       ratios, trends, seasonality stats, and recent rolling values).

    The resulting DataFrame has one row per (group × horizon) and
    is the feature matrix that ``predict.py`` consumes.
    """
    logger.info("Aggregating to planning periods …")

    snapshot_date = df["date"].max()
    logger.info("  Snapshot date: %s", snapshot_date.date())

    records: List[dict] = []

    for horizon in _HORIZONS:
        lookback_start = snapshot_date - pd.Timedelta(days=horizon)
        window_df = df[df["date"] > lookback_start].copy()

        if window_df.empty:
            logger.warning(
                "  No data in %d-day lookback window — skipping.", horizon
            )
            continue

        for gran_name, group_cols in _GRANULARITIES:
            grouped = window_df.groupby(group_cols, sort=False)

            for group_key, group_data in grouped:
                # Ensure group_key is always a tuple
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)

                record = _build_period_record(
                    group_data=group_data,
                    group_cols=group_cols,
                    group_key=group_key,
                    horizon=horizon,
                    gran_name=gran_name,
                    snapshot_date=snapshot_date,
                    lookback_start=lookback_start,
                )
                records.append(record)

    result = pd.DataFrame(records)
    logger.info(
        "  Aggregation complete: %d period-level rows "
        "(%d horizons × %d granularity levels)",
        len(result),
        len(_HORIZONS),
        len(_GRANULARITIES),
    )
    return result


# =========================================================================
# MAIN
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate features from raw channel-level marketing CSVs."
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing the input CSV files (default: ./data)",
    )
    parser.add_argument(
        "--out",
        default="features.csv",
        help="Output path for the feature file (default: features.csv)",
    )
    args = parser.parse_args()

    # Step 1: Unified schema normalization
    df = load_all_channels(args.data_dir)

    # Step 2: Data cleaning
    df = clean_data(df)

    # Step 3: Feature engineering
    df = engineer_features(df)

    # Step 4: Aggregation to planning periods
    df = aggregate_to_periods(df)

    # Write output
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("Feature file written to %s (%d rows)", args.out, len(df))


if __name__ == "__main__":
    main()

