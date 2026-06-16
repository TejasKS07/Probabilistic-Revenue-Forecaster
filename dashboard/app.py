"""
dashboard/app.py
================
FastAPI backend for the Probabilistic Revenue Forecaster Dashboard.

Loads the pre-trained model bundle at startup, pre-computes default
predictions, and exposes JSON API endpoints for the frontend.
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src/ to path so we can import the pipeline modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_features import load_all_channels, clean_data, engineer_features
from predict import generate_predictions, FORECAST_HORIZONS
from llm_summary import RuleBasedSummaryGenerator

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
# Config
# ---------------------------------------------------------------------------

DATA_DIR = str(PROJECT_ROOT / "data")
MODEL_PATH = str(PROJECT_ROOT / "pickle" / "model.pkl")
STATIC_DIR = str(Path(__file__).resolve().parent / "static")

# ---------------------------------------------------------------------------
# App State (loaded once at startup)
# ---------------------------------------------------------------------------

app_state = {
    "df_daily": None,
    "model_bundle": None,
    "default_predictions": None,
    "feature_importances": None,
    "training_metadata": None,
}

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Probabilistic Revenue Forecaster",
    description="Interactive dashboard for probabilistic ad revenue forecasting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup — load model and pre-compute predictions
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("DASHBOARD STARTUP")
    logger.info("=" * 60)

    # 1. Load and prepare daily data
    logger.info("Loading data from %s …", DATA_DIR)
    df = load_all_channels(DATA_DIR)
    df = clean_data(df)
    df = engineer_features(df)
    app_state["df_daily"] = df
    logger.info("  Daily data: %d rows × %d cols", len(df), len(df.columns))

    # 2. Load model bundle
    logger.info("Loading model from %s …", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    app_state["model_bundle"] = bundle
    app_state["feature_importances"] = bundle.get("feature_importances", pd.DataFrame())
    app_state["training_metadata"] = bundle.get("training_metadata", {})
    logger.info("  Model loaded: %d features", len(bundle["feature_names"]))

    # 3. Pre-compute default predictions (multiplier = 1.0)
    logger.info("Pre-computing default predictions …")
    preds = generate_predictions(df, bundle, budget_multiplier=1.0)
    app_state["default_predictions"] = preds
    logger.info("  Default predictions: %d rows", len(preds))

    logger.info("Dashboard ready!")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    budget_multiplier: float = 1.0


class LLMRequest(BaseModel):
    provider: str = "gemini"
    api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/predictions")
async def get_predictions(
    horizon: Optional[int] = Query(None, description="30, 60, or 90"),
    granularity: Optional[str] = Query(None, description="channel, campaign_type, or campaign"),
    channel: Optional[str] = Query(None, description="Filter by channel"),
):
    """Return filtered predictions."""
    df = app_state["default_predictions"]
    if df is None or df.empty:
        return JSONResponse({"error": "No predictions available"}, status_code=500)

    if horizon:
        df = df[df["forecast_horizon"] == horizon]
    if granularity:
        df = df[df["granularity"] == granularity]
    if channel:
        df = df[df["channel"] == channel]

    # Convert to records — handle NaN/Inf
    records = df.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_dict("records")
    return {"data": records, "total": len(records)}


@app.get("/api/channels")
async def get_channels():
    """Return available channels and campaign types."""
    df = app_state["default_predictions"]
    if df is None:
        return {"channels": [], "campaign_types": []}

    channels = sorted(df["channel"].dropna().unique().tolist())
    # Exclude 'ALL' from campaign types list
    campaign_types = sorted([
        ct for ct in df["campaign_type"].dropna().unique().tolist()
        if ct != "ALL"
    ])

    return {
        "channels": channels,
        "campaign_types": campaign_types,
        "granularities": ["channel", "campaign_type", "campaign"],
        "horizons": FORECAST_HORIZONS,
    }


@app.get("/api/paused-types")
async def get_paused_types(
    horizon: int = Query(30, description="Forecast horizon"),
):
    """Return campaign types that exist in the full dataset but have no
    active data in the lookback window (i.e., paused/ended campaigns)."""
    df_daily = app_state["df_daily"]
    preds = app_state["default_predictions"]
    if df_daily is None or preds is None:
        return {"paused": []}

    # All (channel, campaign_type) combos in the FULL dataset
    all_combos = set()
    for ch in df_daily["channel"].unique():
        for ct in df_daily[df_daily["channel"] == ch]["campaign_type"].unique():
            all_combos.add((ch, ct))

    # Combos that have predictions for this horizon
    ct_preds = preds[
        (preds["granularity"] == "campaign_type") &
        (preds["forecast_horizon"] == horizon)
    ]
    active_combos = set()
    for _, r in ct_preds.iterrows():
        active_combos.add((r["channel"], r["campaign_type"]))

    # The difference = paused types
    paused = []
    channel_names = {"google": "Google Ads", "meta": "Meta Ads", "bing": "Microsoft Ads"}
    for ch, ct in sorted(all_combos - active_combos):
        sub = df_daily[(df_daily["channel"] == ch) & (df_daily["campaign_type"] == ct)]
        paused.append({
            "channel": ch,
            "channel_display": channel_names.get(ch, ch),
            "campaign_type": ct,
            "total_revenue": round(float(sub["revenue"].sum()), 2),
            "total_spend": round(float(sub["spend"].sum()), 2),
            "campaigns": int(sub["campaign_name"].nunique()),
            "last_active": str(sub["date"].max().date()) if not sub.empty else "Unknown",
            "status": "Paused",
        })

    return {"paused": paused, "total": len(paused)}


@app.get("/api/feature-importances")
async def get_feature_importances(
    top_n: int = Query(15, description="Number of top features to return"),
):
    """Return top feature importances with explanations."""
    fi = app_state["feature_importances"]
    if fi is None or fi.empty:
        return {"data": []}

    # Filter for campaign-level P50 model (most representative)
    filtered = fi[
        (fi["granularity"] == "campaign") &
        (fi["quantile"] == 0.5)
    ].sort_values("importance", ascending=False).head(top_n)

    total = filtered["importance"].sum()
    records = []
    for _, row in filtered.iterrows():
        records.append({
            "feature": row["feature"],
            "importance": round(float(row["importance"]), 4),
            "percentage": round(float(row["importance"] / total * 100), 1) if total > 0 else 0,
            "explanation": _explain_feature(row["feature"]),
        })

    return {"data": records}


@app.get("/api/summary")
async def get_summary():
    """Return the rule-based causal summary as structured JSON."""
    preds = app_state["default_predictions"]
    fi = app_state["feature_importances"]
    meta = app_state["training_metadata"]

    if preds is None or preds.empty:
        return {"error": "No predictions available"}

    generator = RuleBasedSummaryGenerator(preds, fi, meta)
    summary_text = generator.generate()

    # Also build structured KPIs for the frontend
    kpis = _build_kpis(preds)

    return {
        "summary_text": summary_text,
        "kpis": kpis,
    }


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    """Re-run predictions with a new budget multiplier."""
    df_daily = app_state["df_daily"]
    bundle = app_state["model_bundle"]

    if df_daily is None or bundle is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    multiplier = max(0.1, min(3.0, req.budget_multiplier))  # Clamp range

    logger.info("Running simulation with budget_multiplier=%.2f …", multiplier)
    preds = generate_predictions(df_daily, bundle, budget_multiplier=multiplier)

    records = preds.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_dict("records")
    kpis = _build_kpis(preds)

    return {
        "data": records,
        "total": len(records),
        "budget_multiplier": multiplier,
        "kpis": kpis,
    }


@app.post("/api/llm-summary")
async def llm_summary(req: LLMRequest):
    """Generate an LLM-powered causal summary (requires API key)."""
    preds = app_state["default_predictions"]
    fi = app_state["feature_importances"]
    meta = app_state["training_metadata"]

    api_key = req.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")

    if not api_key:
        # Fallback to rule-based
        generator = RuleBasedSummaryGenerator(preds, fi, meta)
        return {
            "summary": generator.generate(),
            "source": "rule-based",
            "message": "No API key provided. Showing rule-based summary.",
        }

    try:
        from llm_summary import LLMSummaryGenerator
        generator = LLMSummaryGenerator(
            predictions=preds,
            feature_importances=fi,
            training_metadata=meta,
            provider=req.provider,
            api_key=api_key,
        )
        summary = generator.generate()
        return {"summary": summary, "source": "llm", "provider": req.provider}
    except Exception as e:
        logger.error("LLM generation failed: %s", str(e))
        generator = RuleBasedSummaryGenerator(preds, fi, meta)
        return {
            "summary": generator.generate(),
            "source": "rule-based",
            "message": f"LLM failed ({str(e)}). Showing rule-based summary.",
        }


# ---------------------------------------------------------------------------
# Static file serving — must come AFTER API routes
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    """Serve the main dashboard page."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_kpis(preds: pd.DataFrame) -> dict:
    """Build KPI summary dict from predictions."""
    kpis = {}
    for horizon in FORECAST_HORIZONS:
        ch = preds[
            (preds["forecast_horizon"] == horizon) &
            (preds["granularity"] == "channel")
        ]
        if ch.empty:
            continue

        total_p10 = float(ch["revenue_p10"].sum())
        total_p50 = float(ch["revenue_p50"].sum())
        total_p90 = float(ch["revenue_p90"].sum())
        total_spend = float(ch["projected_spend"].sum())
        blended_roas = total_p50 / total_spend if total_spend > 0 else 0

        kpis[str(horizon)] = {
            "revenue_p10": round(total_p10, 2),
            "revenue_p50": round(total_p50, 2),
            "revenue_p90": round(total_p90, 2),
            "projected_spend": round(total_spend, 2),
            "blended_roas": round(blended_roas, 4),
            "confidence_range": round(total_p90 - total_p10, 2),
            "num_channels": int(ch["channel"].nunique()),
        }

    return kpis


def _explain_feature(name: str) -> str:
    """Plain-English explanation for a feature name."""
    explanations = {
        "roas": "Return on ad spend",
        "spend": "Daily advertising expenditure",
        "clicks": "Ad clicks received",
        "impressions": "Times ads were shown",
        "daily_budget": "Daily budget cap",
        "cpc": "Cost per click",
        "cvr": "Conversion rate",
        "budget_utilization": "Budget utilisation %",
        "month": "Month of year",
        "day_of_week": "Day of week",
        "week_of_year": "Week number",
        "quarter": "Fiscal quarter",
        "is_weekend": "Weekend indicator",
        "days_since_campaign_start": "Campaign age (days)",
        "revenue_momentum": "Revenue trend ratio",
        "spend_momentum": "Spend trend ratio",
    }

    if name in explanations:
        return explanations[name]
    if "roll" in name and "revenue" in name:
        return "Rolling revenue metric"
    if "roll" in name and "spend" in name:
        return "Rolling spend metric"
    if "roll" in name and "roas" in name:
        return "Rolling ROAS metric"
    if "roll" in name and "clicks" in name:
        return "Rolling clicks metric"
    if "channel" in name:
        return "Ad channel"
    if "campaign" in name:
        return "Campaign identifier"
    return "Model feature"


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
