"""
llm_summary.py
==============
Phase 4 of the Probabilistic Revenue Forecasting pipeline.

Generates causal business summaries from forecast predictions and model
feature importances.  Two modes:

1. **Rule-based** (default, offline): Analyses predictions and feature
   importances to produce structured insights without any network calls.
   This is safe for the automated evaluation pipeline.

2. **LLM-powered** (optional, online): Sends a structured prompt to a
   Gemini / OpenAI API for richer natural-language causal narratives.
   Used during development and for the demo presentation only.

Usage (called by run.sh — rule-based only):
    python src/llm_summary.py \\
        --predictions output/predictions.csv \\
        --model pickle/model.pkl \\
        --output output/causal_summary.txt

With LLM (demo only):
    python src/llm_summary.py \\
        --predictions output/predictions.csv \\
        --model pickle/model.pkl \\
        --output output/causal_summary.txt \\
        --use-llm \\
        --llm-provider gemini \\
        --api-key YOUR_API_KEY
"""

import argparse
import json
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# RULE-BASED CAUSAL SUMMARY GENERATOR
# =========================================================================

class RuleBasedSummaryGenerator:
    """Generate structured causal insights from predictions and model data.

    This generator requires no network calls and is safe for the automated
    evaluation pipeline.  It analyses:

    - Feature importances to identify key revenue drivers
    - Prediction trends across horizons and granularities
    - Channel and campaign-type performance comparisons
    - Budget efficiency (ROAS) patterns
    - Risk indicators from prediction intervals
    """

    def __init__(
        self,
        predictions: pd.DataFrame,
        feature_importances: pd.DataFrame,
        training_metadata: Dict,
    ):
        self.predictions = predictions
        self.feature_importances = feature_importances
        self.metadata = training_metadata

    def generate(self) -> str:
        """Produce the full causal summary as a formatted string."""
        sections = [
            self._header(),
            self._executive_summary(),
            self._key_revenue_drivers(),
            self._channel_analysis(),
            self._campaign_type_analysis(),
            self._horizon_trends(),
            self._budget_efficiency(),
            self._risk_assessment(),
            self._recommendations(),
        ]
        return "\n\n".join(sections)

    # -- Header --

    def _header(self) -> str:
        date_range = self.metadata.get("date_range", ["N/A", "N/A"])
        channels = self.metadata.get("channels", [])
        return (
            "=" * 70 + "\n"
            "PROBABILISTIC REVENUE FORECAST — CAUSAL SUMMARY\n"
            "=" * 70 + "\n"
            f"Training Data: {date_range[0]} to {date_range[1]}\n"
            f"Channels: {', '.join(channels)}\n"
            f"Total Forecast Rows: {len(self.predictions)}"
        )

    # -- Executive Summary --

    def _executive_summary(self) -> str:
        lines = ["## EXECUTIVE SUMMARY", ""]

        for horizon in [30, 60, 90]:
            ch_data = self.predictions[
                (self.predictions["forecast_horizon"] == horizon) &
                (self.predictions["granularity"] == "channel")
            ]
            if ch_data.empty:
                continue

            total_p50 = ch_data["revenue_p50"].sum()
            total_p10 = ch_data["revenue_p10"].sum()
            total_p90 = ch_data["revenue_p90"].sum()
            total_spend = ch_data["projected_spend"].sum()
            overall_roas = total_p50 / total_spend if total_spend > 0 else 0

            lines.append(
                f"  {horizon}-Day Forecast:\n"
                f"    Expected Revenue (P50): ${total_p50:,.0f}\n"
                f"    Optimistic Range (P90): ${total_p90:,.0f}\n"
                f"    Conservative Range (P10): ${total_p10:,.0f}\n"
                f"    Projected Spend: ${total_spend:,.0f}\n"
                f"    Blended ROAS: {overall_roas:.2f}x"
            )

        return "\n".join(lines)

    # -- Key Revenue Drivers --

    def _key_revenue_drivers(self) -> str:
        lines = ["## KEY REVENUE DRIVERS", ""]

        # Get P50 (median model) feature importances at campaign level
        campaign_imp = self.feature_importances[
            (self.feature_importances["granularity"] == "campaign") &
            (self.feature_importances["quantile"] == 0.5)
        ].sort_values("importance", ascending=False)

        if campaign_imp.empty:
            return "\n".join(lines + ["  No feature importance data available."])

        top_features = campaign_imp.head(10)
        total_importance = campaign_imp["importance"].sum()

        lines.append("  Top 10 features driving revenue predictions:\n")

        for rank, (_, row) in enumerate(top_features.iterrows(), 1):
            pct = (row["importance"] / total_importance * 100) if total_importance > 0 else 0
            feat_name = row["feature"]
            explanation = self._explain_feature(feat_name)
            lines.append(
                f"    {rank:2d}. {feat_name:<30s} "
                f"({pct:5.1f}% importance) — {explanation}"
            )

        # Causal narrative
        top3 = top_features.head(3)["feature"].tolist()
        lines.append("")
        lines.append("  Causal Interpretation:")
        lines.append(
            f"    Revenue is primarily driven by {top3[0]}, "
            f"{top3[1]}, and {top3[2]}. "
        )

        if "roas" in top3 or "roas_roll_7d_mean" in top3:
            lines.append(
                "    The strong influence of ROAS metrics suggests that "
                "campaigns with historically efficient ad spend are the "
                "best predictors of future revenue."
            )
        if "spend" in top3:
            lines.append(
                "    Spend level is a significant driver, indicating that "
                "budget allocation directly impacts revenue outcomes."
            )
        if any("roll" in f for f in top3):
            lines.append(
                "    Rolling average features rank highly, showing that "
                "recent performance trends are more predictive than "
                "point-in-time metrics."
            )

        return "\n".join(lines)

    def _explain_feature(self, feature_name: str) -> str:
        """Return a plain-English explanation for a feature name."""
        explanations = {
            "roas": "Historical return on ad spend (revenue per dollar spent)",
            "spend": "Daily advertising expenditure",
            "clicks": "Number of ad clicks received",
            "impressions": "Number of times ads were shown",
            "daily_budget": "Configured daily budget cap",
            "cpc": "Cost per click (spend efficiency)",
            "cvr": "Conversion rate (conversions per click)",
            "budget_utilization": "Fraction of daily budget actually spent",
            "month": "Month of year (seasonality signal)",
            "day_of_week": "Day of week (weekly patterns)",
            "week_of_year": "Week number (annual seasonality)",
            "quarter": "Fiscal quarter",
            "is_weekend": "Weekend vs weekday indicator",
            "days_since_campaign_start": "Campaign maturity / age in days",
            "revenue_momentum": "Short-term vs long-term revenue trend ratio",
            "spend_momentum": "Short-term vs long-term spend trend ratio",
        }

        # Check exact match first
        if feature_name in explanations:
            return explanations[feature_name]

        # Rolling features
        if "revenue_roll" in feature_name:
            if "std" in feature_name:
                window = feature_name.split("_")[2]
                return f"Revenue volatility over {window} window"
            window = feature_name.split("_")[2]
            return f"Average daily revenue over {window} window"

        if "spend_roll" in feature_name:
            if "std" in feature_name:
                window = feature_name.split("_")[2]
                return f"Spend volatility over {window} window"
            window = feature_name.split("_")[2]
            return f"Average daily spend over {window} window"

        if "roas_roll" in feature_name:
            window = feature_name.split("_")[2]
            return f"Average ROAS over {window} window"

        if "clicks_roll" in feature_name:
            window = feature_name.split("_")[2]
            return f"Average daily clicks over {window} window"

        if "channel" in feature_name:
            return "Advertising channel (Google, Bing, Meta)"

        if "campaign_type" in feature_name:
            return "Campaign type (Search, PMax, Display, etc.)"

        if "campaign_name" in feature_name:
            return "Individual campaign identifier"

        return "Model feature"

    # -- Channel Analysis --

    def _channel_analysis(self) -> str:
        lines = ["## CHANNEL PERFORMANCE ANALYSIS", ""]

        ch_data = self.predictions[
            (self.predictions["granularity"] == "channel") &
            (self.predictions["forecast_horizon"] == 30)
        ]

        if ch_data.empty:
            return "\n".join(lines + ["  No channel-level data available."])

        # Sort by P50 revenue descending
        ch_data = ch_data.sort_values("revenue_p50", ascending=False)

        total_revenue = ch_data["revenue_p50"].sum()

        for _, row in ch_data.iterrows():
            channel = row["channel"]
            rev_share = (row["revenue_p50"] / total_revenue * 100) if total_revenue > 0 else 0
            uncertainty = row["revenue_p90"] - row["revenue_p10"]
            uncertainty_pct = (uncertainty / row["revenue_p50"] * 100) if row["revenue_p50"] > 0 else 0

            lines.append(f"  {channel.upper()}")
            lines.append(f"    30-Day P50 Revenue: ${row['revenue_p50']:,.0f} ({rev_share:.0f}% of total)")
            lines.append(f"    Revenue Range: ${row['revenue_p10']:,.0f} — ${row['revenue_p90']:,.0f}")
            lines.append(f"    ROAS (P50): {row['roas_p50']:.2f}x")
            lines.append(f"    Projected Spend: ${row['projected_spend']:,.0f}")
            lines.append(f"    Forecast Uncertainty: ±{uncertainty_pct:.0f}%")

            # Performance assessment
            if row["roas_p50"] > 3.0:
                lines.append("    Assessment: HIGH PERFORMER — strong return on investment")
            elif row["roas_p50"] > 1.0:
                lines.append("    Assessment: PROFITABLE — generating positive returns")
            elif row["roas_p50"] > 0.5:
                lines.append("    Assessment: MARGINAL — close to break-even")
            else:
                lines.append("    Assessment: UNDERPERFORMING — low return on spend")

            lines.append("")

        return "\n".join(lines)

    # -- Campaign Type Analysis --

    def _campaign_type_analysis(self) -> str:
        lines = ["## CAMPAIGN TYPE ANALYSIS", ""]

        ct_data = self.predictions[
            (self.predictions["granularity"] == "campaign_type") &
            (self.predictions["forecast_horizon"] == 30)
        ]

        if ct_data.empty:
            return "\n".join(lines + ["  No campaign-type data available."])

        ct_data = ct_data.sort_values("revenue_p50", ascending=False)

        lines.append("  30-Day Revenue Forecast by Campaign Type:\n")
        lines.append(f"  {'Campaign Type':<25s} {'Channel':<10s} {'P50 Revenue':>12s} {'ROAS':>8s} {'Risk':>8s}")
        lines.append("  " + "-" * 65)

        for _, row in ct_data.iterrows():
            uncertainty = row["revenue_p90"] - row["revenue_p10"]
            risk = "HIGH" if row["revenue_p50"] > 0 and uncertainty / max(row["revenue_p50"], 1) > 1.5 else "MED" if row["revenue_p50"] > 0 and uncertainty / max(row["revenue_p50"], 1) > 0.5 else "LOW"

            lines.append(
                f"  {row['campaign_type']:<25s} "
                f"{row['channel']:<10s} "
                f"${row['revenue_p50']:>10,.0f} "
                f"{row['roas_p50']:>7.2f}x "
                f"{risk:>7s}"
            )

        # Find best and worst performers
        best = ct_data.iloc[0]
        worst = ct_data.iloc[-1]

        lines.append("")
        lines.append(f"  Top Performer: {best['campaign_type']} ({best['channel']}) — "
                     f"${best['revenue_p50']:,.0f} revenue, {best['roas_p50']:.2f}x ROAS")
        lines.append(f"  Lowest Revenue: {worst['campaign_type']} ({worst['channel']}) — "
                     f"${worst['revenue_p50']:,.0f} revenue, {worst['roas_p50']:.2f}x ROAS")

        return "\n".join(lines)

    # -- Horizon Trends --

    def _horizon_trends(self) -> str:
        lines = ["## FORECAST HORIZON TRENDS", ""]

        for horizon in [30, 60, 90]:
            ch = self.predictions[
                (self.predictions["granularity"] == "channel") &
                (self.predictions["forecast_horizon"] == horizon)
            ]
            if ch.empty:
                continue

            total_p50 = ch["revenue_p50"].sum()
            total_spend = ch["projected_spend"].sum()
            avg_monthly = total_p50 / (horizon / 30)

            lines.append(
                f"  {horizon}-Day: Total P50 = ${total_p50:,.0f} "
                f"(~${avg_monthly:,.0f}/month), "
                f"Spend = ${total_spend:,.0f}"
            )

        # Trend analysis
        horizons_data = {}
        for h in [30, 60, 90]:
            ch = self.predictions[
                (self.predictions["granularity"] == "channel") &
                (self.predictions["forecast_horizon"] == h)
            ]
            if not ch.empty:
                horizons_data[h] = ch["revenue_p50"].sum()

        if 30 in horizons_data and 90 in horizons_data:
            monthly_30 = horizons_data[30]
            monthly_90 = horizons_data[90] / 3
            if monthly_30 > 0:
                growth = (monthly_90 - monthly_30) / monthly_30 * 100
                lines.append("")
                if growth > 5:
                    lines.append(
                        f"  Trend: GROWING — monthly revenue expected to "
                        f"increase {growth:.0f}% over 90 days"
                    )
                elif growth < -5:
                    lines.append(
                        f"  Trend: DECLINING — monthly revenue expected to "
                        f"decrease {abs(growth):.0f}% over 90 days"
                    )
                else:
                    lines.append(
                        "  Trend: STABLE — monthly revenue expected to "
                        "remain roughly flat over 90 days"
                    )

        return "\n".join(lines)

    # -- Budget Efficiency --

    def _budget_efficiency(self) -> str:
        lines = ["## BUDGET EFFICIENCY ANALYSIS", ""]

        ch_30 = self.predictions[
            (self.predictions["granularity"] == "channel") &
            (self.predictions["forecast_horizon"] == 30)
        ]

        if ch_30.empty:
            return "\n".join(lines + ["  No data available."])

        lines.append("  Channel Efficiency (30-Day Horizon):\n")

        for _, row in ch_30.sort_values("roas_p50", ascending=False).iterrows():
            channel = row["channel"]
            hist_rev = row.get("historical_daily_revenue", 0)
            hist_spend = row.get("historical_daily_spend", 0)
            hist_roas = hist_rev / hist_spend if hist_spend > 0 else 0

            lines.append(f"  {channel.upper()}")
            lines.append(f"    Historical Daily Revenue: ${hist_rev:,.0f}")
            lines.append(f"    Historical Daily Spend: ${hist_spend:,.0f}")
            lines.append(f"    Historical ROAS: {hist_roas:.2f}x")
            lines.append(f"    Forecast P50 ROAS: {row['roas_p50']:.2f}x")

            if row["roas_p50"] > hist_roas * 1.1:
                lines.append("    Outlook: IMPROVING — forecast ROAS exceeds historical")
            elif row["roas_p50"] < hist_roas * 0.9:
                lines.append("    Outlook: DECLINING — forecast ROAS below historical")
            else:
                lines.append("    Outlook: STABLE — forecast ROAS consistent with historical")

            lines.append("")

        return "\n".join(lines)

    # -- Risk Assessment --

    def _risk_assessment(self) -> str:
        lines = ["## RISK ASSESSMENT", ""]

        campaign_30 = self.predictions[
            (self.predictions["granularity"] == "campaign") &
            (self.predictions["forecast_horizon"] == 30)
        ]

        if campaign_30.empty:
            return "\n".join(lines + ["  No campaign-level data available."])

        # High uncertainty campaigns
        campaign_30 = campaign_30.copy()
        campaign_30["uncertainty_range"] = campaign_30["revenue_p90"] - campaign_30["revenue_p10"]
        campaign_30["uncertainty_pct"] = np.where(
            campaign_30["revenue_p50"] > 0,
            campaign_30["uncertainty_range"] / campaign_30["revenue_p50"] * 100,
            0,
        )

        high_risk = campaign_30[campaign_30["uncertainty_pct"] > 150].sort_values(
            "uncertainty_pct", ascending=False
        )

        if len(high_risk) > 0:
            lines.append(f"  HIGH UNCERTAINTY CAMPAIGNS ({len(high_risk)} identified):\n")
            for _, row in high_risk.head(5).iterrows():
                lines.append(
                    f"    - {row['campaign_name']} ({row['channel']}): "
                    f"P10=${row['revenue_p10']:,.0f}, "
                    f"P50=${row['revenue_p50']:,.0f}, "
                    f"P90=${row['revenue_p90']:,.0f} "
                    f"(±{row['uncertainty_pct']:.0f}%)"
                )
        else:
            lines.append("  No high-uncertainty campaigns identified.")

        # Zero-revenue campaigns
        zero_rev = campaign_30[campaign_30["revenue_p50"] == 0]
        if len(zero_rev) > 0:
            lines.append(f"\n  ZERO-REVENUE CAMPAIGNS ({len(zero_rev)} identified):")
            lines.append("    These campaigns show zero median revenue forecast:")
            for _, row in zero_rev.iterrows():
                lines.append(
                    f"    - {row['campaign_name']} ({row['channel']}, "
                    f"{row['campaign_type']}): "
                    f"Spend=${row['projected_spend']:,.0f}"
                )

        return "\n".join(lines)

    # -- Recommendations --

    def _recommendations(self) -> str:
        lines = ["## ACTIONABLE RECOMMENDATIONS", ""]

        ch_30 = self.predictions[
            (self.predictions["granularity"] == "channel") &
            (self.predictions["forecast_horizon"] == 30)
        ]

        if ch_30.empty:
            return "\n".join(lines)

        rec_num = 1

        # Sort channels by ROAS
        ch_sorted = ch_30.sort_values("roas_p50", ascending=False)
        best_ch = ch_sorted.iloc[0]
        worst_ch = ch_sorted.iloc[-1]

        # Recommendation 1: Budget reallocation
        if best_ch["roas_p50"] > worst_ch["roas_p50"] * 2:
            lines.append(
                f"  {rec_num}. BUDGET REALLOCATION: Consider shifting budget from "
                f"{worst_ch['channel']} (ROAS: {worst_ch['roas_p50']:.2f}x) to "
                f"{best_ch['channel']} (ROAS: {best_ch['roas_p50']:.2f}x) "
                f"to improve overall portfolio ROAS."
            )
            rec_num += 1

        # Recommendation 2: Campaign pruning
        campaign_30 = self.predictions[
            (self.predictions["granularity"] == "campaign") &
            (self.predictions["forecast_horizon"] == 30)
        ]
        zero_rev = campaign_30[campaign_30["revenue_p50"] == 0]
        if len(zero_rev) > 0:
            wasted_spend = zero_rev["projected_spend"].sum()
            lines.append(
                f"  {rec_num}. CAMPAIGN PRUNING: {len(zero_rev)} campaigns have zero "
                f"median revenue forecast but ${wasted_spend:,.0f} in projected spend. "
                f"Review these campaigns for potential pause or restructuring."
            )
            rec_num += 1

        # Recommendation 3: Uncertainty reduction
        high_uncertainty_count = len(campaign_30[
            campaign_30.apply(
                lambda r: (r["revenue_p90"] - r["revenue_p10"]) / max(r["revenue_p50"], 1) > 1.5,
                axis=1,
            )
        ])
        if high_uncertainty_count > 0:
            lines.append(
                f"  {rec_num}. DATA QUALITY: {high_uncertainty_count} campaigns have "
                f"forecast uncertainty exceeding 150%. Collecting more data or "
                f"adding conversion tracking may improve prediction confidence."
            )
            rec_num += 1

        # Recommendation 4: Budget simulation
        lines.append(
            f"  {rec_num}. SCENARIO PLANNING: Run the prediction pipeline with "
            f"--budget-multiplier 1.2 and 0.8 to model the impact of "
            f"±20% budget changes on forecasted revenue."
        )

        lines.append("\n" + "=" * 70)
        lines.append("End of Causal Summary Report")
        lines.append("=" * 70)

        return "\n".join(lines)


# =========================================================================
# LLM-POWERED SUMMARY GENERATOR (optional, for demo)
# =========================================================================

class LLMSummaryGenerator:
    """Generate causal summaries using an LLM API.

    Supports Google Gemini and OpenAI.  Requires an API key.
    This class is NOT used in the automated pipeline — it's for
    development and demo purposes only.
    """

    def __init__(
        self,
        predictions: pd.DataFrame,
        feature_importances: pd.DataFrame,
        training_metadata: Dict,
        provider: str = "gemini",
        api_key: Optional[str] = None,
    ):
        self.predictions = predictions
        self.feature_importances = feature_importances
        self.metadata = training_metadata
        self.provider = provider.lower()
        self.api_key = api_key

    def _build_prompt(self) -> str:
        """Build a structured prompt for the LLM."""

        # Summarise predictions for the prompt
        ch_30 = self.predictions[
            (self.predictions["granularity"] == "channel") &
            (self.predictions["forecast_horizon"] == 30)
        ]

        prediction_summary = ch_30.to_dict("records") if not ch_30.empty else []

        # Top features
        top_features = self.feature_importances[
            (self.feature_importances["granularity"] == "campaign") &
            (self.feature_importances["quantile"] == 0.5)
        ].sort_values("importance", ascending=False).head(10)
        feature_list = top_features[["feature", "importance"]].to_dict("records")

        prompt = f"""You are an expert digital marketing analyst. Analyse the following 
probabilistic revenue forecast data and provide a causal business summary.

## Training Data Context
- Date range: {self.metadata.get('date_range', ['N/A', 'N/A'])}
- Channels: {self.metadata.get('channels', [])}
- Campaign types: {self.metadata.get('campaign_types', [])}

## 30-Day Channel-Level Forecasts
{json.dumps(prediction_summary, indent=2, default=str)}

## Top 10 Revenue-Driving Features (by importance)
{json.dumps(feature_list, indent=2, default=str)}

## Instructions
Please provide:
1. An executive summary of the revenue outlook (2-3 sentences)
2. Which channels and campaign types are driving revenue and why
3. Key risks and uncertainties in the forecast
4. Specific, actionable budget optimisation recommendations
5. Any seasonal or trend-based factors affecting the forecast

Format the response as a professional business report. Be specific with 
numbers and percentages. Focus on CAUSAL explanations — not just what is 
happening, but WHY it is happening based on the feature importances."""

        return prompt

    def generate(self) -> str:
        """Generate the LLM-powered summary."""
        prompt = self._build_prompt()

        if self.provider == "gemini":
            return self._call_gemini(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            logger.error("Unknown LLM provider: %s", self.provider)
            return f"Error: Unknown provider '{self.provider}'. Use 'gemini' or 'openai'."

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            return ("Error: google-generativeai package not installed. "
                    "Run: pip install google-generativeai")

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            return ("Error: openai package not installed. "
                    "Run: pip install openai")

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert digital marketing analyst."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        return response.choices[0].message.content


# =========================================================================
# MAIN
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate causal summaries from revenue forecasts."
    )
    parser.add_argument(
        "--predictions",
        default="./output/predictions.csv",
        help="Path to predictions CSV (default: ./output/predictions.csv)",
    )
    parser.add_argument(
        "--model",
        default="./pickle/model.pkl",
        help="Path to model bundle pickle (default: ./pickle/model.pkl)",
    )
    parser.add_argument(
        "--output",
        default="./output/causal_summary.txt",
        help="Output path for the causal summary (default: ./output/causal_summary.txt)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM API instead of rule-based generator (requires --api-key)",
    )
    parser.add_argument(
        "--llm-provider",
        default="gemini",
        choices=["gemini", "openai"],
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM provider (can also set via environment variable)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load predictions
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 4: CAUSAL SUMMARY GENERATION")
    logger.info("=" * 60)

    logger.info("Loading predictions from %s …", args.predictions)

    if not os.path.exists(args.predictions):
        logger.error("Predictions file not found: %s", args.predictions)
        sys.exit(1)

    predictions = pd.read_csv(args.predictions)
    logger.info("  Loaded %d forecast rows", len(predictions))

    # ------------------------------------------------------------------
    # 2. Load model bundle (for feature importances & metadata)
    # ------------------------------------------------------------------
    logger.info("Loading model bundle from %s …", args.model)

    if not os.path.exists(args.model):
        logger.error("Model bundle not found: %s", args.model)
        sys.exit(1)

    with open(args.model, "rb") as f:
        model_bundle = pickle.load(f)

    feature_importances = model_bundle.get("feature_importances", pd.DataFrame())
    training_metadata = model_bundle.get("training_metadata", {})

    logger.info(
        "  Feature importances: %d rows, Training range: %s → %s",
        len(feature_importances),
        training_metadata.get("date_range", ["N/A"])[0],
        training_metadata.get("date_range", ["N/A", "N/A"])[1],
    )

    # ------------------------------------------------------------------
    # 3. Generate summary
    # ------------------------------------------------------------------
    if args.use_llm:
        logger.info("Using LLM-powered generator (%s) …", args.llm_provider)
        api_key = args.api_key or os.environ.get("LLM_API_KEY")
        if not api_key:
            logger.error(
                "No API key provided. Use --api-key or set LLM_API_KEY "
                "environment variable."
            )
            sys.exit(1)

        generator = LLMSummaryGenerator(
            predictions=predictions,
            feature_importances=feature_importances,
            training_metadata=training_metadata,
            provider=args.llm_provider,
            api_key=api_key,
        )
    else:
        logger.info("Using rule-based generator (offline mode) …")
        generator = RuleBasedSummaryGenerator(
            predictions=predictions,
            feature_importances=feature_importances,
            training_metadata=training_metadata,
        )

    summary = generator.generate()

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info("Causal summary written to %s (%d chars)", args.output, len(summary))

    # Also print to stdout for convenience
    print()
    print(summary)

    logger.info("=" * 60)
    logger.info("SUMMARY GENERATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
