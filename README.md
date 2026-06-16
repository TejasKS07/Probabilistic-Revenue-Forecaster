# Probabilistic Revenue Forecaster

This project is a submission for the **AIgnition 2026 Hackathon**. It provides a probabilistic machine learning pipeline to forecast digital advertising revenue and Return on Ad Spend (ROAS) across multiple channels (Google, Bing, Meta).

## 🚀 Quick Start

The pipeline is fully automated and meets the AIgnition submission requirements for the headless scoring environment.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the automated prediction pipeline
# Usage: ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

The script will automatically output:
1. `output/predictions.csv` — The probabilistic P10/P50/P90 forecasts.
2. `output/causal_summary.txt` — A structured business insights report generated completely offline.

## 🧠 Methodology & Architecture

The forecasting pipeline is built around **LightGBM Quantile Regression**, trained on daily aggregated campaign metrics.

### 1. Data Processing & Feature Engineering (`src/generate_features.py`)
- **Cleaning**: Drops zero-activity rows (0 spend, 0 revenue) and forward-fills missing daily budget data to maintain temporal continuity.
- **Feature Engineering**: Calculates rolling windows (7d, 14d, 30d) for revenue, spend, ROAS, and clicks. Extracts momentum indicators (short-term vs long-term ratios) to capture trend velocity.
- **Leakage Prevention**: Safely excludes the Meta `conversions` column, which is treated as a direct revenue proxy to align with other channel structures.

### 2. Probabilistic Modeling (`src/train.py`)
- **Hierarchical Training**: Instead of training one massive model, the pipeline trains 9 distinct models (P10, P50, P90 quantiles across Channel, Campaign Type, and Campaign granularities).
- **Time-Series CV**: Validated using a 5-fold expanding window cross-validation to ensure out-of-sample temporal robustness.
- **Model Bundle**: Models, label encoders, and feature names are bundled into a single `pickle/model.pkl` artifact (7.5MB).

### 3. Prediction & Simulation (`src/predict.py`)
- **Forecast Generation**: Applies the pre-trained quantile models to the most recent lookback windows to generate 30, 60, and 90-day forecasts.
- **Budget What-If Analysis**: Includes a `--budget-multiplier` argument (e.g., `1.2` for +20% budget) that scales projected spend and dynamically shifts the revenue distribution, calculating a new predicted ROAS based on the simulated outcome.

### 4. Causal Summary Engine (`src/llm_summary.py`)
- **Offline Rule-Based Generator**: The default mode used in `run.sh` requires no network calls. It parses the predictions and model feature importances to highlight top revenue drivers, identify wasted spend (zero-revenue campaigns), flag high-uncertainty risks, and formulate actionable recommendations.
- **Online LLM Demo**: Includes an optional `--use-llm` flag that hits the Gemini/OpenAI API with a structured prompt, intended specifically for the live demo presentation.

## ⚠️ Assumptions & Limitations

1. **Meta Conversion Mapping**: The `conversions` column in the Meta Ads dataset contains high-magnitude monetary values rather than integer counts. To unify the schema across channels, we explicitly assume this column represents `revenue`.
2. **Stationarity vs Seasonality**: The dataset captures roughly 2.5 years of data. While week-of-year features capture annual seasonality, the model prioritises 7-30 day momentum features, assuming recent performance is highly indicative of near-term future returns. Extreme black-swan market shifts are not modeled.
3. **Budget Multiplier Linearity**: The budget simulation tool anchors predictions on historical averages. Pushing the `--budget-multiplier` to extreme values (e.g., `5.0x`) will scale projected revenue proportionally but may under-represent the natural diminishing returns (ad fatigue/saturation) found at massive scale.

## 📁 Repository Structure

```text
├── data/                       # Raw input CSVs
├── pickle/
│   └── model.pkl               # Bundled LightGBM models (submitted artifact)
├── src/
│   ├── generate_features.py    # Phase 1: Data prep
│   ├── train.py                # Phase 2: Model training
│   ├── predict.py              # Phase 3: Forecast generation
│   └── llm_summary.py          # Phase 4: Insights engine
├── output/                     # Generated forecasts and summaries
├── run.sh                      # Entry point script
├── requirements.txt            # Pinned dependencies
└── README.md                   # Project documentation
```