# Marketing Mix Modelling in Python

An end-to-end, reproducible Marketing Mix Modelling (MMM) workflow built on a **synthetic** weekly marketing dataset — from multi-file data preparation through data quality auditing, adstock/saturation modelling, and bounded budget scenario simulation.

> **This is an educational project.** The dataset is synthetic and the model is deliberately simplified. Contribution estimates and proxy ROI figures are **directional diagnostics, not causal estimates** and must not be used to make real budget decisions without experimental calibration.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Methodology](#methodology)
- [Outputs](#outputs)
- [Limitations and Caveats](#limitations-and-caveats)
- [Possible Extensions](#possible-extensions)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Marketing Mix Modelling attributes sales to media investment and control factors so that planners can compare channel efficiency and explore reallocation. This repository implements that workflow end to end on a synthetic portfolio dataset.

**Data:** 209 continuous weekly observations (2022-01-03 to 2025-12-29, Monday week-start), covering:

| Group | Fields |
| --- | --- |
| Target | `sales` |
| Media (7 channels) | `instagram_spend`, `google_ads_spend`, `tv_spend`, `youtube_spend`, `newspaper_spend`, `influencer_spend`, `ott_spend` |
| Controls | `competitor_spend`, `holiday`, `sales_promotion` |
| Engineered controls | linear trend, annual Fourier seasonality terms |

**Design principles behind the code:**

- **Reusable modules, thin notebooks.** All logic lives in `src/`; notebooks orchestrate and narrate.
- **Auditable data treatment.** Every imputation is logged rather than silently applied.
- **Chronological validation.** No random shuffling — the holdout window is strictly later than training.
- **Explicit caveats.** Each finding is reported as *Finding → Business meaning → Caution*, so association is never presented as causation.

---

## Key Results

### Model selection

Ten specifications were compared on a chronological 80/20 validation window: one raw-media Ridge baseline plus nine adstock × saturation candidates.

| Model | Decay | Half-point mult. | RMSE | MAE | R² |
| --- | --- | --- | --- | --- | --- |
| **Transformed Ridge (selected)** | **0.5** | **0.5** | **100,985** | **79,425** | **0.941** |
| Transformed Ridge | 0.5 | 1.0 | 112,044 | 82,632 | 0.927 |
| Transformed Ridge | 0.2 | 0.5 | 197,128 | 159,773 | 0.774 |
| Baseline Ridge (raw media) | — | — | 277,454 | 227,162 | 0.553 |

Applying carryover and diminishing-returns transformations reduced validation RMSE by **~63.6%** versus the raw-media baseline.

<img width="635" height="299" alt="image" src="https://github.com/user-attachments/assets/ae782a1c-11db-4505-b175-a05ca6f5d5e2" />


*Actual vs. predicted weekly sales across the held-out validation window.*

### Estimated channel contribution and proxy ROI

Contribution is estimated by zeroing one raw channel at a time and re-scoring the fitted model.

| Channel | Contribution share | Total spend | Proxy ROI |
| --- | --- | --- | --- |
| Instagram | 26.3% | 11,440,220 | 12.35 |
| Google Ads | 22.7% | 12,024,860 | 10.14 |
| OTT | 14.5% | 9,180,190 | 8.50 |
| Influencer | 12.0% | 4,706,220 | 13.72 |
| TV | 10.9% | 7,870,450 | 7.46 |
| Newspaper | 6.9% | 1,069,650 | 34.59 |
| YouTube | 6.5% | 5,989,090 | 5.85 |

Instagram and Google Ads dominate absolute contribution because of scale, while Newspaper shows the highest proxy ROI off a small spend base — a classic signal to test rather than to act on directly.

<img width="1265" height="706" alt="estimated_channel_contribution" src="https://github.com/user-attachments/assets/125f2e41-12a1-4884-bbba-b43513dc9614" />


<img width="1265" height="706" alt="proxy_roi_by_channel" src="https://github.com/user-attachments/assets/0817da1a-5030-43b5-9317-ba8a3ecb1fb7" />


*Contribution ranks channels by absolute modelled impact; proxy ROI rescales that impact by spend, which reorders the list entirely.*

### Budget scenarios

Five bounded weekly-budget scenarios, each channel capped at 2× its historical active-week mean:

| Scenario | Weekly budget | Predicted sales | Change |
| --- | --- | --- | --- |
| Current | 260,358 | 2,592,496 | — |
| **ROI Reallocation** | 260,358 | 2,668,466 | **+2.93%** |
| Digital Focus | 260,358 | 2,523,831 | −2.65% |
| TV Cut | 233,645 | 2,513,632 | −3.04% |
| 20% Budget Reduction | 208,287 | 2,347,383 | −9.46% |

<img width="1424" height="785" alt="budget_scenario_comparison" src="https://github.com/user-attachments/assets/c7be89d3-f152-4161-938f-1d937c416cce" />


*Digital Focus* and *ROI Reallocation* hold total budget constant; the other two intentionally do not. Marginal analysis at the 100%→200% boundary shows Google Ads with the largest modelled uplift, followed by Instagram.

---

## Project Structure

```
marketing-mmm-python/
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Multi-file ingest, weekly alignment, treatment log
│   ├── 02_data_quality_and_eda.ipynb   # Quality audit, issue register, 5 business charts
│   ├── 03_simplified_mmm.ipynb         # Adstock + saturation + Ridge, contribution & ROI
│   └── 04_budget_scenarios.ipynb       # Response curves and 5 bounded scenarios
├── src/
│   ├── config.py                       # Centralised project paths
│   ├── sample_extracts.py              # Splits source into 4 client-style extracts
│   ├── data_prep.py                    # Standardisation, weekly alignment, merge, treatment log
│   ├── quality.py                      # Quality summary, issue register, gap and correlation checks
│   ├── transformations.py              # geometric_adstock, hill_saturation, transform_media
│   ├── modeling.py                     # Control features, chronological split, candidate fitting
│   ├── scenarios.py                    # Response curves, scenario construction and scoring
│   └── visualization.py                # All chart functions
├── data/
│   ├── raw/source/                     # Unmodified source copy (never overwritten)
│   ├── raw/client_extracts/            # Generated multi-format extracts
│   └── processed/                      # modelling_ready_data.csv
└── output/
    ├── reports/                        # 18 CSV summaries
    ├── charts/                         # 10 PNG charts
    └── model/                          # model_bundle.joblib, model_metadata.json
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/marketing-mmm-python.git
cd marketing-mmm-python

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Source data

Notebook 01 expects the source CSV at `data/raw/source/mmm_dataset.csv`. If it is not present, the notebook falls back to a local download path — **update that path before running**, or simply place your CSV at the expected location. The source file is copied, never modified in place.

### Running

Run the notebooks in order; each one depends on the artefacts of the previous.

```bash
jupyter lab notebooks/
```

`01 → 02 → 03 → 04`. Each notebook raises a clear `FileNotFoundError` if a prerequisite output is missing.

---

## Notebook Walkthrough

### 01 — Data Preparation

Splits the source into four client-style extracts (sales, digital media, offline media, market context) to demonstrate a refreshable multi-file workflow, then reconciles them into one continuous weekly table.

- Standardises column names and snaps all dates to canonical Monday week-starts
- Flags original media blanks in a `_was_missing` column **before** zero-filling
- Interpolates competitor-spend gaps in weekly order
- **Retains** 3 missing sales values rather than imputing the target
- Validates with assertions: exactly 209 rows, unique and monotonic dates, complete Monday calendar
- Exports `modelling_ready_data.csv` and `data_treatment_log.csv`

### 02 — Data Quality and EDA

Audits the untouched source **alongside** the prepared table so source issues stay visible after treatment.

- Quality summary per field: missing rate, zero rate, negatives, variance, coefficient of variation, IQR outlier count
- Media missingness in the source ranges from 24% (Instagram, OTT) to 38% (Google Ads); competitor spend 4.3%
- Issue register with severity levels; date-gap check returns zero missing weeks
- Correlation diagnostics at a 0.80 absolute threshold — **no pairs flagged**, supporting the choice of Ridge as a mild guard rather than a necessity
- Descriptive summaries by year, quarter, promotion type, and holiday flag

#### The five business charts

**1. Source missingness by field**

<img width="1265" height="706" alt="data_missingness" src="https://github.com/user-attachments/assets/78e29c22-d013-4f42-8107-d9de62e28f2a" />


Media fields carry substantial source blanks — from ~24% (Instagram, OTT) up to ~38% (Google Ads). Each blank is flagged before any zero-fill, so the treatment stays auditable rather than invisible.

**2. Weekly sales trend**

<img width="1584" height="786" alt="weekly_sales_trend" src="https://github.com/user-attachments/assets/61d71f44-6e3c-4fa4-bc78-7c55b2a1b924" />


The observed 13-week average moves from roughly 1.35M at the start of the series to roughly 2.57M at the end, with visible seasonal structure. Trend and seasonality must be controlled before any media coefficient is interpreted.

**3. Sales by promotion type**

<img width="1264" height="706" alt="promotion_sales_comparison" src="https://github.com/user-attachments/assets/ea03701c-c9b1-4efc-beee-9df47243f34c" />


Coupons weeks show the highest average sales (~2.39M) and Normal weeks the lowest (~2.24M). These are unadjusted group comparisons, not incremental lift — but they justify keeping promotion as a control.

**4. Channel spend and activity**

<img width="1584" height="786" alt="channel_spend_activity" src="https://github.com/user-attachments/assets/6db995ab-6ca3-423e-a413-8d4d12d21285" />


Google Ads and Instagram lead on total spend, but Google Ads is active in only ~62% of weeks versus ~76% for Instagram. Scale and flighting differ across channels, which is exactly why multivariate adjustment is needed.

**5. Correlation heatmap**

<img width="1118" height="945" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/cc89bfa3-9e2d-45f5-b414-d7ba111f9899" />


No pair exceeds the 0.80 absolute threshold, so severe multicollinearity is not present. Google Ads shows the strongest raw association with sales (0.42) — association only, and plausibly driven by shared seasonality or budget timing.

### 03 — Simplified MMM

The modelling core.

1. Apply **geometric adstock** (carryover) then **Hill saturation** (diminishing returns) to each media channel
2. Add controls: competitor spend, holiday, promotion, linear trend, annual Fourier terms
3. Fit `StandardScaler → Ridge(alpha=1.0)` inside a pipeline, on the training period only
4. Compare 1 baseline + 9 candidates over a deterministic decay × half-point grid
5. Estimate contribution by zeroing each raw channel and re-scoring; negative estimates are retained rather than clipped
6. Persist the model bundle and a metadata JSON that records the exact parameters, cutoff dates, metrics, and an explicit `"causal_claim": false` flag

### 04 — Budget Scenarios

1. Build 0%–200% response curves per channel (21 points), varying one channel while holding others at representative recent levels
2. Construct five scenarios from recent weekly budgets, capped at 2× historical active-week mean
3. Assert non-negativity, cap compliance, and budget-neutrality for the two reallocation cases
4. Score each scenario and compare against Current
5. Inspect marginal direction at the 100%→200% boundary as a sensitivity signal

![Modelled response curves from 0% to 200% of recent weekly spend](output/charts/media_response_curves.png)

*Each curve shows modelled sales as one channel's weekly spend varies from 0% to 200% of its recent average, all else held representative. The flattening slope is the saturation term doing its work — and the 200% end is an explicit extrapolation boundary, not a validated prediction.*

---

## Methodology

### Adstock (carryover)

Media effects persist beyond the week of spend. Geometric adstock applies an exponential decay:

```
adstocked[t] = spend[t] + decay × adstocked[t-1]
```

The selected `decay = 0.5` means roughly half of a week's effect carries into the next.

### Hill saturation (diminishing returns)

Doubling spend does not double response. Hill saturation maps adstocked spend to a bounded 0–1 response, with a half-point (the spend level at 50% of maximum response) derived per channel from its own spend distribution scaled by the half-point multiplier.

### Validation

A **chronological** 80/20 split holds out the final ~41 observed weeks. This mimics real forecasting, where you predict forward in time — random k-fold would leak future information into training.

### Controls

Trend and annual Fourier terms absorb baseline growth and seasonality, so media coefficients are not credited with movements that seasonality already explains. Promotion, holiday, and competitor spend cover the main non-media drivers present in the data.

---

## Outputs

**Reports (`output/reports/`, 18 CSVs)** — treatment log, quality summary, issue register, date gaps, correlation matrix and flagged pairs, annual/quarterly/distribution/promotion/holiday sales summaries, channel summary, model comparison, validation predictions, contribution & ROI, response curves, scenario allocations and results.

**Charts (`output/charts/`, 10 PNGs)** — `data_missingness`, `weekly_sales_trend`, `promotion_sales_comparison`, `channel_spend_activity`, `correlation_heatmap`, `actual_vs_predicted_sales`, `estimated_channel_contribution`, `proxy_roi_by_channel`, `media_response_curves`, `budget_scenario_comparison`.

**Model (`output/model/`)** — `model_bundle.joblib` (fitted pipeline, half-points, feature names, cutoffs) and `model_metadata.json` (parameters, metrics, caveats) for reproducible scenario scoring.

> The charts embedded in this README are read from `output/charts/`, so that directory must be committed for the images to render on GitHub.

---

## Limitations and Caveats

These are stated plainly because the point of the project is defensible method, not impressive numbers.

1. **Synthetic data.** All findings describe a simulated dataset and generalise to nothing.
2. **The validation window is not an independent test set.** The same window is used to select transformation parameters, so its metrics are optimistic.
3. **No causal identification.** Ridge coefficients on observational data reflect association. Media budgets are chosen by humans responding to expected demand, so reverse causality and confounding are unaddressed.
4. **No uncertainty quantification.** Point estimates only — no confidence intervals, no posterior distributions, no bootstrap ranges.
5. **Contribution method is coarse.** Zero-out re-scoring ignores interaction effects and can produce negative contributions.
6. **Extrapolation risk in scenarios.** The 200% bound is an explicit boundary, and some channels' recent operating levels sit far below their historical active-week means.
7. **Proxy ROI is not financial ROI.** It is modelled contribution divided by spend, with no margin, cost, or lag-to-revenue accounting.

---

## Possible Extensions

- Hold out a genuinely independent test period, separate from the parameter-selection window
- Bayesian MMM (PyMC-Marketing, Meridian, Robyn) for posterior uncertainty on contributions
- Time-series cross-validation with expanding windows instead of a single split
- Calibrate against geo-experiments or incrementality tests to anchor causal claims
- Constrained optimisation for budget allocation rather than a handful of hand-built scenarios
- Shapley-value or Sobol-based contribution decomposition to capture interactions
- Add operational constraints (channel minimums, contractual commitments, production lead times)

---

## Tech Stack

`pandas` · `numpy` · `scikit-learn` (Ridge, StandardScaler, Pipeline) · `matplotlib` · `joblib` · `openpyxl` · `jupyter`

---

## License

MIT — see [LICENSE](LICENSE).
