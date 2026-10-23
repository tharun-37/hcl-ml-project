# Battery RUL & Health Monitoring System

Machine-learning prognostics for lithium-ion battery fleets: predict **Remaining Useful Life (RUL)** and **State of Health (SOH)** from live sensor readings, and explain every prediction with **SHAP**.

Built with Flask, scikit-learn, XGBoost and SHAP, and deployed as a public web dashboard on Vercel.

<p>
  <a href="https://mini-proj-tau.vercel.app"><img alt="Live Demo" src="https://img.shields.io/badge/Live%20Demo-mini--proj--tau.vercel.app-4F46E5?style=for-the-badge"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-3.1.3-000000?style=for-the-badge&logo=flask&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
</p>

**Live app:** https://mini-proj-tau.vercel.app
**Repository:** https://github.com/tharun-37/hcl-ml-project

---

## Overview

Lithium-ion cells lose capacity with every charge–discharge cycle. Fleet operators running AGVs and forklifts need to know, for each cell, **when it will fail** and **how healthy it is right now** — before an unplanned stall becomes a safety incident.

This project turns the Oxford Battery Degradation Dataset into a working decision-support system:

- 4 regression models estimate **RUL** in remaining cycles
- 3 classifiers flag **end-of-life** (SOH < 80%)
- SHAP explains both **globally** (which features drive degradation) and **locally** (why this specific prediction came out the way it did)
- A Flask dashboard serves live predictions, batch fleet scoring, and the analysis charts

The best regressor, Random Forest, reaches **R² = 0.9896** with an RMSE of 222 cycles on a held-out 104-sample test set.

---

## Team

| S.No | Name | Roll No |
|:---:|---|---|
| 1 | Tharun R | 24CSR323 |
| 2 | Suhas S | 24CSR311 |
| 3 | Syed Muhammad Anas S A | 24CSR323 |

HCL AI/ML College Mini Project · 2025–26

---

## Features

| Feature | Description |
|---|---|
| Live prediction | `/predict` returns RUL, SOH, fleet health status and a maintenance strategy |
| Explainability | SHAP TreeExplainer / KernelExplainer returns the top features driving each prediction |
| Batch scoring | `/batch-predict` accepts a CSV of readings and returns the same file with predictions appended |
| Model selection | Choose between Random Forest, XGBoost, Linear Regression and SVR |
| Fleet health bands | Colour-coded status from Optimal (>90% SOH) to Critical (<75% SOH) |
| Prediction history | SQLite-backed log of recent predictions (last 6 shown on the dashboard) |
| Health endpoint | `/health` reports model-package and database status for uptime checks |
| Serverless-safe | Logs and database automatically fall back to a writable temp directory |

---

## Quick start

```bash
git clone https://github.com/tharun-37/hcl-ml-project.git
cd hcl-ml-project

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

The dashboard opens at **http://localhost:5001**.

The trained model package (`models/model.pkl`) is committed, so no training step is needed to run the app.

### Retraining (optional)

```bash
python train.py
```

This re-engineers features, retrains all 7 models, recomputes SHAP importances and overwrites `models/model.pkl`. It reads the processed dataset at `data/oxford_battery_processed.csv`, so the original `.mat` file is not required.

---

## Dataset

**Oxford Battery Degradation Dataset** — Severson et al., *Nature Energy* (2019),
[Data-driven prediction of battery cycle life before capacity degradation](https://www.nature.com/articles/s41586-019-1612-4).

Eight commercial LFP/graphite cells cycled to failure, converted into a cycle-level feature table.

| Property | Value |
|---|---|
| Records | 519 cycle-level rows × 18 columns |
| Cells | 8 (all fell below 80% SOH) |
| Cycle range | 0 – 8,200 |
| SOH range | 61.64% – 99.88% |
| Capacity range | 456.1 – 739.1 mAh (740 mAh nominal) |
| RUL range | 0 – 8,200 cycles |
| Class balance | 404 healthy (SOH ≥ 80%) / 115 near end-of-life |
| Split | 415 train / 104 test (80:20, stratified, `random_state=42`) |

### Features

Ten raw sensor features plus two engineered physical proxies:

| # | Feature | Notes |
|:--:|---|---|
| 1 | `Cycle` | Cycle count |
| 2 | `Discharge_Capacity_mAh` | Capacity delivered |
| 3 | `Mean_Voltage_V` | Mean discharge voltage |
| 4 | `Min_Voltage_V` | Minimum voltage |
| 5 | `Max_Voltage_V` | Maximum voltage |
| 6 | `Std_Voltage_V` | Voltage spread |
| 7 | `Mean_Temp_C` | Mean temperature |
| 8 | `Max_Temp_C` | Peak temperature |
| 9 | `Discharge_Duration_hr` | Discharge time |
| 10 | `Energy_Wh` | Energy delivered |
| 11 | `Voltage_Drop_Rate` | `(Max − Min) / Duration` |
| 12 | `Internal_Resist_Proxy` | `(Max − Mean) / 0.740` |

**Targets:** `RUL` (regression) and `EOL_Class` (1 when SOH < 80%).

---

## Model performance

### Regression — RUL in cycles

| Model | R² | RMSE | MAE |
|---|--:|--:|--:|
| **Random Forest Regressor** | **0.9896** | **222.09** | **150.26** |
| XGBoost Regressor | 0.9879 | 239.75 | 159.63 |
| Linear Regression | 0.7974 | 982.04 | 840.91 |
| Support Vector Regressor (RBF) | 0.5443 | 1472.65 | 974.23 |

Tree ensembles reduce RMSE by roughly **77%** versus the linear baseline — degradation behaviour is strongly non-linear.

### Classification — end-of-life flag

| Model | Accuracy | Precision | Recall | F1 |
|---|--:|--:|--:|--:|
| **Random Forest Classifier** | **100.00%** | 100.00% | 100.00% | 100.00% |
| Logistic Regression | 100.00% | 100.00% | 100.00% | 100.00% |
| Support Vector Classifier (RBF) | 90.38% | 70.97% | 95.65% | 81.48% |

Random Forest confusion matrix on the test set: `[[81, 0], [0, 23]]` — 81 healthy and 23 end-of-life cells, no errors.
SVC trades precision for recall, which is the safe failure mode: it prefers flagging a healthy cell over missing a failing one.

---

## Explainability (SHAP)

Global mean-absolute SHAP importance from the Random Forest regressor:

| Rank | Feature | Importance |
|:--:|---|--:|
| 1 | `Discharge_Duration_hr` | 21.86% |
| 2 | `Mean_Temp_C` | 16.35% |
| 3 | `Discharge_Capacity_mAh` | 14.71% |
| 4 | `Voltage_Drop_Rate` | 13.01% |
| 5 | `Internal_Resist_Proxy` | 6.11% |
| 6 | `Max_Temp_C` | 6.09% |
| 7 | `Energy_Wh` | 5.94% |
| 8 | `Cycle` | 4.88% |
| 9 | `Mean_Voltage_V` | 4.65% |
| 10 | `Std_Voltage_V` | 3.27% |
| 11 | `Max_Voltage_V` | 2.93% |
| 12 | `Min_Voltage_V` | 0.20% |

Discharge duration and temperature dominate capacity fade, which agrees with known LFP ageing behaviour — so the model is validated physically, not only statistically.

`/predict` also returns a **local** breakdown: for each of the top 5 features, its input value, signed impact on RUL, and direction (`increases_rul` / `reduces_rul`).

---

## API reference

Base URL: `https://mini-proj-tau.vercel.app` (or `http://localhost:5001`)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard — metrics, SHAP ranks, prediction form, charts, history |
| `/health` | GET | Model-package and database status |
| `/predict` | POST | Single-cell prediction with SHAP breakdown (JSON) |
| `/batch-predict` | POST | Upload a CSV, receive the scored CSV |

### `POST /predict`

Request body (`application/json`):

| Field | Type | Range | Required |
|---|---|:--:|:--:|
| `model_name` | string | one of the 4 regressor names | no (default `Random Forest Regressor`) |
| `cycle` | number | 0 – 10,000 | yes |
| `discharge_capacity` | number | 300 – 800 mAh | yes |
| `mean_voltage` | number | 2.5 – 4.5 V | yes |
| `min_voltage` | number | 2.0 – 4.0 V | yes |
| `max_voltage` | number | 3.5 – 4.5 V | yes |
| `std_voltage` | number | 0.0 – 1.0 V | yes |
| `mean_temp` | number | 20 – 60 °C | yes |
| `max_temp` | number | 20 – 65 °C | yes |
| `discharge_duration` | number | 0.01 – 1.0 hr | yes |
| `energy` | number | 0.5 – 4.0 Wh | yes |

Values outside these ranges return **HTTP 422** with a per-field error message.

```bash
curl -X POST https://mini-proj-tau.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cycle": 4200,
    "discharge_capacity": 590,
    "mean_voltage": 3.3,
    "min_voltage": 2.8,
    "max_voltage": 3.85,
    "std_voltage": 0.25,
    "mean_temp": 33.5,
    "max_temp": 45,
    "discharge_duration": 0.9,
    "energy": 2.0
  }'
```

Example response:

```json
{
  "success": true,
  "predicted_rul": 4963,
  "soh": 79.73,
  "health_status": "Degraded (Near End-of-Life)",
  "health_color": "#f59e0b",
  "model_used": "Random Forest Regressor",
  "strategy_recommendation": "Scheduled Rotation: Rotate to light-duty transport; schedule replacement in 200 cycles to avoid mid-shift stall.",
  "estimated_eol_cycle": 9163,
  "local_shap": [
    { "feature": "Discharge_Duration_hr", "value": 0.9, "impact": 762.0, "direction": "increases_rul" },
    { "feature": "Mean_Temp_C", "value": 33.5, "impact": 611.2, "direction": "increases_rul" },
    { "feature": "Discharge_Capacity_mAh", "value": 590.0, "impact": -438.7, "direction": "reduces_rul" }
  ]
}
```

### Health bands

| SOH | Status | Recommended action |
|---:|---|---|
| ≥ 90% | Optimal (Fresh Cell) | Full heavy-load AGV and forklift duty cycles |
| ≥ 80% | Healthy (Normal Degradation) | Routine monitoring; continuous shift operations |
| ≥ 75% | Degraded (Near End-of-Life) | Rotate to light duty; schedule replacement |
| < 75% | Critical (End-of-Life Reached) | Decommission immediately; repurpose or recycle |

### `POST /batch-predict`

Multipart form upload. Field `file` is required (CSV); `model_name` is optional.

```bash
curl -X POST https://mini-proj-tau.vercel.app/batch-predict \
  -F "file=@fleet_readings.csv" \
  -F "model_name=Random Forest Regressor" \
  -o battery_phm_fleet_predictions.csv
```

The CSV must contain the 10 raw feature columns. `Voltage_Drop_Rate` and `Internal_Resist_Proxy` are computed automatically when missing. The response appends `Predicted_RUL_Cycles`, `Estimated_SOH_Pct` and `Fleet_Health_Status`.

---

## Project structure

```
hcl-ml-project/
├── app.py                  # Flask app: routes, prediction logic, SHAP, SQLite history
├── config.py               # Paths + serverless-safe writable-directory detection
├── validators.py           # Pydantic request schema and field ranges
├── train.py                # Training pipeline: 4 regressors, 3 classifiers, SHAP
├── requirements.txt        # Runtime dependencies
├── vercel.json             # Fluid compute configuration
├── .python-version         # Pins Python 3.14 (matches the pickled model)
├── models/
│   └── model.pkl           # Trained models, metrics, feature list, SHAP importances
├── data/
│   └── oxford_battery_processed.csv
├── templates/
│   ├── index.html          # Dashboard UI
│   └── error.html          # Error page
└── static/
    └── charts/             # Analysis figures served by the dashboard
        ├── soh_degradation.png
        ├── actual_vs_predicted.png
        ├── model_comparison.png
        ├── shap_importance.png
        └── confusion_matrix.png
```

### How a prediction flows

1. JSON is validated by the pydantic model in `validators.py`.
2. Engineered features are derived (`Voltage_Drop_Rate`, `Internal_Resist_Proxy`).
3. The selected regressor predicts RUL; negative values are clamped to zero.
4. SOH is computed as `discharge_capacity / 740 × 100`.
5. SOH maps to a health band, colour and maintenance strategy.
6. SHAP produces the local top-5 feature breakdown.
7. The prediction is logged to SQLite (best-effort).
8. JSON is returned to the dashboard.

`model.pkl` contains `reg_models`, `reg_metrics`, `cls_models`, `cls_metrics`, `conf_matrices`, `features`, `importances`, `X_train_summary` and `X_train_sample` (used as the SHAP background sample).

---

## Deployment

The app runs on **Vercel** with Fluid compute.

| Setting | Value |
|---|---|
| Runtime | Python 3.14 (`.python-version`) |
| Framework | Flask (detected from `app.py`) |
| Build | `vercel --prod`, or automatically on every push to `main` |
| Function size | Large Functions enabled — the ML bundle exceeds the 500 MB standard limit |

Required environment variable:

| Variable | Value | Why |
|---|---|---|
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | `1` | Allows the scikit-learn / XGBoost / SHAP bundle to build |

Optional: `SECRET_KEY`, `LOG_DIR`.

Deployment notes:

- Deployment Protection must be **off** for anonymous visitors; otherwise Vercel serves a login page instead of the app.
- On serverless, the project directory is read-only, so `config.py` detects that and redirects logs and `predictions.db` to `tempfile.gettempdir()`.
- Because of that redirect, prediction history does not persist across cold starts in production. It is retained normally when running locally.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| Web | Flask 3.1.3, Jinja2 |
| Validation | pydantic 2.13.4 |
| ML | scikit-learn 1.9.0, XGBoost 3.4.1 |
| Explainability | SHAP 0.52.0 |
| Data | pandas 3.0.5, NumPy 2.5.2 |
| Storage | SQLite (prediction history) |
| Hosting | Vercel (Fluid compute, Large Functions) |

---

## Limitations & future work

- The dataset covers 8 cells; generalisation to other chemistries and formats is untested.
- Batch scoring and live prediction use the same 12 features — real telemetry would need mapping to this schema.
- Prediction history is ephemeral on serverless; a managed database or persistent disk would fix this.
- Sequence models (LSTM / Transformer) over raw cycle curves could outperform engineered-feature regressors.
- Next steps: IoT telemetry ingestion, multi-cell fleet correlation, and on-device inference for AGV controllers.

---

## Acknowledgements

Dataset: Severson, K.A., Attia, P.M., Jin, N. et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy 4, 383–391 (2019).

Built for the HCL AI/ML College Mini Project, 2025–26.
