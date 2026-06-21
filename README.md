# SkyCity Auckland Restaurants & Bars — Profit Intelligence Dashboard

Predictive modeling and profit optimization dashboard for multi-channel restaurant
operations (in-store, Uber Eats, DoorDash, self-delivery), built for the Unified
Mentor "SkyCity Auckland Restaurants & Bars" project brief.

## What's in this folder

- `app.py` — the Streamlit application (self-contained: loads the CSV, engineers
  features, trains 4 regression models in-app with caching, and renders 4 tabs).
- `SkyCity_Auckland_Restaurants___Bars.csv` — the dataset (must stay alongside `app.py`).
- `requirements.txt` — pinned dependency versions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder's contents to a GitHub repo (the CSV is small — 368 KB — so there
   are no file-size or LFS concerns).
2. On https://share.streamlit.io, point a new app at `app.py` in that repo.
3. Streamlit Cloud installs from `requirements.txt` automatically. If you hit a
   `ModuleNotFoundError`, double check the package name matches exactly what's
   imported in `app.py` (this caught out the Plotly import in an earlier project —
   worth a quick visual diff against `requirements.txt` before redeploying).

## App structure

| Tab | Purpose |
|---|---|
| 📊 Overview & EDA | KPI cards, profit distribution, segment/cuisine breakdowns, channel revenue mix, the in-store-share-vs-margin confound |
| 🤖 Predictive Models | Linear Regression / Random Forest / Gradient Boosting / XGBoost comparison (R², RMSE, MAE), feature importance, actual-vs-predicted plot |
| 🎛️ What-If Simulator | Pick a restaurant, move channel-mix and cost sliders, see simulated profit vs. actual, plus single-lever sensitivity sweeps |
| 🎯 Optimization & Recommendations | Channel margin economics, break-even commission rate, randomized search for a better channel mix, narrative recommendations |

## Notes on the modeling approach

- Target: **Total Monthly Net Profit** (sum of the four channel net-profit columns).
- Models are trained on an 80/20 split with `random_state=42`; metrics shown in the
  app are computed on the held-out 20%.
- The "Optimization" tab uses a randomized search (4,000 candidate channel mixes,
  bounded to the ranges actually observed in the data) scored by the trained model —
  this is a practical stand-in for a full constrained optimizer, not a guaranteed
  global optimum, and the app says so.
