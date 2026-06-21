"""
SkyCity Auckland Restaurants & Bars
Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations
Streamlit dashboard — Unified Mentor project deliverable.

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

st.set_page_config(
    page_title="SkyCity Auckland | Profit Intelligence",
    page_icon="🍽️",
    layout="wide",
)

NUMERIC_FEATURES = [
    "GrowthFactor", "AOV", "MonthlyOrders",
    "COGSRate", "OPEXRate", "CommissionRate", "DeliveryRadiusKM", "DeliveryCostPerOrder",
    "InStoreShare", "UE_share", "DD_share", "SD_share",
    "Commission_x_UEshare", "DeliveryCost_x_SDshare", "GrowthAdjustedOrders",
]
CATEGORICAL_FEATURES = ["CuisineType", "Segment", "Subregion"]


# ----------------------------------------------------------------------------
# DATA LOADING + FEATURE ENGINEERING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("SkyCity_Auckland_Restaurants___Bars.csv")

    df["TotalNetProfit"] = (
        df["InStoreNetProfit"] + df["UberEatsNetProfit"]
        + df["DoorDashNetProfit"] + df["SelfDeliveryNetProfit"]
    )
    df["TotalRevenue"] = (
        df["InStoreRevenue"] + df["UberEatsRevenue"]
        + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"]
    )
    df["NetProfitPerOrder"] = df["TotalNetProfit"] / df["MonthlyOrders"]
    df["NetProfitMargin"] = df["TotalNetProfit"] / df["TotalRevenue"]

    df["DeliveryNetProfit"] = df["UberEatsNetProfit"] + df["DoorDashNetProfit"] + df["SelfDeliveryNetProfit"]
    df["DeliveryRevenue"] = df["UberEatsRevenue"] + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"]
    df["DeliveryMargin"] = df["DeliveryNetProfit"] / df["DeliveryRevenue"]

    # Channel margin model (validated against the source data):
    # channel_net_profit = channel_revenue * (1 - COGSRate - OPEXRate [- CommissionRate | - SD cost ratio])
    df["InStoreMargin"] = 1 - df["COGSRate"] - df["OPEXRate"]
    df["UEMargin"] = df["InStoreMargin"] - df["CommissionRate"]
    df["DDMargin"] = df["InStoreMargin"] - df["CommissionRate"]
    df["SD_CostToRevenue"] = df["SD_DeliveryTotalCost"] / df["SelfDeliveryRevenue"]
    df["SDMargin"] = df["InStoreMargin"] - df["SD_CostToRevenue"]
    df["BreakEvenCommissionRate"] = df["InStoreMargin"]

    df["Commission_x_UEshare"] = df["CommissionRate"] * df["UE_share"]
    df["DeliveryCost_x_SDshare"] = df["DeliveryCostPerOrder"] * df["SD_share"]
    df["GrowthAdjustedOrders"] = df["MonthlyOrders"] * df["GrowthFactor"]

    lo, hi = df["NetProfitPerOrder"].quantile([0.01, 0.99])
    df["NetProfitPerOrder"] = df["NetProfitPerOrder"].clip(lo, hi)

    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df["TotalNetProfit"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
    ])

    model_defs = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=0
        ),
    }

    pipelines, metrics, preds_store = {}, {}, {}
    for name, model in model_defs.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
            "MAE": float(mean_absolute_error(y_test, preds)),
            "R2": float(r2_score(y_test, preds)),
        }
        pipelines[name] = pipe
        preds_store[name] = preds

    best_name = max(metrics, key=lambda k: metrics[k]["R2"])

    # Feature importance for best model (if tree-based) else |coef| for linear
    best_pipe = pipelines[best_name]
    ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = NUMERIC_FEATURES + cat_names

    fitted_model = best_pipe.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        importances = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        importances = np.abs(fitted_model.coef_)
    else:
        importances = np.zeros(len(all_names))
    fi = pd.Series(importances, index=all_names).sort_values(ascending=False)

    return pipelines, metrics, best_name, (X_test, y_test, preds_store), fi


def predict_scenario(pipe, baseline_row: dict, overrides: dict):
    row = baseline_row.copy()
    row.update(overrides)
    row_df = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return float(pipe.predict(row_df)[0])


# ----------------------------------------------------------------------------
# LOAD + TRAIN
# ----------------------------------------------------------------------------
df = load_data()
pipelines, metrics, best_name, test_artifacts, feature_importance = train_models(df)
best_pipe = pipelines[best_name]
X_test, y_test, preds_store = test_artifacts
best_rmse = metrics[best_name]["RMSE"]

# ----------------------------------------------------------------------------
# SIDEBAR — GLOBAL FILTERS
# ----------------------------------------------------------------------------
st.sidebar.title("🍽️ SkyCity Auckland")
st.sidebar.caption("Restaurants & Bars — Profit Intelligence")
st.sidebar.markdown("---")

segments = st.sidebar.multiselect("Segment", sorted(df["Segment"].unique()), default=sorted(df["Segment"].unique()))
cuisines = st.sidebar.multiselect("Cuisine Type", sorted(df["CuisineType"].unique()), default=sorted(df["CuisineType"].unique()))
subregions = st.sidebar.multiselect("Subregion", sorted(df["Subregion"].unique()), default=sorted(df["Subregion"].unique()))

filtered = df[
    df["Segment"].isin(segments)
    & df["CuisineType"].isin(cuisines)
    & df["Subregion"].isin(subregions)
]

st.sidebar.markdown("---")
st.sidebar.metric("Restaurants in view", len(filtered))
st.sidebar.caption(f"Best model: **{best_name}** (R² = {metrics[best_name]['R2']:.3f} on held-out test data)")
st.sidebar.caption("Built for: Unified Mentor — SkyCity Auckland Restaurants & Bars project")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("SkyCity Auckland Restaurants & Bars")
st.subheader("Predictive Modeling & Profit Optimization for Multi-Channel Restaurant Operations")
st.caption(
    "Static reports can't answer 'what if' questions. This dashboard adds predictive and "
    "prescriptive intelligence on top of the historical financial data — channel mix, cost "
    "sensitivity, and scenario simulation, all in one place."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & EDA",
    "🤖 Predictive Models",
    "🎛️ What-If Simulator",
    "🎯 Optimization & Recommendations",
])

# ----------------------------------------------------------------------------
# TAB 1 — OVERVIEW & EDA
# ----------------------------------------------------------------------------
with tab1:
    if filtered.empty:
        st.warning("No restaurants match the current filters. Adjust the sidebar selections.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg. Monthly Net Profit", f"${filtered['TotalNetProfit'].mean():,.0f}")
        c2.metric("Avg. Net Profit Margin", f"{filtered['NetProfitMargin'].mean()*100:.1f}%")
        c3.metric("Avg. Net Profit / Order", f"${filtered['NetProfitPerOrder'].mean():,.2f}")
        neg_pct = (filtered["TotalNetProfit"] < 0).mean() * 100
        c4.metric("Restaurants Running at a Loss", f"{neg_pct:.1f}%")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                filtered, x="TotalNetProfit", nbins=40, color="Segment",
                title="Distribution of Monthly Net Profit",
                labels={"TotalNetProfit": "Total Monthly Net Profit ($)"},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            seg_profit = filtered.groupby("Segment")["TotalNetProfit"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(
                seg_profit, x="Segment", y="TotalNetProfit", color="Segment",
                title="Average Monthly Net Profit by Segment",
                labels={"TotalNetProfit": "Avg. Net Profit ($)"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            cui_profit = filtered.groupby("CuisineType")["TotalNetProfit"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(
                cui_profit, x="TotalNetProfit", y="CuisineType", orientation="h",
                title="Average Monthly Net Profit by Cuisine Type",
                labels={"TotalNetProfit": "Avg. Net Profit ($)", "CuisineType": ""},
                color="TotalNetProfit", color_continuous_scale="Tealgrn",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            channel_rev = filtered[["InStoreRevenue", "UberEatsRevenue", "DoorDashRevenue", "SelfDeliveryRevenue"]].sum()
            channel_rev.index = ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"]
            fig = px.pie(
                values=channel_rev.values, names=channel_rev.index, hole=0.45,
                title="Revenue Mix Across Channels",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Channel mix vs. profitability — a confounding pattern worth noting")
        col5, col6 = st.columns(2)
        with col5:
            fig = px.scatter(
                filtered, x="InStoreShare", y="NetProfitMargin", color="Segment",
                title="In-Store Share vs. Net Profit Margin, by Segment",
                labels={"InStoreShare": "In-Store Order Share", "NetProfitMargin": "Net Profit Margin"},
                opacity=0.7,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Overall, higher in-store share correlates with *lower* margin — but that's a "
                "segment effect, not a channel effect. See below."
            )
        with col6:
            seg_cost = filtered.groupby("Segment")[["COGSRate", "OPEXRate"]].mean().reset_index()
            seg_cost["CostBase"] = seg_cost["COGSRate"] + seg_cost["OPEXRate"]
            fig = px.bar(
                seg_cost.sort_values("CostBase", ascending=False),
                x="Segment", y=["COGSRate", "OPEXRate"],
                title="Average Cost Structure by Segment (COGS + OPEX, % of revenue)",
                labels={"value": "Rate", "variable": "Cost Component"},
                barmode="stack",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Full-service venues carry a structurally heavier cost base (~88% of revenue in "
                "COGS+OPEX) than QSR, Cafe, or Ghost Kitchen formats. That's what drives losses — "
                "not their channel mix, which tends to lean in-store."
            )

        with st.expander("Show filtered data table"):
            st.dataframe(
                filtered[[
                    "RestaurantName", "CuisineType", "Segment", "Subregion",
                    "TotalNetProfit", "NetProfitMargin", "MonthlyOrders", "AOV",
                ]].sort_values("TotalNetProfit", ascending=False),
                use_container_width=True,
                height=350,
            )

# ----------------------------------------------------------------------------
# TAB 2 — PREDICTIVE MODELS
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("### Model comparison")
    st.caption(
        "All four models predict **Total Monthly Net Profit** from channel-mix, cost, and "
        "restaurant-profile features. Metrics below are computed on a held-out 20% test split "
        "(not seen during training)."
    )

    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    metrics_df = metrics_df[["Model", "R2", "RMSE", "MAE"]].sort_values("R2", ascending=False)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.dataframe(
            metrics_df.style.format({"R2": "{:.4f}", "RMSE": "${:,.0f}", "MAE": "${:,.0f}"}),
            use_container_width=True, hide_index=True,
        )
        st.success(f"Best performing model: **{best_name}** (R² = {metrics[best_name]['R2']:.4f})")

    with col2:
        fig = px.bar(
            metrics_df, x="Model", y="R2", color="Model",
            title="R² by Model (higher = better)", range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=preds_store[best_name], mode="markers",
            marker=dict(opacity=0.6), name="Predictions",
        ))
        lims = [min(y_test.min(), preds_store[best_name].min()), max(y_test.max(), preds_store[best_name].max())]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="red"), name="Perfect fit"))
        fig.update_layout(
            title=f"Actual vs. Predicted Net Profit — {best_name}",
            xaxis_title="Actual Net Profit ($)", yaxis_title="Predicted Net Profit ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        top_fi = feature_importance.head(10).sort_values()
        fig = px.bar(
            top_fi, x=top_fi.values, y=top_fi.index, orientation="h",
            title=f"Top 10 Feature Importances — {best_name}",
            labels={"x": "Relative importance", "y": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Reading the model")
    st.markdown(
        f"""
- **{best_name}** explains about **{metrics[best_name]['R2']*100:.1f}%** of the variance in monthly net profit
  across restaurants, with a typical prediction error (RMSE) of roughly **${best_rmse:,.0f}**.
- Cost structure (**COGSRate**, **OPEXRate**) dominates the model — confirming the EDA finding that
  margin is driven primarily by a restaurant's underlying cost base, not its channel mix alone.
- Channel-mix variables (**DD_share**, **UE_share**, **SD_share**, the commission interaction term) still
  carry meaningful weight, which is what makes the scenario simulator on the next tab useful for
  channel-mix decisions specifically.
- Linear Regression is kept as an interpretability baseline; the gap between it and the
  tree-based models indicates real non-linearity and interaction effects in how cost rates and
  channel shares combine to produce profit.
"""
    )

# ----------------------------------------------------------------------------
# TAB 3 — WHAT-IF SCENARIO SIMULATOR
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("### Channel-mix & cost scenario simulator")
    st.caption(
        f"Pick a restaurant as your starting point, then move the sliders to simulate a "
        f"strategic change. Predictions come from the **{best_name}** model; the shaded band "
        f"is ± the model's test-set RMSE (≈ ${best_rmse:,.0f}), not a formal confidence interval."
    )

    restaurant_names = sorted(df["RestaurantName"].unique())
    sel_name = st.selectbox("Baseline restaurant", restaurant_names, index=0)
    base_row = df[df["RestaurantName"] == sel_name].iloc[0]

    actual_profit = float(base_row["TotalNetProfit"])

    colA, colB, colC = st.columns(3)
    colA.metric("Segment", base_row["Segment"])
    colB.metric("Cuisine", base_row["CuisineType"])
    colC.metric("Actual Net Profit (current)", f"${actual_profit:,.0f}")

    st.markdown("##### Channel mix (must sum to 100% — auto-normalized)")
    s1, s2, s3, s4 = st.columns(4)
    in_store = s1.slider("In-Store share", 0.0, 1.0, float(base_row["InStoreShare"]), 0.01)
    ue = s2.slider("Uber Eats share", 0.0, 1.0, float(base_row["UE_share"]), 0.01)
    dd = s3.slider("DoorDash share", 0.0, 1.0, float(base_row["DD_share"]), 0.01)
    sd = s4.slider("Self-Delivery share", 0.0, 1.0, float(base_row["SD_share"]), 0.01)

    share_sum = in_store + ue + dd + sd
    if share_sum == 0:
        share_sum = 1.0
    in_store_n, ue_n, dd_n, sd_n = [v / share_sum for v in (in_store, ue, dd, sd)]
    st.caption(
        f"Normalized mix → In-Store {in_store_n*100:.0f}% · Uber Eats {ue_n*100:.0f}% · "
        f"DoorDash {dd_n*100:.0f}% · Self-Delivery {sd_n*100:.0f}%"
    )

    st.markdown("##### Cost & operating levers")
    s5, s6, s7, s8 = st.columns(4)
    commission = s5.slider("Commission rate", 0.20, 0.40, float(base_row["CommissionRate"]), 0.01)
    delivery_cost = s6.slider("Self-delivery cost / order ($)", 0.5, 6.0, float(base_row["DeliveryCostPerOrder"]), 0.1)
    delivery_radius = s7.slider("Delivery radius (km)", 3, 18, int(base_row["DeliveryRadiusKM"]), 1)
    growth = s8.slider("Growth factor", 0.95, 1.10, float(base_row["GrowthFactor"]), 0.01)

    overrides = {
        "InStoreShare": in_store_n, "UE_share": ue_n, "DD_share": dd_n, "SD_share": sd_n,
        "CommissionRate": commission, "DeliveryCostPerOrder": delivery_cost,
        "DeliveryRadiusKM": delivery_radius, "GrowthFactor": growth,
        "Commission_x_UEshare": commission * ue_n,
        "DeliveryCost_x_SDshare": delivery_cost * sd_n,
        "GrowthAdjustedOrders": base_row["MonthlyOrders"] * growth,
    }
    baseline_dict = base_row[NUMERIC_FEATURES + CATEGORICAL_FEATURES].to_dict()
    simulated_profit = predict_scenario(best_pipe, baseline_dict, overrides)

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Actual Net Profit", f"${actual_profit:,.0f}")
    r2.metric(
        "Simulated Net Profit", f"${simulated_profit:,.0f}",
        delta=f"{simulated_profit - actual_profit:,.0f}",
    )
    r3.metric("Model uncertainty (± RMSE)", f"± ${best_rmse:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Actual (current)", "Simulated (what-if)"],
        y=[actual_profit, simulated_profit],
        error_y=dict(type="data", array=[0, best_rmse], visible=True),
        marker_color=["#4C78A8", "#F58518"],
    ))
    fig.update_layout(title="Actual vs. Simulated Monthly Net Profit", yaxis_title="Net Profit ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Sensitivity: one lever at a time")
    st.caption("Holding the chosen baseline restaurant fixed, each line shows predicted profit as a single lever is swept across its observed range.")

    sweep_specs = {
        "CommissionRate": np.linspace(0.20, 0.40, 11),
        "UE_share": np.linspace(0.30, 0.70, 11),
        "SD_share": np.linspace(0.10, 0.50, 11),
        "DeliveryCostPerOrder": np.linspace(0.5, 6.0, 11),
        "DeliveryRadiusKM": np.linspace(3, 18, 11),
    }
    sweep_choice = st.selectbox("Lever to sweep", list(sweep_specs.keys()))
    sweep_vals = sweep_specs[sweep_choice]
    sweep_preds = []
    for v in sweep_vals:
        ov = {sweep_choice: v}
        if sweep_choice == "UE_share":
            ov["Commission_x_UEshare"] = commission * v
        if sweep_choice == "CommissionRate":
            ov["Commission_x_UEshare"] = v * ue_n
        if sweep_choice == "SD_share":
            ov["DeliveryCost_x_SDshare"] = delivery_cost * v
        if sweep_choice == "DeliveryCostPerOrder":
            ov["DeliveryCost_x_SDshare"] = v * sd_n
        sweep_preds.append(predict_scenario(best_pipe, baseline_dict, {**overrides, **ov}))

    fig = px.line(
        x=sweep_vals, y=sweep_preds, markers=True,
        title=f"Predicted Net Profit vs. {sweep_choice}",
        labels={"x": sweep_choice, "y": "Predicted Net Profit ($)"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4 — OPTIMIZATION & RECOMMENDATIONS
# ----------------------------------------------------------------------------
with tab4:
    st.markdown("### Channel economics")
    margin_summary = pd.Series({
        "In-Store": filtered["InStoreMargin"].mean(),
        "Self-Delivery": filtered["SDMargin"].mean(),
        "Uber Eats": filtered["UEMargin"].mean(),
        "DoorDash": filtered["DDMargin"].mean(),
    }).sort_values(ascending=False)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = px.bar(
            x=margin_summary.index, y=margin_summary.values * 100,
            color=margin_summary.index,
            title="Average Channel Margin (% of channel revenue)",
            labels={"x": "Channel", "y": "Margin (%)"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "In-store carries no commission or delivery cost, so it always has the highest "
            "margin ceiling. Self-delivery beats both aggregators on average because its "
            "per-order delivery cost is typically lower than the aggregator commission take rate."
        )

    with col2:
        be_rate = filtered["BreakEvenCommissionRate"].mean()
        actual_rate = filtered["CommissionRate"].mean()
        st.metric("Avg. Break-Even Commission Rate", f"{be_rate*100:.1f}%")
        st.metric("Avg. Actual Commission Rate Paid", f"{actual_rate*100:.1f}%")
        st.metric("Negotiation Headroom", f"{(be_rate-actual_rate)*100:.1f} pts")
        st.caption(
            "Break-even commission = 1 − COGSRate − OPEXRate: the rate at which an aggregator "
            "channel's margin hits zero. The gap to the actual rate paid is the safety buffer "
            "before that channel turns loss-making."
        )

    st.markdown("---")
    st.markdown("### Optimal channel mix for the selected restaurant")
    st.caption(f"Using **{sel_name}** (selected on the What-If tab) as the baseline. Search is constrained to the ranges observed in the historical data, to avoid extrapolating beyond what the model has seen.")

    bounds = {
        "InStoreShare": (df["InStoreShare"].min(), df["InStoreShare"].max()),
        "UE_share": (df["UE_share"].min(), df["UE_share"].max()),
        "DD_share": (df["DD_share"].min(), df["DD_share"].max()),
        "SD_share": (df["SD_share"].min(), df["SD_share"].max()),
    }

    rng = np.random.default_rng(RANDOM_STATE)
    n_trials = 4000
    raw = rng.uniform(0, 1, size=(n_trials, 4))
    raw = raw / raw.sum(axis=1, keepdims=True)  # normalize to sum to 1
    # clip to observed per-channel bounds, then renormalize
    for i, key in enumerate(["InStoreShare", "UE_share", "DD_share", "SD_share"]):
        lo, hi = bounds[key]
        raw[:, i] = np.clip(raw[:, i], lo, hi)
    raw = raw / raw.sum(axis=1, keepdims=True)

    candidate_profits = []
    for i in range(n_trials):
        ov = {
            "InStoreShare": raw[i, 0], "UE_share": raw[i, 1],
            "DD_share": raw[i, 2], "SD_share": raw[i, 3],
            "Commission_x_UEshare": commission * raw[i, 1],
            "DeliveryCost_x_SDshare": delivery_cost * raw[i, 3],
        }
        candidate_profits.append(predict_scenario(best_pipe, baseline_dict, {**overrides, **ov}))
    candidate_profits = np.array(candidate_profits)
    best_idx = candidate_profits.argmax()
    best_mix = raw[best_idx]
    best_mix_profit = candidate_profits[best_idx]

    current_mix_profit = simulated_profit
    uplift_pct = (best_mix_profit - current_mix_profit) / abs(current_mix_profit) * 100 if current_mix_profit != 0 else np.nan

    colm1, colm2 = st.columns(2)
    with colm1:
        mix_compare = pd.DataFrame({
            "Channel": ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"],
            "Current mix (%)": [in_store_n*100, ue_n*100, dd_n*100, sd_n*100],
            "Suggested mix (%)": best_mix * 100,
        })
        fig = px.bar(
            mix_compare, x="Channel", y=["Current mix (%)", "Suggested mix (%)"],
            barmode="group", title="Current vs. Model-Suggested Channel Mix",
        )
        st.plotly_chart(fig, use_container_width=True)

    with colm2:
        st.metric("Current simulated profit", f"${current_mix_profit:,.0f}")
        st.metric("Best found profit (4,000 mixes tested)", f"${best_mix_profit:,.0f}")
        st.metric("Optimization Uplift", f"{uplift_pct:,.1f}%")
        st.caption(
            "This is a randomized search over channel-mix combinations within historically "
            "observed bounds, scored by the predictive model — a practical stand-in for a full "
            "constrained optimizer, not a guaranteed global optimum."
        )

    st.markdown("---")
    st.markdown("### Recommendations")
    fs_share_in_neg = (df[df["TotalNetProfit"] < 0]["Segment"] == "Full-service").mean() * 100 if (df["TotalNetProfit"] < 0).any() else 0
    st.markdown(
        f"""
1. **Fix the cost base before the channel mix, for Full-service venues.** Every loss-making
   restaurant in this dataset is in the Full-service segment ({fs_share_in_neg:.0f}% of
   loss-makers), driven by a COGS+OPEX base around 87–88% of revenue — roughly 12–15 points
   higher than QSR, Cafe, or Ghost Kitchen formats. Reshuffling channels alone won't fix a
   structural cost problem.
2. **Self-delivery is the higher-margin delivery channel on average** — it beats both
   aggregators in this data because per-order delivery cost tends to run below the aggregator
   commission take. Expanding self-delivery capacity (where logistics allow) is generally a
   safer lever than just increasing in-store share.
3. **There is real but limited room to negotiate commission rates.** The average restaurant is
   paying close to its break-even commission rate already — treat the negotiation headroom
   figure above as a ceiling, not a target.
4. **Use the What-If tab before committing to a channel-mix change.** Because cost structure
   dominates the model, the impact of a channel-mix shift varies a lot by restaurant — there is
   no single mix that's optimal for every segment or cuisine.
"""
    )

    st.info(
        "Methodology note: the optimization layer above re-uses the trained predictive model "
        "(not a separate causal model), so its recommendations inherit the model's accuracy "
        f"(R² ≈ {metrics[best_name]['R2']:.2f}, RMSE ≈ ${best_rmse:,.0f}) and should be treated as "
        "directional guidance to test, not a guaranteed outcome."
    )
