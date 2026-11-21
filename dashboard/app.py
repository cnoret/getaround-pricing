"""
Getaround Dashboard Application
This application provides an interactive dashboard for analyzing rental delays and pricing data.
It uses Streamlit for the frontend and Plotly for visualizations.
"""

import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from delay import (
    DELAY,
    GAP,
    filter_scope,
    late_return_stats,
    load_rentals,
    simulate_buffer,
)

st.set_page_config(page_title="Getaround Dashboard", layout="wide", page_icon="🚗")

st.markdown(
    """
    <div style='background: linear-gradient(90deg,#1abc9c,#2980b9); padding: 1.2rem 2rem; border-radius: 16px; color: white; font-size: 1.7rem; font-weight: bold; margin-bottom: 2.5rem; letter-spacing: 1px;'>
        🚗 Getaround Dashboard <span style='font-size:1.1rem; font-weight:400;'>&nbsp;|&nbsp; Delay, Pricing & Prediction Insights</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<h3>Getaround Dashboard</h3>", unsafe_allow_html=True)
    st.caption("Filter, explore and predict in a few clicks.")

# Predictions go through the FastAPI service at API_URL (Docker Compose, or the API Space
# set as a secret on Streamlit Community Cloud). If it is not set or does not answer,
# the dashboard falls back to the saved model, so the form always works.
API_URL = os.environ.get("API_URL")
MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "model" / "model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_price(car):
    """Predicted daily price for one car and where it came from ("api" or "local model")."""
    if API_URL:
        try:
            response = requests.post(API_URL, json={"input": [car]}, timeout=10)
            response.raise_for_status()
            return response.json()["prediction"][0], "api"
        except (requests.RequestException, ValueError, KeyError, IndexError):
            pass
    return float(load_model().predict(pd.DataFrame([car]))[0]), "local model"


@st.cache_data
def load_delay_data():
    return load_rentals()


@st.cache_data
def load_pricing_data():
    df = pd.read_csv("data/get_around_pricing_project.csv")
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


tab1, tab2, tab3 = st.tabs(
    ["⏱️ Delay Analysis", "💰 Pricing Analysis", "💸 Price Prediction"]
)

# ── Tab 1: Delay Analysis ──────────────────────────────────────────────────────
with tab1:
    st.header("⏱️ Delay Analysis")
    st.caption("How often do renters return cars late, and what buffer time minimizes conflicts?")

    rentals = load_delay_data()

    st.sidebar.header("Delay Filters")
    threshold = st.sidebar.slider(
        "⏱️ Minimum time between rentals (min)", 0, 180, 60, step=15
    )
    scope = st.sidebar.radio("🔐 Check-in type", ["All", "Connect only"])
    checkin_type = "connect" if scope == "Connect only" else None
    df = filter_scope(rentals, checkin_type)

    late = late_return_stats(rentals, threshold, checkin_type)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Late returns",
        f"{late['late']:,}",
        delta=f"{late['late_share']:.0f}% of ended rentals",
        delta_color="off",
        help="Ended rentals where the car was returned after the scheduled checkout time",
    )
    col2.metric(f"Absorbed (≤ {threshold} min)", f"{late['absorbed']:,}")
    col3.metric("% of delays absorbed", f"{late['absorbed_share']:.1f}%")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.write(
            f"""
            - **Late returns**: ended rentals with a known return delay above 0 min.
            - **Absorbed**: late returns of at most {threshold} min, which the buffer would cover.
            - **% absorbed**: share of late returns covered by the buffer.
            """
        )

    late_returns = df[df[DELAY] > 0]
    hist_max = 12 * 60
    beyond = (late_returns[DELAY] > hist_max).sum()
    fig_hist = px.histogram(
        late_returns[late_returns[DELAY] <= hist_max],
        x=DELAY,
        nbins=48,
        title="Distribution of checkout delays (late returns up to 12 hours)",
        labels={DELAY: "Delay (minutes)"},
        color_discrete_sequence=["#2980b9"],
    )
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, width="stretch")
    st.caption(
        f"{beyond:,} late returns above 12 hours are not shown, so the chart stays readable. "
        "They are still counted in every metric."
    )

    st.subheader("Business Impact")
    st.caption(f"Simulating a {threshold}-minute minimum gap between consecutive rentals.")

    impact = simulate_buffer(rentals, threshold, checkin_type)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Conflicts avoided",
        f"{impact['avoided_share']:.0f}%",
        delta=f"{impact['avoided']} of {impact['conflicts']} conflicts",
        delta_color="off",
        help="Conflicts where the previous rental ran over the planned gap, and which the buffer would prevent",
    )
    col2.metric(
        "Rentals blocked (revenue at risk)",
        f"{impact['blocked_share']:.1f}%",
        delta=f"{impact['blocked']} of {impact['ended']:,} rentals",
        delta_color="off",
        help="Share of all ended rentals that the buffer would prevent from being booked",
    )
    col3.metric("Conflicts remaining", impact["remaining"])
    col4.metric(
        "Delays absorbed",
        f"{late['absorbed_share']:.1f}%",
        delta=f"{late['absorbed']:,} cases",
        delta_color="off",
    )

    with st.expander("ℹ️ How is the impact computed?"):
        st.write(
            """
            - **Conflict**: the previous rental of the same car was returned later than the
              planned gap before the next rental.
            - **Buffer rule**: bookings with a planned gap shorter than the buffer are not allowed.
              This prevents the conflicts on those bookings, but also blocks the bookings themselves.
            - **Rentals blocked** is computed on all ended rentals, as a proxy for revenue at risk.
            - Figures use the raw rental export: missing delays are left out, long delays are kept as is.
            """
        )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_gap = px.histogram(
            df.dropna(subset=[GAP]),
            x=GAP,
            nbins=50,
            title="Time gap between consecutive rentals",
            labels={GAP: "Time gap (minutes)"},
            color_discrete_sequence=["#1abc9c"],
        )
        fig_gap.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{threshold} min buffer",
        )
        st.plotly_chart(fig_gap, width="stretch")

    with col_b:
        fig_box = px.box(
            late_returns[late_returns[DELAY] < 120],
            x="checkin_type",
            y=DELAY,
            points=False,
            color="checkin_type",
            color_discrete_map={"connect": "#2980b9", "mobile": "#1abc9c"},
            title="Delay by check-in type (late returns under 2 hours)",
            labels={DELAY: "Delay (minutes)", "checkin_type": "Check-in type"},
        )
        st.plotly_chart(fig_box, width="stretch")

    st.download_button(
        "📥 Download late returns",
        late_returns.to_csv(index=False).encode(),
        "late_returns.csv",
    )

# ── Tab 2: Pricing Analysis ────────────────────────────────────────────────────
with tab2:
    st.header("💰 Pricing Analysis")
    st.caption("Explore how car characteristics relate to daily rental prices.")

    df_price = load_pricing_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cars in dataset", f"{len(df_price):,}")
    col2.metric("Avg. daily price", f"{df_price['rental_price_per_day'].mean():.0f} €")
    col3.metric("Avg. mileage", f"{df_price['mileage'].mean() / 1000:.0f}k km")
    col4.metric("Avg. engine power", f"{df_price['engine_power'].mean():.0f} hp")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        fig_km = px.histogram(
            df_price,
            x="mileage",
            nbins=50,
            title="Mileage distribution",
            labels={"mileage": "Mileage (km)"},
            color_discrete_sequence=["#2980b9"],
        )
        fig_km.update_xaxes(tickformat=",d")
        st.plotly_chart(fig_km, width="stretch")

    with col_b:
        fig_price_dist = px.histogram(
            df_price,
            x="rental_price_per_day",
            nbins=40,
            title="Daily rental price distribution",
            labels={"rental_price_per_day": "Price per day (€)"},
            color_discrete_sequence=["#27ae60"],
        )
        st.plotly_chart(fig_price_dist, width="stretch")

    fig_scatter = px.scatter(
        df_price,
        x="mileage",
        y="rental_price_per_day",
        color="fuel",
        opacity=0.45,
        title="Rental price vs. mileage by fuel type",
        labels={
            "mileage": "Mileage (km)",
            "rental_price_per_day": "Price per day (€)",
            "fuel": "Fuel",
        },
    )
    fig_scatter.update_xaxes(tickformat=",d")
    st.plotly_chart(fig_scatter, width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        avg_price_type = (
            df_price.groupby("car_type")["rental_price_per_day"]
            .mean()
            .reset_index()
            .sort_values("rental_price_per_day", ascending=True)
        )
        fig_type = px.bar(
            avg_price_type,
            x="rental_price_per_day",
            y="car_type",
            orientation="h",
            title="Avg. price by car type",
            labels={"car_type": "", "rental_price_per_day": "Avg. price/day (€)"},
            color="rental_price_per_day",
            color_continuous_scale="blues",
        )
        fig_type.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_type, width="stretch")

    with col_d:
        avg_price_fuel = (
            df_price.groupby("fuel")["rental_price_per_day"]
            .mean()
            .reset_index()
            .sort_values("rental_price_per_day", ascending=True)
        )
        fig_fuel = px.bar(
            avg_price_fuel,
            x="rental_price_per_day",
            y="fuel",
            orientation="h",
            title="Avg. price by fuel type",
            labels={"fuel": "", "rental_price_per_day": "Avg. price/day (€)"},
            color="rental_price_per_day",
            color_continuous_scale="teal",
        )
        fig_fuel.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_fuel, width="stretch")

    st.caption("Source: get_around_pricing_project.csv")

# ── Tab 3: Price Prediction ────────────────────────────────────────────────────
with tab3:
    st.header("💸 Price Prediction")
    st.caption("Enter your car's details to get a predicted daily rental price.")

    df_price = load_pricing_data()

    models = sorted(df_price["model_key"].dropna().unique())
    fuels = sorted(df_price["fuel"].dropna().unique())
    colors = sorted(df_price["paint_color"].dropna().unique())
    car_types = sorted(df_price["car_type"].dropna().unique())

    with st.form(key="predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Car specs")
            mileage = st.number_input(
                "Mileage (km)", min_value=0, max_value=1_000_000, value=140_000, step=5_000
            )
            engine_power = st.number_input(
                "Engine power (hp)", min_value=0, max_value=500, value=130, step=10
            )
            model_key = st.selectbox("Brand / Model", models)
            fuel = st.selectbox("Fuel type", fuels)
            paint_color = st.selectbox("Color", colors)
            car_type = st.selectbox("Car type", car_types)
        with c2:
            st.subheader("Options & features")
            private_parking_available = st.checkbox("Private parking available", value=True)
            has_gps = st.checkbox("GPS", value=True)
            has_air_conditioning = st.checkbox("Air conditioning", value=False)
            automatic_car = st.checkbox("Automatic transmission", value=False)
            has_getaround_connect = st.checkbox("Getaround Connect", value=True)
            has_speed_regulator = st.checkbox("Speed regulator", value=False)
            winter_tires = st.checkbox("Winter tires", value=True)

        submit = st.form_submit_button("🔮 Predict price", width="stretch")

    if submit:
        input_data = {
            "mileage": float(mileage),
            "engine_power": float(engine_power),
            "model_key": model_key,
            "fuel": fuel,
            "paint_color": paint_color,
            "car_type": car_type,
            "private_parking_available": private_parking_available,
            "has_gps": has_gps,
            "has_air_conditioning": has_air_conditioning,
            "automatic_car": automatic_car,
            "has_getaround_connect": has_getaround_connect,
            "has_speed_regulator": has_speed_regulator,
            "winter_tires": winter_tires,
        }
        prediction, source = predict_price(input_data)
        if API_URL and source != "api":
            st.caption("The prediction API did not answer, so this estimate comes from the same model loaded in the dashboard.")
        else:
            st.caption(f"Estimate from the {'prediction API' if source == 'api' else 'saved model'}.")
        avg_price = df_price["rental_price_per_day"].mean()
        delta = prediction - avg_price
        arrow = "▲" if delta >= 0 else "▼"
        st.markdown(
            f"""
            <div style='text-align:center; padding:2rem 1rem; background:linear-gradient(135deg,#f0fff4,#e8f8f0); border-radius:16px; border:2px solid #1abc9c; margin:1.5rem 0;'>
                <p style='color:#888; font-size:1rem; margin-bottom:0.3rem;'>Estimated daily rental price</p>
                <p style='font-size:3.8rem; font-weight:800; color:#27ae60; margin:0; line-height:1.1;'>{prediction:.0f} €<span style='font-size:1.2rem; font-weight:400; color:#aaa;'> / day</span></p>
                <p style='color:#999; font-size:0.9rem; margin-top:0.6rem;'>Platform average: {avg_price:.0f} € &nbsp;·&nbsp; {arrow} {abs(delta):.0f} € vs. average</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("ℹ️ About this model"):
        st.write(
            """
            Predictions come from a **gradient boosting model** (scikit-learn `HistGradientBoostingRegressor`)
            trained on ~4,800 Getaround listings.

            - **OneHotEncoder** on categorical features (brand, fuel type, color, car type), rare brands grouped
            - Numeric and boolean features passed to the trees as they are
            - Engine power and mileage drive most of the price, then brand and car type
            """
        )
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("MAE", "10.68 €", help="Mean Absolute Error on the 20% holdout set (median-price baseline: 23.49 €)")
        col_m2.metric("RMSE", "16.61 €", help="Root Mean Squared Error on 20% holdout set")
        col_m3.metric("R²", "0.748", help="Coefficient of determination on 20% holdout set")

# Footer
st.markdown(
    "<hr><div style='text-align:center; color:#bbb;'>Made with ❤️ by Christophe NORET - 2026</div>",
    unsafe_allow_html=True,
)
