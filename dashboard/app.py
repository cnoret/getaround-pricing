"""
Getaround Dashboard Application
This application provides an interactive dashboard for analyzing rental delays and pricing data.
It uses Streamlit for the frontend and Plotly for visualizations.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

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

API_URL = os.environ.get("API_URL", "http://localhost:8001/predict")


@st.cache_data
def load_delay_data():
    df = pd.read_csv("data/get_around_delay_analysis.csv")
    return df[df["state"] == "ended"].dropna(
        subset=[
            "delay_at_checkout_in_minutes",
            "time_delta_with_previous_rental_in_minutes",
        ]
    )


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

    df = load_delay_data().copy()
    df["delay_at_checkout_in_minutes"] = df["delay_at_checkout_in_minutes"].astype(int)
    df["time_delta_with_previous_rental_in_minutes"] = df[
        "time_delta_with_previous_rental_in_minutes"
    ].astype(int)

    st.sidebar.header("Delay Filters")
    threshold = st.sidebar.slider(
        "⏱️ Minimum time between rentals (min)", 0, 180, 60, step=15
    )
    scope = st.sidebar.radio("🔐 Check-in type", ["All", "Connect only"])
    if scope == "Connect only":
        df = df[df["checkin_type"] == "connect"]

    late_returns = df[df["delay_at_checkout_in_minutes"] > 0]
    resolved_cases = late_returns[
        late_returns["delay_at_checkout_in_minutes"] > threshold
    ]
    resolved_share = (
        len(resolved_cases) / len(late_returns) * 100 if len(late_returns) > 0 else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Late returns",
        len(late_returns),
        help="Rentals where the car was returned after the scheduled checkout time",
    )
    col2.metric(f"Resolved (>{threshold} min buffer)", len(resolved_cases))
    col3.metric("% of delays covered", f"{resolved_share:.1f}%")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.write(
            """
            - **Late returns**: rentals with a return delay > 0 min.
            - **Resolved**: cases where the selected buffer time absorbs the delay.
            - **% covered**: share of late returns covered by the buffer.
            """
        )

    sentinel_count = (late_returns["delay_at_checkout_in_minutes"] == 126).sum()
    hist_df = late_returns[late_returns["delay_at_checkout_in_minutes"] != 126]
    fig_hist = px.histogram(
        hist_df,
        x="delay_at_checkout_in_minutes",
        nbins=40,
        title="Distribution of checkout delays",
        labels={"delay_at_checkout_in_minutes": "Delay (minutes)"},
        color_discrete_sequence=["#2980b9"],
    )
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        f"⚠️ {sentinel_count:,} rentals encoded as exactly 126 min are excluded, "
        "this appears to be a sentinel value for delays above 2 hours in the source data."
    )

    st.subheader("Business Impact")
    st.caption(f"Simulating a {threshold}-minute minimum gap between consecutive rentals.")

    conflicts = df[df["time_delta_with_previous_rental_in_minutes"] < threshold]
    total_rentals = len(df)
    impacted_revenue = len(conflicts)
    revenue_share = 100 * impacted_revenue / total_rentals if total_rentals > 0 else 0
    critical_late_returns = late_returns[
        late_returns["time_delta_with_previous_rental_in_minutes"] < threshold
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue at risk", f"{revenue_share:.1f}%", delta=f"{impacted_revenue} rentals", delta_color="off")
    col2.metric("Rentals blocked", len(conflicts), delta_color="off")
    col3.metric("Late returns affecting next", len(critical_late_returns), delta_color="off")
    col4.metric("Delays resolved", f"{resolved_share:.1f}%", delta=f"{len(resolved_cases)} cases", delta_color="off")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_gap = px.histogram(
            df,
            x="time_delta_with_previous_rental_in_minutes",
            nbins=50,
            title="Time gap between consecutive rentals",
            labels={"time_delta_with_previous_rental_in_minutes": "Time gap (minutes)"},
            color_discrete_sequence=["#1abc9c"],
        )
        fig_gap.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{threshold} min buffer",
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    with col_b:
        filtered_df = df[
            (df["delay_at_checkout_in_minutes"] > 0)
            & (df["delay_at_checkout_in_minutes"] < 120)
        ]
        fig_box = px.box(
            filtered_df,
            x="checkin_type",
            y="delay_at_checkout_in_minutes",
            points=False,
            color="checkin_type",
            color_discrete_map={"connect": "#2980b9", "mobile": "#1abc9c"},
            title="Delay by check-in type (under 2 hours)",
            labels={
                "delay_at_checkout_in_minutes": "Delay (minutes)",
                "checkin_type": "Check-in type",
            },
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.download_button(
        "📥 Download filtered delay data",
        filtered_df.to_csv(index=False).encode(),
        "filtered_delays.csv",
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
        st.plotly_chart(fig_km, use_container_width=True)

    with col_b:
        fig_price_dist = px.histogram(
            df_price,
            x="rental_price_per_day",
            nbins=40,
            title="Daily rental price distribution",
            labels={"rental_price_per_day": "Price per day (€)"},
            color_discrete_sequence=["#27ae60"],
        )
        st.plotly_chart(fig_price_dist, use_container_width=True)

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
    st.plotly_chart(fig_scatter, use_container_width=True)

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
        st.plotly_chart(fig_type, use_container_width=True)

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
        st.plotly_chart(fig_fuel, use_container_width=True)

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

        submit = st.form_submit_button("🔮 Predict price", use_container_width=True)

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
        try:
            response = requests.post(API_URL, json={"input": [input_data]}, timeout=10)
            if response.status_code == 200:
                prediction = response.json()["prediction"][0]
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
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Could not reach the prediction API: {e}")

    with st.expander("ℹ️ About this model"):
        st.write(
            """
            Predictions are generated by a **Random Forest Regressor** trained on ~4,800 Getaround listings.

            The preprocessing pipeline applies:
            - **StandardScaler** on numeric features (mileage, engine power)
            - **OneHotEncoder** on categorical features (brand, fuel type, color, car type)
            """
        )
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("MAE", "10.68 €", help="Mean Absolute Error on 20% holdout set")
        col_m2.metric("RMSE", "16.73 €", help="Root Mean Squared Error on 20% holdout set")
        col_m3.metric("R²", "0.734", help="Coefficient of determination on 20% holdout set")

# Footer
st.markdown(
    "<hr><div style='text-align:center; color:#bbb;'>Made with ❤️ by Christophe NORET - 2026</div>",
    unsafe_allow_html=True,
)
