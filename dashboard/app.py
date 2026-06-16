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

# Set up Streamlit page configuration
st.set_page_config(page_title="Getaround Dashboard", layout="wide", page_icon="🚗")

# Header
st.markdown(
    """
    <div style='background: linear-gradient(90deg,#1abc9c,#2980b9); padding: 1.2rem 2rem; border-radius: 16px; color: white; font-size: 1.7rem; font-weight: bold; margin-bottom: 2.5rem; letter-spacing: 1px;'>
        🚗 Getaround Dashboard <span style='font-size:1.1rem; font-weight:400;'>&nbsp;|&nbsp; Delay, Pricing & Prediction Insights</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown(
        "<h3>Getaround Dashboard</h3>",
        unsafe_allow_html=True,
    )
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


# Tabs for different analyses
tab1, tab2, tab3 = st.tabs(
    ["⏱️ Delay Analysis", "💰 Pricing Analysis", "💸 Price Prediction"]
)

# Tab 1: Delay Analysis
with tab1:
    st.header("⏱️ Delay Analysis")
    st.info("Analysis of vehicle return delays on Getaround.")

    df = load_delay_data().copy()
    df["delay_at_checkout_in_minutes"] = df["delay_at_checkout_in_minutes"].astype(int)
    df["time_delta_with_previous_rental_in_minutes"] = df[
        "time_delta_with_previous_rental_in_minutes"
    ].astype(int)


    # Sidebar delay filters
    st.sidebar.header("Delay Filters")
    threshold = st.sidebar.slider(
        "⏱️ Minimum time between rentals (minutes)", 0, 180, 60, step=15
    )
    scope = st.sidebar.radio("🔐 Check-in type filter", ["All", "Connect only"])
    if scope == "Connect only":
        df = df[df["checkin_type"] == "connect"]

    # KPI calculations
    late_returns = df[df["delay_at_checkout_in_minutes"] > 0]
    resolved_cases = late_returns[
        late_returns["delay_at_checkout_in_minutes"] > threshold
    ]
    resolved_share = (
        len(resolved_cases) / len(late_returns) * 100 if len(late_returns) > 0 else 0
    )

    # Main metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div style='background:#e8f0fe;padding:20px 12px;border-radius:10px;text-align:center;'><span style='font-size:2.2rem;color:#2980b9;font-weight:700'>{len(late_returns)}</span><br><span style='color:#666;'>Total late returns</span></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='background:#e8f0fe;padding:20px 12px;border-radius:10px;text-align:center;'><span style='font-size:2.2rem;color:#27ae60;font-weight:700'>{len(resolved_cases)}</span><br><span style='color:#666;'>Resolved (>{threshold} min)</span></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div style='background:#e8f0fe;padding:20px 12px;border-radius:10px;text-align:center;'><span style='font-size:2.2rem;color:#1abc9c;font-weight:700'>{resolved_share:.1f}%</span><br><span style='color:#666;'>% covered</span></div>",
            unsafe_allow_html=True,
        )

    with st.expander("ℹ️ What do these metrics mean?"):
        st.write(
            """
            - **Late returns**: Number of rentals with a return delay (> 0 min).
            - **Resolved**: Number of cases where the selected buffer time covers the delay.
            - **% covered**: Percentage of delays covered by your selected threshold.
            """
        )

    # Delay distribution histogram
    fig = px.histogram(
        late_returns,
        x="delay_at_checkout_in_minutes",
        nbins=40,
        title="Delay at checkout distribution",
        color_discrete_sequence=["#2980b9"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Business impact section
    st.subheader("Business Impact")

    conflicts = df[df["time_delta_with_previous_rental_in_minutes"] < threshold]
    total_rentals = len(df)
    impacted_revenue = len(conflicts)
    revenue_share = 100 * impacted_revenue / total_rentals if total_rentals > 0 else 0

    critical_late_returns = late_returns[
        late_returns["time_delta_with_previous_rental_in_minutes"] < threshold
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "% revenue at risk",
        f"{revenue_share:.1f}%",
        delta=f"{impacted_revenue} rentals",
    )
    col2.metric("Rentals blocked", len(conflicts))
    col3.metric("Late returns affecting next", len(critical_late_returns))
    col4.metric(
        "Resolved returns",
        f"{resolved_share:.1f}%",
        delta=f"{len(resolved_cases)} cases",
    )

    # Time delta between rentals
    st.subheader("Time between two consecutive rentals")
    fig_delay = px.histogram(
        df,
        x="time_delta_with_previous_rental_in_minutes",
        nbins=50,
        title="Time delta distribution between rentals (minutes)",
        color_discrete_sequence=["#1abc9c"],
    )
    st.plotly_chart(fig_delay, use_container_width=True)

    # Boxplot after filtering (delays under 2h)
    filtered_df = df[
        (df["delay_at_checkout_in_minutes"] > 0)
        & (df["delay_at_checkout_in_minutes"] < 120)
    ]

    st.subheader("Late returns by check-in type (< 2 hours)")
    fig_box_filtered = px.box(
        filtered_df,
        x="checkin_type",
        y="delay_at_checkout_in_minutes",
        points=False,
        color="checkin_type",
        color_discrete_map={"connect": "#2980b9", "mobile": "#1abc9c"},
        title="Boxplot of delay by check-in type (filtered)",
    )
    st.plotly_chart(fig_box_filtered, use_container_width=True)

    # Download filtered data
    st.download_button(
        "📥 Download filtered delay data",
        filtered_df.to_csv(index=False).encode(),
        "filtered_delays.csv",
    )

# Tab 2: Pricing Analysis
with tab2:
    st.header("💰 Pricing Analysis")
    st.info("Distribution of vehicle characteristics and prices.")

    df_price = load_pricing_data()

    st.subheader("Mileage distribution")
    fig_km = px.histogram(
        df_price,
        x="mileage",
        nbins=50,
        title="Mileage (km)",
        color_discrete_sequence=["#2980b9"],
    )
    st.plotly_chart(fig_km, use_container_width=True)

    st.subheader("Engine power")
    fig_power = px.histogram(
        df_price,
        x="engine_power",
        nbins=30,
        title="Engine power (HP)",
        color_discrete_sequence=["#1abc9c"],
    )
    st.plotly_chart(fig_power, use_container_width=True)

    st.subheader("Daily rental price")
    fig_price = px.histogram(
        df_price,
        x="rental_price_per_day",
        nbins=40,
        title="Rental price per day (€)",
        color_discrete_sequence=["#27ae60"],
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Fuel type distribution")
    fig_fuel = px.histogram(
        df_price, x="fuel", title="Fuel types", color_discrete_sequence=["#ff7675"]
    )
    st.plotly_chart(fig_fuel, use_container_width=True)

    st.subheader("Car types")
    fig_cartype = px.histogram(
        df_price, x="car_type", title="Car types", color_discrete_sequence=["#636e72"]
    )
    st.plotly_chart(fig_cartype, use_container_width=True)

    with st.expander("ℹ️ About this analysis"):
        st.write(
            """
            - **Mileage**: Distribution of car mileage.
            - **Engine power**: Distribution of car engine power.
            - **Rental price**: Daily prices set by owners.
            - **Fuel/Car types**: Diversity of the offer on the platform.
            """
        )

    st.caption("Source: get_around_pricing_project.csv")

# Tab 3: Price Prediction
with tab3:
    st.header("💸 Price Prediction")
    st.info("Fill out the form to get a rental price prediction for a car.")

    df_price = load_pricing_data()

    models = sorted(df_price["model_key"].dropna().unique())
    fuels = sorted(df_price["fuel"].dropna().unique())
    colors = sorted(df_price["paint_color"].dropna().unique())
    car_types = sorted(df_price["car_type"].dropna().unique())

    with st.form(key="predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            mileage = st.number_input(
                "Mileage (thousands of km)", min_value=0.0, value=7.0, step=0.1
            )
            engine_power = st.number_input(
                "Engine power (hundreds of kW)", min_value=0.0, value=0.27, step=0.01
            )
            model_key = st.selectbox("Model", models)
            fuel = st.selectbox("Fuel type", fuels)
            paint_color = st.selectbox("Color", colors)
            car_type = st.selectbox("Car type", car_types)
        with c2:
            private_parking_available = st.checkbox(
                "Private parking available", value=True
            )
            has_gps = st.checkbox("GPS", value=True)
            has_air_conditioning = st.checkbox("Air conditioning", value=False)
            automatic_car = st.checkbox("Automatic car", value=False)
            has_getaround_connect = st.checkbox("Getaround Connect", value=True)
            has_speed_regulator = st.checkbox("Speed regulator", value=False)
            winter_tires = st.checkbox("Winter tires", value=True)

        submit = st.form_submit_button("Predict price")

    if submit:
        input_data = {
            "mileage": mileage,
            "engine_power": engine_power,
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
            response = requests.post(API_URL, json={"input": [input_data]})
            if response.status_code == 200:
                prediction = response.json()["prediction"][0]
                st.success(f"Predicted rental price: **{prediction:.2f} €**")
            else:
                st.error(f"API error: {response.status_code}\n{response.text}")
        except Exception as e:
            st.error(f"Error calling the prediction API: {e}")

    st.caption("Prediction API uses the trained model on Getaround data.")

# Footer
st.markdown(
    "<hr><div style='text-align:center; color: #bbb;'>Made with ❤️ by Christophe NORET - 2026</div>",
    unsafe_allow_html=True,
)
