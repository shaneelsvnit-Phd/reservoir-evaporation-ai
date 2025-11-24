import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="H2O AI: Evaporation Predictor", page_icon="💧", layout="centered")

# --- Header Section ---
st.title("💧 Reservoir Evaporation Predictor")
st.markdown("""
    **AI-Powered Hydrological Forecasting**

    This tool uses a Random Forest Machine Learning model to estimate daily water loss
    based on meteorological parameters.
    *Developed for reservoir management and water conservation planning.*
""")

st.divider()

# --- Model Loading Logic ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('reservoir_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'reservoir_model.pkl' is in the same directory.")
        return None

model = load_model()

# --- User Input Section ---
st.sidebar.header("⚙️ Input Parameters")

def user_input_features():
    temp = st.sidebar.slider("Air Temperature (°C)", 0.0, 50.0, 30.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 40)
    wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 15.0)
    solar = st.sidebar.slider("Solar Radiation (kWh/m²)", 0.0, 12.0, 6.0)
    area = st.sidebar.number_input("Reservoir Surface Area (km²)", min_value=0.1, value=5.0)

    data = {
        'Air_Temperature_C': temp,
        'Humidity_pct': humidity,
        'Wind_Speed_kmh': wind,
        'Solar_Radiation_kWh': solar,
        'Surface_Area_km2': area
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Main Dashboard ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Conditions")
    st.write(input_df.T.rename(columns={0: "Value"}))

with col2:
    st.subheader("Prediction")
    if st.button("Calculate Loss"):
        if model:
            # Predict
            prediction_mm = model.predict(input_df)[0]

            # Calculate Total Volume Loss (Volume = Depth * Area)
            # 1 mm = 1 Liter / m² -> Scaling to Million Liters (ML)
            # 1 km² = 1,000,000 m²
            total_loss_liters = prediction_mm * (input_df['Surface_Area_km2'][0] * 1e6)
            total_loss_ML = total_loss_liters / 1e6

            st.success(f"📉 Depth Loss: **{prediction_mm:.2f} mm**")
            st.info(f"🌊 Volumetric Loss: **{total_loss_ML:.2f} Million Liters**")
        else:
            st.warning("Model not loaded.")

# --- Footer ---
st.markdown("---")
st.caption("Powered by Scikit-Learn & Streamlit | Google AI Scientist Persona Demo")
