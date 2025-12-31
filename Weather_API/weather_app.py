# =========================================================
# 🌦️ AI PROBABILISTIC WEATHER FORECAST 
# =========================================================

# ---------------- IMPORTS ----------------
import os
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from datetime import datetime, timedelta

# ---------------- ENV FIX ----------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

# ---------------- CONFIG ----------------
API_KEY = st.secrets["OPENWEATHER_API_KEY"]

if "OPENWEATHER_API_KEY" not in st.secrets:
    st.error("API key not configured")
    st.stop()
    
PAST_HOURS = 72
FUTURE_DAYS = 7
QUANTILES = [0.1, 0.5, 0.9]

FEATURES = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "cloud"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weather_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="AI Probabilistic Weather Forecast",
    layout="wide"
)

# ---------------- LOAD MODEL & SCALER ----------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_model_scaler():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found:\n{MODEL_PATH}")
        st.stop()

    if not os.path.exists(SCALER_PATH):
        st.error(f"❌ Scaler file not found:\n{SCALER_PATH}")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model_scaler()

# ---------------- WEATHER API ----------------
@st.cache_data(ttl=600)
def fetch_weather(city: str):
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )

    r = requests.get(url, timeout=10)
    data = r.json()

    if r.status_code != 200:
        raise ValueError(data.get("message", "Weather API error"))

    return {
        "temperature_celsius": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure_mb": data["main"]["pressure"],
        "wind_kph": data["wind"]["speed"] * 3.6,
        "cloud": data["clouds"]["all"],
    }

# ---------------- PREDICTION ----------------
@st.cache_data(ttl=300)
def predict(history_df: pd.DataFrame, current_dict: dict):
    data = pd.concat(
        [history_df, pd.DataFrame([current_dict])],
        ignore_index=True
    )[FEATURES]

    scaled = scaler.transform(data)
    X = scaled[-PAST_HOURS:].reshape(1, PAST_HOURS, len(FEATURES))

    preds = model.predict(X, verbose=0)[0]
    preds = preds.reshape(FUTURE_DAYS, len(QUANTILES))

    def inverse_temp(temp_scaled):
        dummy = np.zeros((FUTURE_DAYS, len(FEATURES)))
        dummy[:, 0] = temp_scaled
        return scaler.inverse_transform(dummy)[:, 0]

    return (
        inverse_temp(preds[:, 0]),
        inverse_temp(preds[:, 1]),
        inverse_temp(preds[:, 2]),
    )

# =========================================================
# ========================= UI ============================
# =========================================================

st.title("🌦️ AI Probabilistic Weather Forecast")

# ---------------- INPUT FORM ----------------
st.subheader("📍 Location")

with st.form("city_form"):
    city = st.text_input(
        "Enter City Name",
        placeholder="Delhi, Tokyo, London, New York"
    )
    submit = st.form_submit_button("🔮 Predict Weather")

# ---------------- RUN ----------------
if submit:
    if not city.strip():
        st.warning("⚠️ Please enter a valid city name")
    else:
        with st.spinner("Fetching weather & predicting..."):
            try:
                current = fetch_weather(city)

                # -------- CURRENT WEATHER --------
                st.subheader("📌 Current Weather")
                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric("🌡️ Temp (°C)", f"{current['temperature_celsius']:.1f}")
                c2.metric("💧 Humidity (%)", current["humidity"])
                c3.metric("🌬️ Wind (kph)", f"{current['wind_kph']:.1f}")
                c4.metric("☁️ Cloud (%)", current["cloud"])
                c5.metric("🔽 Pressure (mb)", current["pressure_mb"])

                # -------- SIMULATED HISTORY --------
                history = []
                for _ in range(PAST_HOURS - 1):
                    history.append({
                        "temperature_celsius": current["temperature_celsius"] + np.random.uniform(-4, 4),
                        "humidity": np.clip(current["humidity"] + np.random.uniform(-10, 10), 0, 100),
                        "pressure_mb": np.clip(current["pressure_mb"] + np.random.uniform(-8, 8), 900, 1100),
                        "wind_kph": max(0, current["wind_kph"] + np.random.uniform(-5, 5)),
                        "cloud": np.clip(current["cloud"] + np.random.uniform(-20, 20), 0, 100),
                    })

                history_df = pd.DataFrame(history)[FEATURES]

                # -------- FORECAST --------
                q10, q50, q90 = predict(history_df, current)

                dates = [
                    datetime.now().date() + timedelta(days=i)
                    for i in range(FUTURE_DAYS + 1)
                ]

                df = pd.DataFrame({
                    "Date": dates,
                    "Lower (10%)": np.insert(q10, 0, current["temperature_celsius"]),
                    "Median (50%)": np.insert(q50, 0, current["temperature_celsius"]),
                    "Upper (90%)": np.insert(q90, 0, current["temperature_celsius"]),
                })

                # -------- VISUALS --------
                st.subheader("📈 7-Day Temperature Forecast")
                st.line_chart(df.set_index("Date")[["Median (50%)"]])
                st.area_chart(df.set_index("Date")[["Lower (10%)", "Upper (90%)"]])

                st.subheader("📋 Forecast Table")
                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")
