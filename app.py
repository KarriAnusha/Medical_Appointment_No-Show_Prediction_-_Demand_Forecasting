"""
Medical Appointment No-Show Prediction & Demand Forecasting
Streamlit Application

Two modules:
1. No-Show Risk Predictor - Input patient details, get no-show probability
2. Demand Forecaster - Select time period/specialty, get demand prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Appointment Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── Model Loading (Cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_classification_model():
    """Load the best no-show classifier and preprocessing artifacts."""
    model_path = os.path.join(MODELS_DIR, "best_classifier.joblib")
    encoders_path = os.path.join(MODELS_DIR, "label_encoders.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_columns.joblib")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    features = joblib.load(features_path) if os.path.exists(features_path) else []

    return model, encoders, scaler, features


@st.cache_resource
def load_forecasting_model():
    """Load the best demand forecasting model."""
    model_path = os.path.join(MODELS_DIR, "best_forecaster.joblib")
    features_path = os.path.join(MODELS_DIR, "forecast_feature_columns.joblib")
    log_flag_path = os.path.join(MODELS_DIR, "forecast_log_transform.joblib")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    features = joblib.load(features_path) if os.path.exists(features_path) else []
    uses_log = joblib.load(log_flag_path) if os.path.exists(log_flag_path) else False

    return model, features, uses_log


@st.cache_data
def load_data():
    """Load processed data for visualizations."""
    data_path = os.path.join(DATA_DIR, "processed_data.csv")
    daily_path = os.path.join(DATA_DIR, "daily_demand.csv")

    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    daily = pd.read_csv(daily_path, parse_dates=["appointment_date"]) if os.path.exists(daily_path) else None

    return df, daily


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏥 Medical Appointment Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 No-Show Predictor", "📈 Demand Forecaster", "📊 Insights Dashboard"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**CER Rehabilitation Center**\n\n"
    "University of Vale do Itajaí\n"
    "Southern Brazil"
)


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏥 Medical Appointment No-Show Prediction & Demand Forecasting")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Total Appointments", "109,593")
    with col2:
        st.metric("📊 No-Show Rate", "~31.8%", delta="-11.8% vs target 20%", delta_color="inverse")
    with col3:
        st.metric("🏙️ Cities Served", "13")

    st.markdown("### About This Application")
    st.markdown("""
    This application provides two key analytics tools for the CER Rehabilitation Center:

    | Module | Description |
    |--------|-------------|
    | **🔮 No-Show Predictor** | Predicts the probability of a patient missing their appointment based on demographics, health conditions, and environmental factors |
    | **📈 Demand Forecaster** | Forecasts daily appointment volume to help optimize staff scheduling and resource allocation |
    | **📊 Insights Dashboard** | Visualizations of key patterns, trends, and model performance |

    ### How It Works
    1. **No-Show Predictor**: Enter patient and appointment details → Get a risk score (0-100%)
    2. **Demand Forecaster**: Select date range and parameters → Get predicted daily appointment volume
    3. **Insights**: Explore historical patterns in no-shows, demand, and key factors
    """)


# ══════════════════════════════════════════════════════════════════════════════
# NO-SHOW PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 No-Show Predictor":
    st.title("🔮 No-Show Risk Predictor")
    st.markdown("Enter patient and appointment details to predict the likelihood of a no-show.")
    st.markdown("---")

    # Load model
    clf_model, encoders, scaler, feature_cols = load_classification_model()

    if clf_model is None:
        st.error("⚠️ Classification model not found. Please run Notebook 03 first to train and save the model.")
        st.stop()

    # Input form
    with st.form("noshow_form"):
        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
            disability = st.number_input("Disability Level", min_value=0, max_value=4, value=0, step=1)

        with col2:
            needs_companion = st.selectbox("Needs Companion", [0, 1], format_func=lambda x: "Yes" if x else "No")
            hipertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col3:
            alcoholism = st.selectbox("Alcoholism", [0, 1], format_func=lambda x: "Yes" if x else "No")
            handcap = st.selectbox("Handicap", [0, 1], format_func=lambda x: "Yes" if x else "No")
            scholarship = st.selectbox("Scholarship (Bolsa Família)", [0, 1], format_func=lambda x: "Yes" if x else "No")

        st.subheader("Appointment Details")
        col4, col5, col6 = st.columns(3)

        with col4:
            sms_received = st.selectbox("SMS Reminder Received", [0, 1], format_func=lambda x: "Yes" if x else "No")
            specialty_options = list(encoders.get("specialty", {}).classes_) if "specialty" in encoders else ["Physiotherapy", "Psychotherapy", "Speech Therapy", "Occupational Therapy", "Unknown"]
            specialty = st.selectbox("Specialty", specialty_options)

        with col5:
            shift_options = ["morning", "afternoon", "evening", "night"]
            appointment_shift = st.selectbox("Appointment Shift", shift_options)
            appointment_time = st.number_input("Appointment Time (hour, 0-23)", min_value=0, max_value=23, value=10)

        with col6:
            appointment_date = st.date_input("Appointment Date", value=datetime.now().date())
            place_options = list(encoders.get("place", {}).classes_) if "place" in encoders else ["City A", "City B", "Unknown"]
            place = st.selectbox("City", place_options)

        st.subheader("Weather Conditions")
        col7, col8, col9 = st.columns(3)

        with col7:
            avg_temp = st.number_input("Average Temperature (°C)", min_value=0.0, max_value=50.0, value=22.0, step=0.5)
            max_temp = st.number_input("Maximum Temperature (°C)", min_value=0.0, max_value=50.0, value=28.0, step=0.5)

        with col8:
            rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.5)
            heat_options = list(encoders.get("heat_intensity", {}).classes_) if "heat_intensity" in encoders else ["low", "moderate", "high"]
            heat_intensity = st.selectbox("Heat Intensity", heat_options)

        with col9:
            rain_options = list(encoders.get("rain_intensity", {}).classes_) if "rain_intensity" in encoders else ["none", "light", "moderate", "heavy"]
            rain_intensity = st.selectbox("Rain Intensity", rain_options)
            rainy_day_before = st.selectbox("Rainy Day Before", [0, 1], format_func=lambda x: "Yes" if x else "No")
            storm_day_before = st.selectbox("Storm Day Before", [0, 1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🔮 Predict No-Show Risk", use_container_width=True)

    if submitted:
        with st.spinner("Calculating risk score..."):
            # Build feature dictionary
            appt_date = pd.Timestamp(appointment_date)
            under_12 = 1 if age < 12 else 0
            over_60 = 1 if age >= 60 else 0

            # Age group
            if age < 12: age_group = "child"
            elif age < 18: age_group = "teen"
            elif age < 35: age_group = "young_adult"
            elif age < 50: age_group = "adult"
            elif age < 65: age_group = "middle_aged"
            else: age_group = "senior"

            shift_map = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}

            input_data = {
                "age": age,
                "under_12": under_12,
                "over_60": over_60,
                "disability": disability,
                "needs_companion": needs_companion,
                "Hipertension": hipertension,
                "Diabetes": diabetes,
                "Alcoholism": alcoholism,
                "Handcap": handcap,
                "Scholarship": scholarship,
                "SMSreceived": sms_received,
                "appointment_time": appointment_time,
                "avg_temp": avg_temp,
                "max_temp": max_temp,
                "rain": rain,
                "rainy_day_before": rainy_day_before,
                "storm_day_before": storm_day_before,
                "day_of_week": appt_date.dayofweek,
                "month": appt_date.month,
                "day_of_month": appt_date.day,
                "week_of_year": appt_date.isocalendar()[1],
                "is_weekend": 1 if appt_date.dayofweek >= 5 else 0,
                "is_monday": 1 if appt_date.dayofweek == 0 else 0,
                "is_month_start": 1 if appt_date.day == 1 else 0,
                "is_month_end": 1 if appt_date.day >= 28 else 0,
                "chronic_condition_count": hipertension + diabetes + alcoholism + handcap,
                "shift_encoded": shift_map.get(appointment_shift, 0),
                "temp_rain_interaction": avg_temp * rain,
                "companion_disability": needs_companion * disability,
            }

            # Encode categorical features
            for col_name, encoder in encoders.items():
                value = locals().get(col_name, "Unknown")
                if value is not None:
                    try:
                        encoded_val = encoder.transform([str(value)])[0]
                    except ValueError:
                        encoded_val = 0
                    input_data[col_name + "_encoded"] = encoded_val

            # Create DataFrame with correct feature columns
            input_df = pd.DataFrame([input_data])

            # Ensure all expected features are present
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[feature_cols].select_dtypes(include=[np.number])

            # Ensure column alignment
            missing_cols = set(feature_cols) - set(input_df.columns)
            for col in missing_cols:
                input_df[col] = 0
            input_df = input_df.reindex(columns=feature_cols, fill_value=0)

            # Predict
            probability = clf_model.predict_proba(input_df)[0][1]
            risk_pct = probability * 100

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            if risk_pct >= 60:
                st.error(f"### 🔴 HIGH RISK\n### {risk_pct:.1f}%")
            elif risk_pct >= 35:
                st.warning(f"### 🟡 MEDIUM RISK\n### {risk_pct:.1f}%")
            else:
                st.success(f"### 🟢 LOW RISK\n### {risk_pct:.1f}%")

        with col_r2:
            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                title={"text": "No-Show Risk Score"},
                delta={"reference": 31.8, "suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 35], "color": "#2ecc71"},
                        {"range": [35, 60], "color": "#f39c12"},
                        {"range": [60, 100], "color": "#e74c3c"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 31.8,
                    },
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_r3:
            st.markdown("### Recommended Actions")
            if risk_pct >= 60:
                st.markdown("""
                - 📱 Send **immediate SMS reminder**
                - 📞 Follow up with **phone call**
                - 📅 Offer **rescheduling** option
                - ⏰ Set **day-before reminder**
                """)
            elif risk_pct >= 35:
                st.markdown("""
                - 📱 Send **SMS reminder** 24h before
                - 📧 Send email confirmation
                - 📅 Consider **waitlist backup**
                """)
            else:
                st.markdown("""
                - ✅ Standard confirmation sufficient
                - 📱 Routine SMS reminder
                """)


# ══════════════════════════════════════════════════════════════════════════════
# DEMAND FORECASTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecaster":
    st.title("📈 Demand Forecaster")
    st.markdown("Predict daily appointment volume for resource planning and staff scheduling.")
    st.markdown("---")

    forecast_model, forecast_features, forecast_uses_log = load_forecasting_model()
    df, daily = load_data()

    if forecast_model is None:
        st.error("⚠️ Forecasting model not found. Please run Notebook 04 first to train and save the model.")
        st.stop()

    # Input form
    with st.form("forecast_form"):
        st.subheader("Forecast Parameters")
        col1, col2, col3 = st.columns(3)

        with col1:
            forecast_date = st.date_input("Forecast Date", value=datetime.now().date() + timedelta(days=1))
            avg_temp_f = st.number_input("Expected Avg Temp (°C)", min_value=0.0, max_value=50.0, value=22.0)

        with col2:
            max_temp_f = st.number_input("Expected Max Temp (°C)", min_value=0.0, max_value=50.0, value=28.0)
            rain_f = st.number_input("Expected Rainfall (mm)", min_value=0.0, max_value=200.0, value=0.0)

        with col3:
            # Use historical averages for lag features
            if daily is not None and len(daily) > 0:
                recent_avg = daily["total_appointments"].tail(7).mean()
                recent_14 = daily["total_appointments"].tail(14).mean()
                recent_30 = daily["total_appointments"].tail(30).mean()
            else:
                recent_avg = 150
                recent_14 = 150
                recent_30 = 150
            st.metric("Recent 7-day Avg", f"{recent_avg:.0f} appts/day")
            st.metric("Recent 14-day Avg", f"{recent_14:.0f} appts/day")

        submitted_f = st.form_submit_button("📈 Generate Forecast", use_container_width=True)

    if submitted_f:
        with st.spinner("Generating demand forecast..."):
            f_date = pd.Timestamp(forecast_date)

            # Build feature vector
            forecast_input = {
                "day_of_week": f_date.dayofweek,
                "month": f_date.month,
                "is_weekend": 1 if f_date.dayofweek >= 5 else 0,
                "week_of_year": f_date.isocalendar()[1],
                "avg_age": 35.0,
            }

            # Add weather
            if "avg_temp" in forecast_features:
                forecast_input["avg_temp"] = avg_temp_f
            if "max_temp" in forecast_features:
                forecast_input["max_temp"] = max_temp_f
            if "avg_rain" in forecast_features:
                forecast_input["avg_rain"] = rain_f

            # Add lag features using recent data
            if daily is not None and len(daily) > 0:
                recent = daily["total_appointments"].values
                for lag in [1, 2, 3, 7, 14]:
                    col_name = f"demand_lag_{lag}"
                    if col_name in forecast_features:
                        forecast_input[col_name] = float(recent[-lag]) if lag <= len(recent) else recent_avg

                if "demand_rolling_7" in forecast_features:
                    forecast_input["demand_rolling_7"] = float(np.mean(recent[-7:]))
                if "demand_rolling_14" in forecast_features:
                    forecast_input["demand_rolling_14"] = float(np.mean(recent[-14:]))
                if "demand_rolling_30" in forecast_features:
                    forecast_input["demand_rolling_30"] = float(np.mean(recent[-30:]))
            else:
                for lag in [1, 2, 3, 7, 14]:
                    col_name = f"demand_lag_{lag}"
                    if col_name in forecast_features:
                        forecast_input[col_name] = recent_avg
                for roll in ["demand_rolling_7", "demand_rolling_14", "demand_rolling_30"]:
                    if roll in forecast_features:
                        forecast_input[roll] = recent_avg

            # No-show related features
            if "no_show_rate" in forecast_features:
                forecast_input["no_show_rate"] = 0.318

            # Enhanced features (log-transformed lags, momentum, etc.)
            if daily is not None and len(daily) > 0:
                lag1_val = forecast_input.get("demand_lag_1", recent_avg)
                if "demand_lag_1_log" in forecast_features:
                    forecast_input["demand_lag_1_log"] = float(np.log1p(lag1_val))
                if "demand_rolling_7_log" in forecast_features:
                    forecast_input["demand_rolling_7_log"] = float(np.log1p(forecast_input.get("demand_rolling_7", recent_avg)))
                if "is_low_prev" in forecast_features:
                    forecast_input["is_low_prev"] = 1 if lag1_val < 10 else 0
                if "demand_momentum" in forecast_features:
                    r7 = forecast_input.get("demand_rolling_7", recent_avg)
                    r14 = forecast_input.get("demand_rolling_14", recent_avg)
                    forecast_input["demand_momentum"] = float(r7 - r14)

            # Create DataFrame
            forecast_df = pd.DataFrame([forecast_input])
            for col in forecast_features:
                if col not in forecast_df.columns:
                    forecast_df[col] = 0
            forecast_df = forecast_df.reindex(columns=forecast_features, fill_value=0)

            # Predict (apply expm1 only if model was trained on log1p-transformed target)
            raw_pred = forecast_model.predict(forecast_df)[0]
            if forecast_uses_log:
                predicted_demand = max(0, int(round(np.expm1(raw_pred))))
            else:
                predicted_demand = max(0, int(round(raw_pred)))

        # Display results
        st.markdown("---")
        st.subheader(f"Forecast for {forecast_date.strftime('%A, %B %d, %Y')}")

        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.metric(
                "📋 Predicted Appointments",
                f"{predicted_demand}",
                delta=f"{predicted_demand - recent_avg:.0f} vs 7-day avg"
            )

        with col_f2:
            estimated_noshow = int(predicted_demand * 0.318)
            st.metric("⚠️ Estimated No-Shows", f"{estimated_noshow}")
            st.metric("✅ Expected Actual Visits", f"{predicted_demand - estimated_noshow}")

        with col_f3:
            st.markdown("### Staffing Recommendations")
            if predicted_demand > recent_avg * 1.15:
                st.warning("📈 **Higher than average** — Consider adding staff")
            elif predicted_demand < recent_avg * 0.85:
                st.info("📉 **Lower than average** — Possible to reduce staff")
            else:
                st.success("📊 **Normal range** — Standard staffing adequate")

    # Historical demand chart
    if daily is not None:
        st.markdown("---")
        st.subheader("Historical Appointment Demand")
        fig = px.line(daily, x="appointment_date", y="total_appointments",
                      title="Daily Appointment Volume",
                      labels={"total_appointments": "Appointments", "appointment_date": "Date"})
        fig.update_traces(line_color="steelblue")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Insights Dashboard":
    st.title("📊 Insights Dashboard")
    st.markdown("Key patterns and model performance insights.")
    st.markdown("---")

    df, daily = load_data()

    if df is None:
        st.error("⚠️ Processed data not found. Please run Notebooks 01-02 first.")
        st.stop()

    # Tabs for different insights
    tab1, tab2, tab3, tab4 = st.tabs(["No-Show Patterns", "Demand Trends", "Model Performance", "Geographic Analysis"])

    with tab1:
        st.subheader("No-Show Rate Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # No-show by health conditions
            health_cols = ["Hipertension", "Diabetes", "Alcoholism", "Handcap", "Scholarship", "SMSreceived"]
            existing = [c for c in health_cols if c in df.columns]
            if existing:
                rates = {}
                for col in existing:
                    if df["no_show"].dtype == "object":
                        rate = df[df[col] == 1]["no_show"].apply(lambda x: 1 if x == "Yes" else 0).mean()
                    else:
                        rate = df[df[col] == 1]["no_show"].mean()
                    rates[col] = rate
                rates_df = pd.DataFrame(list(rates.items()), columns=["Condition", "No-Show Rate"])
                fig = px.bar(rates_df, x="Condition", y="No-Show Rate",
                             title="No-Show Rate by Health Condition",
                             color="No-Show Rate", color_continuous_scale="RdYlGn_r")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # No-show by shift
            if "appointment_shift" in df.columns:
                if df["no_show"].dtype == "object":
                    shift_rate = df.groupby("appointment_shift")["no_show"].apply(
                        lambda x: (x == "Yes").mean()
                    ).reset_index()
                else:
                    shift_rate = df.groupby("appointment_shift")["no_show"].mean().reset_index()
                shift_rate.columns = ["Shift", "No-Show Rate"]
                fig = px.bar(shift_rate, x="Shift", y="No-Show Rate",
                             title="No-Show Rate by Appointment Shift",
                             color="No-Show Rate", color_continuous_scale="RdYlGn_r")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Appointment Demand Trends")

        if daily is not None:
            col1, col2 = st.columns(2)

            with col1:
                # Day of week pattern
                dow_demand = daily.groupby("day_of_week")["total_appointments"].mean().reset_index()
                dow_demand["day_name"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                fig = px.bar(dow_demand, x="day_name", y="total_appointments",
                             title="Average Daily Appointments by Day of Week",
                             color="total_appointments", color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Monthly pattern
                monthly = daily.groupby("month")["total_appointments"].mean().reset_index()
                fig = px.bar(monthly, x="month", y="total_appointments",
                             title="Average Daily Appointments by Month",
                             color="total_appointments", color_continuous_scale="Greens")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Model Performance Summary")

        # Load results if available
        clf_results_path = os.path.join(MODELS_DIR, "classification_results.csv")
        forecast_results_path = os.path.join(MODELS_DIR, "forecast_results.csv")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Classification Models")
            if os.path.exists(clf_results_path):
                clf_results = pd.read_csv(clf_results_path, index_col=0)
                st.dataframe(clf_results.style.format("{:.4f}").highlight_max(axis=0, color="lightgreen"),
                             use_container_width=True)
            else:
                st.info("Run Notebook 03 to see classification results.")

        with col2:
            st.markdown("#### Forecasting Models")
            if os.path.exists(forecast_results_path):
                fc_results = pd.read_csv(forecast_results_path, index_col=0)
                format_dict = {"RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE": "{:.1f}", "R²": "{:.4f}"}
                if "WMAPE" in fc_results.columns:
                    format_dict["WMAPE"] = "{:.1f}"
                min_cols = [c for c in ["RMSE", "MAE", "WMAPE"] if c in fc_results.columns]
                st.dataframe(fc_results.style.format(format_dict)
                             .highlight_min(subset=min_cols, color="lightgreen")
                             .highlight_max(subset=["R²"], color="lightgreen"),
                             use_container_width=True)
            else:
                st.info("Run Notebook 04 to see forecasting results.")

    with tab4:
        st.subheader("Geographic Analysis")

        if "place" in df.columns:
            if df["no_show"].dtype == "object":
                place_stats = df.groupby("place").agg(
                    total=("no_show", "size"),
                    noshow_rate=("no_show", lambda x: (x == "Yes").mean())
                ).reset_index()
            else:
                place_stats = df.groupby("place").agg(
                    total=("no_show", "size"),
                    noshow_rate=("no_show", "mean")
                ).reset_index()
            place_stats = place_stats.sort_values("total", ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(place_stats, x="place", y="total",
                             title="Appointments by City",
                             color="total", color_continuous_scale="Blues")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(place_stats, x="place", y="noshow_rate",
                             title="No-Show Rate by City",
                             color="noshow_rate", color_continuous_scale="RdYlGn_r")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Place/city data not available.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with ❤️ using Streamlit\n\n"
    "**Tech Stack:** Python, Scikit-learn, XGBoost, LightGBM, Plotly"
)
