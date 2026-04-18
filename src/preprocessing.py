"""
Preprocessing utilities for Medical Appointment No-Show Prediction & Demand Forecasting.
Contains functions for data loading, cleaning, feature engineering, and model preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from src.config import (
    RAW_DATA_PATH, MODELS_DIR, TARGET_COL, RANDOM_STATE, TEST_SIZE,
    CATEGORICAL_FEATURES, BINARY_FEATURES, NUMERICAL_FEATURES, COLUMN_RENAME
)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(path=None):
    """Load the raw dataset and rename columns to standard names."""
    if path is None:
        path = RAW_DATA_PATH
    df = pd.read_csv(path)
    # Rename columns to match code conventions
    df = df.rename(columns=COLUMN_RENAME)
    # Clean disability column: treat blank/space as NaN
    if "disability" in df.columns:
        df["disability"] = df["disability"].replace(r'^\s*$', np.nan, regex=True)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ── Missing Value Handling ────────────────────────────────────────────────────

def handle_missing_values(df):
    """Handle missing values with appropriate strategies per column."""
    df = df.copy()

    # Age: fill with median (numerical)
    if "age" in df.columns and df["age"].isnull().sum() > 0:
        median_age = df["age"].median()
        df["age"] = df["age"].fillna(median_age)
        print(f"  age: filled {df['age'].isnull().sum()} NaN with median={median_age:.1f}")

    # Specialty: fill with 'Unknown' (categorical)
    if "specialty" in df.columns and df["specialty"].isnull().sum() > 0:
        df["specialty"] = df["specialty"].fillna("Unknown")
        print(f"  specialty: filled NaN with 'Unknown'")

    # Disability: fill with 'Unknown' (categorical)
    if "disability" in df.columns and df["disability"].isnull().sum() > 0:
        df["disability"] = df["disability"].fillna("Unknown")
        print(f"  disability: filled NaN with 'Unknown'")

    # Place: fill with 'Unknown' (categorical)
    if "place" in df.columns and df["place"].isnull().sum() > 0:
        df["place"] = df["place"].fillna("Unknown")
        print(f"  place: filled NaN with 'Unknown'")

    # Weather features: fill with median
    weather_cols = ["avg_temp", "max_temp", "rain"]
    for col in weather_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  {col}: filled NaN with median={median_val:.2f}")

    # Categorical weather: fill with mode
    cat_weather = ["heat_intensity", "rain_intensity"]
    for col in cat_weather:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  {col}: filled NaN with mode='{mode_val}'")

    # Binary weather: fill with 0
    bin_weather = ["rainy_day_before", "storm_day_before"]
    for col in bin_weather:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)
            print(f"  {col}: filled NaN with 0")

    # Fill remaining numeric NaN with median, categorical with mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown")
            print(f"  {col}: filled remaining NaN")

    print(f"\nTotal missing after handling: {df.isnull().sum().sum()}")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df):
    """Create new features from existing data."""
    df = df.copy()

    # ── Temporal features from appointment_date_continuous ──
    if "appointment_date_continuous" in df.columns:
        df["appointment_date"] = pd.to_datetime(df["appointment_date_continuous"], errors="coerce")
        if df["appointment_date"].notna().any():
            df["day_of_week"] = df["appointment_date"].dt.dayofweek  # 0=Mon, 6=Sun
            df["month"] = df["appointment_date"].dt.month
            df["day_of_month"] = df["appointment_date"].dt.day
            df["week_of_year"] = df["appointment_date"].dt.isocalendar().week.astype(int)
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
            df["is_month_start"] = df["appointment_date"].dt.is_month_start.astype(int)
            df["is_month_end"] = df["appointment_date"].dt.is_month_end.astype(int)
            print("  Created temporal features: day_of_week, month, day_of_month, week_of_year, is_weekend, is_month_start, is_month_end")

    # ── Age-based features ──
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 50, 65, 120],
                                  labels=["child", "teen", "young_adult", "adult", "middle_aged", "senior"],
                                  right=False)
        df["age_group"] = df["age_group"].astype(str)
        print("  Created age_group feature")

    # ── Appointment shift encoding ──
    if "appointment_shift" in df.columns:
        shift_map = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
        df["shift_encoded"] = df["appointment_shift"].str.lower().map(shift_map)
        if df["shift_encoded"].isnull().any():
            df["shift_encoded"] = df["shift_encoded"].fillna(0)
        print("  Created shift_encoded feature")

    # ── Health risk score (composite) ──
    health_cols = ["Hipertension", "Diabetes", "Alcoholism", "Handcap"]
    existing_health = [c for c in health_cols if c in df.columns]
    if existing_health:
        df["health_risk_score"] = df[existing_health].sum(axis=1)
        print(f"  Created health_risk_score from {existing_health}")

    # ── Weather composite ──
    if "avg_temp" in df.columns and "rain" in df.columns:
        df["temp_rain_interaction"] = df["avg_temp"] * df["rain"]
        print("  Created temp_rain_interaction feature")

    # ── Companion + has_disability interaction ──
    if "needs_companion" in df.columns and "disability" in df.columns:
        df["has_disability"] = (df["disability"] != "Unknown").astype(int)
        df["companion_disability"] = df["needs_companion"] * df["has_disability"]
        print("  Created has_disability and companion_disability features")

    return df


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_target(df, target_col=TARGET_COL):
    """Encode the target variable: Yes/yes=1 (no-show), No/no=0 (showed up)."""
    df = df.copy()
    if df[target_col].dtype == "object":
        # Handle both capitalized and lowercase
        df[target_col] = df[target_col].str.lower().map({"yes": 1, "no": 0})
    print(f"Target encoded - No-show distribution:\n{df[target_col].value_counts(normalize=True)}")
    return df


def encode_features(df, label_encoders=None, fit=True):
    """Encode categorical features using Label Encoding."""
    df = df.copy()
    if label_encoders is None:
        label_encoders = {}

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    # Also encode engineered categorical features
    if "age_group" in df.columns:
        cat_cols.append("age_group")

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str)
                # Handle unseen categories
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])

    return df, label_encoders


# ── Train/Test Split ──────────────────────────────────────────────────────────

def split_classification_data(df, target_col=TARGET_COL):
    """Split data for classification task."""
    drop_cols = [target_col]
    # Drop non-feature columns
    non_features = ["appointment_date_continuous", "appointment_date"]
    drop_cols.extend([c for c in non_features if c in df.columns])

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train no-show rate: {y_train.mean():.3f}, Test no-show rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def split_timeseries_data(daily_df, date_col="appointment_date", test_ratio=0.2):
    """Chronological split for time series forecasting."""
    daily_df = daily_df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(daily_df) * (1 - test_ratio))

    train = daily_df.iloc[:split_idx].copy()
    test = daily_df.iloc[split_idx:].copy()

    print(f"Time series split - Train: {len(train)} days, Test: {len(test)} days")
    print(f"Train period: {train[date_col].min()} to {train[date_col].max()}")
    print(f"Test period:  {test[date_col].min()} to {test[date_col].max()}")
    return train, test


# ── Aggregation for Demand Forecasting ────────────────────────────────────────

def create_daily_demand(df):
    """Aggregate appointment data to daily demand counts."""
    if "appointment_date" not in df.columns:
        if "appointment_date_continuous" in df.columns:
            df["appointment_date"] = pd.to_datetime(df["appointment_date_continuous"], errors="coerce")
        else:
            raise ValueError("No date column found for aggregation")

    daily = df.groupby("appointment_date").agg(
        total_appointments=("appointment_date", "size"),
        no_show_count=(TARGET_COL, "sum"),
        avg_age=("age", "mean"),
        avg_temp=("avg_temp", "mean") if "avg_temp" in df.columns else ("age", "count"),
        max_temp=("max_temp", "mean") if "max_temp" in df.columns else ("age", "count"),
        avg_rain=("rain", "mean") if "rain" in df.columns else ("age", "count"),
    ).reset_index()

    daily["no_show_rate"] = daily["no_show_count"] / daily["total_appointments"]
    daily["show_count"] = daily["total_appointments"] - daily["no_show_count"]

    # Temporal features
    daily["day_of_week"] = daily["appointment_date"].dt.dayofweek
    daily["month"] = daily["appointment_date"].dt.month
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["week_of_year"] = daily["appointment_date"].dt.isocalendar().week.astype(int)

    # Lag features
    for lag in [1, 2, 3, 7]:
        daily[f"demand_lag_{lag}"] = daily["total_appointments"].shift(lag)
    daily["demand_rolling_7"] = daily["total_appointments"].rolling(7).mean()
    daily["demand_rolling_14"] = daily["total_appointments"].rolling(14).mean()

    daily = daily.dropna().reset_index(drop=True)

    print(f"Daily demand dataset: {daily.shape[0]} days")
    print(f"Average daily appointments: {daily['total_appointments'].mean():.1f}")
    return daily


# ── Model Save/Load ───────────────────────────────────────────────────────────

def save_model(model, filename, directory=None):
    """Save a model to disk using joblib."""
    if directory is None:
        directory = MODELS_DIR
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")
    return filepath


def load_model(filename, directory=None):
    """Load a model from disk."""
    if directory is None:
        directory = MODELS_DIR
    filepath = os.path.join(directory, filename)
    model = joblib.load(filepath)
    print(f"Model loaded: {filepath}")
    return model


def save_artifacts(artifacts_dict, directory=None):
    """Save multiple artifacts (encoders, scalers, etc.)."""
    if directory is None:
        directory = MODELS_DIR
    os.makedirs(directory, exist_ok=True)
    for name, obj in artifacts_dict.items():
        filepath = os.path.join(directory, f"{name}.joblib")
        joblib.dump(obj, filepath)
    print(f"Saved {len(artifacts_dict)} artifacts to {directory}")


def load_artifacts(names, directory=None):
    """Load multiple artifacts."""
    if directory is None:
        directory = MODELS_DIR
    artifacts = {}
    for name in names:
        filepath = os.path.join(directory, f"{name}.joblib")
        if os.path.exists(filepath):
            artifacts[name] = joblib.load(filepath)
    print(f"Loaded {len(artifacts)} artifacts")
    return artifacts
