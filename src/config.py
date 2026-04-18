import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "Medical_appointment_data.csv")

# Column rename mapping (raw dataset → code names)
COLUMN_RENAME = {
    "under_12_years_old": "under_12",
    "over_60_years_old": "over_60",
    "patient_needs_companion": "needs_companion",
    "average_temp_day": "avg_temp",
    "average_rain_day": "rain",
    "max_temp_day": "max_temp",
    "max_rain_day": "max_rain",
    "SMS_received": "SMSreceived",
}

# Target
TARGET_COL = "no_show"

# Feature groups
PATIENT_FEATURES = ["gender", "age", "under_12", "over_60", "disability", "needs_companion"]  # disability is categorical
APPOINTMENT_FEATURES = ["specialty", "appointment_time", "appointment_shift", "appointment_date_continuous"]
LOCATION_FEATURES = ["place"]
HEALTH_FEATURES = ["Hipertension", "Diabetes", "Alcoholism", "Handcap", "Scholarship", "SMSreceived"]
WEATHER_FEATURES = ["avg_temp", "max_temp", "rain", "heat_intensity", "rain_intensity",
                     "rainy_day_before", "storm_day_before"]

CATEGORICAL_FEATURES = ["gender", "specialty", "appointment_shift", "place",
                        "heat_intensity", "rain_intensity", "disability"]
BINARY_FEATURES = ["under_12", "over_60", "needs_companion", "Hipertension", "Diabetes",
                   "Alcoholism", "Handcap", "Scholarship", "SMSreceived",
                   "rainy_day_before", "storm_day_before"]
NUMERICAL_FEATURES = ["age", "appointment_time", "avg_temp", "max_temp", "rain", "max_rain"]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
