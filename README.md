# 🏥 Medical Appointment No-Show Prediction & Demand Forecasting

**Domain:** Healthcare Operations & Resource Management  
**Facility:** University of Vale do Itajaí - CER Rehabilitation Center, Southern Brazil  
**Dataset:** 109,593 appointments × 26 columns

---

## 📋 Problem Statement

The CER rehabilitation facility faces a **31.8% no-show rate** (much higher than the typical 10-20%), resulting in:
- Significant revenue loss and wasted specialist capacity
- Unpredictable daily appointment demand across specialties and locations
- Inability to proactively identify high-risk patients

This project builds **two complementary ML systems**:
1. **No-Show Predictor** — Binary classifier to flag high-risk appointments
2. **Demand Forecaster** — Time series model to predict daily appointment volume

---

## 🗂️ Project Structure

```
├── data/                          # Dataset folder (CSV files)
│   └── Medical_appointment_data.csv  # Raw dataset (download separately)
├── models/                        # Saved models (.joblib)
│   ├── best_classifier.joblib
│   ├── best_forecaster.joblib
│   ├── label_encoders.joblib
│   ├── scaler.joblib
│   └── ...
├── notebooks/
│   ├── 01_EDA.ipynb                          # Exploratory Data Analysis
│   ├── 02_Preprocessing_Feature_Engineering.ipynb  # Data cleaning & feature engineering
│   ├── 03_NoShow_Classification.ipynb        # Classification model training
│   └── 04_Demand_Forecasting.ipynb           # Time series forecasting
├── src/
│   ├── __init__.py
│   ├── config.py                  # Project configuration & constants
│   └── preprocessing.py           # Shared preprocessing utilities
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔧 Setup & Installation

### 1. Clone the repository
```bash
git clone <repo-url>
cd "medical appointment no show prediction and demand forecasting"
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
- Download the medical appointments dataset (109,593 rows × 26 columns)
- Place the CSV file in the `data/` folder as `Medical_appointment_data.csv`

### 5. Run the notebooks (in order)
```
notebooks/01_EDA.ipynb
notebooks/02_Preprocessing_Feature_Engineering.ipynb
notebooks/03_NoShow_Classification.ipynb
notebooks/04_Demand_Forecasting.ipynb
```

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| `no_show` | Target | Yes/No — whether patient missed appointment |
| `gender` | Categorical | Patient gender (Male/Female) |
| `age` | Numeric | Patient age (~21% missing) |
| `under_12` | Binary | Age < 12 flag |
| `over_60` | Binary | Age ≥ 60 flag |
| `disability` | Numeric | Disability level (~15% missing) |
| `needs_companion` | Binary | Patient needs companion |
| `specialty` | Categorical | Appointment specialty (~18% missing) |
| `appointment_time` | Numeric | Hour of appointment |
| `appointment_shift` | Categorical | Morning/Afternoon/Evening/Night |
| `appointment_date_continuous` | Date | Continuous calendar date (no gaps) |
| `place` | Categorical | City/location (~10.5% missing) |
| `Hipertension` | Binary | Has hypertension |
| `Diabetes` | Binary | Has diabetes |
| `Alcoholism` | Binary | Alcoholism flag |
| `Handcap` | Binary | Handicap flag |
| `Scholarship` | Binary | Bolsa Família scholarship |
| `SMSreceived` | Binary | Received SMS reminder |
| `avg_temp` | Numeric | Average temperature (°C) |
| `max_temp` | Numeric | Maximum temperature (°C) |
| `rain` | Numeric | Rainfall (mm) |
| `heat_intensity` | Categorical | Heat level category |
| `rain_intensity` | Categorical | Rain level category |
| `rainy_day_before` | Binary | Was the previous day rainy |
| `storm_day_before` | Binary | Was there a storm the day before |

---

## 🤖 Models

### Classification (No-Show Prediction)
| Model | Target |
|-------|--------|
| Logistic Regression | Baseline with class_weight='balanced' |
| Random Forest | GridSearchCV tuned, class weights |
| XGBoost | scale_pos_weight for imbalance |
| LightGBM | is_unbalance=True, extended grid search |

**Targets:** F1-Score > 0.70, ROC-AUC > 0.75  
**Achieved:** Best F1 ≈ 0.637 (LightGBM), ROC-AUC = 0.783 ✓  
> *Note: The F1 gap is a data limitation — no-shows are driven by unobserved factors (personal emergencies, transportation) not in the dataset. Threshold optimization confirms 0.50 is already optimal.*

### Regression (Demand Forecasting)
| Model | Approach |
|-------|----------|
| ARIMA | Auto-selected (p,d,q) via AIC |
| Random Forest Regressor | Log-target + lag features + GridSearchCV |
| Gradient Boosting | Log-target + lag features + GridSearchCV |
| Stacking Ensemble | RF + LightGBM + XGBoost → Ridge (log1p target) |
| LSTM | 14-day sequence window (TensorFlow/Keras) |

**Targets:** R² > 0.65, WMAPE < 50%  
**Achieved:** Best R² ≈ 0.687 ✓ (Stacking Ensemble), WMAPE ≈ 38% ✓  
> *Note: Traditional MAPE is inflated by ~114 low-volume days (<10 appointments — holidays/weekends). WMAPE (Weighted MAPE) is the industry-standard metric for demand forecasting with variable-scale data.*

---

## 🖥️ Streamlit App

The app has 4 pages:

| Page | Description |
|------|-------------|
| **Home** | Project overview and key metrics |
| **No-Show Predictor** | Input patient/appointment details → risk score |
| **Demand Forecaster** | Select date/weather → predicted daily volume |
| **Insights Dashboard** | Visualizations of patterns, model performance, geographic analysis |

### Run locally:
```bash
streamlit run app.py
```

---

## 📈 Key Skills Demonstrated

- Data Preprocessing & EDA
- Binary Classification with Class Imbalance (SMOTE, class weights)
- Time Series Forecasting (ARIMA, ML regressors, Stacking Ensemble, LSTM)
- Feature Engineering (temporal, interaction, health composite, log-transforms)
- Model Evaluation & Comparison (F1, AUC, R², WMAPE)
- Streamlit App Development
- Healthcare Analytics & Business Insights

---

## 🛠️ Tech Stack

Python • Pandas • NumPy • Scikit-learn • XGBoost • LightGBM • Statsmodels • TensorFlow/Keras • Imbalanced-learn • Matplotlib • Seaborn • Plotly • Streamlit • Joblib
