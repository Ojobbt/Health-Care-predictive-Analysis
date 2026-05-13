from pathlib import Path

# Project root folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Main folders
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# Create folders if they do not exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Raw data files
PATIENTS_FILE = RAW_DATA_DIR / "patients.csv"
ADMISSIONS_FILE = RAW_DATA_DIR / "admissions.csv"
READMISSIONS_FILE = RAW_DATA_DIR / "readmissions.csv"
LABS_FILE = RAW_DATA_DIR / "lab_results.csv"
MEDICATIONS_FILE = RAW_DATA_DIR / "medications.csv"
DIAGNOSES_FILE = RAW_DATA_DIR / "diagnoses.csv"
ED_VISITS_FILE = RAW_DATA_DIR / "ed_visits.csv"
DATA_DICTIONARY_FILE = RAW_DATA_DIR / "data_dictionary.csv"


# Processed datasets
READMISSION_DATASET = PROCESSED_DATA_DIR / "readmission_dataset.csv"
ED_FORECAST_DATASET = PROCESSED_DATA_DIR / "ed_forecast_dataset.csv"
HIGH_RISK_PATIENT_DATASET = PROCESSED_DATA_DIR / "high_risk_patient_investigation.csv"
DELAYED_DECISION_DATASET = PROCESSED_DATA_DIR / "delayed_decision_investigation.csv"

# Saved models
READMISSION_MODEL = MODEL_DIR / "readmission_model.pkl"
ED_FORECAST_MODEL = MODEL_DIR / "ed_forecast_model.pkl"
HIGH_RISK_PATIENT_MODEL = MODEL_DIR / "high_risk_patient_model.pkl"
DELAYED_DECISION_MODEL = MODEL_DIR / "delayed_decision_model.pkl"