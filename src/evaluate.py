import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .config import (
    READMISSION_DATASET,
    ED_FORECAST_DATASET,
    READMISSION_MODEL,
    ED_FORECAST_MODEL,
)


def evaluate_readmission_model():
    df = pd.read_csv(READMISSION_DATASET)
    model_package = joblib.load(READMISSION_MODEL)

    if isinstance(model_package, dict):
        model = model_package["model"]
        threshold = model_package.get("threshold", 0.50)
    else:
        model = model_package
        threshold = 0.50

    target = "readmitted_within_30d"

    drop_cols = [
        "readmitted_within_30d",
        "readmission_reason",

        "patient_id",
        "admission_id",
        "lab_id",
        "medication_id",
        "diagnosis_id",

        "mrn",
        "first_name",
        "last_name",

        "admission_date",
        "discharge_date",
        "registered_date",
        "date_of_birth",

        "total_cost_usd",
        "insurance_paid_usd",

        "zip_code",
    ]

    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target].astype(int)

    datetime_cols = [
        col for col in X.columns
        if "date" in col.lower() or "datetime" in col.lower()
    ]

    X = X.drop(columns=datetime_cols, errors="ignore")

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    print("\nReadmission Model Evaluation")
    print("----------------------------")
    print(f"Threshold: {threshold}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")


def evaluate_ed_forecast_model():
    df = pd.read_csv(ED_FORECAST_DATASET)
    model = joblib.load(ED_FORECAST_MODEL)

    target = "ed_arrivals"

    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["week_start"])
    df = df.sort_values("week_start")

    drop_cols = ["week_start", target]

    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target]

    split_index = int(len(df) * 0.8)

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("\nED Forecast Model Evaluation")
    print("----------------------------")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")


if __name__ == "__main__":
    evaluate_readmission_model()
    evaluate_ed_forecast_model()