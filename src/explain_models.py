import pandas as pd
import joblib
import matplotlib.pyplot as plt

from .config import (
    READMISSION_DATASET,
    ED_FORECAST_DATASET,
    READMISSION_MODEL,
    ED_FORECAST_MODEL,
)


def explain_readmission_model():
    print("\nREADMISSION MODEL FEATURE IMPORTANCE")
    print("-----------------------------------")

    df = pd.read_csv(READMISSION_DATASET)

    drop_cols = [
        "readmitted_within_30d",
        "readmission_reason",
        "patient_id",
        "admission_id",
        "mrn",
        "first_name",
        "last_name",
        "admission_date",
        "discharge_date",
        "length_of_stay_days",
        "total_cost_usd",
        "insurance_paid_usd",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    pipeline = joblib.load(READMISSION_MODEL)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(importance_df.head(15))

    importance_df.head(15).plot(
        x="feature",
        y="importance",
        kind="barh",
        figsize=(10, 6)
    )

    plt.gca().invert_yaxis()
    plt.title("Top Readmission Features")
    plt.tight_layout()
    plt.show()


def explain_ed_forecast_model():
    print("\nED FORECAST MODEL FEATURE IMPORTANCE")
    print("-----------------------------------")

    df = pd.read_csv(ED_FORECAST_DATASET)

    drop_cols = ["week_start", "ed_arrivals"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    pipeline = joblib.load(ED_FORECAST_MODEL)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(importance_df.head(15))

    importance_df.head(15).plot(
        x="feature",
        y="importance",
        kind="barh",
        figsize=(10, 6)
    )

    plt.gca().invert_yaxis()
    plt.title("Top ED Forecast Features")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    explain_readmission_model()
    explain_ed_forecast_model()