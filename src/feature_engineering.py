import pandas as pd

from .config import (
    PROCESSED_DATA_DIR,
    READMISSION_DATASET,
    ED_FORECAST_DATASET,
)


def build_lab_features(labs):
    labs = labs.copy()

    labs["is_abnormal"] = labs["flag"].fillna("Normal").ne("Normal").astype(int)

    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    labs["reference_low"] = pd.to_numeric(labs["reference_low"], errors="coerce")
    labs["reference_high"] = pd.to_numeric(labs["reference_high"], errors="coerce")

    labs["below_reference"] = (
        labs["value"].notna()
        & labs["reference_low"].notna()
        & (labs["value"] < labs["reference_low"])
    ).astype(int)

    labs["above_reference"] = (
        labs["value"].notna()
        & labs["reference_high"].notna()
        & (labs["value"] > labs["reference_high"])
    ).astype(int)

    lab_features = labs.groupby("admission_id").agg(
        total_labs=("lab_id", "count"),
        abnormal_labs=("is_abnormal", "sum"),
        below_reference_labs=("below_reference", "sum"),
        above_reference_labs=("above_reference", "sum"),
        unique_lab_tests=("test_name", "nunique"),
        avg_lab_value=("value", "mean"),
        max_lab_value=("value", "max"),
    ).reset_index()

    lab_features["abnormal_lab_rate"] = (
        lab_features["abnormal_labs"] / lab_features["total_labs"]
    )

    return lab_features


def build_medication_features(medications):
    medications = medications.copy()

    medications["is_high_alert"] = (
        pd.to_numeric(medications["is_high_alert"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    medication_features = medications.groupby("admission_id").agg(
        total_medications=("medication_id", "count"),
        unique_drug_classes=("drug_class", "nunique"),
        high_alert_medications=("is_high_alert", "sum"),
    ).reset_index()

    medication_features["polypharmacy_flag"] = (
        medication_features["total_medications"] >= 5
    ).astype(int)

    medication_features["high_alert_medication_rate"] = (
        medication_features["high_alert_medications"]
        / medication_features["total_medications"]
    )

    return medication_features


def build_diagnosis_features(diagnoses):
    diagnoses = diagnoses.copy()

    diagnosis_features = diagnoses.groupby("admission_id").agg(
        diagnosis_count=("diagnosis_id", "count")
    ).reset_index()

    return diagnosis_features


def build_ed_patient_features(ed_visits):
    ed_visits = ed_visits.copy()

    if "patient_id" not in ed_visits.columns:
        return pd.DataFrame()

    numeric_ed_cols = [
        "wait_time_minutes",
        "door_to_doctor_min",
        "ed_los_minutes",
    ]

    for col in numeric_ed_cols:
        if col in ed_visits.columns:
            ed_visits[col] = pd.to_numeric(ed_visits[col], errors="coerce")

    agg_dict = {
        "num_ed_visits": ("patient_id", "count"),
    }

    if "wait_time_minutes" in ed_visits.columns:
        agg_dict["avg_wait_time"] = ("wait_time_minutes", "mean")

    if "door_to_doctor_min" in ed_visits.columns:
        agg_dict["avg_door_to_doctor"] = ("door_to_doctor_min", "mean")

    if "ed_los_minutes" in ed_visits.columns:
        agg_dict["avg_ed_los"] = ("ed_los_minutes", "mean")

    ed_features = (
        ed_visits
        .groupby("patient_id")
        .agg(**agg_dict)
        .reset_index()
    )

    return ed_features


def add_risk_features(dataset):
    dataset = dataset.copy()

    if "age" in dataset.columns:
        dataset["age"] = pd.to_numeric(dataset["age"], errors="coerce")
        dataset["older_adult_flag"] = (dataset["age"] >= 65).astype(int)

    if "length_of_stay_days" in dataset.columns:
        dataset["long_stay_flag"] = (
            dataset["length_of_stay_days"] >= 7
        ).astype(int)

    if "abnormal_lab_rate" in dataset.columns:
        dataset["high_abnormal_lab_rate"] = (
            dataset["abnormal_lab_rate"] >= 0.30
        ).astype(int)

        dataset["critical_lab_flag"] = (
            dataset["abnormal_lab_rate"] >= 0.50
        ).astype(int)

    if "total_medications" in dataset.columns:
        dataset["high_medication_burden"] = (
            dataset["total_medications"] >= 5
        ).astype(int)

    if "diagnosis_count" in dataset.columns:
        dataset["multiple_diagnoses_flag"] = (
            dataset["diagnosis_count"] >= 3
        ).astype(int)

    if "num_prior_admissions" in dataset.columns:
        dataset["frequent_admitter"] = (
            dataset["num_prior_admissions"] >= 2
        ).astype(int)

    if "num_prior_ed_visits" in dataset.columns:
        dataset["frequent_ed_user"] = (
            dataset["num_prior_ed_visits"] >= 2
        ).astype(int)

    if "num_ed_visits" in dataset.columns:
        dataset["high_ed_usage_flag"] = (
            dataset["num_ed_visits"] >= 2
        ).astype(int)

    if "avg_wait_time" in dataset.columns:
        dataset["high_wait_time_flag"] = (
            dataset["avg_wait_time"] >= 120
        ).astype(int)

    if "avg_ed_los" in dataset.columns:
        dataset["long_ed_los_flag"] = (
            dataset["avg_ed_los"] >= 240
        ).astype(int)

    if "charlson_comorbidity_index" in dataset.columns:
        dataset["high_comorbidity_flag"] = (
            dataset["charlson_comorbidity_index"] >= 3
        ).astype(int)

    if {"diagnosis_count", "total_medications"}.issubset(dataset.columns):
        dataset["complex_patient_flag"] = (
            (dataset["diagnosis_count"] >= 3)
            | (dataset["total_medications"] >= 5)
        ).astype(int)

    if {
        "abnormal_labs",
        "total_medications",
        "diagnosis_count",
    }.issubset(dataset.columns):
        dataset["overall_clinical_risk_score"] = (
            dataset["abnormal_labs"]
            + dataset["total_medications"]
            + dataset["diagnosis_count"]
        )

    if "admission_type" in dataset.columns:
        dataset["emergency_admission_flag"] = (
            dataset["admission_type"]
            .astype(str)
            .str.lower()
            .str.contains("emergency", na=False)
            .astype(int)
        )

    if "discharge_disposition" in dataset.columns:
        dataset["non_home_discharge_flag"] = (
            ~dataset["discharge_disposition"]
            .astype(str)
            .str.lower()
            .str.contains("home", na=False)
        ).astype(int)

    return dataset


def build_readmission_dataset():
    patients = pd.read_csv(PROCESSED_DATA_DIR / "patients_clean.csv")
    admissions = pd.read_csv(PROCESSED_DATA_DIR / "admissions_clean.csv")
    labs = pd.read_csv(PROCESSED_DATA_DIR / "lab_results_clean.csv")
    medications = pd.read_csv(PROCESSED_DATA_DIR / "medications_clean.csv")
    diagnoses = pd.read_csv(PROCESSED_DATA_DIR / "diagnoses_clean.csv")
    ed_visits = pd.read_csv(PROCESSED_DATA_DIR / "ed_visits_clean.csv")

    admissions["admission_date"] = pd.to_datetime(
        admissions["admission_date"],
        errors="coerce",
    )

    admissions["discharge_date"] = pd.to_datetime(
        admissions["discharge_date"],
        errors="coerce",
    )

    admissions["length_of_stay_days"] = (
        admissions["discharge_date"] - admissions["admission_date"]
    ).dt.days

    admissions["length_of_stay_days"] = (
        admissions["length_of_stay_days"]
        .clip(lower=0)
        .fillna(0)
    )

    dataset = admissions.merge(
        patients,
        on="patient_id",
        how="left",
    )

    lab_features = build_lab_features(labs)
    medication_features = build_medication_features(medications)
    diagnosis_features = build_diagnosis_features(diagnoses)
    ed_patient_features = build_ed_patient_features(ed_visits)

    dataset = dataset.merge(lab_features, on="admission_id", how="left")
    dataset = dataset.merge(medication_features, on="admission_id", how="left")
    dataset = dataset.merge(diagnosis_features, on="admission_id", how="left")

    if not ed_patient_features.empty:
        dataset = dataset.merge(ed_patient_features, on="patient_id", how="left")

    if "readmitted_within_30d" not in dataset.columns:
        raise ValueError(
            "Target column 'readmitted_within_30d' not found in admissions_clean.csv"
        )

    dataset["readmitted_within_30d"] = (
        pd.to_numeric(dataset["readmitted_within_30d"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "readmission_reason" in dataset.columns:
        dataset["readmission_reason"] = dataset["readmission_reason"].fillna(
            "No Readmission"
        )

    dataset = add_risk_features(dataset)

    columns_to_remove = [
        "mrn",
        "first_name",
        "last_name",
        "date_of_birth",
        "registered_date",
        "zip_code",
    ]

    dataset = dataset.drop(
        columns=[col for col in columns_to_remove if col in dataset.columns]
    )

    numeric_cols = dataset.select_dtypes(include=["number"]).columns
    dataset[numeric_cols] = dataset[numeric_cols].fillna(0)

    categorical_cols = dataset.select_dtypes(include=["object"]).columns
    dataset[categorical_cols] = dataset[categorical_cols].fillna("Unknown")

    dataset.to_csv(READMISSION_DATASET, index=False)

    print("Readmission feature dataset created.")
    print(f"Saved to: {READMISSION_DATASET}")
    print(f"Shape: {dataset.shape}")


def build_ed_forecast_dataset():
    ed_visits = pd.read_csv(PROCESSED_DATA_DIR / "ed_visits_clean.csv")

    ed_visits["arrival_datetime"] = pd.to_datetime(
        ed_visits["arrival_datetime"],
        errors="coerce",
    )

    if "hour_of_arrival" in ed_visits.columns:
        ed_visits = ed_visits.drop(columns=["hour_of_arrival"])

    numeric_ed_cols = [
        "wait_time_minutes",
        "door_to_doctor_min",
        "ed_los_minutes",
        "triage_level",
    ]

    for col in numeric_ed_cols:
        if col in ed_visits.columns:
            ed_visits[col] = pd.to_numeric(ed_visits[col], errors="coerce")

    if "delayed_decision" in ed_visits.columns:
        ed_visits["delayed_decision"] = (
            pd.to_numeric(ed_visits["delayed_decision"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    ed_visits = ed_visits.dropna(subset=["arrival_datetime"])

    ed_visits["week_start"] = (
        ed_visits["arrival_datetime"]
        .dt.to_period("W")
        .apply(lambda x: x.start_time)
    )

    ed_visits["month"] = ed_visits["week_start"].dt.month
    ed_visits["quarter"] = ed_visits["week_start"].dt.quarter
    ed_visits["week_of_year"] = (
        ed_visits["week_start"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    ed_visits["is_winter"] = ed_visits["month"].isin([12, 1, 2]).astype(int)
    ed_visits["is_summer"] = ed_visits["month"].isin([6, 7, 8]).astype(int)

    agg_dict = {
        "ed_arrivals": ("ed_visit_id", "count"),
        "is_winter": ("is_winter", "max"),
        "is_summer": ("is_summer", "max"),
    }

    if "wait_time_minutes" in ed_visits.columns:
        agg_dict["avg_wait_time"] = ("wait_time_minutes", "mean")

    if "door_to_doctor_min" in ed_visits.columns:
        agg_dict["avg_door_to_doctor"] = ("door_to_doctor_min", "mean")

    if "ed_los_minutes" in ed_visits.columns:
        agg_dict["avg_ed_los"] = ("ed_los_minutes", "mean")

    if "triage_level" in ed_visits.columns:
        agg_dict["avg_triage_level"] = ("triage_level", "mean")

    if "delayed_decision" in ed_visits.columns:
        agg_dict["delay_decision_rate"] = ("delayed_decision", "mean")

    ed_weekly = (
        ed_visits
        .groupby(["week_start", "month", "quarter", "week_of_year"])
        .agg(**agg_dict)
        .reset_index()
    )

    ed_weekly = ed_weekly.sort_values("week_start")

    ed_weekly["lag_1_week"] = ed_weekly["ed_arrivals"].shift(1)
    ed_weekly["lag_2_week"] = ed_weekly["ed_arrivals"].shift(2)
    ed_weekly["lag_4_week"] = ed_weekly["ed_arrivals"].shift(4)
    ed_weekly["lag_8_week"] = ed_weekly["ed_arrivals"].shift(8)

    ed_weekly["rolling_4_week_avg"] = (
        ed_weekly["ed_arrivals"].shift(1).rolling(4).mean()
    )

    ed_weekly["rolling_8_week_avg"] = (
        ed_weekly["ed_arrivals"].shift(1).rolling(8).mean()
    )

    ed_weekly["rolling_12_week_avg"] = (
        ed_weekly["ed_arrivals"].shift(1).rolling(12).mean()
    )

    ed_weekly = ed_weekly.fillna(0)

    ed_weekly.to_csv(ED_FORECAST_DATASET, index=False)

    print("Weekly ED forecast dataset created.")
    print(f"Saved to: {ED_FORECAST_DATASET}")
    print(f"Shape: {ed_weekly.shape}")


def main():
    build_readmission_dataset()
    build_ed_forecast_dataset()


if __name__ == "__main__":
    main()