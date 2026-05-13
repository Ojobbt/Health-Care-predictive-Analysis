import pandas as pd

from .config import (
    PATIENTS_FILE,
    ADMISSIONS_FILE,
    READMISSIONS_FILE,
    LABS_FILE,
    MEDICATIONS_FILE,
    DIAGNOSES_FILE,
    ED_VISITS_FILE,
    PROCESSED_DATA_DIR,
)


def load_data():
    patients = pd.read_csv(PATIENTS_FILE)
    admissions = pd.read_csv(ADMISSIONS_FILE)
    readmissions = pd.read_csv(READMISSIONS_FILE)
    labs = pd.read_csv(LABS_FILE)
    medications = pd.read_csv(MEDICATIONS_FILE)
    diagnoses = pd.read_csv(DIAGNOSES_FILE)
    ed_visits = pd.read_csv(ED_VISITS_FILE)

    return patients, admissions, readmissions, labs, medications, diagnoses, ed_visits


def clean_dates(df):
    df = df.copy()

    for col in df.columns:
        col_lower = col.lower()

        is_date_column = (
            "date" in col_lower
            or "datetime" in col_lower
            or col_lower.endswith("_at")
        )

        is_duration_column = (
            "minutes" in col_lower
            or "_min" in col_lower
            or "los" in col_lower
            or "duration" in col_lower
            or "wait_time" in col_lower
        )

        if is_date_column and not is_duration_column:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def fix_duration_columns(df):
    df = df.copy()

    duration_cols = [
        "wait_time_minutes",
        "door_to_doctor_min",
        "ed_los_minutes",
    ]

    for col in duration_cols:
        if col in df.columns:
            if df[col].astype(str).str.contains("1970-01-01", na=False).any():
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.extract(r"\.(\d+)$")[0]
                )

            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_dataframe(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = clean_dates(df)

    return df


def clean_ed_visits(ed_visits):
    ed_visits = ed_visits.copy()

    ed_visits["arrival_datetime"] = pd.to_datetime(
        ed_visits["arrival_datetime"],
        errors="coerce"
    )

    ed_visits["departure_datetime"] = pd.to_datetime(
        ed_visits["departure_datetime"],
        errors="coerce"
    )

    # Remove unreliable hour_of_arrival column if it exists
    if "hour_of_arrival" in ed_visits.columns:
        ed_visits = ed_visits.drop(columns=["hour_of_arrival"])

    # Keep reliable date-based features only
    ed_visits["arrival_date"] = ed_visits["arrival_datetime"].dt.date
    ed_visits["day_of_week"] = ed_visits["arrival_datetime"].dt.day_name()
    ed_visits["month"] = ed_visits["arrival_datetime"].dt.month
    ed_visits["year"] = ed_visits["arrival_datetime"].dt.year

    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }

    ed_visits["season"] = ed_visits["month"].map(season_map)

    return ed_visits


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    patients, admissions, readmissions, labs, medications, diagnoses, ed_visits = load_data()

    patients = clean_dataframe(patients)
    admissions = clean_dataframe(admissions)
    readmissions = clean_dataframe(readmissions)
    labs = clean_dataframe(labs)
    medications = clean_dataframe(medications)
    diagnoses = clean_dataframe(diagnoses)

    ed_visits = clean_dataframe(ed_visits)
    ed_visits = clean_ed_visits(ed_visits)

    patients.to_csv(PROCESSED_DATA_DIR / "patients_clean.csv", index=False)
    admissions.to_csv(PROCESSED_DATA_DIR / "admissions_clean.csv", index=False)
    readmissions.to_csv(PROCESSED_DATA_DIR / "readmissions_clean.csv", index=False)
    labs.to_csv(PROCESSED_DATA_DIR / "lab_results_clean.csv", index=False)
    medications.to_csv(PROCESSED_DATA_DIR / "medications_clean.csv", index=False)
    diagnoses.to_csv(PROCESSED_DATA_DIR / "diagnoses_clean.csv", index=False)
    ed_visits.to_csv(PROCESSED_DATA_DIR / "ed_visits_clean.csv", index=False)

    print("Data cleaning completed.")
    print(f"Clean files saved to: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()