import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from .config import READMISSION_DATASET, READMISSION_MODEL


def evaluate_thresholds(y_test, probs):
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    results = []

    print("\nThreshold comparison:")

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)

        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f}"
        )

    return pd.DataFrame(results)


def train_readmission_model():
    print("Loading readmission dataset...")

    df = pd.read_csv(READMISSION_DATASET)

    target = "readmitted_within_30d"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

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
    y = df[target]

    datetime_cols = [
        col for col in X.columns
        if (
            "date" in col.lower()
            or "datetime" in col.lower()
        )
    ]

    X = X.drop(columns=datetime_cols, errors="ignore")

    bool_cols = X.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"Rows: {df.shape[0]}")
    print(f"Features used: {X.shape[1]}")
    print(f"Positive readmissions: {y.sum()}")
    print(f"Negative cases: {(y == 0).sum()}")

    if y.nunique() < 2:
        raise ValueError(
            "The target column has only one class. "
            "The model needs both 0 and 1 values in readmitted_within_30d."
        )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()

    if positives == 0:
        scale_pos_weight = 1
    else:
        scale_pos_weight = negatives / positives

    models = {
        "logistic_regression": {
            "model": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
            "params": {},
        },
        "random_forest": {
            "model": RandomForestClassifier(
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
            "params": {
                "model__n_estimators": [300, 500, 700],
                "model__max_depth": [6, 8, 10, 12, 15, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": ["sqrt", "log2"],
            },
        },
        "xgboost": {
            "model": XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "model__n_estimators": [300, 500, 700],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.8, 0.9],
                "model__colsample_bytree": [0.7, 0.8, 0.9],
                "model__min_child_weight": [1, 3, 5],
                "model__gamma": [0, 0.1, 0.3],
            },
        },
    }

    mlflow.set_experiment("readmission_prediction_tuned")

    best_auc = -1
    best_model_name = None
    best_pipeline = None
    best_threshold = 0.40

    for model_name, model_info in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model_info["model"]),
            ]
        )

        if model_info["params"]:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=model_info["params"],
                n_iter=20,
                scoring="roc_auc",
                cv=5,
                random_state=42,
                n_jobs=-1,
                verbose=1,
            )

            search.fit(X_train, y_train)
            final_pipeline = search.best_estimator_

            print(f"Best parameters for {model_name}:")
            print(search.best_params_)

        else:
            final_pipeline = pipeline
            final_pipeline.fit(X_train, y_train)

        probs = final_pipeline.predict_proba(X_test)[:, 1]

        threshold_results = evaluate_thresholds(y_test, probs)
        selected_threshold = threshold_results.sort_values(
            by="f1",
            ascending=False
        ).iloc[0]["threshold"]

        preds = (probs >= selected_threshold).astype(int)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("selected_threshold", selected_threshold)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("pr_auc", pr_auc)

            mlflow.sklearn.log_model(
                final_pipeline,
                artifact_path=f"{model_name}_model",
            )

        print(f"\n{model_name} results:")
        print(f"Selected threshold: {selected_threshold:.2f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")
        print(f"ROC-AUC: {auc:.3f}")
        print(f"PR-AUC: {pr_auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name
            best_pipeline = final_pipeline
            best_threshold = selected_threshold

    model_package = {
        "model": best_pipeline,
        "threshold": float(best_threshold),
        "model_name": best_model_name,
        "roc_auc": float(best_auc),
    }

    joblib.dump(model_package, READMISSION_MODEL)

    print("\nTraining completed.")
    print(f"Best model: {best_model_name}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best ROC-AUC: {best_auc:.3f}")
    print(f"Saved to: {READMISSION_MODEL}")


if __name__ == "__main__":
    train_readmission_model()