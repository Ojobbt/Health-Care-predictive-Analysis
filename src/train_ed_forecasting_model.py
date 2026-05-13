import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from .config import ED_FORECAST_DATASET, ED_FORECAST_MODEL


def train_ed_forecast_model():
    print("Loading ED forecast dataset...")

    df = pd.read_csv(ED_FORECAST_DATASET)

    target = "ed_arrivals"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["week_start"])
    df = df.sort_values("week_start")

    drop_cols = ["week_start", target]

    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = pd.to_numeric(df[target], errors="coerce").fillna(0)

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

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

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }

    split_index = int(len(df) * 0.8)

    if split_index == 0 or split_index == len(df):
        raise ValueError(
            "Not enough ED weekly rows to split into train and test sets."
        )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    mlflow.set_experiment("ed_demand_forecasting")

    best_r2 = float("-inf")
    best_model_name = None
    best_pipeline = None

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, preds)

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(
                pipeline,
                name=f"{model_name}_ed_forecast_model",
        )

            print(f"{model_name} trained.")
            print(f"MAE: {mae:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"R2: {r2:.3f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
                best_pipeline = pipeline

    joblib.dump(best_pipeline, ED_FORECAST_MODEL)

    print("\nBest ED forecast model saved.")
    print(f"Best model: {best_model_name}")
    print(f"Best R2: {best_r2:.3f}")
    print(f"Saved to: {ED_FORECAST_MODEL}")


if __name__ == "__main__":
    train_ed_forecast_model()