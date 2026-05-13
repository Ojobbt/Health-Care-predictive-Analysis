from . import data_cleaning
from . import feature_engineering
from . import train_readmission_model
from . import train_ed_forecasting_model
from . import evaluate


def main():
    print("Step 1: Cleaning data...")
    data_cleaning.main()

    print("Step 2: Creating features...")
    feature_engineering.main()

    print("Step 3: Training readmission model...")
    train_readmission_model.train_readmission_model()

    print("Step 4: Training ED forecast model...")
    train_ed_forecasting_model.train_ed_forecast_model()

    print("Step 5: Evaluating models...")
    evaluate.evaluate_readmission_model()
    evaluate.evaluate_ed_forecast_model()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()