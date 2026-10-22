from src.data_preprocessing import load_and_preprocess_data
from src.train_model import train_and_evaluate_model
from src.predict import predict_churn


def main():
    print("Loading and preprocessing data...")
    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("Training model...")
    model = train_and_evaluate_model(preprocessor, X_train, X_test, y_train, y_test)

    print("Running prediction example...")
    sample = {
        "tenure": 12,
        "monthly_charges": 90.5,
        "total_charges": 1086.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": 1,
        "dependents": 0,
    }

    prediction = predict_churn(model, sample)
    print("Prediction result:", prediction)


if __name__ == "__main__":
    main()
