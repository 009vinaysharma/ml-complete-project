import joblib
import pandas as pd

MODEL_PATH = "model/churn_model.pkl"


def predict_churn(model, input_data):
    if model is None:
        model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    result = {
        "predicted_class": int(prediction),
        "predicted_label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": float(probability[1]),
        "non_churn_probability": float(probability[0]),
    }

    return result
