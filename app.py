from flask import Flask, jsonify, render_template, request

from src.data_preprocessing import load_and_preprocess_data
from src.predict import predict_churn
from src.train_model import train_and_evaluate_model

app = Flask(__name__)
model = None


def initialize_model():
    global model
    if model is None:
        preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess_data()
        model = train_and_evaluate_model(preprocessor, X_train, X_test, y_train, y_test)
    return model


initialize_model()


@app.route("/")
def home():
    return render_template(
        "index.html",
        default={
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
        },
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    data = {
        "tenure": int(request.form.get("tenure", 12)),
        "monthly_charges": float(request.form.get("monthly_charges", 90.5)),
        "total_charges": float(request.form.get("total_charges", 1086.0)),
        "contract_type": request.form.get("contract_type", "Month-to-month"),
        "internet_service": request.form.get("internet_service", "Fiber optic"),
        "payment_method": request.form.get("payment_method", "Electronic check"),
        "gender": request.form.get("gender", "Female"),
        "senior_citizen": int(request.form.get("senior_citizen", 0)),
        "partner": int(request.form.get("partner", 1)),
        "dependents": int(request.form.get("dependents", 0)),
    }

    result = predict_churn(model, data)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
