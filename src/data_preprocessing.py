import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "churn_data.csv")


def generate_synthetic_churn_data(path):
    np.random.seed(42)
    n_samples = 1000

    tenure = np.random.randint(1, 72, n_samples)
    monthly_charges = np.random.uniform(30, 120, n_samples)
    total_charges = monthly_charges * np.random.uniform(1.5, 18.0, n_samples)
    contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], n_samples)
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n_samples)
    payment_method = np.random.choice([
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card",
    ], n_samples)
    gender = np.random.choice(["Male", "Female"], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    partner = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    dependents = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    churn_prob = (
        0.2 * (tenure < 12)
        + 0.15 * (monthly_charges > 80)
        + 0.1 * (internet_service == "Fiber optic")
        + 0.12 * (payment_method == "Electronic check")
        + 0.08 * (gender == "Female")
        + 0.1 * (senior_citizen == 1)
        + 0.05 * (partner == 0)
        + 0.08 * (dependents == 0)
    )
    churn_prob = np.clip(churn_prob / 1.2, 0.05, 0.95)
    churn = np.random.binomial(1, churn_prob, n_samples)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "churn": churn,
        }
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)


def load_and_preprocess_data():
    if not os.path.exists(DATA_PATH):
        generate_synthetic_churn_data(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    numerical_features = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "senior_citizen",
        "partner",
        "dependents",
    ]
    categorical_features = [
        "contract_type",
        "internet_service",
        "payment_method",
        "gender",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numerical_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    return preprocessor, X_train, X_test, y_train, y_test
