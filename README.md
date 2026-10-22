# Customer Churn Prediction

This project builds a machine learning model to predict whether a customer will churn (leave the service).

## Features
- Synthetic dataset generation for a churn use case
- Data preprocessing
- Model training
- Model evaluation
- Prediction on new records
- Model saving for reuse

## Folder structure
- `main.py` – main workflow script
- `src/data_preprocessing.py` – dataset loading and preprocessing logic
- `src/train_model.py` – training and evaluation logic
- `src/predict.py` – prediction logic for new users

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python main.py
```

## Output
The script will:
- generate a sample churn dataset
- train a model
- print evaluation metrics
- save the trained model to `model/churn_model.pkl`
- make a prediction on sample inputs
