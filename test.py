# predict_random.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib

def main():
    # Load full dataset
    df = pd.read_csv("../data.csv")
    df = df.drop("ESTADO", axis=1)
    df = df.dropna().dropna(axis=1)

    # Load saved model and encoder
    clf = joblib.load("../rf_model.joblib")
    encoder = joblib.load("../encoder.joblib")

    # Select random row
    row = df.sample(n=1, random_state=1).copy()
    print(row)

    # Downcast types
    for col in row.select_dtypes(include=["int"]).columns:
        row[col] = pd.to_numeric(row[col], downcast="integer")
    for col in row.select_dtypes(include=["float"]).columns:
        row[col] = pd.to_numeric(row[col], downcast="float")

    # Prepare input
    X = row.drop(columns=["ASISTIDA"])
    y_true = row["ASISTIDA"].values[0]
    encoded_cat = encoder.transform(X.select_dtypes(include=["object"]))
    numerical = X.select_dtypes(include=["number"]).to_numpy()
    X_input = hstack([numerical, encoded_cat])

    # Predict
    y_pred = clf.predict(X_input)
    print(f"Label: {y_true}, Predicted: {y_pred[0]}")
    result = "Correct" if y_true == y_pred[0] else "Incorrect"
    print(result)

if __name__ == "__main__":
    main()