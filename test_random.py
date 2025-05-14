# predict_random.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib
import time

def test_random(df, sample_size):
    df = df.dropna().dropna(axis=1)
    # Load saved model and encoder
    clf = joblib.load(f"../models/lostless/rf_model_{sample_size}.joblib")
    encoder = joblib.load(f"../models/lostless/encoder_{sample_size}.joblib")

    dfi = pd.read_csv(f"../data/lostless/y_test_{sample_size}.csv")
    indices = dfi["index"].tolist()
    dfTest = df.iloc[indices]

    # Select random row
    row = dfTest.sample(n=1, random_state=1).copy()
    
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
    labels = f"Label: {y_true}, Predicted: {y_pred[0]}"
    result = "Correct" if y_true == y_pred[0] else "Incorrect"
    return labels, result


def main():
    # Load dataset
    df = pd.read_csv("../data/lostless/data.csv")
    start = time.time()
    labels, result = test_random(df, 100000)
    print(labels, result)
    stop = time.time()
    print(f"Testing time: {stop - start:.2f} seconds")


if __name__ == "__main__":
    main()
