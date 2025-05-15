# predict_random.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib
import time

def test_random(df, sample_size):
    # Load saved model and encoder
    clf = joblib.load(f"../lostless_data/models/rf_model_{sample_size}.joblib")
    encoder = joblib.load(f"../lostless_data/encoders/encoder_{sample_size}.joblib")
  
    # Select random row
    row = df.sample(n=1, random_state=1).copy()
    
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

def test_row(row, sample_size):
    # Load saved model and encoder
    clf = joblib.load(f"../lostless_data/models/rf_model_{sample_size}.joblib")
    encoder = joblib.load(f"../lostless_data/encoders/encoder_{sample_size}.joblib")
  
    # Select random row
    #row = df.sample(n=1, random_state=1).copy()
    
    # Downcast types
    for col in row.select_dtypes(include=["int"]).columns:
        row[col] = pd.to_numeric(row[col], downcast="integer")
    for col in row.select_dtypes(include=["float"]).columns:
        row[col] = pd.to_numeric(row[col], downcast="float")

    # Prepare input
    X = row #row.drop(columns=["ASISTIDA"])
    #y_true = row["ASISTIDA"].values[0]
    encoded_cat = encoder.transform(X.select_dtypes(include=["object"]))
    numerical = X.select_dtypes(include=["number"]).to_numpy()
    X_input = hstack([numerical, encoded_cat])

    # Predict
    y_pred = clf.predict(X_input)
    labels = f"Predicted: {y_pred[0]}"
    #result = "Correct" if y_true == y_pred[0] else "Incorrect"
    return labels


def main():
    # Load dataset
    df = pd.read_csv("../lostless_data/data/data.csv")
    df = df.dropna().dropna(axis=1)
    start = time.time()
    labels, result = test_random(df, 100000)
    print(labels, result)
    stop = time.time()
    print(f"Testing time: {stop - start:.2f} seconds")


if __name__ == "__main__":
    main()
