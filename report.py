# train_model.py
import time

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def create_report(df, sample_size):
    # Prepare the dataset
    if sample_size != -1:
        df = df.sample(n=sample_size, random_state=42)
    else:
        sample_size = df.shape[0]

    # Load the saved model and encoder
    model_path = f"../lostless_data/models/rf_model_{sample_size}.joblib"
    encoder_path = f"../lostless_data/encoders/encoder_{sample_size}.joblib"
    clf = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    X = df.drop(columns=["ASISTIDA"])
    y = df["ASISTIDA"]

    # One-hot encode categorical
    encoded_cat = encoder.transform(X.select_dtypes(include=["object"]))
    numerical = X.select_dtypes(include=["number"]).to_numpy()
    X_combined = hstack([numerical, encoded_cat])

    # Use SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_combined, y)

    # Train-test split
    _, X_test, _, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )

    # Generate predictions and the report
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)

    # Save the report
    report_path = f"../lostless_data/reports/classification_report_{sample_size}.txt"
    with open(report_path, "w") as file:
        file.write(report)

    print(f"Classification report saved to {report_path}")
    return report


def main():
    # Load dataset
    df = pd.read_csv("../lostless_data/data/data.csv")
    df = df.dropna().dropna(axis=1)
    start = time.time()
    report = create_report(df, -1)
    print(report)
    stop = time.time()
    print(f"Training time: {stop - start:.2f} seconds")


if __name__ == "__main__":
    main()
