# train_model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
import time


def train(df, sample_size):
    df = df.dropna().dropna(axis=1)
    # Sample for training
    if sample_size != -1:
        # Sample a subset of the data
        df = df.sample(n=sample_size, random_state=42)

    # Downcast numerical types
    for col in df.select_dtypes(include=["int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")

    # Split features and labels
    X = df.drop(columns=["ASISTIDA"])
    y = df["ASISTIDA"]

    # One-hot encode categorical
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    encoded_cat = encoder.fit_transform(X.select_dtypes(include=["object"]))

    # Include numerical features
    numerical = X.select_dtypes(include=["number"]).to_numpy()
    X_combined = hstack([numerical, encoded_cat])

    # Use SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_combined, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )

    y_test_df = y_test.reset_index()
    y_test_df.columns = ["index", "y_test"]
    y_test_df.to_csv(f"../data/lostless/y_test_{sample_size}.csv", index=False)

    # Train and evaluate model
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    # Save model and encoder
    joblib.dump(clf, f"../models/lostless/rf_model_{sample_size}.joblib")
    joblib.dump(encoder, f"../models/lostless/encoder_{sample_size}.joblib")
    return(report)


def main():
    # Load dataset
    df = pd.read_csv("../data/lostless/data.csv")
    start = time.time()
    report = train(df, 300000)
    print(report)
    stop = time.time()
    print(f"Training time: {stop - start:.2f} seconds")


if __name__ == "__main__":
    main()
