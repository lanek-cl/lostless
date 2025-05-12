# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib

# Load dataset
df = pd.read_csv("../data.csv")
df = df.dropna().dropna(axis=1)

# Sample for training
df = df.sample(n=100000, random_state=42)

# Downcast numerical types
for col in df.select_dtypes(include=["int"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")
for col in df.select_dtypes(include=["float"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

# Split features and labels
X = df.drop(columns=["ASISTIDA"])
y = df["ASISTIDA"]

# One-hot encode categorical
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
encoded_cat = encoder.fit_transform(X.select_dtypes(include=["object"]))
numerical = X.select_dtypes(include=["number"]).to_numpy()
X_combined = hstack([numerical, encoded_cat])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train and evaluate model
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and encoder
joblib.dump(clf, "../rf_model.joblib")
joblib.dump(encoder, "../encoder.joblib")
