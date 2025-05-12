import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load dataset
df = pd.read_csv("../data.csv")

# Drop rows with missing values
df = df.dropna()

# Alternatively, drop columns with missing values
df = df.dropna(axis=1)
print(df)

# Downcast numerical columns to reduce memory usage
for col in df.select_dtypes(include=["int"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")
for col in df.select_dtypes(include=["float"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

# Feature engineering
X = df.drop(columns=["ASISTIDA"])
y = df["ASISTIDA"]

# One-hot encode categorical features with sparse output
encoder = OneHotEncoder(sparse_output=True)  # Use sparse matrix to save memory
categorical_features = X.select_dtypes(include=["object"])
encoded_categorical = encoder.fit_transform(categorical_features)

# Combine numerical features and sparse categorical features
numerical_features = X.select_dtypes(include=["number"]).to_numpy()
from scipy.sparse import hstack
X_combined = hstack([numerical_features, encoded_categorical])  # Sparse combination

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
