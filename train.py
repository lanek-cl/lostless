import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load your dataset as a Pandas DataFrame
df = pd.read_csv("data.csv")

# Feature engineering
X = df.drop(columns=["ASISTIDA"])  # Features
y = df["ASISTIDA"]  # Target

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)  # Avoid sparse matrix for ease of combination
categorical_features = X.select_dtypes(include=["object"])
encoded_categorical = encoder.fit_transform(categorical_features)

# Combine encoded categorical features with numerical features
numerical_features = X.select_dtypes(include=["number"]).to_numpy()
X_combined = np.hstack([numerical_features, encoded_categorical])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
