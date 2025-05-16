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



def train_model(df, sample_size, path):
    # Sample for training
    if sample_size != -1:
        df = df.sample(n=sample_size, random_state=42)
    else:
        sample_size = df.shape[0]

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
    categorical_cols = X.select_dtypes(include=["object"]).columns
    encoded_cat = encoder.fit_transform(X[categorical_cols])

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

    # Train and evaluate model
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Save model and encoder
    joblib.dump(clf, f"{path}/models/rf_model_{sample_size}.joblib", compress=("zlib", 3))
    joblib.dump(encoder, f"{path}/encoders/encoder_{sample_size}.joblib", compress=("zlib", 3))

    # Save classification report
    report = classification_report(y_test, y_pred)
    with open(f"{path}/reports/classification_report_{sample_size}.txt", "w") as file:
        file.write(report)

    # --- Save feature importances ---
    # Get feature names (numerical + encoded categorical)
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + encoded_feature_names

    # Get feature importances from model
    importances = clf.feature_importances_

    # Create DataFrame with all features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': importances
    }).sort_values(by='Weight', ascending=False)

    # Save raw feature-level importances
    #importance_df.to_csv(f"{path}/weights/weights_feature_{sample_size}.csv", index=False)

    # --- Aggregate importances by original column ---
    feature_to_column = {}

    # Numerical features map 1:1
    for col in numerical_cols:
        feature_to_column[col] = col

    # Encoded categorical features map to base column
    original_columns = X.columns.tolist()
    for feature in encoded_feature_names:
        # Match the original column name that the encoded feature starts with
        base_col = next((col for col in original_columns if feature.startswith(col + "_")), feature)
        feature_to_column[feature] = base_col

    importance_df['Column'] = importance_df['Feature'].map(feature_to_column)

    # Group by original column and sum importances
    column_importances = (
        importance_df.groupby('Column')['Weight']
        .sum()
        .reset_index()
        .sort_values(by='Weight', ascending=False)
    )

    # Save aggregated column-level importances
    column_importances.to_csv(f"{path}/weights/weight_{sample_size}.csv", index=False)

    return report


def main():
    # Load dataset
    path = "../lostless_data"
    df = pd.read_csv(f"{path}/data/data.csv")
    df = df.dropna().dropna(axis=1)
    start = time.time()
    report = train_model(df, -1)
    print(report)
    stop = time.time()
    print(f"Training time: {stop - start:.2f} seconds")


if __name__ == "__main__":
    main()
