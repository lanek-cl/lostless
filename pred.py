import os
import time
import argparse
import pandas as pd
import joblib
from scipy.sparse import hstack

def get_sizes(path):
    all_models = os.listdir(f"{path}/models/")
    model_sizes = [
        int(model.split(".")[0].split("_")[-1])
        for model in all_models if "rf_model" in model
    ]
    return sorted(model_sizes)

def predict_all(n=-1, seed=42, path="../lostless_data_CLC-V2-MDH"):
    start = time.perf_counter()

    # Load and subsample data
    df = pd.read_csv(f"{path}/data/data.csv")  # keep all rows if n=-1
    if n != -1:
        df = df.sample(n=n, random_state=seed)

    # Load largest model + encoder once
    model_sizes = get_sizes(path)
    sample_size = model_sizes[-1]
    clf = joblib.load(f"{path}/models/rf_model_{sample_size}.joblib")
    encoder = joblib.load(f"{path}/encoders/encoder_{sample_size}.joblib")

    # --- Preprocess ALL data at once ---
    for col in df.select_dtypes(include=["int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    cat = df.select_dtypes(include=["object"])
    num = df.select_dtypes(include=["number"])

    encoded_cat = encoder.transform(cat)
    numerical = num.to_numpy()
    X_input = hstack([numerical, encoded_cat])

    # --- Predict all at once ---
    numeric_labels = clf.predict(X_input)
    probs = clf.predict_proba(X_input)[:, 1]  # probability of class 1 in clf.classes_

    # --- Map numeric labels to 0=False / 1=True ---
    # Determine which index corresponds to True
    true_class_index = list(clf.classes_).index(True)
    labels_01 = (numeric_labels == true_class_index).astype(int)  # 1=True, 0=False

    df["PREDICTED"] = labels_01
    df["PROBABILITY"] = probs  # probability of True

    counts = df["PREDICTED"].value_counts(normalize=True)
    true_ratio = counts.get(1, 0.0)
    false_ratio = counts.get(0, 0.0)

    pred_path = f"{path}/predictions"
    os.makedirs(pred_path, exist_ok=True)
    output_path = f"{pred_path}/predictions-{n}-{seed}.csv"
    df.to_csv(output_path, index=False)

    end = time.perf_counter()
    print(f"✅ Saved predictions to {output_path}")
    print(f"⏱️ Time taken: {end - start:.2f} seconds")
    print(f"📊 Prediction ratios:")
    print(f"   True:  {true_ratio:.2%}")
    print(f"   False: {false_ratio:.2%}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions on a sample of data.")
    parser.add_argument("-n", type=int, default=-1, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--path", type=str, default="../lostless_data_CLC-V2-MDH", help="Data path")

    args = parser.parse_args()
    predict_all(n=args.n, seed=args.seed, path=args.path)
