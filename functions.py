# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs lostless analysis using Polars
@date    : 2025/05/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import polars as pl
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import re
from pygwalker.api.streamlit import StreamlitRenderer
import os
import sys
import logging
import joblib
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
import time
# import sys
# import logging

# # Configure logging
# logging.basicConfig(
#     filename="results.log",
#     filemode="a",  # Append to the file
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )

# # Create a custom stream to redirect print statements
# class PrintLogger:
#     def __init__(self, level):
#         self.level = level

#     def write(self, message):
#         # Skip empty lines
#         if message.strip():
#             self.level(message.strip())

#     def flush(self):
#         pass  # No need to implement flush for this example

# # Redirect stdout and stderr
# sys.stdout = PrintLogger(logging.info)
# sys.stderr = PrintLogger(logging.error)

def make_bool(df, sort, by, name):
    return df.with_columns((df[sort] == by).alias(name))


def save_csv(df, path):
    os.makedirs(f"{path}/data", exist_ok=True)
    os.makedirs(f"{path}/models", exist_ok=True)
    os.makedirs(f"{path}/encoders", exist_ok=True)
    os.makedirs(f"{path}/reports", exist_ok=True)
    os.makedirs(f"{path}/weights", exist_ok=True)
    df.write_csv(f"{path}/data/data.csv")
    st.sidebar.success("File saved!")


def fix_year(date_str):
    try:
        match = re.search(r"(\d{4}|\d{2})", date_str)
        if match:
            year = match.group(0)
            if len(year) == 4 and int(year) < 1000:
                date_str = date_str.replace(year, str(1900 + int(year)))
        return date_str
    except Exception:
        return None
    

def filter_data_cdm(df):
    try:
        df = df.with_columns(
            pl.when(pl.col("HORA_ATENCION").is_not_null())
            .then(pl.lit("ASISTIDA"))
            .otherwise(pl.lit("ANULADA"))
            .alias("ESTADO")
        )

        df = df.with_columns(
            pl.col("FECHA_RESERVA")
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            .alias("FECHA_RESERVA"),
            pl.col("HORA_RESERVA")
            .str.strptime(pl.Time, "%H:%M", strict=False)
            .alias("HORA_RESERVA"),
            pl.col("FECHA_NAC_PACIENTE")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            .alias("FECHA_NAC_PACIENTE"),
        )

        df = df.with_columns(
            pl.col("FECHA_RESERVA").dt.strftime("%B").alias("MES_NAME"),
            pl.col("FECHA_RESERVA").dt.month().alias("MES_NUM"),
            pl.col("FECHA_RESERVA").dt.strftime("%A").alias("DIA_NAME"),
            pl.col("FECHA_RESERVA").dt.weekday().alias("DIA_NUM"),
            pl.col("HORA_RESERVA").dt.hour().alias("HORA_NUM"),
        )

        today = datetime.now()

        df = df.with_columns(
            (
                (pl.lit(today.year) - pl.col("FECHA_NAC_PACIENTE").dt.year())
                - (
                    (pl.col("FECHA_NAC_PACIENTE").dt.month() > today.month)
                    | (
                        (pl.col("FECHA_NAC_PACIENTE").dt.month() == today.month)
                        & (pl.col("FECHA_NAC_PACIENTE").dt.day() > today.day)
                    )
                ).cast(pl.Int32)
            ).alias("EDAD")
        )

        sort = "ESTADO" 
        by = "ASISTIDA" 
        df = make_bool(df=df, sort=sort, by=by, name=by)
        columns = df.columns

        dropDefault = [
            "FECHA_CONFIRMACION",
            "RESCONF_RESERVA",
            "FECHA_ANULACION",
            "FECREC_RESERVA",
            "HORA_ATENCION",
            "MONTO_RESERVA",
            "FECHA_FICHA",
            "ESTADO",
            "MES_NAME",
            "MES_NUM",
            "DIA_NAME",
            "DIA_NUM",
            "HORA_NUM",
            "HORA_RESERVA",
            "FECHA_CREACION_PAC",
            #"FECHA_NAC_PACIENTE",
            #"EDAD",
        ]
        cols2drop = st.multiselect("Columns to drop", options=columns, default=dropDefault)


        df = df.drop(cols2drop)
        df = df.drop_nulls()
        if "FECHA_NAC_PACIENTE" in df.columns:
            filter_date = datetime(2022, 2, 4, 0, 0, 0)
            df = df.filter(pl.col("FECHA_NAC_PACIENTE") != filter_date)
        
        pyg_app = StreamlitRenderer(
            dataset=df, default_tab="data", appearance="light"
        )
        pyg_app.explorer()

        name = st.sidebar.text_input("Save name", "New")
        if st.sidebar.button("Save CSV", type="primary"):
            path = f"../lostless_data_CDM-{name}"
            save_csv(df, path)
    except Exception as e:
        st.error(f"Error leyendo el archivo CSV: {e}")

def filter_data_clc(dfe, dfp):
    try:
        # Drop nulls
        dfe = dfe.drop_nulls()
        dfp = dfp.drop_nulls()
        dfm = dfe.join(dfp, left_on="ID_PACIENTE", right_on="ID", how="right")
        dfm = dfm.drop_nulls()
        df = dfm.drop("ID")

        # Parse 'FECHA_DE_RESERVA' to datetime
        df = df.with_columns(
            [
                pl.col("FECHA_DE_RESERVA").str.strptime(
                    pl.Datetime, "%d/%m/%Y %H:%M", strict=False
                )
            ]
        )

        # Extract month, day name, and hour
        df = df.with_columns(
            [
                pl.col("FECHA_DE_RESERVA").dt.month().alias("MES"),
                pl.col("FECHA_DE_RESERVA").dt.strftime("%A").alias("DIA"),
                pl.col("FECHA_DE_RESERVA").dt.hour().alias("HORA"),
            ]
        )

        sort = "ESTADO_ULTIMO" 
        by = "ASISTIDA"
        df = make_bool(df=df, sort=sort, by=by, name=by)
        columns = df.columns

        dropDefault = [
            "ESTADO_ULTIMO",
            "MES",
            "DIA",
            "HORA"
        ]
        cols2drop = st.multiselect("Columns to drop", options=columns, default=dropDefault)
        df = df.drop(cols2drop)
        df = df.drop_nulls()
        pyg_app = StreamlitRenderer(
            dataset=df, default_tab="data", appearance="light"
        )
        pyg_app.explorer()

        name = st.sidebar.text_input("Save name", "New")
        if st.sidebar.button("Save CSV", type="primary"):
            path = f"../lostless_data_CLC-{name}"
            save_csv(df, path)

    except Exception as e:
        st.error(f"Error leyendo el archivo CSV: {e}")


def test_row(row, sample_size, path):
    # Load saved model and encoder
    clf = joblib.load(f"{path}/models/rf_model_{sample_size}.joblib")
    encoder = joblib.load(f"{path}/encoders/encoder_{sample_size}.joblib")
    
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
    
    labels = y_pred[0]
    #result = "Correct" if y_true == y_pred[0] else "Incorrect"
    return labels


def test_random(df, sample_size, path):
    df = pd.read_csv(f"{path}/data/data.csv")
    # Load saved model and encoder
    clf = joblib.load(f"{path}/models/rf_model_{sample_size}.joblib")
    encoder = joblib.load(f"{path}/encoders/encoder_{sample_size}.joblib")
  
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


def train_model(sample_size, path):
    print("Starting training process...")

    # Load dataset
    print(f"Loading dataset from {path}/data/data.csv...")
    df = pd.read_csv(f"{path}/data/data.csv")
    print(f"Dataset loaded with shape: {df.shape}")

    # Drop missing values
    #print("Dropping rows and columns with missing values...")
    #df = df.dropna().dropna(axis=1)
    #print(f"Data after dropping missing values: {df.shape}")

    # Sample for training
    if sample_size != -1:
        print(f"Sampling {sample_size} rows for training...")
        df = df.sample(n=sample_size, random_state=42)
    else:
        sample_size = df.shape[0]
        print(f"Using the entire dataset with {sample_size} rows for training.")

    # Downcast numerical types
    print("Downcasting numerical types for memory efficiency...")
    for col in df.select_dtypes(include=["int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")
    print("Numerical downcasting complete.")

    # Split features and labels
    print("Splitting features (X) and labels (y)...")
    X = df.drop(columns=["ASISTIDA"])
    y = df["ASISTIDA"]
    print(f"Feature set shape: {X.shape}, Label shape: {y.shape}")

    # One-hot encode categorical variables
    print("Encoding categorical features using one-hot encoding...")
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    categorical_cols = X.select_dtypes(include=["object"]).columns
    encoded_cat = encoder.fit_transform(X[categorical_cols])
    print(f"Encoded categorical features shape: {encoded_cat.shape}")

    # Combine numerical and encoded categorical features
    print("Combining numerical and encoded categorical features...")
    numerical = X.select_dtypes(include=["number"]).to_numpy()
    X_combined = hstack([numerical, encoded_cat])
    print(f"Combined feature matrix shape: {X_combined.shape}")

    # Use SMOTE to balance the classes
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_combined, y)
    print(f"Balanced dataset shape: {X_balanced.shape}, {y_balanced.shape}")

    # Train-test split
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

    # Train the model
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("Making predictions on the test set...")
    y_pred = clf.predict(X_test)

    # Save the model and encoder
    print("Saving the trained model and encoder...")
    joblib.dump(clf, f"{path}/models/rf_model_{sample_size}.joblib", compress=("zlib", 3))
    joblib.dump(encoder, f"{path}/encoders/encoder_{sample_size}.joblib", compress=("zlib", 3))
    print("Model and encoder saved.")

    # Save classification report
    print("Generating classification report...")
    report = classification_report(y_test, y_pred)
    report_path = f"{path}/reports/classification_report_{sample_size}.txt"
    with open(report_path, "w") as file:
        file.write(report)
    print(f"Classification report saved to {report_path}")

    # Save feature importances
    print("Calculating feature importances...")
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + encoded_feature_names
    importances = clf.feature_importances_

    print("Aggregating feature importances...")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': importances
    }).sort_values(by='Weight', ascending=False)
    #importance_df.to_csv(f"{path}/weights/weights_feature_{sample_size}.csv", index=False)

    # Map features to original columns
    print("Mapping encoded features to original columns...")
    feature_to_column = {col: col for col in numerical_cols}
    for feature in encoded_feature_names:
        base_col = next((col for col in X.columns if feature.startswith(col + "_")), feature)
        feature_to_column[feature] = base_col

    importance_df['Column'] = importance_df['Feature'].map(feature_to_column)

    # Aggregate importances
    print("Aggregating importances by original columns...")
    column_importances = (
        importance_df.groupby('Column')['Weight']
        .sum()
        .reset_index()
        .sort_values(by='Weight', ascending=False)
    )
    weight_path = f"{path}/weights/weight_{sample_size}.csv"
    column_importances.to_csv(weight_path, index=False)
    print(f"Aggregated column importances saved to {weight_path}")

    print("Training process complete.")
    return report

