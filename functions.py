# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs lostless analysis using Polars
@date    : 2025/05/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import os
from datetime import datetime

import joblib
import pandas as pd
import polars as pl
import streamlit as st
from imblearn.over_sampling import SMOTE
from pygwalker.api.streamlit import StreamlitRenderer
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pio.templates.default = "plotly"
pio.templates[pio.templates.default].layout.colorway = (
    px.colors.qualitative.Plotly
)  # Dark2

def make_bool(df, sort, by, name):
    return df.with_columns((df[sort] == by).alias(name))


def save_csv(df, path):
    os.makedirs(f"{path}/data", exist_ok=True)
    os.makedirs(f"{path}/models", exist_ok=True)
    os.makedirs(f"{path}/encoders", exist_ok=True)
    os.makedirs(f"{path}/reports", exist_ok=True)
    os.makedirs(f"{path}/weights", exist_ok=True)
    os.makedirs(f"{path}/predictions", exist_ok=True)
    df.write_csv(f"{path}/data/data.csv")
    st.sidebar.success("File saved!")


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
            # "FECHA_NAC_PACIENTE",
            # "EDAD",
        ]
        cols2drop = st.multiselect(
            "Columns to drop", options=columns, default=dropDefault
        )

        df = df.drop(cols2drop)
        df = df.drop_nulls()
        if "FECHA_NAC_PACIENTE" in df.columns:
            filter_date = datetime(2022, 2, 4, 0, 0, 0)
            df = df.filter(pl.col("FECHA_NAC_PACIENTE") != filter_date)

        pyg_app = StreamlitRenderer(dataset=df, default_tab="data", appearance="light")
        pyg_app.explorer()

        name = st.sidebar.text_input("Save name", "New")
        if st.sidebar.button("Save CSV", type="primary"):
            path = f"../lostless_data_CDM-{name}"
            save_csv(df, path)
    except Exception as e:
        st.error(f"Error leyendo el archivo CSV: {e}")


def filter_data_clc(dfe, dfp, v):
    try:
        # Drop nulls
        dfe = dfe.drop_nulls()
        dfp = dfp.drop_nulls()
        dfm = dfe.join(dfp, left_on="ID_PACIENTE", right_on="ID", how="right")
        dfm = dfm.drop_nulls()
        df = dfm.drop("ID")

        #st.write(df)

        # Parse 'FECHA_DE_RESERVA' to datetime
        if v == "1":
            df = df.with_columns(
                [
                    pl.col("FECHA_DE_RESERVA").str.strptime(
                        pl.Datetime, "%d/%m/%Y %H:%M", strict=False
                    )
                ]
            )
            sort = "ESTADO_ULTIMO"
            by = "ASISTIDA"
            df = make_bool(df=df, sort=sort, by=by, name=by)
        if v == "2":
            df = df.with_columns(
                pl.col("FECHA_DE_RESERVA").str.strptime(
                    pl.Datetime, "%Y-%m-%d %H:%M:%S %Z", strict=False
                )
            )
    

        # Extract month, day name, and hour
        df = df.with_columns(
            [
                pl.col("FECHA_DE_RESERVA").dt.month().alias("MES"),
                pl.col("FECHA_DE_RESERVA").dt.strftime("%A").alias("DIA"),
                pl.col("FECHA_DE_RESERVA").dt.hour().alias("HORA"),
            ]
        )
        
        columns = df.columns

        dropDefault = []
        cols2drop = st.multiselect(
            "Columns to drop", options=columns, default=dropDefault
        )
        df = df.drop(cols2drop)
        df = df.drop_nulls()
        pyg_app = StreamlitRenderer(dataset=df, default_tab="data", appearance="light")
        pyg_app.explorer()

        name = st.sidebar.text_input("Save name", "New")
        if st.sidebar.button("Save CSV", type="primary"):
            path = f"../lostless_data_CLC-{name}"
            save_csv(df, path)

    except Exception as e:
        st.error(f"Error leyendo el archivo CSV: {e}")


def test_row(row, sample_size, path, model):
    # Load saved model and encoder
    clf = joblib.load(f"{path}/models/rf_model_{model}_{sample_size}.joblib")
    encoder = joblib.load(f"{path}/encoders/encoder_{model}_{sample_size}.joblib")

    # Downcast types
    for col in row.select_dtypes(include=["int"]).columns:
        row[col] = pd.to_numeric(row[col], downcast="integer")
    for col in row.select_dtypes(include=["float"]).columns:
        row[col] = pd.to_numeric(row[col], downcast="float")

    # Prepare input
    X = row  # row.drop(columns=["ASISTIDA"])
    # y_true = row["ASISTIDA"].values[0]
    encoded_cat = encoder.transform(X.select_dtypes(include=["object"]))
    numerical = X.select_dtypes(include=["number"]).to_numpy()
    X_input = hstack([numerical, encoded_cat])

    # Predict
    y_pred = clf.predict(X_input)

    labels = y_pred[0]
    # result = "Correct" if y_true == y_pred[0] else "Incorrect"
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


def train_model(sample_size, path, model):
    print("Starting training process...")

    # Load dataset
    print(f"Loading dataset from {path}/data/data.csv...")
    df = pd.read_csv(f"{path}/data/data.csv")
    print(f"Dataset loaded with shape: {df.shape}")

    # Drop missing values
    # print("Dropping rows and columns with missing values...")
    # df = df.dropna().dropna(axis=1)
    # print(f"Data after dropping missing values: {df.shape}")

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
    if model == "RandomForest":
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)

    elif model == "XGBoost":
        clf = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric="logloss")

    elif model == "LightGBM":
        clf = LGBMClassifier(random_state=42, n_jobs=-1)

    elif model == "CatBoost":
        clf = CatBoostClassifier(random_state=42, verbose=0)

    elif model == "ExtraTrees":
        clf = ExtraTreesClassifier(random_state=42, n_jobs=-1)

    else:
        raise ValueError(f"Unknown model: {model}")

    clf.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("Making predictions on the test set...")
    y_pred = clf.predict(X_test)

    # Save the model and encoder
    print("Saving the trained model and encoder...")
    joblib.dump(
        clf, f"{path}/models/rf_model_{model}_{sample_size}.joblib", compress=("zlib", 3)
    )
    joblib.dump(
        encoder, f"{path}/encoders/encoder_{model}_{sample_size}.joblib", compress=("zlib", 3)
    )
    print("Model and encoder saved.")

    # Save classification report
    print("Generating classification report...")
    report = classification_report(y_test, y_pred)
    report_path = f"{path}/reports/classification_report_{model}_{sample_size}.txt"
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
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Weight": importances}
    ).sort_values(by="Weight", ascending=False)
    # importance_df.to_csv(f"{path}/weights/weights_feature_{sample_size}.csv", index=False)

    # Map features to original columns
    print("Mapping encoded features to original columns...")
    feature_to_column = {col: col for col in numerical_cols}
    for feature in encoded_feature_names:
        base_col = next(
            (col for col in X.columns if feature.startswith(col + "_")), feature
        )
        feature_to_column[feature] = base_col

    importance_df["Column"] = importance_df["Feature"].map(feature_to_column)

    # Aggregate importances
    print("Aggregating importances by original columns...")
    column_importances = (
        importance_df.groupby("Column")["Weight"]
        .sum()
        .reset_index()
        .sort_values(by="Weight", ascending=False)
    )
    weight_path = f"{path}/weights/weight_{model}_{sample_size}.csv"
    column_importances.to_csv(weight_path, index=False)
    print(f"Aggregated column importances saved to {weight_path}")

    print("Training process complete.")
    return report


def summary(df, sort, var, by, ex=False):
    if ex and var == "EDAD":
        df = df.filter(pl.col("EDAD") != 3)
    summary = (
        df.group_by(var)
        .agg(
            [
                pl.col(by).sum().alias("Count_True"),
                (1 - pl.col(by)).sum().alias("Count_False"),
            ]
        )
        .with_columns(
            [
                (pl.col("Count_True") / pl.col("Count_False").replace(0, None)).alias(
                    "True/False Ratio"
                ),
                (
                    pl.col("Count_True")
                    / (pl.col("Count_True") + pl.col("Count_False"))
                    * 100
                ).alias("True/Total Ratio"),
                (
                    pl.col("Count_False")
                    / (pl.col("Count_True") + pl.col("Count_False"))
                    * 100
                ).alias("False/Total Ratio"),
            ]
        )
    )
    summary = summary.drop_nulls()
    summary = summary.sort(var)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=summary[var],
            y=summary["Count_True"],
            name=f"# {by}",
            # marker_color='green'
        )
    )

    fig.add_trace(
        go.Bar(
            x=summary[var],
            y=summary["Count_False"],
            name=f"# !{by}",
            # marker_color='red'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary[var],
            y=summary["True/Total Ratio"],
            name=f"% {by}",
            yaxis="y2",
            mode="lines+markers",
            line=dict(dash="dash"),
        )
    )

    # Line plot for True/Total Ratio on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=summary[var],
            y=summary["False/Total Ratio"],
            name=f"% !{by}",
            yaxis="y2",
            mode="lines+markers",
            line=dict(dash="dash", color="orange"),
        )
    )

    fig.update_layout(
        title=f"{sort}: {by} V/S {var}",
        xaxis=dict(title=var, tickangle=-30),  # Tilt labels by -45 degrees
        yaxis=dict(title="Cantidad"),
        yaxis2=dict(title=f"% {by}", overlaying="y", side="right"),
        barmode="group",
        legend=dict(x=0.05, y=0.5),
        height=600,
        width=1000,
    )
    st.write(fig)


def make_bool(df, sort, by, name):
    return df.with_columns((df[sort] == by).alias(name))


def clear_page(title="Lanek"):
    try:
        # im = Image.open('assets/logos/favicon.png')
        st.set_page_config(
            page_title=title,
            # page_icon=im,
            layout="wide",
        )
        hide_streamlit_style = """
            <style>
                .reportview-container {
                    margin-top: -2em;
                }
                #MainMenu {visibility: hidden;}
                .stDeployButton {display:none;}
                footer {visibility: hidden;}
                #stDecoration {display:none;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    except Exception:
        pass