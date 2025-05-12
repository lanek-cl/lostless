import streamlit as st
import pandas as pd
from io import BytesIO

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
import joblib
from scipy.sparse import hstack




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


def train(df, drop_columns, sample_size):
    # Drop specified columns
    df = df.drop(columns=drop_columns, axis=1)
    # Sample for training
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
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    encoded_cat = encoder.fit_transform(X.select_dtypes(include=["object"]))

    # Include numerical features
    numerical = X.select_dtypes(include=["number"]).to_numpy()
    X_combined = hstack([numerical, encoded_cat])

    # Use SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_combined, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Train and evaluate model
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    st.code(classification_report(y_test, y_pred))

    # Save model and encoder
    joblib.dump(clf, "../models/lostless/rf_model.joblib")
    joblib.dump(encoder, "../models/lostless/encoder.joblib")


def test(df, drop_columns, row_number):
    # Load saved model and encoder
    clf = joblib.load("../models/lostless/rf_model.joblib")
    encoder = joblib.load("../models/lostless/encoder.joblib")

    df = df.drop(columns=drop_columns, axis=1)
    row = df[df.index == row_number].copy()

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
    st.info(f"Label: {y_true}, Predicted: {y_pred[0]}")
    result = y_true == y_pred[0]
    if result:
        st.success("Correct!")
    else:
        st.error("Incorrect!")



def main():

    clear_page("Train")
    # Streamlit app title
    st.title("Data trainer")

    # File uploader
    uploaded_file = st.file_uploader("Upload an CSV file", type=["csv"])


    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.dropna().dropna(axis=1)
        numerical_columns = ["DIA_NUM", "HORA", "EDAD"]
        for col in numerical_columns:
            if df[col].dtype not in ["int64", "float64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if st.toggle("Test"):
            with st.form("testing"):
                drop_columns = st.multiselect(
                    "Columns to drop",
                    df.columns,
                    default=["ESTADO"],
                )
                row_number = st.number_input("Row Number", min_value=0, max_value=df.shape[0], value=1, step=1, key="row_num")
                submitted = st.form_submit_button("Test Model")
                if submitted:
                    try:
                        test(df=df, drop_columns=drop_columns, row_number=row_number)
                        st.success("Testing finished!")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            with st.form("training"):
                drop_columns = st.multiselect(
                    "Columns to drop",
                    df.columns,
                    default=["ESTADO"],
                )
                sample_size = st.number_input("Sample Size", min_value=10000, max_value=df.shape[0], value=10000, step=10000, key="sample_size")
                submitted = st.form_submit_button("Train Model")
                if submitted:
                    try:
                        train(df=df, drop_columns=drop_columns, sample_size=sample_size)
                        st.success("Training finished!")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()