import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import concurrent.futures
from scipy.sparse import hstack
import joblib

from functions import filter_data_cdm, filter_data_clc, test_row


def run_background_command(command):
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


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


# def run_train_in_background(df, sample_size, path):
# report = train_model(df=df, sample_size=sample_size, path=path)
# st.code(report)
# st.write("Training completed.")


def get_sizes(path, mod):
    all_models = os.listdir(f"{path}/models/")
    model_sizes = []
    for model in all_models:
        if "rf_model" in model and mod in model:
            model_sizes.append(int(model.split(".")[0].split("_")[-1]))
    model_sizes.sort()
    return model_sizes


def get_paths():
    all_models = os.listdir("../")
    model_sizes = []
    for model in all_models:
        if "lostless_data_" in model:
            model_sizes.append(model.split("_")[-1])
    model_sizes.sort()
    return model_sizes


def display_log_file(
    file_path, lines_to_display=100, refresh_interval=1, reverse=False
):
    placeholder = st.empty()

    while True:
        # Check if file exists
        log_file = Path(file_path)
        if log_file.exists():
            # Read the last `lines_to_display` lines of the file
            with log_file.open("r") as f:
                lines = f.readlines()[-lines_to_display:]
                if reverse:
                    lines = lines[::-1]
        else:
            lines = ["Log file not found."]

        # Display the lines
        with placeholder.container():
            # unique_key = str(uuid.uuid4())
            # st.text_area("Log Content", "".join(lines), height=600, key=unique_key)
            st.code("".join(lines), height=600)

        # Wait for the specified refresh interval
        time.sleep(refresh_interval)


def train_mode(path):
    df = pd.read_csv(f"{path}/data/data.csv")
    df = df.dropna().dropna(axis=1)
    # with st.form("training"):
    # dataset = st.selectbox("Dataset", ["data_trim.csv", "data.csv"], index=0)

    sample_size = st.sidebar.number_input(
        "Sample Size",
        min_value=-1,
        max_value=df.shape[0],
        value=10000,
        step=10000,
        key="sample_size",
    )

    model = st.sidebar.selectbox("Model", ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "ExtraTrees"], index=0)
    submitted = st.sidebar.button("Train", type="primary")
    if submitted:
        model_sizes = get_sizes(path, model)
        if sample_size not in model_sizes:
            try:
                # thread = threading.Thread(target=run_train_in_background, args=(df, sample_size, path))
                # thread.start()
                run_background_command(
                    [
                        "conda",
                        "run",
                        "-n",
                        "lostless",
                        "python",
                        "train.py",
                        f"{sample_size}",
                        f"{path}",
                        f"{model}",
                    ]
                )

                st.sidebar.info("Training running in background")
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")
        else:
            st.sidebar.warning("A model of that size already exists")

    log_file_path = "results.log"
    display_log_file(log_file_path, 25, 1, False)


def report_mode(path):
    model = st.sidebar.selectbox("Model", ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "ExtraTrees"], index=0)
    model_sizes = get_sizes(path, model)
    sample_size = st.sidebar.selectbox(
        "Model Size",
        model_sizes[::-1],
    )

    # submitted = st.sidebar.button("Report", type="primary")
    # if submitted:
    report_path = f"{path}/reports/classification_report_{model}_{sample_size}.txt"
    with st.spinner("Testing...", show_time=True):
        try:
            with open(report_path, "r") as file:
                report = file.read()
            st.code(report)
            df = pd.read_csv(f"{path}/weights/weight_{sample_size}.csv")
            df = df.reset_index(drop=True)
            st.table(df)
            hide_index_js = """
            <script>
                const tables = window.parent.document.querySelectorAll('table');
                tables.forEach(table => {
                    const indexColumn = table.querySelector('thead th:first-child');
                    if (indexColumn) {
                        indexColumn.style.display = 'none';
                    }
                    const indexCells = table.querySelectorAll('tbody th');
                    indexCells.forEach(cell => {
                        cell.style.display = 'none';
                    });
                });
            </script>
            """

            # Use components.html to inject the JavaScript
            st.components.v1.html(hide_index_js, height=0)

        except Exception:
            st.info("No report found")
        st.success("Reporting finished!")


def predict_mode(path):
    df = pd.read_csv(f"{path}/data/data.csv")
    df = df.dropna().dropna(axis=1)
    with st.form("testing"):
        model = st.sidebar.selectbox(
            "Model", 
            ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "ExtraTrees"], 
            index=0)

        model_sizes = get_sizes(path, model)
        sample_size = st.sidebar.selectbox(
            "Model Size",
            model_sizes[::-1],
        )
        data = df.sample(n=1, random_state=sample_size).copy()
        try:
            data = data.drop(columns=["ASISTIDA"])
            real = data["ASISTIDA"].tolist()[0]
        except:
            real = True
        data = data.to_dict(orient="records")[0]

        editable_data = {}
        num_cols = 4
        cols = st.columns(num_cols)
        for i, (key, value) in enumerate(data.items()):
            # Select the column
            col = cols[i % num_cols]

            with col:
                if isinstance(value, (int, np.int64)):
                    editable_data[key] = st.number_input(key, value=value)
                elif isinstance(value, bool):
                    editable_data[key] = st.checkbox(key, value=value)
                elif isinstance(value, float):
                    editable_data[key] = st.number_input(
                        key, value=int(value), step=1, min_value=0
                    )
                elif (key.startswith("FECHA") or "FECHA" in key) and (
                    key != "FECHA_DE_CITA"
                ):
                    editable_data[key] = st.date_input(
                        key, value=pd.to_datetime(value).date()
                    )
                elif key.startswith("HORA") or "HORA" in key:
                    editable_data[key] = st.time_input(
                        key, value=pd.to_datetime(value).time()
                    )
                else:
                    editable_data[key] = st.text_input(key, value=value)

        submitted = st.form_submit_button("Predict")
        if submitted:
            row = pd.DataFrame([editable_data])

            with st.spinner("Predicting...", show_time=True):
                predicted = test_row(row=row, sample_size=sample_size, path=path, model=model)
                st.code(f"Predicted: {predicted} | Correct: {real}")
                if predicted == real:
                    st.success("Prediction successful!")
                else:
                    st.error("Prediction failed!")


def predict_all(n=-1, seed=42, path="../lostless_data_CLC-V2-MDH"):
    df = pd.read_csv(f"{path}/data/data.csv")
    testPath = path
    if "V2" in path:
        path = path.replace("V2", "V1")

    model = st.sidebar.selectbox(
        "Model", 
        ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "ExtraTrees"], 
        index=0
    )

    model_sizes = get_sizes(path, model)
    sample_size = st.sidebar.selectbox(
        "Model Size",
        model_sizes[::-1],
    )
    with st.form("testing"):
        submitted = st.form_submit_button("Predict")
        if submitted:
            start = time.perf_counter()

            # Load and subsample data
            
            if n != -1:
                df = df.sample(n=n, random_state=seed)

            # Load largest model + encoder once
            model_sizes = get_sizes(path, model)
            sample_size = model_sizes[-1]
            clf = joblib.load(f"{path}/models/rf_model_{model}_{sample_size}.joblib")
            encoder = joblib.load(f"{path}/encoders/encoder_{model}_{sample_size}.joblib")

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

            pred_path = f"{testPath}/predictions"
            os.makedirs(pred_path, exist_ok=True)
            output_path = f"{pred_path}/prediction_{model}_{sample_size}.csv"
            df.to_csv(output_path, index=False)

            end = time.perf_counter()
            st.write(f"✅ Saved predictions to {output_path}")
            st.write(f"⏱️ Time taken: {end - start:.2f} seconds")
            st.write(f"📊 Prediction ratios:")
            st.write(f"True:  {true_ratio:.2%}")
            st.write(f"False: {false_ratio:.2%}")

            return df

def main():
    clear_page("Train")
    st.title("Data trainer")

    df = None

    # upload = st.sidebar.toggle("Upload Data?")
    # if upload:
    #    # File uploader
    #    uploaded_file = st.sidebar.file_uploader("Upload an CSV file", type=["csv"])
    #    if uploaded_file:
    #        df = pd.read_csv(uploaded_file)
    #        df = df.dropna().dropna(axis=1)

    mode = st.sidebar.selectbox(
        "Option",
        ["Train", "Report", "Filter", "Predict", "FULL"],
    )

    if mode == "Filter":
        dataset = st.sidebar.selectbox("Dataset", ["Ciudad del Mar", "Los Carrera"])
        if dataset == "Ciudad del Mar":
            df = pl.read_csv("../lostless_dataset/data/CDM.csv")
            filter_data_cdm(df)
        elif dataset == "Los Carrera":
            v = st.sidebar.selectbox("Version", ["1", "2"])

            dfe = pl.read_csv(
                f"../lostless_dataset/data/Eventos_v{v}.csv", separator=",", ignore_errors=True
            )
            dfp = pl.read_csv(
                f"../lostless_dataset/data/Pacientes_v{v}.csv", separator=",", ignore_errors=True
            )
            filter_data_clc(dfe, dfp, v)

    else:
        dataset = st.sidebar.selectbox("Dataset", get_paths(), index=0)
        path = f"../lostless_data_{dataset}"
        if mode == "Train":
            train_mode(path)
        elif mode == "Report":
            report_mode(path)
        elif mode == "Predict":
            predict_mode(path)
        elif mode == "FULL":
            predict_all(-1, 42, path)


if __name__ == "__main__":
    main()
