import os
import streamlit as st
import pandas as pd
from train import train_model
from test_random import test_random, test_row
import threading
import numpy as np


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


def run_train_in_background(df, sample_size):
    report = train_model(df=df, sample_size=sample_size)
    st.code(report)
    st.write("Training completed.")


def get_sizes():
    all_models = os.listdir("../lostless_data/models/")
    model_sizes = []
    for model in all_models:
        if "rf_model" in model:
            model_sizes.append(int(model.split(".")[0].split("_")[-1]))
    model_sizes.sort()
    return model_sizes


def train_mode(df):
    with st.form("training"):
        sample_size = st.number_input(
            "Sample Size",
            min_value=100,
            max_value=df.shape[0],
            value=10000,
            step=10000,
            key="sample_size",
        )
        submitted = st.form_submit_button("Train Model")
        if submitted:
            model_sizes = get_sizes()
            if sample_size not in model_sizes:
                try:
                    with st.spinner("Training", show_time=True):
                        thread = threading.Thread(target=run_train_in_background, args=(df, sample_size))
                        thread.start()
                    st.info("Training running in background")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.info("Model of that size already exists")


def report_mode():
    with st.form("testing"):
        model_sizes = get_sizes()
        sample_size = st.selectbox(
            "Model Size",
            model_sizes[::-1],
        )
        
        submitted = st.form_submit_button("Model Report")
        if submitted:
            report_path = f"../lostless_data/reports/classification_report_{sample_size}.txt"
            with st.spinner("Testing...", show_time=True):
                try:
                    with open(report_path, "r") as file:
                        report = file.read()
                        st.code(report)
                except Exception as e:
                    st.info(f"No report found")
                st.success("Reporting finished!")

def predict_mode(df):
    with st.form("testing"):
        model_sizes = get_sizes()
        sample_size = st.selectbox(
            "Model Size",
            model_sizes[::-1],
        )

        #data = {col: np.random.choice(df[col].values) for col in df.columns}
        #del data["ASISTIDA"]
        data = df.sample(n=1, random_state=1).copy()
        real = data["ASISTIDA"].tolist()[0]
        data = data.drop(columns=["ASISTIDA"])
        data = data.to_dict(orient="records")[0]

        

        editable_data = {}
        #for key, value in data.items():
        #    if isinstance(value, int) or isinstance(value, np.int64):
        #        editable_data[key] = st.number_input(key, value=value)
        #    elif isinstance(value, bool):
        #        editable_data[key] = st.checkbox(key, value=value)
        #    elif isinstance(value, float):
        #        editable_data[key] = st.number_input(key, value=value, format="%.2f")
        #    elif key.startswith("FECHA") or "FECHA" in key:
        #        editable_data[key] = st.date_input(key, value=pd.to_datetime(value).date())
        #    elif key.startswith("HORA") or "HORA" in key:
        #        editable_data[key] = st.time_input(key, value=pd.to_datetime(value).time())
        #    else:
        #        editable_data[key] = st.text_input(key, value=value)


        num_cols = 5
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
                    editable_data[key] = st.number_input(key, value=value, format="%.2f")
                elif key.startswith("FECHA") or "FECHA" in key:
                    editable_data[key] = st.date_input(key, value=pd.to_datetime(value).date())
                elif key.startswith("HORA") or "HORA" in key:
                    editable_data[key] = st.time_input(key, value=pd.to_datetime(value).time())
                else:
                    editable_data[key] = st.text_input(key, value=value)

        
        submitted = st.form_submit_button("Predict Row")
        if submitted:
            #st.json(editable_data)
            row = pd.DataFrame([editable_data])
            #st.write(row)
            
            with st.spinner("Predicting...", show_time=True):
                labels = test_row(row=row, sample_size=sample_size)             
                st.code(labels+f" | Correct: {real}")
                st.success("Prediction finished!")


def main():
    clear_page("Train")
    st.title("Data trainer")

    model_sizes = get_sizes()
    df = None
    upload = st.sidebar.toggle("Upload Data?")

    if upload:
        # File uploader
        uploaded_file = st.sidebar.file_uploader("Upload an CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df = df.dropna().dropna(axis=1)
    else:
        df = pd.read_csv("../lostless_data/data/data.csv")
        df = df.dropna().dropna(axis=1)

    
    if df is not None:
        mode = st.sidebar.selectbox(
            "Option",
            ["Predict", "Report", "Train"],
        )
        if mode == "Train":
            train_mode(df)

        elif mode == "Report":
            report_mode()
        
        elif mode == "Predict":
            predict_mode(df)
                 


if __name__ == "__main__":
    main()
