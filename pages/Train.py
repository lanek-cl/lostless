import os
import streamlit as st
import pandas as pd
from train import train
from test_random import test_random
import threading


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
    report = train(df=df, sample_size=sample_size)
    st.code(report)
    st.write("Training completed.")


def main():
    clear_page("Train")
    # Streamlit app title
    st.title("Data trainer")

    all_models = os.listdir("../lostless_data/models/")
    model_sizes = []
    for model in all_models:
        if "rf_model" in model:
            model_sizes.append(int(model.split(".")[0].split("_")[-1]))
    model_sizes.sort()

    df = None
    upload = st.toggle("Upload Data?")

    if upload:
        # File uploader
        uploaded_file = st.file_uploader("Upload an CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df = df.dropna().dropna(axis=1)
    else:
        df = pd.read_csv("../lostless_data/data/data.csv")
        df = df.dropna().dropna(axis=1)

    
    if df is not None:
        if st.toggle("Train?"):
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
        else:
            with st.form("testing"):
                sample_size = st.selectbox(
                    "Model Size",
                    model_sizes[::-1],
                )
                
                submitted = st.form_submit_button("Test Model")
                if submitted:
                    report_path = f"../lostless_data/reports/classification_report_{sample_size}.txt"
                    with st.spinner("Testing...", show_time=True):
                        #labels, result = test_random(df=df, sample_size=sample_size)
                        #st.code(labels)
                        #st.code(result)
                        try:
                            with open(report_path, "r") as file:
                                report = file.read()
                                st.code(report)
                        except Exception as e:
                            st.info(f"No report found")
                        st.success("Testing finished!")     


if __name__ == "__main__":
    main()
