import streamlit as st
import pandas as pd
from io import BytesIO


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

clear_page("Converter")
# Streamlit app title
st.title("Excel to CSV Converter")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])

if uploaded_file:
    try:
        # Load the Excel file
        excel_data = pd.ExcelFile(uploaded_file)

        # List to store dataframes
        dfs = []

        # Process each sheet
        for sheet_name in excel_data.sheet_names:
            st.write(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(excel_data, sheet_name=sheet_name)
            dfs.append(df)

        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Prepare CSV for download
        csv_buffer = BytesIO()
        combined_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Download button
        st.download_button(
            label="Download CSV",
            data=csv_buffer,
            file_name=f"{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )

        st.success("All sheets have been combined and are ready for download!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
