# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs main controller simulation
@date    : 2025/04/29
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

def clear_page(title='Lanek'):
    try:
        #im = Image.open('assets/logos/favicon.png')
        st.set_page_config(
            page_title=title,
            #page_icon=im,
            layout='wide',
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


def main():
    clear_page('LostLess')
    st.markdown("# Exploración datos LostLess")
    uploaded_files = st.file_uploader(
        "Choose a CSV file", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files:
        uids = []
        aids = []
        for file in uploaded_files:
            try:
                df = pd.read_csv(file,delimiter=';')
                df = df.dropna()
                #st.dataframe(df)
                try:
                    uids = list(set(df["ID"]))
                except:
                    aids = list(set(df["ID_PACIENTE"]))

                pyg_app = StreamlitRenderer(dataset=df, default_tab='Data')
                pyg_app.explorer()
            except Exception as e:
                    st.error(f"An error occurred while reading the file: {e}")

        countt = 0
        countf = 0
        for id in uids:
            if id in aids:
                countt = countt + 1
            else:
                countf = countf + 1
        st.write(countt, countf)





if __name__ == '__main__':
    main()
