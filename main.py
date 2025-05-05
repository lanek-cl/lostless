# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs lostless analysis
@date    : 2025/04/29
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

pio.templates.default = 'plotly'
pio.templates[pio.templates.default].layout.colorway = px.colors.qualitative.Plotly#Dark2

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


def summary(df, var):
    summary = df.groupby(var)['ASISTIDO'].value_counts().unstack(fill_value=0)
    summary = summary.rename(columns={True: 'Count_True', False: 'Count_False'})
    summary['True/False Ratio'] = summary['Count_True'] / summary['Count_False'].replace(0, float('nan'))
    summary['True/Total Ratio'] = summary['Count_True'] / (summary['Count_True'] + summary['Count_False'])
    summary['False/Total Ratio'] = summary['Count_False'] / (summary['Count_True'] + summary['Count_False'])

    # Optional: reset index to make it easier to export or view
    summary = summary.reset_index()

    #st.write(summary)

    # --- Plotting ---
    fig = go.Figure()

    # Bar plots for True and False counts
    fig.add_trace(go.Bar(
        x=summary[var],
        y=summary['Count_True'],
        name='Asiste (True)',
        #marker_color='green'
    ))

    fig.add_trace(go.Bar(
        x=summary[var],
        y=summary['Count_False'],
        name='No Asiste (False)',
        #marker_color='red'
    ))

    # Line plot for True/Total Ratio on secondary y-axis
    fig.add_trace(go.Scatter(
        x=summary[var],
        y=summary['True/Total Ratio'],
        name='Ratio Asistencia',
        yaxis='y2',
        mode='lines+markers',
        line=dict(dash='dash')
    ))

    # Line plot for True/Total Ratio on secondary y-axis
    fig.add_trace(go.Scatter(
        x=summary[var],
        y=summary['False/Total Ratio'],
        name='Ratio Inasistencia',
        yaxis='y2',
        mode='lines+markers',
        line=dict(dash='dash')
    ))

    # --- Layout ---
    fig.update_layout(
        title=f'Asistencia por {var}',
        xaxis=dict(
            title=var,
            tickangle=-30  # Tilt labels by -45 degrees
        ),
        yaxis=dict(title='Cantidad'),
        yaxis2=dict(title='Ratio Asistencia', overlaying='y', side='right'),
        barmode='group',
        legend=dict(x=0.05, y=0.5),
        height=600,
        width=1000
    )

    st.write(fig)
    #fig.show()


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
                    estado = df['ESTADO_ULTIMO']
                    asistido = []
                    for i in estado:
                        if i == "ASISTIDA":
                            asistido.append(True)
                        else:
                            asistido.append(False)
                    df["ASISTIDO"] = asistido
                    st.write(df)
                    summary(df=df, var='ESPACIALIDAD')
                    summary(df=df, var='MEDIO')
                    summary(df=df, var='MOTIVO_CONSULTA')
                    df['ID_CORRELATIVO'] = pd.factorize(df['ID_DE_PROFESIONAL'])[0]

                    summary(df=df, var='ID_CORRELATIVO')
                    

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
