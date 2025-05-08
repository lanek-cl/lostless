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

pio.templates.default = "plotly"
pio.templates[pio.templates.default].layout.colorway = (
    px.colors.qualitative.Plotly
)  # Dark2


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


def summary(df, sort, var, by):
    summary = df.groupby(var)[by].value_counts().unstack(fill_value=0)
    summary = summary.rename(columns={True: "Count_True", False: "Count_False"})
    summary["True/False Ratio"] = summary["Count_True"] / summary[
        "Count_False"
    ].replace(0, float("nan"))
    summary["True/Total Ratio"] = (
        summary["Count_True"] / (summary["Count_True"] + summary["Count_False"])
    ) * 100
    summary["False/Total Ratio"] = (
        summary["Count_False"] / (summary["Count_True"] + summary["Count_False"])
    ) * 100
    summary = summary.reset_index()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=summary[var],
            y=summary["Count_True"],
            name=by,
            # marker_color='green'
        )
    )

    fig.add_trace(
        go.Bar(
            x=summary[var],
            y=summary["Count_False"],
            name=f"No {by}",
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
            name=f"% no {by}",
            yaxis="y2",
            mode="lines+markers",
            line=dict(dash="dash"),
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
    estado = df[sort]
    asistido = []
    for i in estado:
        if i == by:
            asistido.append(True)
        else:
            asistido.append(False)
    df[name] = asistido
    return df


def main():
    clear_page("LostLess")
    st.markdown("# Exploración datos LostLess")

    patients = st.file_uploader(
        "Seleccionar archivo de pacientes",
        type=["csv"],
        accept_multiple_files=False,
        key="patients",
    )

    events = st.file_uploader(
        "Seleccionar archivo de eventos",
        type=["csv"],
        accept_multiple_files=False,
        key="events",
    )

    if events and patients:
        try:
            dfe = pd.read_csv(events, delimiter=";", low_memory=False)
            dfe = dfe.dropna()
            dfe["PROFESIONAL_ID"] = pd.factorize(dfe["ID_DE_PROFESIONAL"])[0]
            dfp = pd.read_csv(patients, delimiter=";", low_memory=False)
            dfp = dfp.dropna()
            dfm = pd.merge(dfe, dfp, left_on="ID_PACIENTE", right_on="ID", how="left")
            df = dfm.drop(columns=["ID"])
            df["FECHA_DE_RESERVA"] = pd.to_datetime(
                df["FECHA_DE_RESERVA"], format="%d/%m/%Y %H:%M"
            )
            df["MES"] = df["FECHA_DE_RESERVA"].dt.month
            df["DIA"] = df["FECHA_DE_RESERVA"].dt.day_name()
            df["HORA"] = df["FECHA_DE_RESERVA"].dt.hour

            cols = []
            for col in df.columns.tolist():
                if "ID_" not in col:
                    cols.append(col)
            sort = st.selectbox(
                "Sort column",
                cols,
                index = 8
            )
            cols2 = [x for x in cols if x != sort]
            unique_vals = df[sort].unique()

            by = st.selectbox(
                "Sort value",
                unique_vals
            )

            cols2 = [x for x in cols if x != sort]

            var = st.selectbox(
                "V/S column",
                cols2,
                index = 9
            )

            if st.button("Sort", type="primary"):
                df = make_bool(df=df, sort=sort, by=by, name=by)
                summary(df=df, sort=sort, var=var, by=by)
                # st.write(df)
                pyg_app = StreamlitRenderer(dataset=df, default_tab="Data")
                pyg_app.explorer()

        except Exception as e:
            st.error(f"Error leyendo el archivo CSV: {e}")


if __name__ == "__main__":
    main()
