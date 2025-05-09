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
from datetime import datetime
import re
import time

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

def fix_year(date_str):
    try:
        # Extract the year using regex
        match = re.search(r"(\d{4}|\d{2})", date_str)
        if match:
            year = match.group(0)
            # Fix years less than 1000 (e.g., 0957 -> 1957)
            if len(year) == 4 and int(year) < 1000:
                date_str = date_str.replace(year, str(1900 + int(year)))
        return date_str
    except Exception:
        return None

def main():
    clear_page("LostLess")
    st.markdown("# Exploración datos LostLess")

    file = st.sidebar.file_uploader(
        "Seleccionar archivo CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="patients",
    )

    if file:
        try:
            df = pd.read_csv(file, delimiter=",", low_memory=False)

            #if st.toggle("Drop NaN?"):
            #    df = df.dropna()

            df['Asistencia'] = np.where(df['HORA_ATENCION'].notna(), 'Asiste', 'No Asiste')
            df["FECHA_RESERVA"] = pd.to_datetime(df["FECHA_RESERVA"], format="%Y-%m-%d")
            df["HORA_RESERVA"] = pd.to_datetime(df["HORA_RESERVA"], format="%H:%M")
            df["MES"] = df["FECHA_RESERVA"].dt.month
            df["DIA"] = df["FECHA_RESERVA"].dt.day_name()
            df["HORA"] = df["HORA_RESERVA"].dt.hour

            df['FECHA_NAC_PACIENTE'] = pd.to_datetime(df['FECHA_NAC_PACIENTE'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
            today = datetime.now()
            df['EDAD'] = df['FECHA_NAC_PACIENTE'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))
            #df = df[df["EDAD"] > 3]
            #df = df[df["EDAD"] < 100]

            cols = []
            for col in df.columns.tolist():
                if "ID_" not in col:
                    cols.append(col)
            sort = st.sidebar.selectbox(
                "Sort column",
                cols,
                index = 25
            )
            cols2 = [x for x in cols if x != sort]
            unique_vals = df[sort].unique()

            by = st.sidebar.selectbox(
                "Sort value",
                unique_vals,
                index = 1
            )

            cols2 = [x for x in cols if x != sort]

            var = st.sidebar.selectbox(
                "V/S column",
                cols2,
                index = 28
            )

            if st.sidebar.button("Filter", type="primary"):
                df = make_bool(df=df, sort=sort, by=by, name=by)
                summary(df=df, sort=sort, var=var, by=by)
                st.dataframe(df)
                #pyg_app = StreamlitRenderer(dataset=df, default_tab="data")
                #pyg_app.explorer()


        except Exception as e:
            st.error(f"Error leyendo el archivo CSV: {e}")


if __name__ == "__main__":
    main()
