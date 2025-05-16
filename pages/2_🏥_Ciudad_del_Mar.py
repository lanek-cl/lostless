# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs lostless analysis using Polars
@date    : 2025/05/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import polars as pl
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime, date
import re
import time
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer


pio.templates.default = "plotly"
pio.templates[pio.templates.default].layout.colorway = (
    px.colors.qualitative.Plotly
)  # Dark2


def clear_page(title="Lanek"):
    try:
        st.set_page_config(
            page_title=title,
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
    if var == "EDAD":
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
    return df.with_columns((df[sort] == by).alias(name))


def fix_year(date_str):
    try:
        match = re.search(r"(\d{4}|\d{2})", date_str)
        if match:
            year = match.group(0)
            if len(year) == 4 and int(year) < 1000:
                date_str = date_str.replace(year, str(1900 + int(year)))
        return date_str
    except Exception:
        return None
    

def filter_data(df):
    try:
        # Read CSV file
        
        # Ensure required columns are present
        required_columns = [
            "HORA_ATENCION",
            "FECHA_RESERVA",
            "HORA_RESERVA",
            "FECHA_NAC_PACIENTE",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return

        # Create a new column "Asistencia" to avoid any naming conflict
        df = df.with_columns(
            # Create "Asistencia" column
            pl.when(pl.col("HORA_ATENCION").is_not_null())
            .then(pl.lit("ASISTIDA"))
            .otherwise(pl.lit("ANULADA"))
            .alias("ESTADO")
        )

        # Parse dates and times
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


        # Extract date/time components
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

        # Filter columns for sidebar options
        cols = [col for col in df.columns if "ID_" not in col]
        sort = st.sidebar.selectbox("Sort column", cols, index=25)
        unique_vals = df[sort].unique().to_list()
        by = st.sidebar.selectbox("Sort value", unique_vals, index=0)
        cols2 = [x for x in cols if x != sort]
        var = st.sidebar.selectbox("V/S column", cols2, index=28)

        # Apply filter and display results
        if st.sidebar.button("Filter", type="primary"):
            df = make_bool(df=df, sort=sort, by=by, name=by)
            summary(df=df, sort=sort, var=var, by=by)
            pyg_app = StreamlitRenderer(
                dataset=df, default_tab="data", appearance="light"
            )
            pyg_app.explorer()

    except Exception as e:
        st.error(f"Error leyendo el archivo CSV: {e}")



def main():
    clear_page("CDM")
    st.markdown("# Exploración datos LostLess")

    df = None
    upload = st.sidebar.toggle("Upload Data?")

    if upload:
        file = st.sidebar.file_uploader(
            "Seleccionar archivo CSV",
            type=["csv"],
            accept_multiple_files=False,
            key="patients",
        )

        if file:
            df = pl.read_csv(file)
    
    else:
        df = pl.read_csv("../lostless_dataset/data/CDM.csv")

    if df is not None:
        filter_data(df)


if __name__ == "__main__":
    main()
