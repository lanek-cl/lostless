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
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
import polars as pl

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
    return df.with_columns((df[sort] == by).alias(name))


def main():
    clear_page("LostLess")
    st.markdown("# Exploración datos LostLess")

    patients = st.sidebar.file_uploader(
        "Seleccionar archivo de pacientes",
        type=["csv"],
        accept_multiple_files=False,
        key="patients",
    )

    events = st.sidebar.file_uploader(
        "Seleccionar archivo de eventos",
        type=["csv"],
        accept_multiple_files=False,
        key="events",
    )

    if events and patients:
        try:
            # Read CSV files
            dfe = pl.read_csv(events, separator=";", ignore_errors=True)
            dfp = pl.read_csv(patients, separator=";", ignore_errors=True)

            # Drop nulls
            dfe = dfe.drop_nulls()
            dfp = dfp.drop_nulls()

            # Factorize 'ID_DE_PROFESIONAL' to create 'PROFESIONAL_ID'
            # dfe = dfe.with_columns([
            #    pl.col("ID_DE_PROFESIONAL").cast(pl.Categorical).cat.codes().alias("PROFESIONAL_ID")
            # ])

            # Merge dataframes on 'ID_PACIENTE' and 'ID'
            dfm = dfe.join(dfp, left_on="ID_PACIENTE", right_on="ID", how="right")
            # dfm = dfp.join(dfe, left_on="ID", right_on="ID_PACIENTE", how="left")
            dfm = dfm.drop_nulls()

            # Drop 'ID' column
            df = dfm.drop("ID")

            # Parse 'FECHA_DE_RESERVA' to datetime
            df = df.with_columns(
                [
                    pl.col("FECHA_DE_RESERVA").str.strptime(
                        pl.Datetime, "%d/%m/%Y %H:%M", strict=False
                    )
                ]
            )

            # Extract month, day name, and hour
            df = df.with_columns(
                [
                    pl.col("FECHA_DE_RESERVA").dt.month().alias("MES"),
                    pl.col("FECHA_DE_RESERVA").dt.strftime("%A").alias("DIA"),
                    pl.col("FECHA_DE_RESERVA").dt.hour().alias("HORA"),
                ]
            )

            cols = []
            for col in df.columns:
                if "ID_" not in col:
                    cols.append(col)
            sort = st.sidebar.selectbox("Sort column", cols, index=8)
            cols2 = [x for x in cols if x != sort]
            unique_vals = df[sort].unique()

            by = st.sidebar.selectbox("Sort value", unique_vals, index=1)

            cols2 = [x for x in cols if x != sort]

            var = st.sidebar.selectbox("V/S column", cols2, index=8)

            if st.sidebar.button("Filter", type="primary"):
                df = make_bool(df=df, sort=sort, by=by, name=by)
                summary(df=df, sort=sort, var=var, by=by)
                pyg_app = StreamlitRenderer(
                    dataset=df, default_tab="data", appearance="light"
                )
                pyg_app.explorer()

        except Exception as e:
            st.error(f"Error leyendo el archivo CSV: {e}")


if __name__ == "__main__":
    main()
