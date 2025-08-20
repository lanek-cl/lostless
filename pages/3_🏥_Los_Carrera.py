# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs lostless analysis
@date    : 2025/04/29
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import polars as pl
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer


from functions import summary, make_bool, clear_page


def filter_data(dfe, dfp):
    try:
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


def main():
    clear_page("CLC")
    st.markdown("# Exploración datos LostLess")

    dfe = None
    dfp = None
    upload = st.sidebar.toggle("Upload Data?")

    if upload:

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
            dfe = pl.read_csv(events, separator=";", ignore_errors=True)
            dfp = pl.read_csv(patients, separator=";", ignore_errors=True)

    else:
        dfe = pl.read_csv(
            "../lostless_dataset/data/Eventos_v2.csv", separator=",", ignore_errors=True
        )
        dfp = pl.read_csv(
            "../lostless_dataset/data/Pacientes_v2.csv", separator=",", ignore_errors=True
        )
        


    if dfe is not None and dfp is not None:
        filter_data(dfe, dfp)


if __name__ == "__main__":
    main()
