# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs lostless analysis using Polars
@date    : 2025/05/12
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

from datetime import datetime
import polars as pl
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

from functions import summary, make_bool, clear_page


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
            summary(df=df, sort=sort, var=var, by=by, ex=True)
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
