# -*- coding: utf-8 -*-
"""
@file    : 👋_Inicio.py
@brief   : Handles welcome page and code update
@date    : 2024/08/22
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import streamlit as st


def clear_page(title="GSP"):
    try:
        # im = Image.open('assets/logos/favicon.png')
        st.set_page_config(
            page_title=title,
            # page_icon=im,
            layout="wide",
        )

        # add_logo("assets/logos/ap75.png", height=75)

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

        # with open("./source/style.css") as f:
        #    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


clear_page("Procesar")


def main():

    st.write("# ¡Bienvenido/a a EEG Graph Classification! 👀👋")

    st.markdown(
        """
        Seleccione una de las opciones de la barra lateral.
        1. **Ciudad del Mar**.
        2. **Los Carrera**.
        """
    )


if __name__ == "__main__":
    main()
