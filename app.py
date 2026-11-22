"""Interface Streamlit do FluxoVC.

Execute com:  streamlit run app.py

Este arquivo contém apenas a interface; toda a lógica fica em ``src/``.
"""

from __future__ import annotations

import streamlit as st

from src import __version__
from src.logs import configurar_logging

configurar_logging()

st.set_page_config(page_title="FluxoVC", page_icon="🚦", layout="wide")

st.title("🚦 FluxoVC")
st.caption("Monitor de fluxo de veículos em vias urbanas — visão computacional")

st.info("Aplicação em construção. As próximas fases adicionam aquisição, detecção e visualização.")

st.sidebar.header("Configurações")
st.sidebar.caption(f"Versão {__version__}")
