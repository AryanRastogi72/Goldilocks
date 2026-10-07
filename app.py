from pathlib import Path

import streamlit as st

from engine.search import SearchIndex
from ui import render_app


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Goldilocks Search",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_resource(show_spinner=False)
def load_index(index_path):
    return SearchIndex(index_path, max_cached_shards=1)


render_app(ROOT, load_index)
