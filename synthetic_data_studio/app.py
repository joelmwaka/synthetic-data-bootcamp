"""Synthetic Data Studio — Streamlit entry point.

Run from this folder (so ``.streamlit/config.toml`` binds to localhost)::

    uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

st.set_page_config(page_title="Synthetic Data Studio", page_icon=":material/train:", layout="wide")

pages = {
    "Data": [
        st.Page("views/browse.py", title="Browse", icon=":material/photo_library:", default=True),
        st.Page("views/inspect.py", title="Inspect", icon=":material/search:"),
    ],
    "Synthesis": [
        st.Page("views/generate.py", title="Generate", icon=":material/auto_awesome:"),
        st.Page("views/review.py", title="Review", icon=":material/fact_check:"),
        st.Page("views/batch.py", title="Batch", icon=":material/dynamic_feed:"),
    ],
    "Records": [
        st.Page("views/runs.py", title="Runs & export", icon=":material/inventory_2:"),
    ],
}
st.navigation(pages).run()
