import io
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64
from html import escape
from uuid import uuid4

try:
    from scipy.stats import pearsonr
    SCIPY_IMPORT_ERROR = None
except Exception as e:
    pearsonr = None
    SCIPY_IMPORT_ERROR = str(e)


OMNI_DATA_FILENAME = "1964-may 2026.txt"
@st.cache_data(show_spinner=False)
def load_local_omni_text(filename: str = OMNI_DATA_FILENAME) -> str:
    """Load the OMNI data file bundled next to app.py in the GitHub repo."""
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidate_paths = [base_dir / filename]

    cwd_candidate = Path.cwd() / filename
    if cwd_candidate not in candidate_paths:
        candidate_paths.append(cwd_candidate)

    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore")

    searched = "\n".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f'OMNI data file "{filename}" was not found. Upload it to the same folder as the running Python file.\n\nSearched:\n{searched}'
    )



def set_bg_image_auto():
    candidates = [
        "background.png",
    ]

    image_file = None
    for name in candidates:
        base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        p = base_dir / name
        if p.exists() and p.is_file():
            image_file = p
            break

    if image_file is None:
        return

    suffix = image_file.suffix.lower()
    mime = {
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".webp": "webp",
    }.get(suffix, "png")

    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/{mime};base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# ============================================================
# STREAMLIT APP
# Combines the 3 uploaded scripts into one website workflow:
#   1) Storm detection
#   2) Storm filtering for main-phase calculation
#   3) Data-complete storms + peak delays + peaks/integrals + correlations
#
# Input OMNI format expected (8 columns):
#   Year DOY Hour IMF Bz Vsw Ey Dst
#
# This app preserves the same analysis flow used in the uploaded scripts.
# ============================================================

st.set_page_config(page_title="Magnetic Storm Detection Program", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Move main content higher */
.block-container {
    padding-top: 0rem !important;  /* default is ~3rem */
    margin-top: -1.8rem !important;
}

/* Extra nudge for title and first widgets */
h1 {
    margin-top: -34px !important;
}



.guide-pdf-button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.35rem !important;
    padding: 0.45rem 0.75rem !important;
    margin: 0 0 0.45rem 0 !important;
    border-radius: 9px !important;
    border: 1px solid rgba(0,0,0,0.22) !important;
    background: rgba(255,255,255,0.92) !important;
    color: #111111 !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    text-decoration: none !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
}

.guide-pdf-button:hover {
    background: #ffffff !important;
    border-color: rgba(0,0,0,0.35) !important;
    color: #000000 !important;
    text-decoration: none !important;
}


/* Keep Streamlit Guide popover visually like the original small Guide button. */
div[data-testid="stPopover"] {
    width: fit-content !important;
    max-width: fit-content !important;
    display: inline-block !important;
    margin: 0 0 0.45rem 0 !important;
}

/* Streamlit adds a built-in down-chevron to popovers. Hide the whole
   internal button content and redraw only the original Guide label. */
div[data-testid="stPopover"] > button {
    width: auto !important;
    min-width: 0 !important;
    max-width: fit-content !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0.45rem 0.75rem !important;
    margin: 0 !important;
    border-radius: 9px !important;
    border: 1px solid rgba(0,0,0,0.22) !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #111111 !important;
    opacity: 1 !important;
    filter: none !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
    font-size: 0 !important;
    line-height: 1.1 !important;
}

div[data-testid="stPopover"] > button:hover,
div[data-testid="stPopover"] > button:focus,
div[data-testid="stPopover"] > button:active,
div[data-testid="stPopover"] > button[aria-expanded="true"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: rgba(0,0,0,0.35) !important;
    color: #111111 !important;
    opacity: 1 !important;
    filter: none !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
}

/* Remove Streamlit's original popover children, including the down arrow. */
div[data-testid="stPopover"] > button > * {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
}

/* Re-create the button text without Streamlit's chevron. */
div[data-testid="stPopover"] > button::before {
    content: "📘 Guide" !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #111111 !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    opacity: 1 !important;
    filter: none !important;
}

div[data-testid="stPopover"] > button:hover::before,
div[data-testid="stPopover"] > button:focus::before,
div[data-testid="stPopover"] > button:active::before,
div[data-testid="stPopover"] > button[aria-expanded="true"]::before {
    color: #111111 !important;
    opacity: 1 !important;
    filter: none !important;
}


</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
section[data-testid="stSidebar"] {
    width: 300px !important;
    flex: 0 0 300px !important;
}

section[data-testid="stSidebar"] > div {
    width: 300px !important;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* Remove Streamlit header anchor/action links everywhere */
[data-testid="stHeaderActionElements"],
.stHeadingActionElements,
.stHeaderActionElements,
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
    visibility: hidden !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Strong dimming on the BACKGROUND only */
.stApp {
    background-image:
        linear-gradient(rgba(0, 0, 0, 0.82), rgba(0, 0, 0, 0.82)),
        url("https://images.unsplash.com/photo-1473929735477-6e6f6b5c5b4a?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Keep main content above the background normally */
.block-container {
    position: relative;
    z-index: 1;
}

/* Yellow sidebar, but keep value fields white */
section[data-testid="stSidebar"] {
    background-color: #F8E463 !important;
}

section[data-testid="stSidebar"] > div {
    background-color: #F8E463 !important;
}

section[data-testid="stSidebar"] [data-baseweb="input"] {
    background: white !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background: white !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-baseweb="base-input"] > div,
section[data-testid="stSidebar"] input {
    background: white !important;
}

/* White metric cards */
div[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 10px !important;
    padding: 8px !important;
}

/* White expanders */
details {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 10px !important;
    padding: 6px !important;
}

/* White tables / dataframes */
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    background: #F0DC59 !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 10px !important;
    padding: 8px !important;
    width: 100% !important;
    overflow: hidden !important;
}

/* White inner frame */
div[data-testid="stDataFrame"] > div,
div[data-testid="stTable"] > div {
    background: white !important;
    border-radius: 8px !important;
    width: 100% !important;
    max-width: 100% !important;
}

/* Keep normal full-width behavior on normal screens */
div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"],
div[data-testid="stTable"] [data-testid="stDataFrameResizable"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;   /* keep the horizontal bar */
    overflow-y: hidden !important; /* remove the extra outer vertical bar */
}

/* Important: do NOT force fixed width; allow content to grow so scrollbars appear on smaller screens */
div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] > div,
div[data-testid="stTable"] [data-testid="stDataFrameResizable"] > div {
    min-width: max-content !important;
}

/* But on wider screens, let tables still occupy the full container width */
@media (min-width: 1100px) {
    div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] > div,
    div[data-testid="stTable"] [data-testid="stDataFrameResizable"] > div {
        width: 100% !important;
        min-width: 100% !important;
    }
}

/* White tab panels */
div[data-baseweb="tab-panel"] {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.10) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-top: 8px !important;
}

/* White content panels: needed after replacing st.tabs() with a single-section selector */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.10) !important;
    border-radius: 12px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: white !important;
    border-radius: 12px !important;
}

/* White chart wrappers */
div[data-testid="stPlotlyChart"],
div[data-testid="stPyplot"] {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 10px !important;
    padding: 6px !important;
}






/* Make tabs transparent again */
div[data-baseweb="tab-list"] {
    background: transparent !important;
    padding: 6px !important;
}

button[data-baseweb="tab"] {
    background: transparent !important;
    color: white !important;
    font-weight: 600 !important;
}

/* Active tab */
button[aria-selected="true"] {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    font-weight: 700 !important;
}

/* Improve readability */
button[data-baseweb="tab"] * {
    color: white !important;
    font-weight: 600 !important;
}

/* Line under tabs */
div[data-baseweb="tab-highlight"] {
    background: rgba(255,255,255,0.6) !important;
    height: 2px !important;
}



/* Add thin black border (stroke) to tab text */
button[data-baseweb="tab"] {
    -webkit-text-stroke: 0.5px black;
    text-stroke: 0.5px black;
}

/* Ensure children inherit it */
button[data-baseweb="tab"] * {
    -webkit-text-stroke: 0px black;
    text-stroke: 0px black;
}



/* Stronger tab text */
button[data-baseweb="tab"] {
    font-weight: 800 !important;   /* bold */
    font-size: 20px !important;    /* change this value to control size */
}

/* Ensure children inherit */
button[data-baseweb="tab"] * {
    font-weight: 800 !important;
    font-size: 20px !important;
}



/* Tabs: remove fill, add white border */
button[data-baseweb="tab"] {
    background: transparent !important;
    border: 2px solid transparent !important;
    border-radius: 10px !important;
    padding: 6px 10px !important;
}

/* Active tab: white border, no fill */
button[aria-selected="true"] {
    background: transparent !important;
    border: 3px solid white !important;  /* thickness here */
    color: white !important;
}

/* Optional: hover subtle border */
button[data-baseweb="tab"]:hover {
    border: 2px solid rgba(255,255,255,0.5) !important;
}



/* Remove underline highlight bar */
div[data-baseweb="tab-highlight"] {
    display: none !important;
}



/* Remove any tab underline / separator line completely */
div[data-baseweb="tab-highlight"] {
    display: none !important;
    height: 0 !important;
    border: none !important;
    background: transparent !important;
}

div[data-baseweb="tab-list"] {
    border-bottom: none !important;
    box-shadow: none !important;
}

div[data-baseweb="tab-border"] {
    display: none !important;
    border: none !important;
    background: transparent !important;
}

/* Safety: remove generic borders under tab wrappers */
div[data-testid="stTabs"] > div,
div[data-testid="stTabs"] div[role="tablist"] {
    border-bottom: none !important;
    box-shadow: none !important;
}



/* Separate white box for file uploader */
div[data-testid="stFileUploader"] {
    margin-top: -1.25rem !important;
    background: white !important;
    border: 1px solid rgba(0,0,0,0.10) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}

/* Keep uploader text readable */
div[data-testid="stFileUploader"] * {
    color: black !important;
    -webkit-text-stroke: 0 !important;
    text-stroke: 0 !important;
}

/* The drag-and-drop inner area */
div[data-testid="stFileUploader"] section {
    background: white !important;
    border-radius: 10px !important;
}



/* FORCE title white + keep the app title to exactly two lines on smaller screens. */
h1 {
    color: white !important;
    font-weight: 800 !important;
}

.app-main-title {
    color: white !important;
    font-weight: 800 !important;
    line-height: 1.08 !important;
    margin-top: -34px !important;
    margin-bottom: 0.35rem !important;
    font-size: clamp(1.55rem, 2.65vw, 2.75rem) !important;
}

.app-main-title .title-line {
    white-space: nowrap !important;
}


/* START button: thin white border */
div[data-testid="stButton"] > button[kind="primary"] {
    border: 2px solid white !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:focus,
div[data-testid="stButton"] > button[kind="primary"]:active {
    border: 2px solid white !important;
}


/* Compact header controls for smaller laptop screens. */
div[data-testid="stButton"] > button[kind="primary"] {
    white-space: nowrap !important;
    min-width: 86px !important;
    padding-left: 0.85rem !important;
    padding-right: 0.85rem !important;
}

div[data-testid="stButton"] > button[kind="primary"] p {
    white-space: nowrap !important;
}

div[data-testid="stFileUploader"] {
    padding: 8px 10px !important;
    margin-top: -1.45rem !important;
}

div[data-testid="stFileUploader"] label p {
    font-size: 0.86rem !important;
    line-height: 1.05 !important;
    margin-bottom: 0.2rem !important;
}

div[data-testid="stFileUploader"] section {
    min-height: 66px !important;
    padding: 0.35rem !important;
}

div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzone"] {
    padding: 0.35rem !important;
}

@media (max-width: 1250px) {
    .app-main-title {
        font-size: clamp(1.25rem, 2.45vw, 2.05rem) !important;
        line-height: 1.05 !important;
    }

    div[data-testid="stFileUploader"] {
        padding: 6px 8px !important;
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
}


</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Section selector replacement for tabs: keep the old transparent/white tab look.
   IMPORTANT: scope this ONLY to the main section selector, so normal table
   sorting radios keep the original single-dot radio-button appearance. */
.st-key-active_section div[data-testid="stRadio"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.st-key-active_section div[data-testid="stRadio"] > label {
    display: none !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: transparent !important;
    border: 2px solid transparent !important;
    border-radius: 10px !important;
    padding: 6px 10px !important;
    margin: 0 !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    -webkit-text-stroke: 0px black !important;
    text-stroke: 0px black !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    border: 2px solid rgba(255,255,255,0.5) !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    border: 3px solid white !important;
    color: white !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label * {
    color: white !important;
    font-weight: 800 !important;
    font-size: 16px !important;
}

/* Visible text size for the main section options (Overview, Extras, etc.). */
.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
    font-size: 22px !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    margin-bottom: 0 !important;
}

/* Hide the radio dots only for the main section selector. */

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Active section panel after replacing st.tabs(): force the old white tab-body background.
   Uses both the Streamlit key class when available and broad fallback selectors. */
.st-key-active_section_panel,
.st-key-active_section_panel > div,
.st-key-active_section_panel [data-testid="stVerticalBlock"],
.st-key-active_section_panel [data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-radius: 12px !important;
}

.st-key-active_section_panel {
    border: 1px solid rgba(0,0,0,0.14) !important;
    box-shadow: none !important;
    padding: 8px 16px 16px 16px !important;
    margin-top: 0px !important;
}

.st-key-active_section + div {
    margin-top: 0px !important;
    padding-top: 0px !important;
}

.st-key-active_section_panel > div {
    padding-top: 0px !important;
    margin-top: 0px !important;
}

/* Fallback for Streamlit versions where st.container(key=...) is not supported. */
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stTable"]),
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stDataFrame"]),
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h1),
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h2),
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h3) {
    background-color: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.14) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stTable"]) > div,
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stDataFrame"]) > div,
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h1) > div,
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h2) > div,
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h3) > div {
    background-color: #ffffff !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Keep ONLY table sorting radio options on one line on smaller laptop screens.
   The main section selector (Overview / Storm detection / etc.) keeps its original wrapping behavior. */
div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] {
    flex-wrap: nowrap !important;
    gap: 0.45rem !important;
    align-items: center !important;
}

div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    white-space: nowrap !important;
    min-width: max-content !important;
    padding-right: 0.40rem !important;
}

div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"],
div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
    white-space: nowrap !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
}

/* Sorting rows are intentionally compact; this prevents wrapping inside the sorting option column only. */
div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] {
    min-width: max-content !important;
}

/* Restore section selector wrapping explicitly, even when other radio styling is active. */
.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] {
    flex-wrap: wrap !important;
}

/* Table sorting radios: make ONLY the unselected circle outline a bit thicker.
   Keeps the circle size and selected red state unchanged. */
div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:not(:checked)) > div:first-child div {
    box-shadow: inset 0 0 0 1px currentColor !important;
}
</style>
""", unsafe_allow_html=True)






set_bg_image_auto()

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 10px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if pearsonr is None:
    st.error(
        "SciPy is not available in this Python environment, so Pearson p-values cannot be computed the same way as in your Colab file. Install SciPy in this exact environment, then rerun the app."
    )
    st.code('python -m pip install scipy', language='bash')
    if SCIPY_IMPORT_ERROR:
        st.caption(f"SciPy import error: {SCIPY_IMPORT_ERROR}")
    st.stop()


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class Row:
    t: datetime
    imf: float
    bz: float
    vsw: float
    ey: float
    dst: float


def dst_int(x: float) -> int:
    return int(round(float(x)))


def fmt_dt(dt) -> str:
    if dt is None or pd.isna(dt):
        return "NA"
    return pd.to_datetime(dt, utc=True).strftime("%Y-%m-%d %H:%M")


def parse_line_new8(line: str) -> Optional[Tuple[int, int, int, float, float, float, float, float]]:
    parts = line.strip().split()
    if not parts:
        return None
    if not parts[0].lstrip("+-").isdigit():
        return None
    if len(parts) < 8:
        return None
    parts = parts[:8]
    try:
        y = int(parts[0])
        doy = int(parts[1])
        hour = int(parts[2])
        imf = float(parts[3])
        bz = float(parts[4])
        vsw = float(parts[5])
        ey = float(parts[6])
        dst = float(parts[7])
        return y, doy, hour, imf, bz, vsw, ey, dst
    except ValueError:
        return None


def to_time_utc(year: int, doy: int, hour: int, shift_plus_1_hour: bool) -> datetime:
    t = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, hours=hour)
    if shift_plus_1_hour:
        t += timedelta(hours=1)
    return t


def parse_user_datetime(s: str, is_end: bool) -> datetime:
    s = str(s).strip()
    formats = ["%Y-%m-%d %H:%M", "%Y-%m-%d"]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            if fmt == "%Y-%m-%d":
                if is_end:
                    dt = dt + timedelta(hours=23)
            return dt
        except ValueError:
            pass
    raise ValueError(f"Invalid date format: {s!r}. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'.")


def filter_rows_by_date(rows: List[Row], date_start: str, date_end: str) -> List[Row]:
    t_start = parse_user_datetime(date_start, is_end=False)
    t_end = parse_user_datetime(date_end, is_end=True)
    if t_end < t_start:
        raise ValueError("DATE_END must be >= DATE_START.")
    return [r for r in rows if t_start <= r.t <= t_end]


def read_rows_new8_from_text(text: str, shift_plus_1_hour: bool) -> Tuple[List[Row], int]:
    rows: List[Row] = []
    bad = 0
    for line in text.splitlines():
        p = parse_line_new8(line)
        if p is None:
            bad += 1
            continue
        y, doy, hour, imf, bz, vsw, ey, dst = p
        rows.append(
            Row(
                t=to_time_utc(y, doy, hour, shift_plus_1_hour),
                imf=imf,
                bz=bz,
                vsw=vsw,
                ey=ey,
                dst=dst,
            )
        )
    rows.sort(key=lambda r: r.t)
    return rows, bad


# ---------------------------
# Stage 1: storm detection
# ---------------------------
def build_episodes_contiguous(rows: List[Row], episode_level: float) -> List[Tuple[int, int]]:
    episodes: List[Tuple[int, int]] = []
    in_ep = False
    start = 0

    for i, r in enumerate(rows):
        qualifies = r.dst <= episode_level

        if not in_ep:
            if qualifies:
                in_ep = True
                start = i
            continue

        prev = rows[i - 1]
        dt_hours = (r.t - prev.t).total_seconds() / 3600.0
        if (not qualifies) or (dt_hours > 1.0 + 1e-9):
            episodes.append((start, i - 1))
            in_ep = False
            if qualifies:
                in_ep = True
                start = i

    if in_ep:
        episodes.append((start, len(rows) - 1))
    return episodes


def find_episode_min_idx(rows: List[Row], a: int, b: int) -> int:
    imin = a
    for i in range(a, b + 1):
        if rows[i].dst < rows[imin].dst:
            imin = i
    return imin


def is_local_min_radius_stage1(rows: List[Row], i: int, a: int, b: int, radius_h: int) -> bool:
    lo = max(a, i - radius_h)
    hi = min(b, i + radius_h)
    v = rows[i].dst

    min_val = rows[lo].dst
    for j in range(lo + 1, hi + 1):
        if rows[j].dst < min_val:
            min_val = rows[j].dst

    if v != min_val:
        return False

    for j in range(lo, i):
        if rows[j].dst == min_val:
            return False
    return True


def storms_from_episodes(
    rows: List[Row],
    episodes_raw: List[Tuple[int, int]],
    minima_level: float,
    local_min_radius_hours: int,
    minima_split_factor: float,
    storm_limit: float,
) -> Tuple[List[Dict], List[Tuple[int, int]], List[Dict]]:
    storms: List[Dict] = []
    episodes_kept: List[Tuple[int, int]] = []
    multi_storm_details: List[Dict] = []

    new_ep_id = 0

    for (a, b) in episodes_raw:
        imin_global = find_episode_min_idx(rows, a, b)
        global_min = rows[imin_global].dst

        if global_min > minima_level:
            continue

        new_ep_id += 1
        episodes_kept.append((a, b))

        cand = []
        for i in range(a, b + 1):
            if rows[i].dst <= minima_level and is_local_min_radius_stage1(rows, i, a, b, local_min_radius_hours):
                cand.append(i)

        if not cand:
            cand = [imin_global]

        accepted = [cand[0]]
        separation_tests_used = []

        for nxt in cand[1:]:
            prev = accepted[-1]
            lo = min(prev, nxt)
            hi = max(prev, nxt)

            if hi - lo <= 1:
                if rows[nxt].dst < rows[prev].dst:
                    accepted[-1] = nxt
                continue

            inter_minimum_peak = max(rows[j].dst for j in range(lo + 1, hi))
            m1 = rows[prev].dst
            m2 = rows[nxt].dst
            lowest_min = min(m1, m2)
            higher_min = max(m1, m2)
            peak_height = inter_minimum_peak - higher_min
            separation_threshold = (-lowest_min) * minima_split_factor

            if peak_height >= separation_threshold:
                accepted.append(nxt)
                separation_tests_used.append(
                    {
                        "m1_idx": prev,
                        "m2_idx": nxt,
                        "m1": m1,
                        "m2": m2,
                        "lowest_min": lowest_min,
                        "higher_min": higher_min,
                        "inter_minimum_peak": inter_minimum_peak,
                        "peak_height": peak_height,
                        "separation_threshold": separation_threshold,
                    }
                )
            else:
                if rows[nxt].dst < rows[prev].dst:
                    accepted[-1] = nxt

        minima_indices = [idx for idx in accepted if rows[idx].dst <= minima_level]
        if not minima_indices:
            minima_indices = [imin_global]

        minima_indices = [idx for idx in minima_indices if rows[idx].dst >= storm_limit]
        if not minima_indices:
            continue

        if len(minima_indices) >= 2:
            multi_storm_details.append(
                {
                    "episode_id": new_ep_id,
                    "start": rows[a].t,
                    "end": rows[b].t,
                    "global_min": global_min,
                    "minima": [(rows[i].t, rows[i].dst, i) for i in minima_indices],
                    "separation_tests": separation_tests_used,
                }
            )

        for idx in minima_indices:
            r = rows[idx]
            storms.append(
                {
                    "episode_id": new_ep_id,
                    "tmin_utc": r.t,
                    "minDst": r.dst,
                    "row_index": idx,
                    "imf": r.imf,
                    "bz": r.bz,
                    "vsw": r.vsw,
                    "ey": r.ey,
                }
            )

    storms.sort(key=lambda s: s["tmin_utc"])
    for sid, s in enumerate(storms, start=1):
        s["storm_id"] = sid

    return storms, episodes_kept, multi_storm_details


def rows_to_df(rows: List[Row]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "t_utc": r.t,
                "IMF": r.imf,
                "Bz": r.bz,
                "Vsw": r.vsw,
                "Ey": r.ey,
                "Dst": r.dst,
            }
            for r in rows
        ]
    )


def storms_to_df(storms: List[Dict]) -> pd.DataFrame:
    if not storms:
        return pd.DataFrame(columns=["storm_id", "episode_id", "minDst", "tmin_utc"])
    out = []
    for s in storms:
        out.append(
            {
                "storm_id": s["storm_id"],
                "episode_id": s["episode_id"],
                "minDst": s["minDst"],
                "tmin_utc": s["tmin_utc"],
            }
        )
    return pd.DataFrame(out)


def episodes_to_df(rows: List[Row], episodes: List[Tuple[int, int]], storms: List[Dict]) -> pd.DataFrame:
    storm_count_by_episode: Dict[int, int] = {}
    for s in storms:
        ep_id = int(s.get("episode_id", 0))
        if ep_id > 0:
            storm_count_by_episode[ep_id] = storm_count_by_episode.get(ep_id, 0) + 1

    out = []
    for ep_id, (a, b) in enumerate(episodes, start=1):
        t0 = rows[a].t
        t1 = rows[b].t
        dur_h = (t1 - t0).total_seconds() / 3600.0 + 1.0
        imin = find_episode_min_idx(rows, a, b)
        global_min = rows[imin].dst
        out.append(
            {
                "episode_id": ep_id,
                "start_utc": t0,
                "end_utc": t1,
                "duration_hours": dur_h,
                "storm_count": int(storm_count_by_episode.get(ep_id, 0)),
                "global_min": dst_int(global_min),
                "t_min": rows[imin].t,
            }
        )
    return pd.DataFrame(out, columns=["episode_id", "start_utc", "end_utc", "duration_hours", "storm_count", "global_min", "t_min"])


def multi_storms_to_df(multi_storms: List[Dict]) -> pd.DataFrame:
    columns = [
        "episode_id",
        "start_utc",
        "end_utc",
        "duration_hours",
        "storm_count",
        "global_min",
        "storm_number",
        "Dst_min",
        "t_min",
    ]
    out = []
    for d in multi_storms:
        start_utc = d["start"]
        end_utc = d["end"]
        duration_hours = (end_utc - start_utc).total_seconds() / 3600.0 + 1.0
        storm_count = len(d.get("minima", []))
        for k, (t, v, _idx) in enumerate(d["minima"], start=1):
            out.append(
                {
                    "episode_id": d["episode_id"],
                    "start_utc": start_utc,
                    "end_utc": end_utc,
                    "duration_hours": duration_hours,
                    "storm_count": int(storm_count),
                    "global_min": dst_int(d["global_min"]),
                    "storm_number": k,
                    "Dst_min": dst_int(v),
                    "t_min": t,
                }
            )
    return pd.DataFrame(out, columns=columns)


# ---------------------------
# Stage 2: filtering
# ---------------------------
def is_local_max_radius(rows: List[Row], i: int, radius_h: int) -> bool:
    lo = max(0, i - radius_h)
    hi = min(len(rows) - 1, i + radius_h)
    v = float(rows[i].dst)

    max_val = float(rows[lo].dst)
    for j in range(lo + 1, hi + 1):
        if float(rows[j].dst) > max_val:
            max_val = float(rows[j].dst)

    if v != max_val:
        return False

    for j in range(i + 1, hi + 1):
        if float(rows[j].dst) == max_val:
            return False
    return True


def is_local_min_radius_stage2(rows: List[Row], i: int, radius_h: int) -> bool:
    lo = max(0, i - radius_h)
    hi = min(len(rows) - 1, i + radius_h)
    v = float(rows[i].dst)

    min_val = float(rows[lo].dst)
    for j in range(lo + 1, hi + 1):
        if float(rows[j].dst) < min_val:
            min_val = float(rows[j].dst)

    if v != min_val:
        return False

    for j in range(i + 1, hi + 1):
        if float(rows[j].dst) == min_val:
            return False
    return True


def calculate_weak_mark_factor(strong_storm_threshold: float, strong_mark: float) -> float:
    if abs(float(strong_storm_threshold)) < 1e-12:
        return float("nan")
    return float(strong_mark) / float(strong_storm_threshold)


def compute_mark(minDst: float, strong_storm_threshold: float, strong_mark: float) -> float:
    weak_mark_factor = calculate_weak_mark_factor(strong_storm_threshold, strong_mark)
    if minDst <= strong_storm_threshold:
        return strong_mark
    return minDst * weak_mark_factor


def find_tstart_simple(
    rows: List[Row],
    imin: int,
    minDst: float,
    radius_max: int,
    strong_storm_threshold: float,
    strong_mark: float,
) -> Tuple[Optional[int], float]:
    mark = compute_mark(minDst, strong_storm_threshold, strong_mark)

    if imin <= 0:
        return None, mark

    reached_mark = False

    for i in range(imin - 1, -1, -1):
        v = float(rows[i].dst)
        if not reached_mark:
            if v >= mark:
                reached_mark = True
            else:
                continue
        if is_local_max_radius(rows, i, radius_max):
            return i, mark

    return None, mark


def stage2_filter(
    rows: List[Row],
    storms: List[Dict],
    multi_storm_episodes: List[Dict],
    strong_storm_threshold: float,
    strong_mark: float,
    local_max_radius_hours: int,
    disturbance_filter: bool,
    disturbance_level: float,
    disturb_dip_count: int,
    disturb_dip_radius_hours: int,
    boundary_storm_threshold: float,
):
    multi_storm_episode_ids = {d["episode_id"] for d in multi_storm_episodes}
    storms_ordered = sorted(storms, key=lambda s: (s["tmin_utc"], int(s["row_index"]), int(s["storm_id"])))

    kept = []
    excluded = []
    disturbances = []
    previous_all_storms = []

    previous_storm_excluded_count = 0
    left_censored_boundary_count = 0
    no_tstart_skipped_count = 0

    left_censored_episode_id = None
    if rows and storms_ordered and float(rows[0].dst) <= float(boundary_storm_threshold):
        left_censored_episode_id = int(storms_ordered[0]["episode_id"])

    for s in storms_ordered:
        imin = int(s["row_index"])
        minDst = float(s["minDst"])
        tmin = s["tmin_utc"]
        ep_id = int(s["episode_id"])
        is_multi_storm = ep_id in multi_storm_episode_ids

        # If the selected date range starts while Dst is already below the
        # storm threshold, the first detected storm/episode is left-censored:
        # its onset may lie before the selected interval, so it is excluded
        # from the clean main-phase analysis. It is not shown in the visible
        # "previous storm" exclusion table.
        if left_censored_episode_id is not None and ep_id == left_censored_episode_id:
            left_censored_boundary_count += 1
            previous_all_storms.append(s)
            continue

        istart, mark = find_tstart_simple(
            rows,
            imin,
            minDst,
            local_max_radius_hours,
            strong_storm_threshold,
            strong_mark,
        )

        # Non-boundary storms without a valid tstart cannot enter the clean
        # main-phase sample, but they are no longer reported as a separate
        # visible exclusion reason.
        if istart is None:
            no_tstart_skipped_count += 1
            previous_all_storms.append(s)
            continue

        tstart = rows[istart].t
        dst_start = float(rows[istart].dst)
        mainphase_duration = (tmin - tstart).total_seconds() / 3600.0

        blocking_storm = None
        for ps in previous_all_storms:
            prev_tmin = ps["tmin_utc"]
            if tstart <= prev_tmin < tmin:
                blocking_storm = ps
                break

        if blocking_storm is not None:
            previous_storm_excluded_count += 1
            excluded.append(
                {
                    **s,
                    "is_multi_storm": is_multi_storm,
                    "mark": mark,
                    "tstart_utc": tstart,
                    "dst_start": dst_start,
                    "mainphase_duration": mainphase_duration,
                    "blocking_storm_id": blocking_storm["storm_id"],
                    "blocking_storm_tmin_utc": blocking_storm["tmin_utc"],
                    "reason": "excluded_previous_storm_between_tstart_and_tmin",
                }
            )
            previous_all_storms.append(s)
            continue

        if disturbance_filter and minDst > disturbance_level:
            secondary_dip_count = 0
            disturb_dip_index = None
            disturb_dip_dst = None
            disturb_dip_time = None

            for i in range(imin - 1, istart, -1):
                if is_local_min_radius_stage2(rows, i, disturb_dip_radius_hours):
                    secondary_dip_count += 1
                    if disturb_dip_index is None:
                        disturb_dip_index = i
                        disturb_dip_dst = float(rows[i].dst)
                        disturb_dip_time = rows[i].t

            if secondary_dip_count >= disturb_dip_count:
                disturbances.append(
                    {
                        **s,
                        "is_multi_storm": is_multi_storm,
                        "mark": mark,
                        "tstart_utc": tstart,
                        "dst_start": dst_start,
                        "mainphase_duration": mainphase_duration,
                        "disturb_dip_count": secondary_dip_count,
                        "disturb_dip_index": disturb_dip_index,
                        "disturb_dip_utc": disturb_dip_time,
                        "disturb_dip_dst": disturb_dip_dst,
                        "reason": f"disturbance_dipcount>={disturb_dip_count}_before_tstart_dipradius{disturb_dip_radius_hours}h",
                    }
                )
                previous_all_storms.append(s)
                continue

        kept.append(
            {
                **s,
                "is_multi_storm": is_multi_storm,
                "mark": mark,
                "tstart_utc": tstart,
                "dst_start": dst_start,
                "mainphase_duration": mainphase_duration,
            }
        )
        previous_all_storms.append(s)

    summary = {
        "total_stage1_storms": len(storms),
        "kept_storms": len(kept),
        "disturbances": len(disturbances),
        "excluded_previous_storm": previous_storm_excluded_count,
        "left_censored_boundary": left_censored_boundary_count,
        "skipped_no_tstart": no_tstart_skipped_count,
        "avg_mainphase_kept": float(np.mean([x["mainphase_duration"] for x in kept])) if kept else np.nan,
    }

    return kept, disturbances, excluded, summary


# ---------------------------
# Stage 3: Data-complete storms + metrics
# ---------------------------
def is_missing(val) -> bool:
    if val is None:
        return True
    try:
        v = float(val)
    except Exception:
        return True
    return (
        abs(v - 999.9) < 1e-6
        or abs(v - 999.99) < 1e-6
        or abs(v - 9999.0) < 1e-6
    )


def pearson_safe(x: pd.Series, y: pd.Series):
    df = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    n = len(df)
    if n < 3:
        return np.nan, np.nan, n
    if df.iloc[:, 0].nunique() < 2 or df.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, n
    r, p = pearsonr(df.iloc[:, 0].values, df.iloc[:, 1].values)
    return float(r), float(p), n


def sem(series) -> float:
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    n = len(s)
    if n <= 1:
        return np.nan
    return float(s.std(ddof=1) / np.sqrt(n))


def class_name(minDst: float) -> str:
    if pd.isna(minDst):
        return "other"
    if minDst <= -250.0:
        return "super-storm (Dst_min <= -250)"
    if minDst <= -100.0:
        return "intense (-250 < Dst_min <= -100)"
    if minDst <= -50.0:
        return "moderate (-100 < Dst_min <= -50)"
    return "other"


def class_order_key(name: str) -> int:
    if str(name).startswith("moderate"):
        return 0
    if str(name).startswith("intense"):
        return 1
    if str(name).startswith("super-storm"):
        return 2
    return 99


def count_by_class(df: pd.DataFrame, count_label: str) -> pd.DataFrame:
    classes = [
        "moderate (-100 < Dst_min <= -50)",
        "intense (-250 < Dst_min <= -100)",
        "super-storm (Dst_min <= -250)",
    ]
    out = pd.DataFrame({"class": classes})
    if df.empty or "minDst" not in df.columns:
        out[count_label] = 0
        return out

    tmp = df.copy()
    tmp["class"] = tmp["minDst"].apply(class_name)
    tmp = tmp[tmp["class"] != "other"]

    if tmp.empty:
        out[count_label] = 0
        return out

    grp = tmp.groupby("class", dropna=False).size().reset_index(name=count_label)
    out = out.merge(grp, on="class", how="left")
    out[count_label] = out[count_label].fillna(0).astype(int)
    out["__order"] = out["class"].map(class_order_key)
    out = out.sort_values(["__order", "class"]).drop(columns="__order")
    return out


def summarize_by_class(df: pd.DataFrame, count_label: str, avg_cols: List[Tuple[str, str]]) -> pd.DataFrame:
    classes = [
        "moderate (-100 < Dst_min <= -50)",
        "intense (-250 < Dst_min <= -100)",
        "super-storm (Dst_min <= -250)",
    ]
    out = pd.DataFrame({"class": classes})

    if df.empty or "class" not in df.columns:
        out[count_label] = 0
        for _, label in avg_cols:
            out[label] = 0
        return out

    tmp = df.copy()
    tmp = tmp[tmp["class"] != "other"]

    if tmp.empty:
        out[count_label] = 0
        for _, label in avg_cols:
            out[label] = 0
        return out

    grouped = tmp.groupby("class", dropna=False)
    grp = grouped.size().reset_index(name=count_label)
    out = out.merge(grp, on="class", how="left")

    for src_col, label in avg_cols:
        if src_col in tmp.columns:
            means = grouped[src_col].mean().reset_index(name=label)
            out = out.merge(means, on="class", how="left")
        else:
            out[label] = 0

    out[count_label] = out[count_label].fillna(0).astype(int)
    for _, label in avg_cols:
        out[label] = out[label].fillna(0)

    out["__order"] = out["class"].map(class_order_key)
    out = out.sort_values(["__order", "class"]).drop(columns="__order")
    return out


def correlations_selected(df_metrics: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        ("bzp", "Bzp"),
        ("eyp", "Eyp"),
        ("eyi", "Eyi"),
    ]
    rows = []
    if df_metrics.empty:
        return pd.DataFrame(
            [{"Parameter vs |Dst_min|": label, "r": 0, "n": 0} for _, label in wanted]
        )
    for src, label in wanted:
        if src not in df_metrics.columns:
            rows.append({"Parameter vs |Dst_min|": label, "r": 0, "n": 0})
            continue
        r, _p, n = pearson_safe(df_metrics[src], df_metrics["abs_minDst"])
        rows.append({"Parameter vs |Dst_min|": label, "r": 0 if pd.isna(r) else r, "n": n})
    return pd.DataFrame(rows)



def format_val(label, val):
    try:
        label_l = str(label).lower()
        # treat anything that looks like a count as integer
        if any(k in label_l for k in ["count", "total storms", "storms", "episodes"]):
            return "—" if pd.isna(val) else str(int(round(float(val))))
        # if actual numeric integer value
        if isinstance(val, (int,)) or (isinstance(val, float) and float(val).is_integer()):
            return "—" if pd.isna(val) else str(int(val))
        return f"{float(val):.3f}"
    except:
        return str(val)


def format_mean_pm_error(mean_val, err_val, digits: int = 2) -> str:
    if mean_val is None or pd.isna(mean_val):
        return "—"
    if err_val is None or pd.isna(err_val):
        return f"{float(mean_val):.{digits}f}"
    return f"{float(mean_val):.{digits}f} ± {float(err_val):.{digits}f}"
def pretty_value(x):
    if x is None:
        return "—"
    if isinstance(x, (pd.Timestamp, datetime)):
        return fmt_dt(x)
    if isinstance(x, str):
        xs = x.strip()
        if xs:
            dt = pd.to_datetime(xs, utc=True, errors="coerce")
            if not pd.isna(dt):
                return fmt_dt(dt)
        return x
    if pd.isna(x):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        if abs(float(x) - round(float(x))) < 1e-9:
            return f"{int(round(float(x))):,}"
        return f"{float(x):.3f}"
    return x






def build_stage3_peak_delays_table(clean_storms_df: pd.DataFrame, peak_delays_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["storm_id", "minDst", "tmin_utc", "mainphase_duration", "tmin-tbzp", "tmin-teyp", "abs(tbzp-teyp)"]
    if peak_delays_df is None or peak_delays_df.empty:
        return pd.DataFrame(columns=cols)

    df = peak_delays_df.copy()
    if "mainphase_duration" not in df.columns and "mainphase_duration" in df.columns:
        df["mainphase_duration"] = df["mainphase_duration"]
    if "tmin-tbzp" not in df.columns and "delay_bzp_to_tmin_hours" in df.columns:
        df["tmin-tbzp"] = df["delay_bzp_to_tmin_hours"]
    if "tmin-teyp" not in df.columns and "delay_eyp_to_tmin_hours" in df.columns:
        df["tmin-teyp"] = df["delay_eyp_to_tmin_hours"]

    if "abs(tbzp-teyp)" not in df.columns:
        if "t_eyp_utc" in df.columns and "t_bzp_utc" in df.columns:
            te = pd.to_datetime(df["t_eyp_utc"], utc=True, errors="coerce")
            tb = pd.to_datetime(df["t_bzp_utc"], utc=True, errors="coerce")
            df["abs(tbzp-teyp)"] = (tb - te).abs().dt.total_seconds() / 3600.0
        else:
            df["abs(tbzp-teyp)"] = np.nan

    desired = [c for c in cols if c in df.columns]
    return df[desired].copy()


def build_stage3_full_peak_data_table(clean_storms_df: pd.DataFrame, clean_rows_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["storm_id", "minDst", "tstart_utc", "tmin_utc", "timfp", "tbzp", "tspeedp", "teyp"]
    if clean_storms_df is None or clean_storms_df.empty:
        return pd.DataFrame(columns=cols)
    if clean_rows_df is None or clean_rows_df.empty:
        out = clean_storms_df.copy()
        keep = [c for c in ["storm_id", "minDst", "tstart_utc", "tmin_utc"] if c in out.columns]
        out = out[keep].copy()
        for c in ["timfp", "tbzp", "tspeedp", "teyp"]:
            out[c] = pd.NaT
        return out[cols]

    rows = clean_rows_df.copy()
    rows["t_utc"] = pd.to_datetime(rows["t_utc"], utc=True, errors="coerce")
    for c in ["IMF", "Bz", "Vsw", "Ey"]:
        if c in rows.columns:
            rows[c] = pd.to_numeric(rows[c], errors="coerce")

    out_rows = []
    for _, s in clean_storms_df.iterrows():
        sid = int(s["storm_id"])
        minDst = s["minDst"] if "minDst" in s else np.nan
        tstart = pd.to_datetime(s["tstart_utc"], utc=True, errors="coerce")
        tmin = pd.to_datetime(s["tmin_utc"], utc=True, errors="coerce")
        g = rows[rows["storm_id"] == sid].copy()
        if not pd.isna(tstart):
            g = g[g["t_utc"] >= tstart]
        if not pd.isna(tmin):
            g = g[g["t_utc"] <= tmin]

        timfp = pd.NaT
        tbzp = pd.NaT
        tspeedp = pd.NaT
        teyp = pd.NaT

        if not g.empty:
            if "IMF" in g.columns and g["IMF"].notna().any():
                timfp = g.loc[g["IMF"].idxmax(), "t_utc"]
            if "Bz" in g.columns and g["Bz"].notna().any():
                tbzp = g.loc[g["Bz"].idxmin(), "t_utc"]
            if "Vsw" in g.columns and g["Vsw"].notna().any():
                tspeedp = g.loc[g["Vsw"].idxmax(), "t_utc"]
            if "Ey" in g.columns and g["Ey"].notna().any():
                eyinj = g["Ey"].clip(lower=0)
                if eyinj.notna().any():
                    teyp = g.loc[eyinj.idxmax(), "t_utc"]

        out_rows.append({
            "storm_id": sid,
            "minDst": minDst,
            "tstart_utc": tstart,
            "tmin_utc": tmin,
            "timfp": timfp,
            "tbzp": tbzp,
            "tspeedp": tspeedp,
            "teyp": teyp,
        })

    return pd.DataFrame(out_rows, columns=cols)







def build_all_missing_values_table(omni_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["row_index", "parameter", "t_utc", "value"]
    if omni_df is None or omni_df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for idx, r in omni_df.iterrows():
        for param, sentinel in [("IMF", 999.9), ("Bz", 999.9), ("Vsw", 9999.0), ("Ey", 999.99)]:
            val = r.get(param)
            if pd.isna(val) or (val == sentinel):
                rows.append({
                    "row_index": idx,
                    "parameter": param,
                    "t_utc": r.get("t_utc"),
                    "value": val
                })

    return pd.DataFrame(rows, columns=cols)

def build_stage3_rejected_quality_table(rejected_quality_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["storm_id", "minDst", "tmin_utc", "missing_value_hours", "failed_parameters"]
    if rejected_quality_df is None or rejected_quality_df.empty:
        return pd.DataFrame(columns=cols)
    df = rejected_quality_df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c == "failed_parameters" else np.nan
    return df[cols].copy()

def build_stage3_replacements_table(replacements_df: pd.DataFrame, clean_storms_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["storm_id", "minDst", "parameter", "t_utc", "error_value", "replaced_value"]
    if replacements_df is None or replacements_df.empty:
        return pd.DataFrame(columns=cols)

    df = replacements_df.copy()

    # --- normalize column names ---
    rename_map = {}

    # time
    if "tutc" in df.columns:
        rename_map["tutc"] = "t_utc"

    # parameter
    if "param" in df.columns:
        rename_map["param"] = "parameter"

    # original value → error_value
    if "orig_value" in df.columns:
        rename_map["orig_value"] = "error_value"
    elif "original_value" in df.columns:
        rename_map["original_value"] = "error_value"

    # replaced value
    if "new_value" in df.columns:
        rename_map["new_value"] = "replaced_value"

    # row index
    if "idx" in df.columns:
        rename_map["idx"] = "row_index"

    df = df.rename(columns=rename_map)

    # attach minDst if missing
    if "minDst" not in df.columns and "storm_id" in df.columns:
        if clean_storms_df is not None and not clean_storms_df.empty:
            df = df.merge(clean_storms_df[["storm_id", "minDst"]], on="storm_id", how="left")

    # ensure all columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NaT if c == "t_utc" else None

    return df[cols].copy()


def build_stage3_all_clean_storms_table(clean_storms_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    if clean_storms_df is None or clean_storms_df.empty:
        return pd.DataFrame(columns=[
            "storm_id", "minDst", "tmin_utc", "dst_start", "tstart_utc", "mainphase_duration",
            "imf_peak", "bz_peak", "speed_peak", "ey_peak", "eyi"
        ])

    base_cols = ["storm_id", "minDst", "tmin_utc", "dst_start", "tstart_utc"]
    base = clean_storms_df.copy()

    # Normalize main phase column name for display
    if "mainphase_duration" not in base.columns and "mainphase_duration" in base.columns:
        base["mainphase_duration"] = base["mainphase_duration"]

    keep_base = [c for c in ["storm_id", "minDst", "tmin_utc", "dst_start", "tstart_utc", "mainphase_duration"] if c in base.columns]
    base = base[keep_base].copy()

    if metrics_df is not None and not metrics_df.empty:
        metrics_keep = [c for c in ["storm_id", "imf_peak", "bz_peak", "speed_peak", "ey_peak", "eyi"] if c in metrics_df.columns]
        metrics_part = metrics_df[metrics_keep].copy()
        merged = base.merge(metrics_part, on="storm_id", how="left")
    else:
        merged = base.copy()
        for c in ["imf_peak", "bz_peak", "speed_peak", "ey_peak", "eyi"]:
            if c not in merged.columns:
                merged[c] = np.nan

    desired = ["storm_id", "minDst", "tmin_utc", "dst_start", "tstart_utc", "mainphase_duration",
               "imf_peak", "bz_peak", "speed_peak", "ey_peak", "eyi"]
    desired = [c for c in desired if c in merged.columns]
    rest = [c for c in merged.columns if c not in desired]
    return merged[desired + rest]

def prettify_table_headers(df: pd.DataFrame, correlation_first_col_name: str = None) -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    rename_map = {}
    for i, col in enumerate(out.columns):
        col_str = "" if col is None else str(col)
        col_clean = col_str.strip().lower().replace("_", " ")
        col_key = col_str.strip().lower()

        if col_clean in {"mainphase duration", "main phase duration"}:
            rename_map[col] = "Mp duration"
        elif col_clean in {"dst start", "dst_start"}:
            rename_map[col] = "Dst_start"
        elif col_key == "mindst" or col_clean == "mindst":
            rename_map[col] = "Dst_min"
        elif col_key in {"tstart_utc", "t_start"} or col_clean in {"tstart utc", "t start"}:
            rename_map[col] = "t_start"
        elif col_key in {"tmin_utc", "tmin", "t_min"} or col_clean in {"tmin utc", "tmin", "t min"}:
            rename_map[col] = "t_min"
        elif col_key in {"tbzp", "t_bzp"} or col_clean in {"tbzp", "t bzp"}:
            rename_map[col] = "t_Bzp"
        elif col_key in {"timfp", "t_imfp"} or col_clean in {"timfp", "t imfp"}:
            rename_map[col] = "t_IMFp"
        elif col_key in {"teyp", "t_eyp"} or col_clean in {"teyp", "t eyp"}:
            rename_map[col] = "t_Eyp"
        elif col_key in {"tspeedp", "t_speedp", "t_vp"} or col_clean in {"tspeedp", "t speedp", "t vp"}:
            rename_map[col] = "t_Vp"
        elif col_key in {"tmin-tbzp", "t_min-t_bzp"}:
            rename_map[col] = "t_min-t_Bzp"
        elif col_key in {"tmin-teyp", "t_min-t_eyp"}:
            rename_map[col] = "t_min-t_Eyp"
        elif col_key in {"abs(tbzp-teyp)", "abs(t_bzp-t_eyp)"}:
            rename_map[col] = "abs(t_Bzp-t_Eyp)"
        elif col_clean == "parameter vs |mindst|":
            rename_map[col] = "Parameter vs |Dst_min|"
        elif col_key in {"imfp", "imf_peak"} or col_clean == "imf peak":
            rename_map[col] = "IMFp"
        elif col_key in {"bzp", "bz_peak"} or col_clean == "bz peak":
            rename_map[col] = "Bzp"
        elif col_key in {"eyp", "ey_peak"} or col_clean == "ey peak":
            rename_map[col] = "Eyp"
        elif col_key in {"speedp", "speed_peak"} or col_clean in {"speed peak", "vsw peak"}:
            rename_map[col] = "Vp"
        elif col_key == "eyi" or col_clean == "ey integral":
            rename_map[col] = "Eyi"
        elif col_clean == "date":
            rename_map[col] = "Date"
        elif col_clean == "n":
            rename_map[col] = "N"
        elif col_clean == "r":
            rename_map[col] = "R"
        elif col_clean == "p":
            rename_map[col] = "p-value"
        elif correlation_first_col_name and i == 0 and col_clean in {"", "unnamed: 0", "metric"}:
            rename_map[col] = correlation_first_col_name

    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def format_corr_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()

    # enforce desired order
    desired_order = ["imf_peak", "bz_peak", "ey_peak", "eyi", "speed_peak", "mainphase_duration"]
    order_map = {k:i for i,k in enumerate(desired_order)}
    if df.shape[1] > 0:
        df["__order"] = df.iloc[:,0].map(lambda x: order_map.get(str(x).lower(), 999))
        df = df.sort_values("__order").drop(columns="__order")
    # remove delay correlations
    df = df[~df.iloc[:,0].astype(str).str.contains("delay", case=False, na=False)]
    if df.empty:
        return df
    show_df = df.copy()
    if "r" in show_df.columns:
        show_df["r"] = show_df["r"].map(lambda v: "—" if pd.isna(v) else f"{float(v):.3f}")
    if "p" in show_df.columns:
        show_df["p"] = show_df["p"].map(lambda v: "—" if pd.isna(v) else f"{float(v):.2e}")
    if "n" in show_df.columns:
        show_df["n"] = show_df["n"].map(lambda v: "—" if pd.isna(v) else str(int(v)))
    for col in show_df.columns:
        if col not in {"r", "p", "n"}:
            show_df[col] = show_df[col].map(pretty_value)
    if show_df.shape[1] > 0:
        first_col = show_df.columns[0]
        show_df[first_col] = show_df[first_col].replace({
            "mainphase_duration": "Mp duration",
            "main phase duration": "Mp duration",
            "Main phase duration": "Mp duration",
            "minDst": "Dst_min",
            "Dst_min": "Dst_min",
            "imfp": "IMFp",
            "imf_peak": "IMFp",
            "bzp": "Bzp",
            "bz_peak": "Bzp",
            "eyp": "Eyp",
            "ey_peak": "Eyp",
            "speedp": "Vp",
            "Speedp": "Vp",
            "speed_peak": "Vp",
            "eyi": "Eyi",
            "tmin": "t_min",
            "tmin_utc": "t_min",
            "tstart_utc": "t_start",
            "tbzp": "t_Bzp",
            "timfp": "t_IMFp",
            "teyp": "t_Eyp",
            "tspeedp": "t_Vp",
        })
    show_df = prettify_table_headers(show_df, correlation_first_col_name="Parameter vs |Dst_min|")
    return show_df




def _static_table_cell_html(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "&mdash;"
    text_value = escape(str(value))
    return text_value.replace("\n", "<br>")


def render_static_scroll_table(df: pd.DataFrame, max_height: int = 420, key: Optional[str] = None, fit_to_container: bool = False, equal_col_widths: bool = False):
    if df is None or df.empty:
        st.info("No data.")
        return

    table_id = key or f"static_table_{uuid4().hex}"
    table_width_css = "100%" if fit_to_container else "max-content"
    table_min_width_css = "100%" if fit_to_container else "100%"
    table_layout_css = "fixed" if (fit_to_container and equal_col_widths) else "auto"
    col_count = max(1, len(df.columns))
    col_width_css = f"{100.0 / col_count:.6f}%" if (fit_to_container and equal_col_widths) else "auto"
    header_html = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)

    body_rows = []
    for row in df.itertuples(index=False, name=None):
        cells = "".join(f"<td>{_static_table_cell_html(value)}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "".join(body_rows)

    html = f"""
    <style>
    div#{table_id}_outer {{
        width: 100%;
        max-width: 100%;
        background: #white;
        border: 1px solid #000000;
        border-radius: 10px;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.05);
        padding: 0;
        overflow: hidden;
    }}

    div#{table_id}_scroll {{
        width: 100%;
        max-width: 100%;
        background: transparent;
        border-radius: 7px;
        overflow-x: auto;
        overflow-y: auto;
        max-height: {max_height}px;
    }}

    table#{table_id} {{
        border-collapse: separate;
        border-spacing: 0;
        width: {table_width_css};
        min-width: {table_min_width_css};
        table-layout: {table_layout_css};
        font-size: 0.95rem;
        background: white;
    }}

    table#{table_id} thead th {{
        position: sticky;
        top: 0;
        z-index: 2;
        background: white;
        color: black;
        font-weight: 600;
        cursor: default !important;
        pointer-events: none !important;
        user-select: none !important;
    }}

    table#{table_id} th,
    table#{table_id} td {{
        width: {col_width_css};
        max-width: {col_width_css};
        padding: 0.45rem 0.65rem;
        text-align: left;
        white-space: nowrap;
        border-bottom: 1px solid #e5e7eb;
        border-right: 1px solid #f3f4f6;
        background: white;
    }}

    table#{table_id} th:last-child,
    table#{table_id} td:last-child {{
        border-right: none;
    }}

    table#{table_id} tbody tr:last-child td {{
        border-bottom: 1px solid #e5e7eb;
    }}

    table#{table_id} tbody tr:nth-child(even) td {{
        background: #fafafa;
    }}
    </style>
    <div id="{table_id}_outer">
        <div id="{table_id}_scroll">
            <table id="{table_id}">
                <thead>
                    <tr>{header_html}</tr>
                </thead>
                <tbody>
                    {body_html}
                </tbody>
            </table>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('<div style="height:30px"></div>', unsafe_allow_html=True)


def render_table(title: str, df: pd.DataFrame, one_decimal_columns: Optional[set] = None):
    if df.empty:
        st.info("No data.")
    else:
        show_df = df.copy()
        one_decimal_columns = {str(c).lower() for c in (one_decimal_columns or set())}
        for col in show_df.columns:
            col_l = str(col).lower()
            if col_l in one_decimal_columns and pd.api.types.is_numeric_dtype(show_df[col]):
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):.1f}")
            elif "avg" in col_l and pd.api.types.is_numeric_dtype(show_df[col]):
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):.2f}")
            else:
                show_df[col] = show_df[col].map(pretty_value)
        show_df = prettify_table_headers(show_df)
        render_static_scroll_table(show_df, key=f"{title.lower().replace(' ', '_')}_{uuid4().hex}")



def render_table_stage3_all_clean(title: str, df: pd.DataFrame):
    if df.empty:
        st.info("No data.")
    else:
        show_df = df.copy()
        float_cols_2dec = {"bzp", "imfp", "eyp", "bz_peak", "imf_peak", "ey_peak", "eyi"}
        for col in show_df.columns:
            col_l = str(col).lower()
            if pd.api.types.is_datetime64_any_dtype(show_df[col]):
                show_df[col] = pd.to_datetime(show_df[col], utc=True, errors="coerce").map(fmt_dt)
            elif col_l in float_cols_2dec:
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):.2f}")
            elif col_l in {"speedp", "speed_peak"}:
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else str(int(round(float(x)))))
            elif col == "storm_id":
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else f"{int(x)}")
            else:
                show_df[col] = show_df[col].map(pretty_value)
        show_df = prettify_table_headers(show_df)
        render_static_scroll_table(show_df, key=f"{title.lower().replace(' ', '_')}_{uuid4().hex}")


def sort_stage1_storm_minima(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    out = df.copy()
    if order_choice.lower() in ("time", "date") and "tmin_utc" in out.columns:
        return out.sort_values("tmin_utc", ascending=True).reset_index(drop=True)
    if order_choice in ("minDst", "Dst_min") and "minDst" in out.columns:
        # stronger -> weaker means more negative -> less negative
        return out.sort_values("minDst", ascending=True).reset_index(drop=True)
    return out.reset_index(drop=True)


def sort_stage1_episodes(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    out = df.copy()

    if order_choice.lower() in ("time", "date", "data"):
        time_col = None
        for candidate in ["start_utc", "start", "episode_start", "start_time"]:
            if candidate in out.columns:
                time_col = candidate
                break
        if time_col:
            return out.sort_values(by=time_col, ascending=True).reset_index(drop=True)
        return out.reset_index(drop=True)

    if order_choice.lower() in ("duration", "duration_hours"):
        duration_col = None
        for candidate in ["duration_hours", "duration_h", "duration"]:
            if candidate in out.columns:
                duration_col = candidate
                break
        if duration_col:
            return out.sort_values(by=duration_col, ascending=False).reset_index(drop=True)
        return out.reset_index(drop=True)

    if order_choice in ("global_min", "global min") and "global_min" in out.columns:
        sort_cols = ["global_min"]
        ascending = [True]
        for candidate in ["start_utc", "start", "episode_start", "start_time"]:
            if candidate in out.columns:
                sort_cols.append(candidate)
                ascending.append(True)
                break
        return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out.reset_index(drop=True)


def sort_stage1_multi_storm_minima(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    out = df.copy()
    time_col = "t_min" if "t_min" in out.columns else "tmin_utc" if "tmin_utc" in out.columns else None

    if order_choice.lower() in ("time", "date", "data") and time_col:
        return out.sort_values(time_col, ascending=True).reset_index(drop=True)

    if order_choice.lower() in ("duration", "duration_hours"):
        duration_col = None
        for candidate in ["duration_hours", "duration_h", "duration"]:
            if candidate in out.columns:
                duration_col = candidate
                break
        if duration_col:
            sort_cols = [duration_col]
            ascending = [False]
            if time_col:
                sort_cols.append(time_col)
                ascending.append(True)
            return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        return out.reset_index(drop=True)

    if order_choice in ("global_min", "global min") and "global_min" in out.columns:
        # stronger -> weaker means more negative -> less negative
        sort_cols = ["global_min"]
        ascending = [True]
        if time_col:
            sort_cols.append(time_col)
            ascending.append(True)
        return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out.reset_index(drop=True)


def sort_stage2_table(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    out = df.copy()

    if order_choice.lower() in ("time", "date") and "tmin_utc" in out.columns:
        return out.sort_values("tmin_utc", ascending=True).reset_index(drop=True)

    if order_choice in ("minDst", "Dst_min") and "minDst" in out.columns:
        # stronger -> weaker means more negative -> less negative
        sort_cols = ["minDst"]
        ascending = [True]
        if "tmin_utc" in out.columns:
            sort_cols.append("tmin_utc")
            ascending.append(True)
        return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if order_choice in ("Mp duration", "main phase duration", "mainphase duration"):
        duration_col = None
        for candidate in ["mainphase_duration", "mainphase_duration", "main_phase_duration", "duration_hours"]:
            if candidate in out.columns:
                duration_col = candidate
                break
        if duration_col:
            sort_cols = [duration_col]
            ascending = [False]
            if "tmin_utc" in out.columns:
                sort_cols.append("tmin_utc")
                ascending.append(True)
            return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out.reset_index(drop=True)



def sort_by_time_generic(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for col in ["tmin_utc", "tstart_utc", "utc", "time", "datetime"]:
        if col in df.columns:
            return df.sort_values(by=col, ascending=True).reset_index(drop=True)
    return df

def sort_stage3_all_clean_storms(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    normalized_order_choice = {
        "date": "date",
        "time": "date",
        "dst_min": "Dst_min",
        "minDst": "Dst_min",
        "mindst": "Dst_min",
        "imfp": "imfp",
        "bzp": "bzp",
        "eyp": "eyp",
        "vp": "speedp",
        "speedp": "speedp",
        "eyi": "eyi",
    }.get(order_choice.lower(), order_choice)
    out = df.copy()

    if normalized_order_choice == "date" and "tmin_utc" in out.columns:
        return out.sort_values("tmin_utc", ascending=True).reset_index(drop=True)

    if normalized_order_choice in ("minDst", "Dst_min") and "minDst" in out.columns:
        sort_cols = ["minDst"]
        ascending = [True]
        if "tmin_utc" in out.columns:
            sort_cols.append("tmin_utc")
            ascending.append(True)
        return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if normalized_order_choice in ("Mp duration", "main phase duration", "mainphase duration"):
        for candidate in ["mainphase_duration", "mainphase_duration", "main_phase_duration", "duration_hours"]:
            if candidate in out.columns:
                sort_cols = [candidate]
                ascending = [False]
                if "tmin_utc" in out.columns:
                    sort_cols.append("tmin_utc")
                    ascending.append(True)
                return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    # Solar wind variables: stronger -> weaker
    # imfp / speedp / eyp / eyi: larger value = stronger => descending
    # bzp: more southward = more negative => ascending
    order_map = {
        "imfp": ("imf_peak", False),
        "bzp": ("bz_peak", True),
        "speedp": ("speed_peak", False),
        "eyp": ("ey_peak", False),
        "eyi": ("eyi", False),
    }
    if normalized_order_choice in order_map:
        col, asc = order_map[normalized_order_choice]
        if col in out.columns:
            sort_cols = [col]
            ascending = [asc]
            if "tmin_utc" in out.columns:
                sort_cols.append("tmin_utc")
                ascending.append(True)
            return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out.reset_index(drop=True)


def reorder_stage2_display(df: pd.DataFrame, excluded: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    desired = ["storm_id", "minDst", "tmin_utc", "dst_start", "tstart_utc", "mainphase_duration"]
    front = [c for c in desired if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    if excluded and len(rest) >= 3:
        tail = rest[-3:]
        middle = rest[:-3]
        return df[front + middle + tail]
    return df[front + rest]


def compute_correlations(df_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mainphase_duration",
        "imf_peak",
        "bz_peak",
        "speed_peak",
        "ey_peak",
        "eyi",
        "delay_eyp_to_tmin_hours",
        "delay_bzp_to_tmin_hours",
    ]
    out = []
    if df_metrics.empty:
        return pd.DataFrame([{"metric": m, "r": np.nan, "p": np.nan, "n": 0} for m in metrics])
    for m in metrics:
        if m not in df_metrics.columns:
            out.append({"metric": m, "r": np.nan, "p": np.nan, "n": 0})
            continue
        r, p, n = pearson_safe(df_metrics[m], df_metrics["abs_minDst"])
        out.append({
            "metric": m,
            "r": r,
            "p": p,
            "n": n
        })
    return pd.DataFrame(out)




def render_correlation_plots(metrics_df: pd.DataFrame):
    plot_specs = [
        ("imf_peak", "IMFp"),
        ("bz_peak", "Bzp"),
        ("ey_peak", "Eyp"),
        ("eyi", "Eyi"),
        ("speed_peak", "Vp"),
        ("mainphase_duration", "Mp duration"),
    ]

    # Change plot size here:
    # figsize=(width, height) in inches
    plot_figsize = (4.5, 3.2)

    if metrics_df is None or metrics_df.empty or "abs_minDst" not in metrics_df.columns:
        st.info("No correlation plots available.")
        return

    plots = []
    for metric_col, label in plot_specs:
        if metric_col not in metrics_df.columns:
            continue

        plot_df = metrics_df[["abs_minDst", metric_col]].dropna().copy()
        if plot_df.empty:
            continue

        plots.append((metric_col, label, plot_df))

    for i in range(0, len(plots), 2):
        cols = st.columns(2)

        for j in range(2):
            if i + j >= len(plots):
                break

            metric_col, label, plot_df = plots[i + j]
            x = plot_df["abs_minDst"].to_numpy(dtype=float)
            y = plot_df[metric_col].to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=plot_figsize)
            fig.set_layout_engine(None)
            ax.set_position([0.16, 0.18, 0.78, 0.68])  # fixed axes box for identical visible size/alignment
            ax.scatter(x, y, color='red', marker='o', s=20)

            r, _, n = pearson_safe(plot_df[metric_col], plot_df["abs_minDst"])
            title = f"{label} vs |Dst_min|"
            if pd.notna(r):
                title += f" (R={float(r):.3f}, N={int(n)})"
            ax.set_title(title)
            ax.set_xlabel("|Dst_min|")
            ax.set_ylabel(label)

            if pd.notna(r) and abs(float(r)) > 0.5 and len(plot_df) >= 2:
                try:
                    m, b = np.polyfit(x, y, 1)
                    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    yfit = m * xfit + b
                    ax.plot(xfit, yfit, color='black')

                    # Print line equation close to the fitted line
                    xeq = xfit[int(len(xfit) * 0.60)]
                    yeq = m * xeq + b
                    sign = "+" if b >= 0 else "-"
                    eq_text = f"y = {m:.3f}x {sign} {abs(b):.3f}"

                    ax.text(xeq, yeq, eq_text)
                except Exception:
                    pass

            cols[j].pyplot(fig, clear_figure=True)


def _clean_one_parameter_values(mp: pd.DataFrame, col: str) -> Tuple[bool, Optional[np.ndarray]]:
    """Return Data-completeed values for one solar-wind parameter during one storm main phase.

    The rule matches the existing Stage 3 quality rule, but applies it to only
    one parameter: a single internal missing value is replaced by the average of
    its two neighbours; edge missing values or consecutive missing values fail.
    """
    if mp is None or mp.empty or col not in mp.columns:
        return False, None

    values = pd.to_numeric(mp[col], errors="coerce").to_numpy(copy=True)
    missing_idx = [i for i, v in enumerate(values) if is_missing(v)]

    if not missing_idx:
        return True, values

    run_len = 1
    for i in range(1, len(missing_idx)):
        if missing_idx[i] == missing_idx[i - 1] + 1:
            run_len += 1
            if run_len >= 2:
                return False, None
        else:
            run_len = 1

    for i in missing_idx:
        if i == 0 or i == len(values) - 1:
            return False, None
        prev = values[i - 1]
        nxt = values[i + 1]
        if is_missing(prev) or is_missing(nxt):
            return False, None
        values[i] = (float(prev) + float(nxt)) / 2.0

    return True, values


PARAMETER_PEAK_TABLE_CONFIGS = {
    "imf": {"source_col": "IMF", "peak_col": "imfp", "mode": "max", "columns": ["storm", "minDst", "imfp"]},
    "bz": {"source_col": "Bz", "peak_col": "bzp", "mode": "min", "columns": ["storm", "minDst", "bzp"]},
    "ey": {"source_col": "Ey", "peak_col": "eyp", "mode": "ey_positive_max", "columns": ["storm", "minDst", "eyp"]},
    "speed": {"source_col": "Vsw", "peak_col": "Speedp", "mode": "max", "columns": ["storm", "minDst", "Speedp"]},
}


def build_parameter_clean_peak_table(kept_df: pd.DataFrame, omni_df: pd.DataFrame, parameter_key: str) -> pd.DataFrame:
    """Build one per-parameter Data-complete peak table for the Extras section.

    This is intentionally parameter-specific so the Extras tab can calculate only
    the selected table instead of building all long tables on every analysis run.
    """
    cfg = PARAMETER_PEAK_TABLE_CONFIGS.get(parameter_key)
    if cfg is None:
        return pd.DataFrame(columns=["storm", "minDst"])

    out_columns = cfg["columns"]
    if kept_df is None or kept_df.empty or omni_df is None or omni_df.empty:
        return pd.DataFrame(columns=out_columns)

    kept = kept_df.copy()
    omni = omni_df.copy()

    if "tstart_utc" not in kept.columns or "tmin_utc" not in kept.columns or "t_utc" not in omni.columns:
        return pd.DataFrame(columns=out_columns)

    kept["tstart_utc"] = pd.to_datetime(kept["tstart_utc"], utc=True, errors="coerce")
    kept["tmin_utc"] = pd.to_datetime(kept["tmin_utc"], utc=True, errors="coerce")
    omni["t_utc"] = pd.to_datetime(omni["t_utc"], utc=True, errors="coerce")

    source_col = cfg["source_col"]
    if source_col not in omni.columns:
        return pd.DataFrame(columns=out_columns)

    for c in [source_col, "Dst"]:
        if c in omni.columns:
            omni[c] = pd.to_numeric(omni[c], errors="coerce")

    out_rows = []
    peak_col = cfg["peak_col"]
    mode = cfg["mode"]

    for _, s in kept.iterrows():
        try:
            sid = int(s["storm_id"])
        except Exception:
            continue

        tstart = pd.to_datetime(s.get("tstart_utc"), utc=True, errors="coerce")
        tmin = pd.to_datetime(s.get("tmin_utc"), utc=True, errors="coerce")
        if pd.isna(tstart) or pd.isna(tmin):
            continue

        mp = omni[(omni["t_utc"] >= tstart) & (omni["t_utc"] <= tmin)].copy()
        if mp.empty:
            continue
        mp.sort_values("t_utc", inplace=True)

        ok, values = _clean_one_parameter_values(mp, source_col)
        if not ok or values is None or len(values) == 0:
            continue

        if mode == "min":
            peak_value = np.nanmin(values)
        elif mode == "ey_positive_max":
            peak_value = np.nanmax(np.clip(values.astype(float), 0, None))
        else:
            peak_value = np.nanmax(values)

        if pd.isna(peak_value):
            continue

        minDst = float(s["minDst"]) if "minDst" in s and pd.notna(s["minDst"]) else np.nan
        out_rows.append({"storm": sid, "minDst": minDst, peak_col: float(peak_value)})

    df = pd.DataFrame(out_rows, columns=out_columns)
    if not df.empty:
        df.sort_values(["storm"], ascending=[True], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(show_spinner=False, max_entries=16)
def build_parameter_clean_peak_table_cached(kept_df: pd.DataFrame, omni_df: pd.DataFrame, parameter_key: str) -> pd.DataFrame:
    return build_parameter_clean_peak_table(kept_df, omni_df, parameter_key)


def build_parameter_clean_peak_tables(kept_df: pd.DataFrame, omni_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build all per-parameter Data-complete peak tables.

    Kept for the explicit Extras option: All tables.
    """
    return {
        key: build_parameter_clean_peak_table(kept_df, omni_df, key)
        for key in PARAMETER_PEAK_TABLE_CONFIGS.keys()
    }


def sort_parameter_peak_table(df: pd.DataFrame, peak_col: str, order_choice: str) -> pd.DataFrame:
    """Sort an Extras parameter-Data-complete table by storm or by its peak strength."""
    if df is None or df.empty:
        return df

    out = df.copy()

    if order_choice == "peak value" and peak_col in out.columns:
        # For Bz, strongest means most negative southward Bz.
        ascending = True if peak_col == "bzp" else False
        return out.sort_values([peak_col, "storm"], ascending=[ascending, True]).reset_index(drop=True)

    if "storm" in out.columns:
        return out.sort_values("storm", ascending=True).reset_index(drop=True)

    return out.reset_index(drop=True)


def format_parameter_peak_correlation(df: pd.DataFrame, peak_col: str) -> str:
    """Return a compact Pearson correlation text for minDst vs one Extras peak column."""
    if df is None or df.empty or "minDst" not in df.columns or peak_col not in df.columns:
        return "Correlation Dst_min vs parameter peak: R = NA | p = NA | N = 0"

    r, p, n = pearson_safe(df["minDst"], df[peak_col])

    if pd.isna(r):
        return f"Correlation Dst_min vs parameter peak: R = NA | p = NA | N = {n}"

    return f"Correlation Dst_min vs parameter peak: R = {r:.3f} | p = {p:.3g} | N = {n}"


def stage3_clean_and_metrics(kept_df: pd.DataFrame, omni_df: pd.DataFrame):
    if kept_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, pd.DataFrame(), {"Data-complete_count": 0, "storms_with_single_missing_replaced": 0}

    omni = omni_df.copy()
    kept = kept_df.copy()

    kept["tstart_utc"] = pd.to_datetime(kept["tstart_utc"], utc=True)
    kept["tmin_utc"] = pd.to_datetime(kept["tmin_utc"], utc=True)
    omni["t_utc"] = pd.to_datetime(omni["t_utc"], utc=True)

    for c in ["IMF", "Bz", "Vsw", "Ey", "Dst"]:
        omni[c] = pd.to_numeric(omni[c], errors="coerce")

    clean_storms = []
    clean_rows = []
    replaced_values_log = []
    rejected_quality_log = []
    storms_with_single_missing_replaced = 0

    for _, s in kept.iterrows():
        sid = int(s["storm_id"])
        tstart = s["tstart_utc"]
        tmin = s["tmin_utc"]

        mp = omni[(omni["t_utc"] >= tstart) & (omni["t_utc"] <= tmin)].copy()
        if mp.empty:
            continue
        mp.sort_values("t_utc", inplace=True)

        reject = False
        had_single_missing_replaced = False
        storm_replacements = []
        storm_missing_records = []

        for col in ["IMF", "Bz", "Vsw", "Ey"]:
            values = mp[col].to_numpy(copy=True)
            times = mp["t_utc"].to_numpy(copy=True)
            missing_idx = [i for i, v in enumerate(values) if is_missing(v)]
            for i in missing_idx:
                storm_missing_records.append(
                    {
                        "storm_id": sid,
                        "episode_id": int(s["episode_id"]),
                        "minDst": float(s["minDst"]) if pd.notna(s["minDst"]) else np.nan,
                        "parameter": col,
                        "t_utc": pd.to_datetime(times[i], utc=True),
                        "missing_value": values[i],
                        "row_index": i,
                    }
                )

            col_reject = False
            if missing_idx:
                runs = 1
                for i in range(1, len(missing_idx)):
                    if missing_idx[i] == missing_idx[i - 1] + 1:
                        runs += 1
                        if runs >= 2:
                            reject = True
                            col_reject = True
                            break
                    else:
                        runs = 1

            if not col_reject:
                for i in missing_idx:
                    if i == 0 or i == len(values) - 1:
                        reject = True
                        col_reject = True
                        break
                    prev = values[i - 1]
                    nxt = values[i + 1]
                    if is_missing(prev) or is_missing(nxt):
                        reject = True
                        col_reject = True
                        break
                    replaced_val = (float(prev) + float(nxt)) / 2.0
                    storm_replacements.append(
                        {
                            "storm_id": sid,
                            "episode_id": int(s["episode_id"]),
                            "parameter": col,
                            "t_utc": pd.to_datetime(times[i], utc=True),
                            "orig_value": values[i],
                            "replaced_value": replaced_val,
                            "row_index": i,
                        }
                    )
                    values[i] = replaced_val
                    had_single_missing_replaced = True

            if not col_reject:
                mp[col] = values

        if reject:
            rejected_quality_log.append(
                {
                    "storm_id": sid,
                    "minDst": float(s["minDst"]) if pd.notna(s["minDst"]) else np.nan,
                    "tmin_utc": pd.to_datetime(s["tmin_utc"], utc=True) if "tmin_utc" in s else pd.NaT,
                    "missing_value_hours": len({pd.to_datetime(r["t_utc"], utc=True) for r in storm_missing_records}) if storm_missing_records else 0,
                    "failed_parameters": ",".join(sorted({r["parameter"] for r in storm_missing_records})) if storm_missing_records else "",
                }
            )
            continue

        clean_storms.append(s.to_dict())
        if had_single_missing_replaced:
            storms_with_single_missing_replaced += 1
            replaced_values_log.extend(storm_replacements)

        mp["storm_id"] = sid
        mp["episode_id"] = int(s["episode_id"])
        clean_rows.append(mp)

    clean_storms_df = pd.DataFrame(clean_storms) if clean_storms else pd.DataFrame(columns=kept.columns)
    clean_rows_df = pd.concat(clean_rows, ignore_index=True) if clean_rows else pd.DataFrame()
    replacements_df = pd.DataFrame(replaced_values_log)
    rejected_quality_df = pd.DataFrame(rejected_quality_log)

    if not clean_storms_df.empty:
        clean_storms_df.sort_values("minDst", inplace=True)
        clean_storms_df["tstart_utc"] = pd.to_datetime(clean_storms_df["tstart_utc"], utc=True)
        clean_storms_df["tmin_utc"] = pd.to_datetime(clean_storms_df["tmin_utc"], utc=True)

    if clean_rows_df.empty:
        return clean_storms_df, clean_rows_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), rejected_quality_df, {
            "Data-complete_count": len(clean_storms_df),
            "storms_with_single_missing_replaced": storms_with_single_missing_replaced,
        }

    clean_rows_df["t_utc"] = pd.to_datetime(clean_rows_df["t_utc"], utc=True)
    for c in ["Dst", "IMF", "Bz", "Vsw", "Ey"]:
        clean_rows_df[c] = pd.to_numeric(clean_rows_df[c], errors="coerce")

    rows_by_storm = {int(k): g.sort_values("t_utc") for k, g in clean_rows_df.groupby("storm_id")}

    peak_delays_out = []
    peaks_integrals_out = []

    for _, s in clean_storms_df.iterrows():
        sid = int(s["storm_id"])
        ep = int(s["episode_id"])
        tstart = pd.to_datetime(s["tstart_utc"], utc=True)
        tmin = pd.to_datetime(s["tmin_utc"], utc=True)
        minDst = float(s["minDst"]) if pd.notna(s["minDst"]) else np.nan

        g = rows_by_storm.get(sid)
        if g is None or g.empty:
            continue
        mp = g[(g["t_utc"] >= tstart) & (g["t_utc"] <= tmin)].copy()
        if mp.empty:
            continue

        mainphase_duration = (tmin - tstart).total_seconds() / 3600.0
        mp["Ey_inj"] = mp["Ey"].clip(lower=0)

        if mp["Ey_inj"].notna().any():
            i_eyp = mp["Ey_inj"].idxmax()
            eyp = float(mp.loc[i_eyp, "Ey_inj"])
            t_eyp = mp.loc[i_eyp, "t_utc"]
            delay_ey_h = (tmin - t_eyp).total_seconds() / 3600.0
        else:
            eyp = np.nan
            t_eyp = pd.NaT
            delay_ey_h = np.nan

        if mp["Bz"].notna().any():
            i_bzp = mp["Bz"].idxmin()
            bzp = float(mp.loc[i_bzp, "Bz"])
            t_bzp = mp.loc[i_bzp, "t_utc"]
            delay_bz_h = (tmin - t_bzp).total_seconds() / 3600.0
        else:
            bzp = np.nan
            t_bzp = pd.NaT
            delay_bz_h = np.nan

        peak_delays_out.append(
            {
                "storm_id": sid,
                "episode_id": ep,
                "tstart_utc": tstart,
                "tmin_utc": tmin,
                "minDst": minDst,
                "mainphase_duration": mainphase_duration,
                "eyp": eyp,
                "t_eyp_utc": t_eyp,
                "delay_eyp_to_tmin_hours": delay_ey_h,
                "bzp": bzp,
                "t_bzp_utc": t_bzp,
                "delay_bzp_to_tmin_hours": delay_bz_h,
                "mainphase_rows": len(mp),
            }
        )

        peaks_integrals_out.append(
            {
                "storm_id": sid,
                "episode_id": ep,
                "tstart_utc": tstart,
                "tmin_utc": tmin,
                "minDst": minDst,
                "abs_minDst": abs(minDst) if pd.notna(minDst) else np.nan,
                "mainphase_duration": mainphase_duration,
                "imf_peak": mp["IMF"].max(skipna=True),
                "bz_peak": mp["Bz"].min(skipna=True),
                "speed_peak": mp["Vsw"].max(skipna=True),
                "ey_peak": mp["Ey_inj"].max(skipna=True),
                "eyi": mp["Ey_inj"].sum(skipna=True),
                "mainphase_rows": len(mp),
            }
        )

    df_delays = pd.DataFrame(peak_delays_out)
    df_metrics = pd.DataFrame(peaks_integrals_out)

    if not df_metrics.empty:
        df_metrics = df_metrics.merge(
            df_delays[["storm_id", "delay_eyp_to_tmin_hours", "delay_bzp_to_tmin_hours", "eyp", "bzp", "t_eyp_utc", "t_bzp_utc"]],
            on="storm_id",
            how="left",
        )
        df_metrics["class"] = df_metrics["minDst"].apply(class_name)
    else:
        df_metrics["class"] = pd.Series(dtype="object")

    summary = {
        "Data-complete_count": len(clean_storms_df),
        "storms_with_single_missing_replaced": storms_with_single_missing_replaced,
        "avg_mainphase_Data-complete": float(df_delays["mainphase_duration"].mean()) if not df_delays.empty else np.nan,
        "err_mainphase_Data-complete": sem(df_delays["mainphase_duration"]) if not df_delays.empty else np.nan,
        "avg_delay_eyp": float(df_delays["delay_eyp_to_tmin_hours"].mean()) if not df_delays.empty else np.nan,
        "err_delay_eyp": sem(df_delays["delay_eyp_to_tmin_hours"]) if not df_delays.empty else np.nan,
        "avg_delay_bzp": float(df_delays["delay_bzp_to_tmin_hours"].mean()) if not df_delays.empty else np.nan,
        "err_delay_bzp": sem(df_delays["delay_bzp_to_tmin_hours"]) if not df_delays.empty else np.nan,
    }

    return clean_storms_df, clean_rows_df, df_delays, df_metrics, replacements_df, rejected_quality_df, summary


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    out = prettify_table_headers(out)
    return out.to_csv(index=False).encode("utf-8")


def zip_outputs(files: Dict[str, pd.DataFrame]) -> bytes:
    import zipfile

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in files.items():
            zf.writestr(name, to_csv_bytes(df))
    bio.seek(0)
    return bio.read()


@st.cache_data(show_spinner=False, max_entries=8)
def run_storm_analysis_cached(
    raw_text: str,
    shift_plus_1_hour: bool,
    date_start: str,
    date_end: str,
    episode_level: float,
    storm_level: float,
    local_min_radius_hours: int,
    minima_split_factor: float,
    storm_limit: float,
    strong_storm_threshold: float,
    strong_mark: float,
    local_max_radius_hours: int,
    disturbance_filter: bool,
    disturbance_level: float,
    disturb_dip_count: int,
    disturb_dip_radius_hours: int,
) -> Dict[str, object]:
    """Run the expensive storm analysis once per unique input/settings set.

    Sorting controls still trigger a normal Streamlit rerun, but this cached
    function returns the already computed dataframes instead of recalculating
    storm detection, filtering, Stage 3 Data-completeing, and correlations.
    Extras parameter peak tables are calculated lazily and kept in session state
    after the user selects them for the first time.
    """
    all_rows, bad_rows = read_rows_new8_from_text(raw_text, bool(shift_plus_1_hour))
    if not all_rows:
        raise ValueError(f"No valid data rows were read from the OMNI data file: {OMNI_DATA_FILENAME}")

    filtered_rows = filter_rows_by_date(all_rows, date_start, date_end)
    if not filtered_rows:
        raise ValueError("No rows remain after the date filter.")

    weak_mark_factor_cached = calculate_weak_mark_factor(float(strong_storm_threshold), float(strong_mark))
    if pd.isna(weak_mark_factor_cached):
        raise ValueError("Invalid filtering settings: STRONG_STORM_THRESHOLD must not be 0.")

    episodes_raw = build_episodes_contiguous(filtered_rows, float(episode_level))

    # Left-censored boundary storm rule:
    # If the selected date range starts while Dst is already below the storm
    # threshold, the first detected episode may be the recovery/main phase of a
    # storm that began before DATE_START. Remove that first episode before
    # Stage 1 storm tables are built, so it is not displayed as a normal storm
    # in "Storm detection" and it cannot enter the clean main-phase analysis.
    left_censored_boundary_episodes = 0
    if episodes_raw and filtered_rows and float(filtered_rows[0].dst) <= float(storm_level):
        first_a, _first_b = episodes_raw[0]
        if int(first_a) == 0:
            episodes_raw = episodes_raw[1:]
            left_censored_boundary_episodes = 1

    storms, episodes, multi_storms = storms_from_episodes(
        filtered_rows,
        episodes_raw,
        float(storm_level),
        int(local_min_radius_hours),
        float(minima_split_factor),
        float(storm_limit),
    )

    omni_df = rows_to_df(filtered_rows)
    storms_df = storms_to_df(storms)
    episodes_df = episodes_to_df(filtered_rows, episodes, storms)
    multi_df = multi_storms_to_df(multi_storms)

    run_info = {
        "rows_read": int(len(all_rows)),
        "rows_analyzed": int(len(filtered_rows)),
        "bad_rows": int(bad_rows),
        "full_span_start": all_rows[0].t,
        "full_span_end": all_rows[-1].t,
        "analyzed_span_start": filtered_rows[0].t,
        "analyzed_span_end": filtered_rows[-1].t,
        "left_censored_boundary_episodes": int(left_censored_boundary_episodes),
        "left_censored_boundary_note": (
            "! There is a storm event preceding the selected start date (first Dst value is below storm_level so whole episode gets rejected)"
        ) if left_censored_boundary_episodes else "",
    }

    kept, disturbances, excluded, stage2_summary = stage2_filter(
        filtered_rows,
        storms,
        multi_storms,
        float(strong_storm_threshold),
        float(strong_mark),
        int(local_max_radius_hours),
        bool(disturbance_filter),
        float(disturbance_level),
        int(disturb_dip_count),
        int(disturb_dip_radius_hours),
        float(storm_level),
    )
    stage2_summary["left_censored_boundary"] = int(stage2_summary.get("left_censored_boundary", 0)) + int(left_censored_boundary_episodes)

    kept_df = pd.DataFrame(kept)
    disturbances_df = pd.DataFrame(disturbances)
    excluded_df = pd.DataFrame(excluded)

    clean_storms_df, clean_rows_df, peak_delays_df, metrics_df, replacements_df, rejected_quality_df, stage3_summary = stage3_clean_and_metrics(
        kept_df, omni_df
    )

    corr_all_df = compute_correlations(metrics_df)
    corr_mod_df = compute_correlations(metrics_df[metrics_df["class"].str.startswith("moderate", na=False)]) if not metrics_df.empty else pd.DataFrame()
    corr_int_df = compute_correlations(metrics_df[metrics_df["class"].str.startswith("intense", na=False)]) if not metrics_df.empty else pd.DataFrame()
    corr_sup_df = compute_correlations(metrics_df[metrics_df["class"].str.startswith("super-storm", na=False)]) if not metrics_df.empty else pd.DataFrame()

    # Cached display/export source tables. These are independent of the selected
    # sort order; radio buttons only sort copies of these dataframes later.
    stage2_cols_to_remove = ["bz", "imf", "vsw", "ey", "is_multi_storm", "mark", "disturb_dip_count", "disturb_dip_index", "disturb_dip_utc", "disturb_dip_dst", "episode_id", "row_index"]
    kept_display_df = reorder_stage2_display(
        kept_df.drop(columns=["reason", *stage2_cols_to_remove], errors="ignore"),
        excluded=False,
    )
    disturbances_display_df = reorder_stage2_display(
        disturbances_df.drop(columns=["reason", *stage2_cols_to_remove], errors="ignore"),
        excluded=False,
    )
    excluded_display_df = reorder_stage2_display(
        excluded_df.drop(columns=["reason", *stage2_cols_to_remove], errors="ignore"),
        excluded=True,
    )

    stage3_all_clean_storms_df = build_stage3_all_clean_storms_table(clean_storms_df, metrics_df)
    stage3_peak_delays_df = build_stage3_peak_delays_table(clean_storms_df, peak_delays_df)
    stage3_full_peak_data_df = build_stage3_full_peak_data_table(clean_storms_df, clean_rows_df)
    stage3_replacements_df = build_stage3_replacements_table(replacements_df, clean_storms_df)
    stage3_rejected_quality_df = build_stage3_rejected_quality_table(rejected_quality_df)
    stage3_all_missing_values_df = build_all_missing_values_table(omni_df)

    return {
        "run_info": run_info,
        "omni_df": omni_df,
        "storms_df": storms_df,
        "episodes_df": episodes_df,
        "multi_df": multi_df,
        "kept_df": kept_df,
        "disturbances_df": disturbances_df,
        "excluded_df": excluded_df,
        "stage2_summary": stage2_summary,
        "Data-complete_storms_df": clean_storms_df,
        "Data-complete_rows_df": clean_rows_df,
        "peak_delays_df": peak_delays_df,
        "metrics_df": metrics_df,
        "replacements_df": replacements_df,
        "rejected_quality_df": rejected_quality_df,
        "stage3_summary": stage3_summary,
        "corr_all_df": corr_all_df,
        "corr_mod_df": corr_mod_df,
        "corr_int_df": corr_int_df,
        "corr_sup_df": corr_sup_df,
        "kept_display_df": kept_display_df,
        "disturbances_display_df": disturbances_display_df,
        "excluded_display_df": excluded_display_df,
        "stage3_all_Data-complete_storms_df": stage3_all_clean_storms_df,
        "stage3_peak_delays_df": stage3_peak_delays_df,
        "stage3_full_peak_data_df": stage3_full_peak_data_df,
        "stage3_replacements_df": stage3_replacements_df,
        "stage3_rejected_quality_df": stage3_rejected_quality_df,
        "stage3_all_missing_values_df": stage3_all_missing_values_df,
    }


# ---------------------------
# UI
# ---------------------------

# Change these numbers to resize/bolden only the main "Analysis settings" expander label.
ANALYSIS_SETTINGS_LABEL_SIZE_PX = 20
ANALYSIS_SETTINGS_LABEL_FONT_WEIGHT = 800

# Change this single line to move the Guide button and Analysis settings bar down/up from the top.
GUIDE_ANALYSIS_TOP_SPACER = "5.5rem"

# Change this number to resize the section titles inside Analysis options
# such as "Storm detection" and "Filtering (for main phase)".
ANALYSIS_OPTIONS_TITLE_SIZE_PX = 19

# Change this symbol if you want a different bullet before Analysis options section titles.
ANALYSIS_OPTIONS_TITLE_BULLET = "•"

# Change this color to control the Analysis options background.
# Examples: "#F8E463" yellow, "#FFFFFF" white, "#F3F4F6" light gray.
ANALYSIS_OPTIONS_BACKGROUND_COLOR = "#D9FFB0"

# Change this number to control the width of the Analysis options column.
# Bigger number = wider left options area. Smaller number = narrower left options area.
ANALYSIS_OPTIONS_WIDTH_RATIO = 1.05

# Change this number to control the empty distance between Analysis options
# and the main program area (title, START button, results, etc.).
# Bigger number = more space between the left options and the main content.
# Smaller number = less space.
ANALYSIS_TO_CONTENT_GAP_RATIO = 0.08

# Change this number to control the empty distance from the right edge of the screen.
# Bigger number = more empty space on the right. Smaller number = less empty space.
RIGHT_SCREEN_GAP_RATIO = 0.25

# Change this number to control the width of the main title/results area.
# Bigger number = wider main content area. Smaller number = narrower main content area.
MAIN_CONTENT_WIDTH_RATIO = 5.35



def render_section_heading(title: str, unit_items: List[str]):
    """Render a section heading with an inline unit legend beside the title."""
    units_html = escape(" | ".join(str(item) for item in unit_items))
    title_html = escape(title)
    st.markdown(
        '<div class="section-heading-row">'
        f'<h3 class="section-heading-title">{title_html}</h3>'
        f'<span class="section-heading-units">{units_html}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def analysis_options_title(title: str):
    st.markdown(
        f"""
        <div class='analysis-options-title'>
            <span class='analysis-options-title-bullet'>{escape(ANALYSIS_OPTIONS_TITLE_BULLET)}</span>
            <span>{escape(title)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pdf_guide_button(pdf_filename: str = "parameters.pdf"):
    try:
        app_dir = Path(__file__).resolve().parent
    except NameError:
        app_dir = Path.cwd()

    # Try the expected filename first, then a few common spelling/case variants.
    # The PDF must be in the same folder as this .py file when the app runs.
    candidate_names = [
        pdf_filename,
        "paramaters.pdf",
        "Parameters.pdf",
        "Paramaters.pdf",
        "PARAMETERS.pdf",
    ]

    pdf_path = None
    for name in candidate_names:
        candidate = app_dir / name
        if candidate.exists() and candidate.is_file():
            pdf_path = candidate
            break

    if pdf_path is None:
        st.caption(f"Guide PDF not found. Put {pdf_filename} in the same folder as this app file.")
        return

    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    # Hide the browser PDF viewer toolbar so the data-URI title is not shown above the Guide PDF.
    pdf_src = f"data:application/pdf;base64,{pdf_b64}#toolbar=0&navpanes=0&scrollbar=1&view=FitH"
    guide_id = f"custom_guide_toggle_{uuid4().hex}"
    safe_pdf_name = escape(pdf_path.name)

    # Use a CSS-only checkbox toggle instead of JavaScript.
    # This lets the X button close the in-app PDF panel reliably inside Streamlit.
    guide_html = f'''
    <style>
    .custom-guide-wrapper {{
        position: relative !important;
        width: fit-content !important;
        max-width: fit-content !important;
        margin: 0 0 0.45rem 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        overflow: visible !important;
        z-index: 2147483000 !important;
    }}

    .custom-guide-toggle {{
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}

    .custom-guide-open.guide-pdf-button {{
        list-style: none !important;
        cursor: pointer !important;
        user-select: none !important;
        background: #ffffff !important;
        background-color: #ffffff !important;
        opacity: 1 !important;
        filter: none !important;
    }}

    .custom-guide-open.guide-pdf-button:hover,
    .custom-guide-open.guide-pdf-button:focus,
    .custom-guide-open.guide-pdf-button:active {{
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        opacity: 1 !important;
        filter: none !important;
        text-decoration: none !important;
    }}

    /* Floating Guide panel: open directly below the Guide button, but stay above the app content. */
    .custom-guide-body {{
        display: none !important;
        position: absolute !important;
        top: calc(100% + 6px) !important;
        left: 0 !important;
        z-index: 2147483000 !important;
        width: 780px !important;
        max-width: calc(100vw - 60px) !important;
        max-height: calc(100vh - 150px) !important;
        margin-top: 0 !important;
        padding: 0.6rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0,0,0,0.14) !important;
        background: #ffffff !important;
        box-shadow: 0 12px 34px rgba(0,0,0,0.32) !important;
        overflow: auto !important;
    }}

    .custom-guide-toggle:checked ~ .custom-guide-body {{
        display: block !important;
    }}

    .custom-guide-toolbar {{
        position: sticky !important;
        top: 0 !important;
        z-index: 2147483001 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        gap: 8px !important;
        padding: 0 0 0.5rem 0 !important;
        margin: 0 !important;
        background: #ffffff !important;
    }}

    .custom-guide-download,
    .custom-guide-close {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 34px !important;
        padding: 0 0.75rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(0,0,0,0.22) !important;
        background: #ffffff !important;
        color: #111111 !important;
        font-size: 0.92rem !important;
        font-weight: 800 !important;
        line-height: 1 !important;
        text-decoration: none !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
        cursor: pointer !important;
        user-select: none !important;
    }}

    .custom-guide-close {{
        width: 34px !important;
        padding: 0 !important;
        font-size: 1.15rem !important;
    }}

    .custom-guide-download:hover,
    .custom-guide-close:hover {{
        background: #f9fafb !important;
        border-color: rgba(0,0,0,0.35) !important;
        color: #000000 !important;
        text-decoration: none !important;
    }}

    .custom-guide-body iframe {{
        width: 100% !important;
        height: min(790px, calc(100vh - 250px)) !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        background: white !important;
    }}
    </style>

    <div class="custom-guide-wrapper">
        <input id="{guide_id}" class="custom-guide-toggle" type="checkbox">
        <label for="{guide_id}" class="custom-guide-open guide-pdf-button">Algorithm parameters</label>
        <div class="custom-guide-body">
            <div class="custom-guide-toolbar">
                <a class="custom-guide-download" href="{pdf_src}" download="{safe_pdf_name}">Download PDF</a>
                <label for="{guide_id}" class="custom-guide-close" title="Close PDF" aria-label="Close PDF">×</label>
            </div>
            <iframe src="{pdf_src}"></iframe>
        </div>
    </div>
    '''
    st.markdown(guide_html, unsafe_allow_html=True)


st.markdown(f"""
<style>
/* Keep the options panel flush to the left and let the right column hold all output. */
.block-container {{
    max-width: 100% !important;
    padding-top: 0rem !important;
    padding-left: 0rem !important;
    padding-right: 1rem !important;
    margin-top: -1.8rem !important;
}}

/* Remove the tiny inherited offset before the first layout column. */
div[data-testid="stHorizontalBlock"] > div:first-child {{
    padding-left: 0rem !important;
}}

/* Analysis options dropdown background. */
div[data-testid="stHorizontalBlock"] > div:first-child details:not(.custom-guide-details) {{
    background: {ANALYSIS_OPTIONS_BACKGROUND_COLOR} !important;
    border: 1px solid rgba(0,0,0,0.16) !important;
    border-radius: 10px !important;
}}

div[data-testid="stHorizontalBlock"] > div:first-child details:not(.custom-guide-details) summary {{
    background: {ANALYSIS_OPTIONS_BACKGROUND_COLOR} !important;
    border-radius: 8px !important;
}}

/* Only the main "Analysis settings" expander label. */
div[data-testid="stHorizontalBlock"] > div:first-child details:not(.custom-guide-details) summary p {{
    font-size: {ANALYSIS_SETTINGS_LABEL_SIZE_PX}px !important;
    font-weight: {ANALYSIS_SETTINGS_LABEL_FONT_WEIGHT} !important;
    line-height: 1.2 !important;
}}

div[data-testid="stHorizontalBlock"] > div:first-child details:not(.custom-guide-details) [data-testid="stExpanderDetails"] {{
    background: {ANALYSIS_OPTIONS_BACKGROUND_COLOR} !important;
}}

.analysis-options-title {{
    display: flex !important;
    align-items: center !important;
    gap: 0.45rem !important;
    font-size: {ANALYSIS_OPTIONS_TITLE_SIZE_PX}px !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    margin: 1.0rem 0 0.45rem 0 !important;
}}

.analysis-options-title-bullet {{
    font-size: calc({ANALYSIS_OPTIONS_TITLE_SIZE_PX}px + 1px) !important;
    line-height: 1 !important;
    display: inline-block !important;
}}

/* Section headings: title on the left, unit badge on the far right, without adding extra top-page layout space. */
.section-heading-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    flex-wrap: nowrap;
    width: 100%;
    margin: 0 0 0.8rem 0;
    padding: 0;
}}

.section-heading-title {{
    flex: 1 1 auto;
    min-width: 0;
    margin: 0 !important;
    padding: 0 !important;
    color: #111827 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
}}

.section-heading-units {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 auto;
    margin: 0 0 0 auto !important;
    padding: 0.28rem 0.62rem;
    border: 1px solid #111111;
    border-radius: 8px;
    background: #ffffff;
    color: #111111;
    font-size: 0.92rem;
    font-weight: 600;
    line-height: 1.2;
    white-space: nowrap;
}}

</style>
""", unsafe_allow_html=True)

# Keep the custom Guide wrapper transparent and allow the PDF panel to overflow above other app content.
st.markdown("""
<style>
.custom-guide-wrapper,
.custom-guide-wrapper:has(.custom-guide-toggle:checked) {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    overflow: visible !important;
    z-index: 2147483000 !important;
}

.custom-guide-wrapper .guide-pdf-button,
.custom-guide-wrapper .guide-pdf-button:hover,
.custom-guide-wrapper .guide-pdf-button:focus,
.custom-guide-wrapper .guide-pdf-button:active {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    opacity: 1 !important;
}

/* Allow the custom Guide PDF panel to overflow above other app content. */
div[data-testid="stElementContainer"]:has(.custom-guide-wrapper),
div[data-testid="element-container"]:has(.custom-guide-wrapper),
div[data-testid="stMarkdownContainer"]:has(.custom-guide-wrapper),
div[data-testid="stVerticalBlock"]:has(.custom-guide-wrapper),
div[data-testid="stHorizontalBlock"]:has(.custom-guide-wrapper) {
    overflow: visible !important;
    z-index: 2147483000 !important;
    position: relative !important;
}
</style>
""", unsafe_allow_html=True)
# The analysis controls are kept in the left expander column.
# The top spacer places the Guide immediately above Analysis settings.
# Analysis settings is positioned to line up with the DATE_START / DATE_END / START row.
# The second column is an intentional empty gap between the options and the main content.
# The fourth column is an intentional empty right margin.
main_options_col, analysis_gap_col, main_content_col, right_gap_col = st.columns(
    [ANALYSIS_OPTIONS_WIDTH_RATIO, ANALYSIS_TO_CONTENT_GAP_RATIO, MAIN_CONTENT_WIDTH_RATIO, RIGHT_SCREEN_GAP_RATIO],
    gap="small",
)

with main_options_col:
    st.markdown(f"<div style='height:{GUIDE_ANALYSIS_TOP_SPACER}'></div>", unsafe_allow_html=True)
    render_pdf_guide_button("parameters.pdf")
    with st.expander("Analysis settings", expanded=False):
        shift_plus_1_hour = st.checkbox("Shift timestamps by +1 hour", value=True)

        analysis_options_title("Storm detection")
        episode_level = st.number_input("EPISODE_LEVEL", value=-20, step=1)
        storm_level = st.number_input("STORM_LEVEL", value=-50, step=1)
        local_min_radius_hours = st.number_input("LOCAL_MIN_RADIUS_HOURS", value=3, step=1, min_value=0)
        minima_split_factor = st.number_input("MINIMA_SPLIT_FACTOR", value=0.4, step=0.05, format="%.2f")
        storm_limit = st.number_input("STORM_LIMIT", value=-1000, step=10)

        analysis_options_title("Filtering (for main phase)")
        strong_storm_threshold = st.number_input("STRONG_STORM_THRESHOLD", value=-100, step=1)
        strong_mark = st.number_input("STRONG_MARK", value=-50, step=1)
        weak_mark_factor = calculate_weak_mark_factor(strong_storm_threshold, strong_mark)
        if pd.isna(weak_mark_factor):
            st.error("STRONG_STORM_THRESHOLD cannot be 0 because WEAK_MARK is calculated from STRONG_MARK / STRONG_STORM_THRESHOLD.")
        else:
            st.caption(f"WEAK_MARK = Dst_min × {weak_mark_factor:.3f}")
        local_max_radius_hours = st.number_input("LOCAL_MAX_RADIUS_HOURS", value=3, step=1, min_value=0)
        disturbance_filter = st.checkbox("DISTURBANCE_FILTER", value=True)
        disturbance_level = st.number_input(
            "DISTURB_LEVEL",
            value=-100,
            step=1,
            disabled=not disturbance_filter,
        )
        disturb_dip_count = st.number_input(
            "DISTURB_DIP_COUNT",
            value=2,
            step=1,
            min_value=1,
            disabled=not disturbance_filter,
        )
        disturb_dip_radius_hours = st.number_input(
            "DISTURB_DIP_RADIUS_HOURS",
            value=3,
            step=1,
            min_value=0,
            disabled=not disturbance_filter,
        )

with main_content_col:
    header_left, header_mid, header_right = st.columns([4.35, 0.18, 1.47], gap="small")

    with header_left:
        st.markdown(
            "<h1 class='app-main-title'><span class='title-line'>Magnetic Storm Detection</span><br><span class='title-line'>and Main Phase Analysis</span></h1>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
        # Keep DATE_START / DATE_END / START automatically aligned.
        try:
            date_start_col, date_end_col, spacer_col, start_col, loading_col = st.columns(
                [1.38, 1.38, 0.04, 0.95, 1.25],
                vertical_alignment="bottom",
            )
        except TypeError:
            # Compatibility with older Streamlit versions.
            date_start_col, date_end_col, spacer_col, start_col, loading_col = st.columns([1.38, 1.38, 0.04, 0.95, 1.25])
        with date_start_col:
            st.markdown(
                "<div style='color:white; font-weight:800; font-size:0.95rem; margin-bottom:0.25rem;'>DATE_START</div>",
                unsafe_allow_html=True
            )
            date_start = st.text_input(
                "DATE_START",
                value="1964-01-01 00:00",
                label_visibility="collapsed",
                key="date_start_input"
            )
        with date_end_col:
            st.markdown(
                "<div style='color:white; font-weight:800; font-size:0.95rem; margin-bottom:0.25rem;'>DATE_END</div>",
                unsafe_allow_html=True
            )
            date_end = st.text_input(
                "DATE_END",
                value="2026-06-01 00:00",
                label_visibility="collapsed",
                key="date_end_input"
            )
        with start_col:
            run = st.button("START", type="primary")

        with loading_col:
            # Keep the working indicator in the same visual row as
            # DATE_START, DATE_END and START. The fixed-height slot also
            # prevents the header row from jumping when the indicator appears.
            loading_indicator = st.empty()
            loading_indicator.markdown(
                "<div class='working-indicator-slot working-indicator-placeholder'></div>",
                unsafe_allow_html=True,
            )

    
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    with header_mid:
        st.markdown("<div style='height:3.2rem;'></div>", unsafe_allow_html=True)
    
    with header_right:
        # The OMNI data file is bundled with the app in the GitHub repository.
        # No user upload is required.
        st.empty()

    if run:
        st.session_state.analysis_started = True

    if st.session_state.analysis_started:
        # Show this immediately at the beginning of every rerun after START has been pressed.
        # It stays visible while Streamlit recalculates/renders tables, tabs, Extras, downloads, etc.
        loading_indicator.markdown(
            """
            <style>
            @keyframes stormLoaderSpin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .working-indicator-slot {
                height: 46px;
                min-height: 46px;
                display: flex;
                align-items: center;
                justify-content: flex-start;
                padding: 0;
                margin-top: -10px !important;
                transform: translateY(-15px) !important;
                box-sizing: border-box;
                line-height: 1;
            }
            .working-indicator-placeholder {
                visibility: hidden;
            }
            .storm-working-indicator {
                gap: 0.55rem;
                color: black !important;
                font-weight: 900;
                font-size: 20px;
                white-space: nowrap;
                border: none !important;
                border-radius: 12px;
                background: #ffffff !important;
                padding: 7px 13px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.35);
            }
            .storm-loader-dot {
                width: 22px;
                height: 22px;
                border: 4px solid rgba(0,0,0,0.22);
                border-top-color: #000000;
                border-radius: 50%;
                animation: stormLoaderSpin 0.8s linear infinite;
                display: inline-block;
                flex: 0 0 auto;
            }
            </style>
            <div class="working-indicator-slot storm-working-indicator">
                <span class="storm-loader-dot"></span>
                <span style="font-size:20px; font-weight:900; letter-spacing:0.02em; color:black;">Working</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not st.session_state.analysis_started:
        st.stop()

    try:
        raw_text = load_local_omni_text(OMNI_DATA_FILENAME)
        analysis_cache_key = (
            hashlib.sha256(raw_text.encode("utf-8", errors="ignore")).hexdigest(),
            bool(shift_plus_1_hour),
            str(date_start),
            str(date_end),
            float(episode_level),
            float(storm_level),
            int(local_min_radius_hours),
            float(minima_split_factor),
            float(storm_limit),
            float(strong_storm_threshold),
            float(strong_mark),
            int(local_max_radius_hours),
            bool(disturbance_filter),
            float(disturbance_level),
            int(disturb_dip_count),
            int(disturb_dip_radius_hours),
        )
        analysis_results = run_storm_analysis_cached(
            raw_text,
            bool(shift_plus_1_hour),
            date_start,
            date_end,
            float(episode_level),
            float(storm_level),
            int(local_min_radius_hours),
            float(minima_split_factor),
            float(storm_limit),
            float(strong_storm_threshold),
            float(strong_mark),
            int(local_max_radius_hours),
            bool(disturbance_filter),
            float(disturbance_level),
            int(disturb_dip_count),
            int(disturb_dip_radius_hours),
        )

        # Keep the indicator visible until the whole Streamlit rerun has finished rendering.
        # This also covers table sorting, tab changes, and Extras rerenders.

        # Persistent widget helper: Streamlit removes widget state when a widget is not
        # rendered during a rerun. Since the section selector renders only one section
        # at a time, sort radios inside hidden sections would otherwise reset when the
        # user leaves the section and comes back. Store the selected value in a separate
        # permanent session_state key, and use a temporary widget key only for the radio.
        def _remember_widget_value(widget_key: str, persistent_key: str) -> None:
            st.session_state[persistent_key] = st.session_state[widget_key]

        def persistent_radio(label, options, persistent_key: str, default=None, **kwargs):
            options = list(options)
            if not options:
                raise ValueError("persistent_radio requires at least one option")
            if default is None:
                default = options[0]
            if default not in options:
                default = options[0]

            saved_value = st.session_state.get(persistent_key, default)
            if saved_value not in options:
                saved_value = default
                st.session_state[persistent_key] = saved_value

            widget_key = f"_{persistent_key}_widget"
            index = options.index(saved_value)
            return st.radio(
                label,
                options,
                index=index,
                key=widget_key,
                on_change=_remember_widget_value,
                args=(widget_key, persistent_key),
                **kwargs,
            )

        def render_table_title(title: str):
            """Render table section titles with larger text."""
            clean_title = str(title).replace("**", "")
            st.markdown(
                f'<div style="font-size:1.25rem; font-weight:700; margin:0.45rem 0 0.20rem 0; line-height:1.25;">{escape(clean_title)}</div>',
                unsafe_allow_html=True,
            )

        def render_sort_radio(options, persistent_key: str, default=None):
            """Render sorting controls with the label above the radio options."""
            st.markdown(
                '<div style="margin-bottom:0.15rem; white-space:nowrap; font-weight:400;">Sort by:</div>',
                unsafe_allow_html=True,
            )
            return persistent_radio(
                "",
                options,
                persistent_key=persistent_key,
                default=default if default is not None else options[0],
                label_visibility="collapsed",
                horizontal=True,
            )

        run_info = analysis_results["run_info"]
        omni_df = analysis_results["omni_df"]
        storms_df = analysis_results["storms_df"]
        episodes_df = analysis_results["episodes_df"]
        multi_df = analysis_results["multi_df"]
        kept_df = analysis_results["kept_df"]
        disturbances_df = analysis_results["disturbances_df"]
        excluded_df = analysis_results["excluded_df"]
        stage2_summary = analysis_results["stage2_summary"]
        clean_storms_df = analysis_results["Data-complete_storms_df"]
        clean_rows_df = analysis_results["Data-complete_rows_df"]
        peak_delays_df = analysis_results["peak_delays_df"]
        metrics_df = analysis_results["metrics_df"]
        replacements_df = analysis_results["replacements_df"]
        rejected_quality_df = analysis_results["rejected_quality_df"]
        stage3_summary = analysis_results["stage3_summary"]
        corr_all_df = analysis_results["corr_all_df"]
        corr_mod_df = analysis_results["corr_mod_df"]
        corr_int_df = analysis_results["corr_int_df"]
        corr_sup_df = analysis_results["corr_sup_df"]
        kept_display_df = analysis_results["kept_display_df"]
        disturbances_display_df = analysis_results["disturbances_display_df"]
        excluded_display_df = analysis_results["excluded_display_df"]
        stage3_all_clean_storms_df = analysis_results["stage3_all_Data-complete_storms_df"]
        stage3_peak_delays_df = analysis_results["stage3_peak_delays_df"]
        stage3_full_peak_data_df = analysis_results["stage3_full_peak_data_df"]
        stage3_replacements_df = analysis_results["stage3_replacements_df"]
        stage3_rejected_quality_df = analysis_results["stage3_rejected_quality_df"]
        stage3_all_missing_values_df = analysis_results["stage3_all_missing_values_df"]

    except FileNotFoundError as e:
        loading_indicator.empty()
        st.error(str(e))
        st.info('Upload the OMNI data file named "1964-may 2026.txt" to the same folder as the running Python file.')
        st.stop()
    except ValueError as e:
        loading_indicator.empty()
        st.error(str(e))
        st.stop()
    except Exception as e:
        loading_indicator.empty()
        st.exception(e)
        st.stop()


    # ---------------------------
    # Dashboard summary
    # ---------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total storms", f"{len(storms_df):,}")
    col2.metric("Filtered storms", f"{len(kept_df):,}")
    col3.metric("Data-complete storms", f"{len(clean_storms_df):,}")

    with st.expander("Run summary", expanded=False):
        s1a, s1b = st.columns(2)
        s1a.write(f"**Rows read:** {int(run_info.get('rows_read', 0)):,}")
        s1a.write(f"**Rows analyzed:** {int(run_info.get('rows_analyzed', 0)):,}")
        s1a.write(f"**Bad/non-data lines skipped:** {int(run_info.get('bad_rows', 0)):,}")
        s1b.write(f"**Full file span:** {fmt_dt(run_info.get('full_span_start'))} → {fmt_dt(run_info.get('full_span_end'))}")
        s1b.write(f"**Analyzed span:** {fmt_dt(run_info.get('analyzed_span_start'))} → {fmt_dt(run_info.get('analyzed_span_end'))}")


    # ---------------------------
    # Sections
    # ---------------------------
    # Streamlit tabs execute every tab body on every rerun. For this app, that means
    # long tables in inactive sections can still slow down unrelated actions such as
    # sorting a visible table. Use a single section selector instead, so only the
    # active section is rendered during each rerun.
    section_options = [
        "Overview",
        "Storm detection",
        "Main phase filtering",
        "Data-complete",
        "Correlations",
        "Downloads",
        "Extras",
    ]
    selected_tab = st.radio(
        "Section",
        section_options,
        index=section_options.index(st.session_state.get("active_section", "Overview"))
        if st.session_state.get("active_section", "Overview") in section_options else 0,
        horizontal=True,
        key="active_section",
        label_visibility="collapsed",
    )

    try:
        section_panel = st.container(border=True, key="active_section_panel")
    except TypeError:
        section_panel = st.container(border=True)
    with section_panel:
        if selected_tab == "Overview":
            stage1_class_df = count_by_class(storms_df, "storm count")

            filtered_summary_source = kept_df.copy()
            if not filtered_summary_source.empty:
                filtered_summary_source["class"] = filtered_summary_source["minDst"].apply(class_name)
            filtered_class_df = summarize_by_class(
                filtered_summary_source,
                "filtered storm count",
                [("mainphase_duration", "avg Mp duration (h)")],
            )

            clean_summary_source = metrics_df.copy()
            clean_class_df = summarize_by_class(
                clean_summary_source,
                "Data-complete storm count",
                [
                    ("mainphase_duration", "avg Mp duration (h)"),
                    ("delay_eyp_to_tmin_hours", "avg Eyp delay (h)"),
                    ("delay_bzp_to_tmin_hours", "avg Bzp delay (h)"),
                ],
            )
            clean_corr_df = correlations_selected(clean_summary_source)

            with st.container(border=True):
                left, right = st.columns(2)
                with left:
                    st.markdown(
                        """
                        <h3 style="margin:0 0 0.5rem 0; padding:0;">Results of storm detection</h3>
                        """,
                        unsafe_allow_html=True,
                    )
                    summary_left_df = pd.DataFrame(
                        [
                            {"": "Total storms", " ": len(storms_df)},
                            {"": "Storm episodes", " ": len(episodes_df)},
                            {"": "Multistorm episodes", " ": len(multi_df["episode_id"].unique()) if not multi_df.empty else 0},
                        ]
                    )
                    for _, row in summary_left_df.iterrows():
                        st.markdown(f"{row['']}: <span style='font-weight:bold'>&nbsp;&nbsp;{format_val(row[''], row[' '])}</span>", unsafe_allow_html=True)
                with right:
                    render_static_scroll_table(prettify_table_headers(stage1_class_df.copy()), key="summary_stage1_class_table", fit_to_container=True, equal_col_widths=True)

            with st.container(border=True):
                left, right = st.columns(2)
                with left:
                    st.markdown(
                        """
                        <h3 style="margin:0 0 0.5rem 0; padding:0;">Results after filtering</h3>
                        """,
                        unsafe_allow_html=True,
                    )
                    previous_storm_exclusions = int(stage2_summary["excluded_previous_storm"])
                    filtered_left_df = pd.DataFrame(
                        [
                            {"": "Filtered storms", " ": len(kept_df)},
                            {"": "Excluded as disturbances", " ": len(disturbances_df)},
                            {"": "Exclusion due to previous-storm overlap", " ": previous_storm_exclusions},
                            {"": "Average Mp duration (hours)", " ": format_mean_pm_error(stage2_summary["avg_mainphase_kept"], sem(pd.to_numeric(kept_df["mainphase_duration"], errors="coerce")) if ("mainphase_duration" in kept_df.columns and not kept_df.empty) else np.nan)},
                        ]
                    )
                    for _, row in filtered_left_df.iterrows():
                        st.markdown(f"{row['']}: <span style='font-weight:bold'>&nbsp;&nbsp;{format_val(row[''], row[' '])}</span>", unsafe_allow_html=True)
                with right:
                    filtered_class_show_df = filtered_class_df.copy()

                    # For the second Overview table, show avg Mp duration per class as
                    # mean ± standard error, matching the formatting used on the left.
                    filtered_mp_error_by_class = {}
                    if (
                        filtered_summary_source is not None
                        and not filtered_summary_source.empty
                        and "class" in filtered_summary_source.columns
                        and "mainphase_duration" in filtered_summary_source.columns
                    ):
                        for class_label, class_group in filtered_summary_source.groupby("class", dropna=False):
                            filtered_mp_error_by_class[str(class_label)] = sem(
                                pd.to_numeric(class_group["mainphase_duration"], errors="coerce")
                            )

                    if "avg Mp duration (h)" in filtered_class_show_df.columns:
                        filtered_class_show_df["avg Mp duration (h)"] = filtered_class_show_df.apply(
                            lambda r: format_mean_pm_error(
                                r.get("avg Mp duration (h)", np.nan),
                                filtered_mp_error_by_class.get(str(r.get("class", "")), np.nan),
                            ),
                            axis=1,
                        )

                    # In the second Overview table only, show compact storm-class names.
                    if "class" in filtered_class_show_df.columns:
                        filtered_class_show_df["class"] = filtered_class_show_df["class"].replace({
                            "moderate (-100 < Dst_min <= -50)": "moderate",
                            "intense (-250 < Dst_min <= -100)": "intense",
                            "super-storm (Dst_min <= -250)": "super-storm",
                        })
                    render_static_scroll_table(prettify_table_headers(filtered_class_show_df), key="summary_filtered_class_table", fit_to_container=True, equal_col_widths=True)

            with st.container(border=True):
                left, right = st.columns(2)
                with left:
                    st.markdown(
                        """
                        <div style="display:flex; align-items:center; gap:8px; margin:0 0 0.5rem 0;">
                            <h3 style="margin:0; padding:0;">Results of Data-complete storms</h3>
                            <button
                                type="button"
                                title="Data-complete storms are filtered storms for which the solar wind parameters do not contain consecutive missing OMNIWeb values during the main phase. Filtered storms with isolated missing values are also retained, with those values replaced using single-point linear interpolation."
                                style="
                                    width:24px; height:24px; border-radius:0; border:1px solid #000000;
                                    background:#F8E463; color:#111827; font-weight:700; cursor:help;
                                    line-height:22px; text-align:center; padding:0;
                                "
                                aria-label="Data-complete storms hint"
                            >?</button>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Reuse the already computed clean_class_df when available;
                    # otherwise fall back to zero.
                    clean_counts = {
                        "moderate": 0,
                        "intense": 0,
                        "super-storm": 0,
                    }
                    if clean_class_df is not None and not clean_class_df.empty:
                        count_col = "Data-complete storm count"
                        for _, class_row in clean_class_df.iterrows():
                            class_label = str(class_row.get("class", "")).lower()
                            try:
                                class_count = int(class_row.get(count_col, 0))
                            except Exception:
                                class_count = 0
                            if class_label.startswith("moderate"):
                                clean_counts["moderate"] = class_count
                            elif class_label.startswith("intense"):
                                clean_counts["intense"] = class_count
                            elif class_label.startswith("super-storm"):
                                clean_counts["super-storm"] = class_count

                    clean_left_rows = [
                        {"label": "Data-complete storms", "value": len(clean_storms_df), "kind": "normal"},
                        {"label": "class_counts", "value": clean_counts, "kind": "class_counts"},
                        {"label": "Average min Dst delay from Eyp (hours)", "value": format_mean_pm_error(stage3_summary.get("avg_delay_eyp", np.nan), stage3_summary.get("err_delay_eyp", np.nan)), "kind": "normal"},
                        {"label": "Average min Dst delay from Bzp (hours)", "value": format_mean_pm_error(stage3_summary.get("avg_delay_bzp", np.nan), stage3_summary.get("err_delay_bzp", np.nan)), "kind": "normal"},
                    ]
                    for row in clean_left_rows:
                        if row["kind"] == "class_counts":
                            counts = row["value"]
                            st.markdown(
                                "Moderate:&nbsp;&nbsp;&nbsp;<span style='font-weight:bold'>{}</span>,&nbsp;&nbsp;&nbsp;"
                                "Intense:&nbsp;&nbsp;&nbsp;<span style='font-weight:bold'>{}</span>,&nbsp;&nbsp;&nbsp;"
                                "Super-storm:&nbsp;&nbsp;&nbsp;<span style='font-weight:bold'>{}</span>".format(
                                    format_val("storms", counts["moderate"]),
                                    format_val("storms", counts["intense"]),
                                    format_val("storms", counts["super-storm"]),
                                ),
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(f"{row['label']}: <span style='font-weight:bold'>&nbsp;&nbsp;{format_val(row['label'], row['value'])}</span>", unsafe_allow_html=True)
                with right:
                    clean_corr_show_df = clean_corr_df.copy()
                    if not clean_corr_show_df.empty and clean_corr_show_df.shape[1] > 0:
                        first_corr_col = clean_corr_show_df.columns[0]
                        clean_corr_show_df[first_corr_col] = clean_corr_show_df[first_corr_col].replace({
                            "imfp": "IMFp",
                            "imf_peak": "IMFp",
                            "bzp": "Bzp",
                            "bz_peak": "Bzp",
                            "eyp": "Eyp",
                            "ey_peak": "Eyp",
                            "speedp": "Vp",
                            "speed_peak": "Vp",
                            "eyi": "Eyi",
                            "mainphase_duration": "Mp duration",
                        })
                    if "r" in clean_corr_show_df.columns:
                        clean_corr_show_df["r"] = pd.to_numeric(clean_corr_show_df["r"], errors="coerce").map(
                            lambda x: "—" if pd.isna(x) else f"{float(x):.3f}"
                        )
                    render_static_scroll_table(prettify_table_headers(clean_corr_show_df), key="summary_Data-complete_correlations_table", fit_to_container=True, equal_col_widths=True)

        elif selected_tab == "Storm detection":
            render_section_heading("Storm detection", ["Dst: nT"])
            render_table_title("Total storms")

            stage1_storm_minima_order = render_sort_radio(
                ["Date", "Dst_min"],
                persistent_key="stage1_storm_minima_order",
                default="Date",
            )

            stage1_storm_minima_df = sort_stage1_storm_minima(storms_df, stage1_storm_minima_order)
            render_table("Storm minima", stage1_storm_minima_df)

            if int(run_info.get("left_censored_boundary_episodes", 0)) > 0:
                st.warning(run_info.get("left_censored_boundary_note", "! There is a storm event preceeding the selected date start"))
    

            render_table_title("Episodes")

            stage1_episodes_order = render_sort_radio(
                ["Date", "duration", "global_min"],
                persistent_key="stage1_episodes_order",
                default="Date",
            )

            episodes_sorted_df = sort_stage1_episodes(episodes_df, stage1_episodes_order)
            render_table("Episodes", episodes_sorted_df)
            render_table_title("Multi-storm episodes")

            stage1_multi_order = render_sort_radio(
                ["Date", "duration", "global_min"],
                persistent_key="stage1_multi_order",
                default="Date",
            )

            multi_sorted_df = sort_stage1_multi_storm_minima(multi_df, stage1_multi_order)
            render_table("Multi-storm episode minima", multi_sorted_df)

        elif selected_tab == "Main phase filtering":
            render_section_heading("Main-phase filtering", ["Dst: nT", "Mp duration: hours"])
            render_table_title("Filtered storms")

            stage2_kept_order = render_sort_radio(
                ["Date", "Dst_min", "Mp duration"],
                persistent_key="stage2_kept_order",
                default="Date",
            )

            kept_sorted_df = sort_stage2_table(kept_display_df, stage2_kept_order)
            render_table("Kept storms", kept_sorted_df)

            render_table_title("Excluded as disturbances")

            stage2_disturbances_order = render_sort_radio(
                ["Date", "Dst_min", "Mp duration"],
                persistent_key="stage2_disturbances_order",
                default="Date",
            )

            disturbances_sorted_df = sort_stage2_table(disturbances_display_df, stage2_disturbances_order)
            render_table("Disturbances", disturbances_sorted_df)

            render_table_title("Excluded due to previous storm found between t_start and t_min")
            render_table("Excluded", sort_by_time_generic(excluded_display_df))

        elif selected_tab == "Data-complete":
            render_section_heading("Data-complete storms and main-phase metrics", ["Dst, IMF, Bz: nT", "Mp duration: hours", "V: km/s", "Ey: mV/m"])

            render_table_title("Data-complete storms")

            stage3_all_clean_order = render_sort_radio(
                ["Date", "Dst_min", "Mp duration", "IMFp", "Bzp", "Vp", "Eyp", "Eyi"],
                persistent_key="stage3_all_Data-complete_order",
                default="Date",
            )

            stage3_all_clean_sorted_df = sort_stage3_all_clean_storms(stage3_all_clean_storms_df, stage3_all_clean_order)
            render_table_stage3_all_clean("All Data-complete storms", stage3_all_clean_sorted_df)

            render_table_title("Peak delays")
            render_table("Peak delays", sort_by_time_generic(stage3_peak_delays_df))

            render_table_title("Full peak timeseries")
            render_table("Full peak data", sort_by_time_generic(stage3_full_peak_data_df))

            render_table_title("Replaced single missing values")
            render_table("Replaced single missing values", sort_by_time_generic(stage3_replacements_df))

            render_table_title("Rejected storms (multiple missing values)")
            rejected_storms_show_df = sort_by_time_generic(stage3_rejected_quality_df).copy()
            if not rejected_storms_show_df.empty:
                for col in rejected_storms_show_df.columns:
                    col_l = str(col).lower()
                    if pd.api.types.is_datetime64_any_dtype(rejected_storms_show_df[col]):
                        rejected_storms_show_df[col] = pd.to_datetime(rejected_storms_show_df[col], utc=True, errors="coerce").map(fmt_dt)
                    else:
                        rejected_storms_show_df[col] = rejected_storms_show_df[col].map(pretty_value)
                rejected_storms_show_df = prettify_table_headers(rejected_storms_show_df)
            render_static_scroll_table(
                rejected_storms_show_df,
                max_height=260,
                key="rejected_storms_multiple_missing_values_table",
                fit_to_container=True,
                equal_col_widths=False,
            )

            # All missing OMNI values table is kept for CSV/download only, not rendered in the app,
            # to avoid slowing down page load for large OMNI files.

        elif selected_tab == "Correlations":
            st.subheader("Pearson correlations vs |Dst_min|")
            render_table_title("Correlations")
            render_static_scroll_table(format_corr_df(corr_all_df), key="correlations_table", fit_to_container=True)
            render_table_title("Plots")
            render_correlation_plots(metrics_df)

        elif selected_tab == "Downloads":
            st.subheader("Download outputs")

            # The app no longer renders every section on every rerun. Therefore the
            # sorted DataFrames used by downloads are prepared here, only when the
            # Downloads section is opened.
            stage1_storm_minima_download_df = sort_stage1_storm_minima(storms_df, "time")

            time_col = None
            for candidate in ["start_utc", "start", "episode_start", "start_time"]:
                if candidate in episodes_df.columns:
                    time_col = candidate
                    break
            episodes_download_df = episodes_df.sort_values(by=time_col, ascending=True) if time_col else episodes_df

            multi_download_df = sort_stage1_multi_storm_minima(multi_df, "time")
            kept_download_df = sort_stage2_table(kept_display_df, "time")
            disturbances_download_df = sort_stage2_table(disturbances_display_df, "time")
            stage3_all_clean_download_df = sort_stage3_all_clean_storms(stage3_all_clean_storms_df, "time")

            stage1_downloads = [
                ("Total storms", "Total storms.csv", stage1_storm_minima_download_df),
                ("Episodes", "Episodes.csv", episodes_download_df),
                ("Multi-storm episodes", "Multi-storm episodes.csv", multi_download_df),
            ]

            stage2_downloads = [
                ("Filtered storms", "Filtered storms.csv", kept_download_df),
                ("Excluded as disturbances", "Excluded as disturbances.csv", disturbances_download_df),
                ("Excluded due to previous storm found between t_start and t_min", "Excluded due to previous storm found between t_start and t_min.csv", sort_by_time_generic(excluded_display_df)),
            ]

            stage3_downloads = [
                ("Data-complete storms", "Data-complete storms.csv", stage3_all_clean_download_df),
                ("Peak delays", "Peak delays.csv", sort_by_time_generic(stage3_peak_delays_df)),
                ("Full peak timeseries", "Full peak timeseries.csv", sort_by_time_generic(stage3_full_peak_data_df)),
                ("Replaced single missing values", "Replaced single missing values.csv", sort_by_time_generic(stage3_replacements_df)),
                ("Rejected storms (multiple missing values)", "Rejected storms (multiple missing values).csv", sort_by_time_generic(stage3_rejected_quality_df)),
                ("All missing OMNI values", "All missing OMNI values.csv", stage3_all_missing_values_df),
            ]

            correlation_downloads = [
                ("Correlations", "Correlations.csv", corr_all_df),
            ]

            ordered_output_files = stage1_downloads + stage2_downloads + stage3_downloads + correlation_downloads

            output_files = {filename: df for _title, filename, df in ordered_output_files}

            zip_bytes = zip_outputs(output_files)
            st.download_button(
                "Download all outputs as ZIP",
                data=zip_bytes,
                file_name="magnetic_storm_outputs.zip",
                mime="application/zip",
                use_container_width=True,
                on_click="ignore",
            )

            for title, filename, df in ordered_output_files:
                st.download_button(
                    f"Download {title}",
                    data=to_csv_bytes(df),
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}",
                    on_click="ignore",
                )


        elif selected_tab == "Extras":
            st.subheader("Extras")
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:8px; margin:0.45rem 0 0.20rem 0; line-height:1.25;">
                    <div style="font-size:1.25rem; font-weight:700; margin:0; padding:0;">Parameter-specific data-complete peak tables</div>
                    <button
                        type="button"
                        title="Tables show peak values of solar wind parameters during the main phase separately. A storm may be included in the analysis for one parameter, such as Bz, even if it contains missing values in another parameter, such as V or Ey. As a result, the sample size may differ between parameters and is generally larger than the fully data-complete sample."
                        style="
                            width:24px; height:24px; border-radius:0; border:1px solid #000000;
                            background:#F8E463; color:#111827; font-weight:700; cursor:help;
                            line-height:22px; text-align:center; padding:0;
                        "
                        aria-label="Parameter-Data-complete peak tables hint"
                    >?</button>
                </div>
                """,
                unsafe_allow_html=True,
            )

            extras_table_options = {
                "IMF": {
                    "key": "imf",
                    "title": "**Data-complete for IMF only**",
                    "table_name": "Extras IMF values",
                    "peak_col": "imfp",
                    "sort_key": "extras_imf_peak_order",
                },
                "Bz": {
                    "key": "bz",
                    "title": "**Data-complete for Bz only**",
                    "table_name": "Extras Bz values",
                    "peak_col": "bzp",
                    "sort_key": "extras_bz_peak_order",
                },
                "V": {
                    "key": "speed",
                    "title": "**Data-complete for V only**",
                    "table_name": "Extras V values",
                    "peak_col": "Speedp",
                    "sort_key": "extras_speed_peak_order",
                },
            }

            total_filtered_storms = int(len(kept_df)) if kept_df is not None else 0

            # Controls the width of the Extras parameter selector.
            # Increase/decrease this value to change the dropdown width.
            extras_parameter_selector_width_px = 430

            # Use a popover-based selector instead of st.selectbox here.
            # This avoids Streamlit's searchable text-input behavior and makes
            # the control behave like simple clickable choices: click to open,
            # click again to close, click an option to select it.
            if "extras_selected_parameter" not in st.session_state:
                st.session_state.extras_selected_parameter = list(extras_table_options.keys())[0]
            old_parameter_label_map = {
                "IMFp values": "IMF",
                "Bzp values": "Bz",
                "Vp values": "V",
                "IMFp": "IMF",
                "Bzp": "Bz",
                "Vp": "V",
            }
            st.session_state.extras_selected_parameter = old_parameter_label_map.get(
                st.session_state.extras_selected_parameter,
                st.session_state.extras_selected_parameter,
            )
            if st.session_state.extras_selected_parameter not in extras_table_options:
                st.session_state.extras_selected_parameter = list(extras_table_options.keys())[0]

            selected_parameter_label = st.session_state.extras_selected_parameter
            selected_parameter_label_css = selected_parameter_label.replace('\\', '\\\\').replace('"', '\\"')

            st.markdown(
                f"""
                <style>
                div[data-testid="stPopover"] {{
                    width: {extras_parameter_selector_width_px}px !important;
                    max-width: 100% !important;
                    display: block !important;
                }}

                div[data-testid="stPopover"] > button {{
                    width: 100% !important;
                    max-width: 100% !important;
                    justify-content: space-between !important;
                    cursor: pointer !important;
                    caret-color: transparent !important;
                    user-select: none !important;
                    background: #ffffff !important;
                    background-color: #ffffff !important;
                    color: #111111 !important;
                    border: 1px solid rgba(0,0,0,0.22) !important;
                    border-width: 1px !important;
                    border-radius: 9px !important;
                    padding: 0.45rem 0.75rem !important;
                    outline: none !important;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
                    font-size: 0 !important;
                    line-height: 1.1 !important;
                }}

                div[data-testid="stPopover"] > button::before {{
                    content: "{selected_parameter_label_css}" !important;
                    display: inline-flex !important;
                    align-items: center !important;
                    justify-content: flex-start !important;
                    color: #111111 !important;
                    font-size: 0.95rem !important;
                    font-weight: 700 !important;
                    line-height: 1.1 !important;
                    opacity: 1 !important;
                    filter: none !important;
                }}

                div[data-testid="stPopover"] > button::after {{
                    content: "▾" !important;
                    display: inline-flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    color: #111111 !important;
                    font-size: 0.85rem !important;
                    font-weight: 700 !important;
                    line-height: 1 !important;
                    margin-left: 0.5rem !important;
                }}

                div[data-testid="stPopover"] > button:hover,
                div[data-testid="stPopover"] > button:focus,
                div[data-testid="stPopover"] > button:active,
                div[data-testid="stPopover"] > button[aria-expanded="true"] {{
                    cursor: pointer !important;
                    caret-color: transparent !important;
                    background: #ffffff !important;
                    background-color: #ffffff !important;
                    color: #111111 !important;
                    border-color: rgba(0,0,0,0.22) !important;
                    border-width: 1px !important;
                    outline: none !important;
                    opacity: 1 !important;
                    filter: none !important;
                }}

                div[data-testid="stPopover"] > button > * {{
                    display: none !important;
                    visibility: hidden !important;
                    width: 0 !important;
                    min-width: 0 !important;
                    max-width: 0 !important;
                    height: 0 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    overflow: hidden !important;
                    opacity: 0 !important;
                }}

                /* Extras parameter menu: keep the opened choices the same width and style as the selected-value button. */
                [data-testid="stPopoverBody"],
                [data-baseweb="popover"] {{
                    width: {extras_parameter_selector_width_px}px !important;
                    max-width: {extras_parameter_selector_width_px}px !important;
                    box-sizing: border-box !important;
                }}

                [data-testid="stPopoverBody"] div[data-testid="stButton"],
                [data-baseweb="popover"] div[data-testid="stButton"] {{
                    width: 100% !important;
                    max-width: 100% !important;
                    box-sizing: border-box !important;
                    margin: 0 !important;
                }}

                [data-testid="stPopoverBody"] div[data-testid="stButton"] > button,
                [data-baseweb="popover"] div[data-testid="stButton"] > button {{
                    width: 100% !important;
                    max-width: 100% !important;
                    min-height: 38px !important;
                    justify-content: flex-start !important;
                    cursor: pointer !important;
                    caret-color: transparent !important;
                    user-select: none !important;
                    background: #ffffff !important;
                    background-color: #ffffff !important;
                    color: #111111 !important;
                    border: 1px solid rgba(0,0,0,0.22) !important;
                    border-width: 1px !important;
                    border-radius: 9px !important;
                    padding: 0.45rem 0.75rem !important;
                    outline: none !important;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
                    font-size: 0.95rem !important;
                    font-weight: 700 !important;
                    line-height: 1.1 !important;
                    box-sizing: border-box !important;
                }}

                [data-testid="stPopoverBody"] div[data-testid="stButton"] > button p,
                [data-baseweb="popover"] div[data-testid="stButton"] > button p {{
                    color: #111111 !important;
                    font-size: 0.95rem !important;
                    font-weight: 700 !important;
                    line-height: 1.1 !important;
                    margin: 0 !important;
                }}

                [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:hover,
                [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:focus,
                [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:active,
                [data-baseweb="popover"] div[data-testid="stButton"] > button:hover,
                [data-baseweb="popover"] div[data-testid="stButton"] > button:focus,
                [data-baseweb="popover"] div[data-testid="stButton"] > button:active {{
                    background: #ffffff !important;
                    background-color: #ffffff !important;
                    color: #111111 !important;
                    border-color: rgba(0,0,0,0.22) !important;
                    border-width: 1px !important;
                    outline: none !important;
                    cursor: pointer !important;
                    caret-color: transparent !important;
                }}

                /* No permanent selected/focus look on the selected parameter button or option buttons. */
                div[data-testid="stPopover"] > button:focus-visible,
                div[data-testid="stPopover"] > button:focus:not(:focus-visible),
                [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:focus-visible,
                [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:focus:not(:focus-visible),
                [data-baseweb="popover"] div[data-testid="stButton"] > button:focus-visible,
                [data-baseweb="popover"] div[data-testid="stButton"] > button:focus:not(:focus-visible) {{
                    border: 1px solid rgba(0,0,0,0.22) !important;
                    outline: none !important;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
                }}
                div[data-testid="stPopover"],
                div[data-testid="stPopover"] *,
                [data-testid="stPopoverBody"],
                [data-testid="stPopoverBody"] *,
                [data-baseweb="popover"],
                [data-baseweb="popover"] * {{
                    cursor: pointer !important;
                    caret-color: transparent !important;
                    user-select: none !important;
                }}

                .extras-parameter-summary {{
                    color: #111827 !important;
                    font-size: 1rem !important;
                    font-weight: 400 !important;
                    line-height: 1.45 !important;
                    margin-top: 0.35rem !important;
                    margin-bottom: 0.35rem !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Choose parameter for parameter-specific Data-completeness**")
            with st.popover(selected_parameter_label):
                for option_label in extras_table_options.keys():
                    if st.button(option_label, key=f"extras_parameter_choice_{extras_table_options[option_label]['key']}", use_container_width=True):
                        st.session_state.extras_selected_parameter = option_label
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass

            selected_parameter_label = st.session_state.extras_selected_parameter
            selected_option = extras_table_options[selected_parameter_label]

            if "extras_peak_table_cache" not in st.session_state:
                st.session_state.extras_peak_table_cache = {}

            def get_extras_peak_table_once(parameter_key: str, table_name: str) -> pd.DataFrame:
                cache_key = (analysis_cache_key, parameter_key)
                cache = st.session_state.extras_peak_table_cache
                if cache_key not in cache:
                    with st.spinner(f"Calculating {table_name}..."):
                        cache[cache_key] = build_parameter_clean_peak_table(kept_df, omni_df, parameter_key)
                return cache[cache_key]

            def render_parameter_peak_table(title: str, table_name: str, parameter_key: str, peak_col: str, sort_key: str):
                table_df = get_extras_peak_table_once(parameter_key, table_name)

                accepted_count = int(len(table_df)) if table_df is not None else 0
                rejected_count = max(total_filtered_storms - accepted_count, 0)
                render_table_title(title)

                order_choice = render_sort_radio(
                    ["storm", "peak value"],
                    persistent_key=sort_key,
                    default="storm",
                )

                sorted_df = sort_parameter_peak_table(table_df, peak_col, order_choice)
                one_decimal_columns = {peak_col} if str(peak_col).lower() in {"imfp", "bzp"} else set()
                render_table(table_name, sorted_df, one_decimal_columns=one_decimal_columns)
                corr_text = format_parameter_peak_correlation(table_df, peak_col)
                st.markdown(
                    "<div class='extras-parameter-summary'>"
                    f"{escape(str(corr_text))}"
                    "</div>",
                    unsafe_allow_html=True,
                )

            render_parameter_peak_table(
                selected_option["title"],
                selected_option["table_name"],
                selected_option["key"],
                selected_option["peak_col"],
                selected_option["sort_key"],
            )

    # The app has finished computing and rendering this rerun, so hide the busy indicator.
    try:
        loading_indicator.empty()
    except Exception:
        pass

    # === AUTO ADD storm_id for Stage 2 tables ===
    def add_storm_id(df):
        if df is not None and len(df) > 0:
            df = df.copy()
            df.insert(0, "storm_id", range(1, len(df)+1))
        return df

    try:
        kept_storms_df = add_storm_id(kept_storms_df)
        disturbances_df = add_storm_id(disturbances_df)
        excluded_df = add_storm_id(excluded_df)
    except Exception:
        pass
    # === END AUTO ADD ===


    # === AUTO ORDER COLUMNS Stage 2 ===
    def reorder_stage2(df, is_excluded=False):
        if df is None or len(df) == 0:
            return df
        df = df.copy()

        base_order = ["storm_id", "dst_start", "minDst", "tstart_utc", "tmin_utc", "mainphase_duration"]

        # keep only columns that exist
        base_order = [c for c in base_order if c in df.columns]

        remaining = [c for c in df.columns if c not in base_order]

        if is_excluded:
            # keep last 3 columns at the end
            tail = remaining[-3:] if len(remaining) >= 3 else remaining
            middle = [c for c in remaining if c not in tail]
            new_order = base_order + middle + tail
        else:
            new_order = base_order + remaining

        return df[new_order]

    try:
        kept_storms_df = reorder_stage2(kept_storms_df, False)
        disturbances_df = reorder_stage2(disturbances_df, False)
        excluded_df = reorder_stage2(excluded_df, True)
    except Exception:
        pass
    # === END ORDER ===
