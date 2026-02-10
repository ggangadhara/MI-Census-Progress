# ============================================================
# MI Census Pro – ULTRA CLEAN STREAMLIT-SAFE VERSION
# Version: V200_ENTERPRISE_LOW_BANDWIDTH (FINAL)
# Author: Gangadhar
# ============================================================

import streamlit as st
import os, io, json, re, base64, hashlib, textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

class AppConfig:
    VERSION = "V200_ENTERPRISE_LOW_BANDWIDTH"
    GLOBAL_PASSWORD = "mandya"

    IMAGE_DPI_GRAPH = 80
    IMAGE_DPI_CARD = 100
    MAX_FILE_MB = 15
    CHUNK_SIZE = 1024 * 1024

    COLORS = {
        "primary": "#1a73e8",
        "success": "#34A853",
        "danger": "#EA4335",
        "neutral": "#DADCE0",
        "text": "#202124",
        "subtext": "#5f6368",
        "bg_secondary": "#f8f9fa",
        "table_green": "#92D050"
    }

    TALUK_COLORS = {
        "Malavalli Taluk": "#1967d2",
        "Mandya Taluk": "#d93025",
        "Srirangapatna Taluk": "#188038",
        "Maddur Taluk": "#e37400",
        "K.R. Pete Taluk": "#007b83",
        "Nagamangala Taluk": "#3f51b5",
        "Pandavapura Taluk": "#9334e6"
    }

    USER_MAP = {
        "Chethan_NGM": "Nagamangala Taluk",
        "Gangadhar_MLV": "Malavalli Taluk",
        "Nagarjun_KRP": "K.R. Pete Taluk",
        "Prashanth_SRP": "Srirangapatna Taluk",
        "Purushottam_PDV": "Pandavapura Taluk",
        "Siddaraju_MDY": "Mandya Taluk",
        "Sunil_MDR": "Maddur Taluk",
        "Mandya_Admin": "District Admin"
    }

    AUTHORIZED_USERS = sorted(USER_MAP.keys())


# ============================================================
# 2. STREAMLIT PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="MI Census Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 3. SAFE CACHING UTILITIES
# ============================================================

@st.cache_data(ttl=600, max_entries=10, show_spinner=False)
def load_dataframe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".csv"):
            return pd.read_csv(path, low_memory=True, dtype_backend="numpy_nullable")
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return None


@lru_cache(maxsize=1000)
def clean_name(name) -> str:
    if pd.isna(name):
        return "UNKNOWN"
    name = str(name).upper()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^A-Z\s]", "", name)
    return " ".join(name.split())


def validate_upload(uploaded_file):
    if uploaded_file.size > AppConfig.MAX_FILE_MB * 1024 * 1024:
        st.error("❌ File too large. Upload under 15 MB.")
        st.stop()


# ============================================================
# 4. CORE COMPUTATION (CACHED – NO IMAGES)
# ============================================================

@st.cache_data(ttl=600, max_entries=10, show_spinner=False)
def compute_report(df_assign, df_monitor, taluk, v_completed, v_submitted):

    df_assign["CK"] = df_assign["User"].apply(clean_name)
    df_monitor["CK"] = df_monitor["Enu name"].apply(clean_name)

    tcol = next(c for c in df_assign.columns if "Total schemes" in c)
    df_assign[tcol] = pd.to_numeric(df_assign[tcol], errors="coerce").fillna(0)

    grp_a = df_assign.groupby("CK")[tcol].sum()
    grp_m = df_monitor.groupby("CK").size()

    final = pd.concat([grp_a, grp_m], axis=1).fillna(0)
    final.columns = ["Assigned", "Completed"]

    final["% Completed"] = np.where(
        final["Assigned"] > 0,
        final["Completed"] / final["Assigned"],
        0
    )

    final = final.reset_index()
    final.insert(0, "S. No.", final.index + 1)

    metrics = {
        "total_villages": len(df_monitor),
        "completed_v": v_completed,
        "submitted_v": v_submitted
    }

    ts = (datetime.now(timezone.utc) + timedelta(hours=5.5)).strftime("%d-%m-%Y %I:%M %p")

    title = f"{taluk}: VAO wise Ground Water Census Progress\n(Generated on {ts})"

    return final, metrics, title


# ============================================================
# 5. IMAGE GENERATION (NOT CACHED)
# ============================================================

def generate_graph(df, title):
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.5)))

    df = df.sort_values("Completed")
    y = np.arange(len(df))

    colors = [
        AppConfig.COLORS["success"] if p > 0.1 else AppConfig.COLORS["danger"]
        for p in df["% Completed"]
    ]

    ax.barh(y, df["Assigned"], color=AppConfig.COLORS["neutral"])
    ax.barh(y, df["Completed"], color=colors, height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(df["CK"])
    ax.set_title(title, fontsize=14, weight="bold")

    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, dpi=AppConfig.IMAGE_DPI_GRAPH, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================
# 6. UI
# ============================================================

def inject_css():
    st.markdown("""
    <style>
    #MainMenu, footer, header {display:none;}
    .block-container {max-width:1200px;}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 7. MAIN APP
# ============================================================

def main():
    inject_css()

    if "logged" not in st.session_state:
        st.session_state.logged = False

    if not st.session_state.logged:
        st.title("7th Minor Irrigation Census")
        user = st.selectbox("Officer", ["Select"] + AppConfig.AUTHORIZED_USERS)
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if pwd == AppConfig.GLOBAL_PASSWORD and user != "Select":
                st.session_state.logged = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid login")
        return

    taluk = AppConfig.USER_MAP[st.session_state.user]
    st.header(taluk)

    base = os.path.join("user_data", st.session_state.user)
    os.makedirs(base, exist_ok=True)

    master_path = os.path.join(base, "master.csv")

    st.subheader("Master Assignment File")
    mf = st.file_uploader("Upload Master File", type=["csv", "xlsx"])
    if mf:
        validate_upload(mf)
        with open(master_path, "wb") as f:
            f.write(mf.getbuffer())
        st.success("Master file saved")

    if not os.path.exists(master_path):
        st.stop()

    st.subheader("Daily Monitoring File")
    tf = st.file_uploader("Upload Monitoring CSV", type=["csv"])
    if not tf:
        st.stop()

    validate_upload(tf)

    v1 = st.number_input("Completed Villages", min_value=0)
    v2 = st.number_input("Submitted Villages", min_value=0)

    if st.button("Generate Report"):
        df_assign = load_dataframe(master_path)
        df_monitor = pd.read_csv(tf, low_memory=True, dtype_backend="numpy_nullable")

        final, metrics, title = compute_report(
            df_assign, df_monitor, taluk, v1, v2
        )

        graph = generate_graph(final, title)

        st.success("Report Generated")
        st.image(graph, use_column_width=True)
        st.download_button("Download Graph", graph, "progress.png")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()