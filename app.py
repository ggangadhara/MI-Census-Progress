"""
7th Minor Irrigation Census — Progress Monitoring System
Version: V185_PRODUCTION
Stable Clean Build
"""

import streamlit as st
import os, sys, logging, re, textwrap, json, gc, time, hashlib, traceback, io, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple

import gspread
try:
    from google.oauth2.service_account import Credentials
    USE_NEW_AUTH = True
except ImportError:
    from oauth2client.service_account import ServiceAccountCredentials
    USE_NEW_AUTH = False


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MI_Census")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
class AppConfig:
    VERSION = "V185_PRODUCTION"
    SESSION_TIMEOUT_MINUTES = 180
    MAX_UPLOAD_MB = 10
    MAX_LOGIN_TRIES = 5

    USER_MAP = {
        "Chethan_NGM": "Nagamangala Taluk",
        "Gangadhar_MLV": "Malavalli Taluk",
        "Nagarjun_KRP": "K.R. Pete Taluk",
        "Prashanth_SRP": "Srirangapatna Taluk",
        "Purushottam_PDV": "Pandavapura Taluk",
        "Siddaraju_MDY": "Mandya Taluk",
        "Sunil_MDR": "Maddur Taluk",
        "Mandya_Admin": "District Admin",
    }

    AUTHORIZED_USERS = sorted(USER_MAP.keys())


# ─────────────────────────────────────────────
# SECURITY
# ─────────────────────────────────────────────
def get_password() -> str:
    try:
        return str(st.secrets["app"]["password"]).strip()
    except Exception:
        st.error("Password missing in Streamlit Secrets")
        st.stop()


def check_session_timeout():
    if "last_active" not in st.session_state:
        st.session_state["last_active"] = time.time()
        return

    if time.time() - st.session_state["last_active"] > AppConfig.SESSION_TIMEOUT_MINUTES * 60:
        st.session_state.clear()
        st.warning("Session expired. Login again.")
        st.rerun()

    st.session_state["last_active"] = time.time()


# ─────────────────────────────────────────────
# FILE LOADER
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def smart_load_dataframe(content: bytes) -> Optional[pd.DataFrame]:
    buf = io.BytesIO(content)
    for reader in [pd.read_excel, pd.read_csv]:
        try:
            buf.seek(0)
            df = reader(buf)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────
# REPORT ENGINE
# ─────────────────────────────────────────────
def generate_all_reports(df_assign: pd.DataFrame,
                         df_monitor: pd.DataFrame,
                         taluk: str) -> Dict:

    if df_assign is None or df_assign.empty:
        raise RuntimeError("Master file empty")

    if df_monitor is None or df_monitor.empty:
        raise RuntimeError("Monitoring file empty")

    df_a = df_assign.copy()
    df_m = df_monitor.copy()

    if "User" not in df_a.columns:
        raise RuntimeError("Master missing 'User' column")

    total_col = next((c for c in df_a.columns if "Total schemes" in c), None)
    if total_col is None:
        raise RuntimeError("Master missing 'Total schemes' column")

    df_a[total_col] = pd.to_numeric(df_a[total_col], errors="coerce").fillna(0)
    grouped = df_a.groupby("User")[total_col].sum().reset_index()

    if "Enu name" not in df_m.columns:
        raise RuntimeError("Monitoring missing 'Enu name'")

    df_m["GW"] = pd.to_numeric(df_m.iloc[:, 9], errors="coerce").fillna(0)

    completed = df_m.groupby("Enu name")["GW"].sum().reset_index()

    final = pd.merge(grouped, completed,
                     left_on="User",
                     right_on="Enu name",
                     how="left").fillna(0)

    final.rename(columns={
        total_col: "Assigned",
        "GW": "Completed"
    }, inplace=True)

    final["Pct"] = np.where(
        final["Assigned"] > 0,
        final["Completed"] / final["Assigned"],
        0
    )

    # Simple graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(final["User"], final["Assigned"], color="#DADCE0")
    ax.barh(final["User"], final["Completed"], color="#34A853")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=80)
    img.seek(0)
    plt.close()

    return {"graph": img}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    st.set_page_config(page_title="MI Census V185", layout="wide")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        check_session_timeout()

    # LOGIN
    if not st.session_state["logged_in"]:
        st.title("7th Minor Irrigation Census")

        usr = st.selectbox("Select Office", ["Select…"] + AppConfig.AUTHORIZED_USERS)
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if usr != "Select…" and pwd == get_password():
                st.session_state["logged_in"] = True
                st.session_state["user"] = usr
                st.session_state["last_active"] = time.time()
                st.rerun()
            else:
                st.error("Invalid credentials")

        return

    user = st.session_state["user"]

    if user == "Mandya_Admin":
        st.success("Admin Mode")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
        return

    taluk = AppConfig.USER_MAP[user]
    st.header(f"{taluk}")

    master_file = st.file_uploader("Upload Master File", type=["xlsx", "csv"])
    monitor_file = st.file_uploader("Upload Monitoring File", type=["xlsx", "csv"])

    if st.button("Generate Reports"):
        try:
            da = smart_load_dataframe(master_file.getvalue()) if master_file else None
            dm = smart_load_dataframe(monitor_file.getvalue()) if monitor_file else None

            if da is None or da.empty:
                st.error("Master file invalid")
                return

            if dm is None or dm.empty:
                st.error("Monitoring file invalid")
                return

            result = generate_all_reports(da, dm, taluk)
            st.image(result["graph"], use_container_width=True)

        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()