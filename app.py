"""
7th Minor Irrigation Census — Progress Monitoring System
Version: V185_PRODUCTION
Fixes applied: memory optimisation, security hardening, error handling, input validation, logging
"""

import streamlit as st
import os, sys, logging, hmac, re, textwrap, json, gc, time, hashlib, traceback, io, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

# ── ReportLab (PDF generation) ───────────────────────────────────────
from reportlab.lib.pagesizes import A4, landscape as rl_landscape
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer)
from reportlab.platypus.flowables import KeepInFrame, KeepTogether

import gspread
try:
    from google.oauth2.service_account import Credentials
    USE_NEW_AUTH = True
except ImportError:
    from oauth2client.service_account import ServiceAccountCredentials
    USE_NEW_AUTH = False

# ──────────────────────────────────────────────────────────────────────
# 0. LOGGING (must be first)
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MI_Census")

# ──────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
class AppConfig:
    VERSION = "V185_PRODUCTION"
    SESSION_TIMEOUT_MINUTES = 180
    MAX_UPLOAD_MB   = 10    # hard upload cap
    MAX_LOGIN_TRIES = 5     # brute-force guard per browser session
    HISTORY_DAYS    = 90    # rolling window for daily_history.csv

    USER_MAP = {
        "Chethan_NGM":     "Nagamangala Taluk",
        "Gangadhar_MLV":   "Malavalli Taluk",
        "Nagarjun_KRP":    "K.R. Pete Taluk",
        "Prashanth_SRP":   "Srirangapatna Taluk",
        "Purushottam_PDV": "Pandavapura Taluk",
        "Siddaraju_MDY":   "Mandya Taluk",
        "Sunil_MDR":       "Maddur Taluk",
        "Mandya_Admin":    "District Admin",
    }
    _officers = sorted(u for u in USER_MAP if u != "Mandya_Admin")
    AUTHORIZED_USERS = _officers + ["Mandya_Admin"]

    # Named constants for file column positions (prevents silent breakage if format changes)
    MONITOR_COL_MAPPED      = 4   # col E: enumerator mapping indicator
    MONITOR_COL_GW          = 9   # col J: Total GW schedules
    MONITOR_COL_SW          = 10  # col K: Total SW schedules
    MONITOR_COL_WB          = 11  # col L: Total WB schedules
    MONITOR_COL_NOT_STARTED = 21  # col V: Not started flag
    MONITOR_COL_ENumerator  = 5   # col F: Enumerator name (fallback index)
    ASSIGN_COL_VILLAGE      = 1   # col B: Village name in assignment file
    MONITOR_COL_VILLAGE     = 3   # col D: Village name in monitoring file

    # Regex: only allow safe characters in strings used as file-path segments
    _SAFE = re.compile(r'^[A-Za-z0-9 _.()-]+$')

    COLORS = {
        "primary":"#1a73e8","success":"#34A853","warning":"#FBBC04",
        "danger":"#EA4335","light_red":"#EE675C","neutral":"#DADCE0",
        "text":"#202124","subtext":"#5f6368","bg_light":"#ffffff","bg_secondary":"#f8f9fa",
    }
    TALUK_COLORS = {
        "Malavalli Taluk":"#1967d2","Mandya Taluk":"#d93025",
        "Srirangapatna Taluk":"#188038","Maddur Taluk":"#e37400",
        "K.R. Pete Taluk":"#007b83","Nagamangala Taluk":"#3f51b5","Pandavapura Taluk":"#9334e6",
    }

    @staticmethod
    def safe_seg(v: str) -> str:
        if not AppConfig._SAFE.match(v):
            raise ValueError(f"Unsafe path segment: {v!r}")
        return v

_TALUK_ORDER = [
    "K.R. Pete Taluk","Maddur Taluk","Malavalli Taluk","Mandya Taluk",
    "Nagamangala Taluk","Pandavapura Taluk","Srirangapatna Taluk",
]
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "central_data")

# ──────────────────────────────────────────────────────────────────────
# 2. SECURITY HELPERS
# ──────────────────────────────────────────────────────────────────────
def get_password() -> str:
    """Read password ONLY from Streamlit Secrets — no hardcoded fallback."""
    try:
        pwd = str(st.secrets["app"]["password"]).strip()
        if len(pwd) < 4:
            st.error("⛔ Password too short in Streamlit Secrets. Contact administrator.")
            st.stop()
        return pwd
    except (KeyError, FileNotFoundError):
        st.error("⛔ [app] password missing from Streamlit Secrets. Contact administrator.")
        st.stop()

def check_login_attempts() -> bool:
    n = st.session_state.get("_login_tries", 0)
    if n >= AppConfig.MAX_LOGIN_TRIES:
        st.error(f"⛔ Too many failed attempts. Refresh the page to try again.")
        return False
    return True

def record_failed_login():
    st.session_state["_login_tries"] = st.session_state.get("_login_tries", 0) + 1
    logger.warning("Failed login attempt #%d", st.session_state["_login_tries"])

def check_session_timeout():
    if "last_active" not in st.session_state:
        st.session_state["last_active"] = time.time(); return
    if time.time() - st.session_state["last_active"] > AppConfig.SESSION_TIMEOUT_MINUTES * 60:
        logger.info("Session expired: %s", st.session_state.get("user","?"))
        st.session_state.clear()
        st.session_state["logged_in"] = False
        st.warning("⏱️ Session expired — please log in again.")
        st.rerun()
    st.session_state["last_active"] = time.time()

# ──────────────────────────────────────────────────────────────────────
# 3. BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────
def setup_config():
    d = ".streamlit"
    os.makedirs(d, exist_ok=True)
    # NOTE: enableXsrfProtection = true (never set false — CSRF vulnerability)
    # NOTE: maxUploadSize aligned with MAX_UPLOAD_MB + small buffer
    content = f"""
[server]
maxUploadSize = {AppConfig.MAX_UPLOAD_MB + 5}
enableCORS = false
enableXsrfProtection = true
headless = true
enableWebsocketCompression = false
websocketPingTimeout = 300
[theme]
primaryColor = "{AppConfig.COLORS['primary']}"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "{AppConfig.COLORS['text']}"
font = "sans serif"
[browser]
gatherUsageStats = false
"""
    with open(os.path.join(d, "config.toml"), "w") as f:
        f.write(content.strip())

setup_config()
st.set_page_config(page_title="MI Census V185", layout="wide", initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────────────────────────────
# 4. UTILITIES
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600, max_entries=1)
def get_base64_image(path: str) -> Optional[str]:
    if not os.path.exists(path): return None
    try:
        with open(path,"rb") as f: return base64.b64encode(f.read()).decode()
    except OSError as e:
        logger.warning("Logo load: %s", e); return None

def validate_upload(f, max_mb=AppConfig.MAX_UPLOAD_MB) -> Tuple[bool,str]:
    if f is None: return False, "No file"
    mb = len(f.getvalue()) / 1048576
    if mb > max_mb: return False, f"File too large: {mb:.1f} MB (max {max_mb} MB)"
    name = getattr(f,"name","").lower()
    if not any(name.endswith(e) for e in (".csv",".xlsx",".xls")):
        return False, "Only CSV / XLSX / XLS allowed"
    return True, "OK"

# max_entries=5 → 5×~5 MB = 25 MB peak (7 officers but files rarely all in cache at once)
@st.cache_data(show_spinner=False, ttl=300, max_entries=5)
def smart_load_dataframe(content: bytes, fhash: str) -> Optional[pd.DataFrame]:
    buf = io.BytesIO(content)
    for reader, kw in [(pd.read_excel,{}),(pd.read_csv,{"encoding":"utf-8"}),(pd.read_csv,{"encoding":"latin1"})]:
        try:
            buf.seek(0); df = reader(buf, **kw)
            if not df.empty: return df
        except Exception: continue
    logger.error("smart_load_dataframe failed for hash=%s", fhash[:8])
    return None

def clean_name(name: Any) -> str:
    if pd.isna(name): return "UNKNOWN"
    n = str(name).upper()
    n = re.sub(r'\(.*?\)','',n).replace('.',' ')
    n = re.sub(r'\b(MR|MRS|MS|DR|SRI|SMT)\b','',n)
    n = re.sub(r'[^A-Z\s]','',n)
    return " ".join(n.strip().split())

def save_file(f, path: str) -> bool:
    if f is None: return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"wb") as fh: fh.write(f.getvalue())
        return True
    except OSError as e:
        logger.error("save_file %s: %s", path, e); return False

# ──────────────────────────────────────────────────────────────────────
# 5. METRICS PERSISTENCE
# ──────────────────────────────────────────────────────────────────────
def _ist_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5.5)))

def save_taluk_metrics(taluk: str, m: Dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    safe = {}
    for k,v in m.items():
        if isinstance(v,(np.integer,np.int64)):    safe[k] = int(v)
        elif isinstance(v,(np.floating,np.float64)): safe[k] = float(v)
        else: safe[k] = v
    safe.update({"timestamp": _ist_now().isoformat(), "taluk": taluk})

    # Write per-taluk JSON (latest only)
    try:
        with open(os.path.join(_DATA_DIR, f"{AppConfig.safe_seg(taluk).replace(' ','_')}.json"),"w") as f:
            json.dump(safe, f)
    except (OSError, ValueError) as e:
        logger.error("save_taluk_metrics JSON: %s", e)

    # Rolling history CSV (keep last HISTORY_DAYS)
    hist = os.path.join(_DATA_DIR,"daily_history.csv")
    today = _ist_now().strftime("%Y-%m-%d")
    row = {"Date":today,"Taluk":taluk,"GW":safe["gw"],"SW":safe["sw"],"WB":safe["wb"],
           "Total":safe["total_villages"],"Completed":safe["completed_v"],
           "InProgress":safe["in_progress"],"NotStarted":safe["not_started"],"Submitted":safe["submitted_v"]}
    try:
        if os.path.exists(hist):
            df = pd.read_csv(hist)
            df = df[~((df["Date"]==today)&(df["Taluk"]==taluk))]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            cutoff = (_ist_now()-timedelta(days=AppConfig.HISTORY_DAYS)).strftime("%Y-%m-%d")
            df = df[df["Date"]>=cutoff]
        else:
            df = pd.DataFrame([row])
        df.to_csv(hist, index=False)
    except Exception as e:
        logger.error("save_taluk_metrics history: %s", e)

@st.cache_data(show_spinner=False, ttl=60)
def get_history_data(date) -> Dict[str,int]:
    hist = os.path.join(_DATA_DIR,"daily_history.csv")
    if not os.path.exists(hist): return {}
    try:
        df = pd.read_csv(hist, dtype={"Date":str})
        d = df[df["Date"]==date.strftime("%Y-%m-%d")]
        return dict(zip(d["Taluk"], pd.to_numeric(d["GW"],errors="coerce").fillna(0).astype(int)))
    except Exception as e:
        logger.error("get_history_data: %s", e); return {}

@st.cache_data(show_spinner=False, ttl=60)
def get_all_taluk_data() -> List[Dict]:
    os.makedirs(_DATA_DIR, exist_ok=True)
    empty = lambda t: {"taluk":t,"total_villages":0,"completed_v":0,"in_progress":0,
                       "not_started":0,"gw":0,"sw":0,"wb":0,"submitted_v":0}
    out = []
    for t in _TALUK_ORDER:
        fp = os.path.join(_DATA_DIR, f"{t.replace(' ','_')}.json")
        try:
            if os.path.exists(fp):
                with open(fp) as f: out.append(json.load(f))
            else: out.append(empty(t))
        except (OSError, json.JSONDecodeError) as e:
            logger.error("get_all_taluk_data %s: %s", t, e); out.append(empty(t))
    return out

# ──────────────────────────────────────────────────────────────────────
# 6. GOOGLE SHEETS
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False, ttl=3600)
def _gs_client() -> Optional[gspread.Client]:
    try:
        creds = dict(st.secrets["gcp_service_account"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        if USE_NEW_AUTH:
            c = Credentials.from_service_account_info(creds, scopes=scopes)
        else:
            c = ServiceAccountCredentials.from_json_keyfile_dict(creds, scopes)
        return gspread.authorize(c)
    except KeyError:
        logger.warning("gcp_service_account not in secrets"); return None
    except Exception as e:
        logger.error("_gs_client: %s", e); return None

def _gs_retry(fn, retries=3, delay=2):
    """Retry a Google Sheets API call on transient errors (429, 503)."""
    import time as _t
    for attempt in range(retries):
        try:
            return fn()
        except gspread.exceptions.APIError as e:
            code = e.response.status_code if hasattr(e, "response") else 0
            if code in (429, 503) and attempt < retries - 1:
                wait = delay * (2 ** attempt)
                logger.warning("GSheets %d — retrying in %ds (attempt %d/%d)", code, wait, attempt+1, retries)
                _t.sleep(wait)
            else:
                raise
        except Exception:
            raise
    return None


@st.cache_data(ttl=180, show_spinner=False, max_entries=7)
def load_master_from_sheets(user: str, url: str) -> Optional[pd.DataFrame]:
    try:
        cl = _gs_client()
        if cl is None: return None
        ws = cl.open_by_url(url).worksheet(f"{user}_master")
        data = ws.get_all_values()
        if len(data) < 2: return None
        df = pd.DataFrame(data[1:], columns=data[0]).replace("", np.nan)
        return df
    except gspread.WorksheetNotFound: return None
    except gspread.exceptions.APIError as e:
        logger.error("load_master API: %s", e); return None
    except Exception as e:
        logger.error("load_master: %s", e); return None

def save_master_to_sheets(user: str, df: pd.DataFrame, url: str) -> bool:
    try:
        cl = _gs_client()
        if cl is None: return False
        ss = cl.open_by_url(url)
        tab = f"{user}_master"
        try:   ws = ss.worksheet(tab)
        except gspread.WorksheetNotFound:
            ws = ss.add_worksheet(tab, rows=1000, cols=20)
        ws.clear()
        clean = df.copy().fillna("").replace([np.inf,-np.inf],"")
        _gs_retry(lambda: ws.update([clean.columns.tolist()]+clean.values.tolist(), value_input_option="USER_ENTERED"))
        logger.info("Saved %d rows master for %s", len(df), user)
        return True
    except Exception as e:
        logger.error("save_master: %s", e); return False

def sync_district_to_sheets(df: pd.DataFrame, url: str, tab: str) -> str:
    try:
        cl = _gs_client()
        if cl is None: return "❌ Google Sheets not configured"
        ss = cl.open_by_url(url)
        try:   ws = ss.worksheet(tab)
        except gspread.WorksheetNotFound:
            ws = ss.add_worksheet(tab, rows=200, cols=30)
        ws.clear()
        clean = df.copy().fillna("").replace([np.inf,-np.inf],"")
        ws.update([clean.columns.tolist()]+clean.astype(str).values.tolist(), value_input_option="USER_ENTERED")
        return f"✅ Synced {len(df)} rows → tab '{tab}'"
    except Exception as e:
        logger.error("sync_district: %s", e)
        return f"❌ Sync error: {e}"

# ──────────────────────────────────────────────────────────────────────
# 7. END-OF-DAY AUTO SYNC
# ──────────────────────────────────────────────────────────────────────
_AUTO_SYNC_FILE = os.path.join(_DATA_DIR, "auto_sync.json")

def _load_auto_sync_record() -> Dict:
    try:
        if os.path.exists(_AUTO_SYNC_FILE):
            with open(_AUTO_SYNC_FILE) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return {"last_sync_date": "", "last_sync_time": ""}

def _save_auto_sync_record(date_str: str, time_str: str) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        with open(_AUTO_SYNC_FILE, "w") as f:
            json.dump({"last_sync_date": date_str, "last_sync_time": time_str}, f)
    except OSError as e:
        logger.error("_save_auto_sync_record: %s", e)

def check_and_auto_sync() -> None:
    """
    Called on every page load.
    If IST time is between 18:45–18:59 (6:45–6:59 PM) and today has not been
    auto-synced yet, silently sync district abstract to Google Sheets.
    This ensures the end-of-day snapshot is captured before 7 PM IST as long
    as someone uses the app during the 18:45–18:59 window.
    """
    try:
        sheet_url = st.secrets["sheets"]["master_sheet_url"]
    except (KeyError, FileNotFoundError, Exception):
        return   # Google Sheets not configured — nothing to sync

    now = _ist_now()
    # Window: 18:45:00 → 18:59:59 (syncs before 7 PM IST)
    if not (now.hour == 18 and now.minute >= 45):
        return

    today_str = now.strftime("%Y-%m-%d")
    rec = _load_auto_sync_record()
    if rec.get("last_sync_date") == today_str:
        return   # already synced today

    # Build the district abstract DataFrame (same as admin dashboard)
    try:
        taluk_data   = get_all_taluk_data()
        # Use today as both prev and curr for the nightly snapshot
        prev_data_map = get_history_data(now.date() - timedelta(days=1))
        rows = []
        for idx, t in enumerate(taluk_data):
            cg = t["gw"]; pg = prev_data_map.get(t["taluk"], 0)
            rows.append({
                "Sl.No": idx + 1, "State": "KARNATAKA", "District": "Mandya",
                "Taluk": t["taluk"].replace(" Taluk",""),
                "Total Villages": t["total_villages"],
                "Completed Villages": t["completed_v"],
                "In Progress": t["in_progress"],
                "Not Started": t["not_started"],
                "GW Submitted": t["gw"],
                "SW Submitted": t["sw"],
                "WB Submitted": t["wb"],
                "Villages Submitted": t["submitted_v"],
                f"GW {(now.date()-timedelta(days=1)).strftime('%d.%m.%Y')}": pg,
                f"GW {now.strftime('%d.%m.%Y')}": cg,
                "Difference": cg - pg,
            })
        if not rows:
            return
        df_snap = pd.DataFrame(rows)
        tr: Dict = {"Sl.No":"Total","State":"","District":"","Taluk":"Grand Total"}
        for col in df_snap.columns[4:]:
            try:    tr[col] = df_snap[col].sum()
            except (TypeError, ValueError): tr[col] = ""
        df_snap = pd.concat([df_snap, pd.DataFrame([tr])], ignore_index=True)

        tab_name = f"EOD_{now.strftime('%d_%m_%Y')}"   # e.g. EOD_12_02_2026
        msg = sync_district_to_sheets(df_snap, sheet_url, tab_name)
        if "✅" in msg:
            _save_auto_sync_record(today_str, now.strftime("%H:%M"))
            logger.info("Auto end-of-day sync SUCCESS → tab %s at %s IST", tab_name, now.strftime("%H:%M"))
        else:
            logger.error("Auto end-of-day sync FAILED: %s", msg)
    except Exception as e:
        logger.error("check_and_auto_sync: %s", e)

# ──────────────────────────────────────────────────────────────────────
# 8. REPORT GENERATION  (max_entries=2 → 2×~12 MB = 24 MB peak; safe for 1 GB RAM)
@st.cache_data(show_spinner=False, ttl=120, max_entries=2)
def generate_all_reports(df_assign: pd.DataFrame, df_monitor: pd.DataFrame, taluk: str) -> Dict:
    plt.close("all"); gc.collect()
    plt.rcParams["font.family"]     = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Roboto","Arial","sans-serif"]
    try:
        # ── Validate inputs ──────────────────────────────────────────
        if df_assign is None or df_monitor is None:
            raise ValueError("Master or monitoring data is missing.")
        df_a = df_assign.copy();  df_a.columns  = df_a.columns.str.strip()
        df_m = df_monitor.copy(); df_m.columns  = df_m.columns.str.strip()
        if df_a.empty or df_m.empty:
            raise ValueError("An uploaded file is empty.")
        if "User" not in df_a.columns:
            raise ValueError(f"Master file missing 'User' column. Found: {list(df_a.columns)}")
        t_col = next((c for c in df_a.columns if "Total schemes" in c), None)
        if t_col is None:
            raise ValueError(f"Master file has no 'Total schemes' column. Found: {list(df_a.columns)}")

        # ── Parse monitoring file ────────────────────────────────────
        nc = df_m.shape[1]
        def _ns(kw, idx):   # numeric series by column keyword or fallback index
            c = next((col for col in df_m.columns if kw in col), None)
            src = df_m[c] if c else (df_m.iloc[:,idx] if nc>idx else pd.Series(dtype=float))
            return pd.to_numeric(src, errors="coerce").fillna(0).astype(int)

        gw_s = _ns("Total schedules GW", AppConfig.MONITOR_COL_GW)
        sw_s = _ns("Total schedules SW", AppConfig.MONITOR_COL_SW)
        wb_s = _ns("Total schedules WB", AppConfig.MONITOR_COL_WB)
        total_v = len(df_m)
        mapped  = int(df_m.iloc[:,AppConfig.MONITOR_COL_MAPPED].notna().sum()) if nc>AppConfig.MONITOR_COL_MAPPED else 0
        gw_v = int(gw_s.sum()); sw_v = int(sw_s.sum()); wb_v = int(wb_s.sum())

        ns_v = 0
        if nc > 21:
            ns_v = int((df_m.iloc[:,AppConfig.MONITOR_COL_NOT_STARTED].astype(str).str.strip().str.lower()=="true").sum())

        c_status = next((c for c in df_m.columns if "Present status of village schedule" in c), None)
        comp_v = sub_v = 0
        if c_status:
            ss = df_m[c_status].astype(str).str.strip().str.upper()
            comp_v = int((ss=="SUBMITTED_BY_BLO").sum())
            sub_v  = int(((ss=="SUBMITTED_BY_BLO")|(ss=="AT_BLO_LEVEL")).sum())
        else:
            logger.warning("'Present status' column not found in %s monitoring file", taluk)

        ip_v = max(0, total_v - (comp_v + ns_v))

        metrics = {"total_villages":total_v,"mapped":mapped,"gw":gw_v,"sw":sw_v,"wb":wb_v,
                   "completed_v":comp_v,"submitted_v":sub_v,"in_progress":ip_v,"not_started":ns_v}
        save_taluk_metrics(taluk, metrics)

        # ── Village-code extraction helper ──────────────────────────
        # Both files embed a unique 6-digit census code in the village name:
        #   Assignment  : "614718 - M.Shivapura"  → 614718
        #   Monitoring  : "Aslishivapura(614722)"  → 614722
        # This code is the ONLY reliable join key — name spellings differ.
        _code_re = re.compile(r'\b(\d{5,6})\b')
        def _vcode(s: Any) -> Optional[str]:
            m = _code_re.search(str(s))
            return m.group(1) if m else None

        vil_a_col = df_a.columns[AppConfig.ASSIGN_COL_VILLAGE]  if df_a.shape[1] > AppConfig.ASSIGN_COL_VILLAGE  else None
        vil_m_col = df_m.columns[AppConfig.MONITOR_COL_VILLAGE] if df_m.shape[1] > AppConfig.MONITOR_COL_VILLAGE else None

        if vil_a_col is None or vil_m_col is None:
            raise ValueError("Village column not found in one of the files.")

        # Assign village codes to both frames
        df_a["_VCode"] = df_a[vil_a_col].astype(str).apply(_vcode)
        df_m["_VCode"] = df_m[vil_m_col].astype(str).apply(_vcode)
        df_m["_GW"]    = gw_s   # Total GW schedules (col J) — already parsed above

        codes_a = df_a["_VCode"].notna().sum()
        codes_m = df_m["_VCode"].notna().sum()
        logger.info("Village codes extracted — Assignment: %d/%d, Monitoring: %d/%d",
                    codes_a, len(df_a), codes_m, len(df_m))

        # Monitoring: sum GW completed per village code
        mon_by_code = (df_m.groupby("_VCode")["_GW"]
                       .sum().reset_index()
                       .rename(columns={"_GW": "GW_Done"}))
        mon_vil_name = (df_m.groupby("_VCode")[vil_m_col]
                        .first().reset_index()
                        .rename(columns={vil_m_col: "VillageRaw_M"}))

        # Assignment prep
        df_a[t_col] = pd.to_numeric(df_a[t_col], errors="coerce").fillna(0)
        df_a["VAO_CK"] = df_a["User"].astype(str).apply(clean_name)
        km = df_a.groupby("VAO_CK")["User"].first().to_dict()

        # ── Village-level joined frame (kept for village Excel) ──────
        vil_fin = df_a[["VAO_CK", "_VCode", vil_a_col, t_col]].copy()
        vil_fin = vil_fin.merge(mon_by_code, on="_VCode", how="left")
        vil_fin = vil_fin.merge(mon_vil_name, on="_VCode", how="left")
        vil_fin["GW_Done"] = vil_fin["GW_Done"].fillna(0).astype(int)
        vil_fin.rename(columns={t_col: "GW_Assigned"}, inplace=True)
        # Display name: prefer monitoring village name, fall back to assignment name
        vil_fin["Village"] = vil_fin["VillageRaw_M"].combine_first(vil_fin[vil_a_col]).astype(str).str.strip()
        vil_fin["VAO_Name"] = vil_fin["VAO_CK"].map(km).fillna(vil_fin["VAO_CK"]).str.title()
        vil_fin["Pct_v"] = np.where(
            vil_fin["GW_Assigned"] > 0,
            vil_fin["GW_Done"] / vil_fin["GW_Assigned"],
            np.where(vil_fin["GW_Done"] > 0, 1.0, 0.0)
        )
        vil_fin = vil_fin.sort_values(["VAO_Name", "Village"]).reset_index(drop=True)
        have_village_data = True

        # ── VAO-wise aggregation (derived from village-level join) ───
        ga = (df_a.groupby("VAO_CK")[t_col].sum()
              .reset_index().rename(columns={t_col: "Assigned"}))
        gm = (vil_fin.groupby("VAO_CK")["GW_Done"].sum()
              .reset_index().rename(columns={"GW_Done": "Completed"}))

        logger.info("VAO join → Assigned: %d, Completed: %d (via village-code join)",
                    int(df_a[t_col].sum()), int(gm["Completed"].sum()))

        del df_m; gc.collect()

        fin = pd.merge(ga, gm, on="VAO_CK", how="left").fillna(0)
        fin["Name"] = fin["VAO_CK"].map(km).fillna(fin["VAO_CK"]).str.title()
        del df_a; gc.collect()
        fin["Pct"]  = np.where(fin["Assigned"]>0, fin["Completed"]/fin["Assigned"],
                               np.where(fin["Completed"]>0,1.0,0.0))
        fin = fin.sort_values("Completed",ascending=False).reset_index(drop=True)
        fin.insert(0,"S.No",fin.index+1)
        tot_a = float(fin["Assigned"].sum())
        tot_c = float(fin["Completed"].sum())
        prog  = (tot_c/tot_a) if tot_a>0 else 0.0

        ts = (_ist_now()).strftime("%d-%m-%Y %I:%M %p")
        title_txt = (f"{taluk}: VAO wise progress of Ground Water Schemes (tube well) census "
                     f"wrt 6th Minor Irrigation Census upto 2018-19.\n(Generated on: {ts})")

        # ── VAO-wise PDF (A4 landscape, auto-scaled to ONE page) ─────
        _PW, _PH = rl_landscape(A4)          # 841.89 × 595.28 pt
        _M  = 12 * mm                         # 12 mm margin all sides
        _UW = _PW - 2 * _M                    # usable width  ≈ 769 pt
        _UH = _PH - 2 * _M                    # usable height ≈ 524 pt

        # Shared styles
        _sty_title = ParagraphStyle('vao_title', fontSize=10,
            fontName='Helvetica-Bold', alignment=TA_CENTER, leading=14,
            spaceAfter=4, textColor=rl_colors.HexColor('#202124'))
        _sty_gen = ParagraphStyle('vao_gen', fontSize=7,
            fontName='Helvetica', alignment=TA_CENTER, leading=9,
            textColor=rl_colors.HexColor('#5f6368'))

        # Build table rows + per-cell style commands
        _hdr = ['S. No.', 'VAO Full Name',
                'GW Assigned\n(6th MI Census)',
                'GW Completed\n(7th MI Census)', '% Completed']
        _rows = [_hdr]
        _sc = [  # base style commands
            ('BACKGROUND',  (0, 0), (-1, 0), rl_colors.HexColor('#D3D3D3')),
            ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',    (0, 0), (-1,-1), 8),
            ('LEADING',     (0, 0), (-1,-1), 10),
            ('ALIGN',       (0, 0), ( 0,-1), 'CENTER'),
            ('ALIGN',       (1, 0), ( 1,-1), 'LEFT'),
            ('ALIGN',       (2, 0), (-1,-1), 'CENTER'),
            ('VALIGN',      (0, 0), (-1,-1), 'MIDDLE'),
            ('GRID',        (0, 0), (-1,-1), 0.3, rl_colors.HexColor('#BDBDBD')),
            ('LINEBELOW',   (0, 0), (-1, 0), 0.8, rl_colors.HexColor('#9E9E9E')),
            ('ROWHEIGHT',   (0, 0), (-1,-1), 11),
            ('TOPPADDING',  (0, 0), (-1,-1), 2),
            ('BOTTOMPADDING',(0,0), (-1,-1), 2),
            ('LEFTPADDING', (0, 0), (-1,-1), 3),
            ('RIGHTPADDING',(0, 0), (-1,-1), 3),
        ]
        _vao_rows = fin[["S.No","Name","Assigned","Completed","Pct"]].copy()
        for _ri, _r in enumerate(_vao_rows.itertuples(), 1):
            _pct = float(_r.Pct)
            _good = _pct > 0.25 or (float(_r.Assigned) == 0 and float(_r.Completed) > 0)
            _rows.append([
                str(int(_r._1)) if pd.notna(_r._1) else '',
                str(_r.Name),
                str(int(_r.Assigned)),
                str(int(_r.Completed)),
                f"{_pct*100:.1f}%",
            ])
            _bg = rl_colors.white if _ri % 2 == 1 else rl_colors.HexColor('#F8F9FA')
            _sc.append(('BACKGROUND', (0, _ri), (-1, _ri), _bg))
            _sc.append(('BACKGROUND', (4, _ri), (4, _ri),
                        rl_colors.HexColor('#C6EFCE') if _good else rl_colors.HexColor('#FFC7CE')))
            _sc.append(('TEXTCOLOR', (4, _ri), (4, _ri),
                        rl_colors.HexColor('#006100') if _good else rl_colors.HexColor('#9C0006')))
            _sc.append(('FONTNAME', (4, _ri), (4, _ri), 'Helvetica-Bold'))

        # Grand total row
        _gt = len(_rows)
        _rows.append(['', 'Grand Total', str(int(tot_a)), str(int(tot_c)), f"{prog*100:.1f}%"])
        _sc += [
            ('BACKGROUND', (0, _gt), (-1, _gt), rl_colors.HexColor('#F2F2F2')),
            ('FONTNAME',   (0, _gt), (-1, _gt), 'Helvetica-Bold'),
            ('LINEABOVE',  (0, _gt), (-1, _gt), 0.8, rl_colors.HexColor('#757575')),
        ]

        # Column widths (proportional, sum = _UW)
        _cw = [_UW * p for p in [0.05, 0.42, 0.17, 0.18, 0.16]]
        _tbl = Table(_rows, colWidths=_cw, repeatRows=1)
        _tbl.setStyle(TableStyle(_sc))

        # Title + table, shrink-wrapped into one page
        _title_line = title_txt.split('\n')[0]
        _gen_line   = title_txt.split('\n')[1] if '\n' in title_txt else ''
        _title_para = Paragraph(_title_line, _sty_title)
        _gen_para   = Paragraph(_gen_line, _sty_gen)
        _title_h    = 35   # pt reserved for title block
        _kif = KeepInFrame(_UW, _UH - _title_h, [_tbl], mode='shrink')

        b_xl = io.BytesIO()
        SimpleDocTemplate(b_xl, pagesize=rl_landscape(A4),
            leftMargin=_M, rightMargin=_M, topMargin=_M, bottomMargin=_M
        ).build([_title_para, _gen_para, Spacer(1, 4), _kif])
        b_xl.seek(0)
        del _rows, _sc, _vao_rows; gc.collect()

        # ── Village-wise PDF (A4 landscape, columns fit to width) ────
        b_vill = None
        if have_village_data:
            try:
                vill_title_line = (f"{taluk}: VAO & Village wise GW Schedules — "
                                   f"Assigned (6th MI Census) vs Completed (7th MI Census)")

                # Column widths: S.No | VAO Name | Village Name | Assigned | Completed | %
                _vw = [_UW * p for p in [0.04, 0.26, 0.36, 0.11, 0.12, 0.10]]

                # Shared cell style
                _v_base = [
                    ('FONTSIZE',      (0, 0), (-1,-1), 8),
                    ('LEADING',       (0, 0), (-1,-1), 10),
                    ('VALIGN',        (0, 0), (-1,-1), 'MIDDLE'),
                    ('GRID',          (0, 0), (-1,-1), 0.25, rl_colors.HexColor('#BDBDBD')),
                    ('TOPPADDING',    (0, 0), (-1,-1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1,-1), 2),
                    ('LEFTPADDING',   (0, 0), (-1,-1), 3),
                    ('RIGHTPADDING',  (0, 0), (-1,-1), 3),
                ]

                # Styles for different row types
                def _vstyle(extra): return TableStyle(_v_base + extra)

                _hdr_row = [['S. No.', 'VAO Full Name', 'Village Name',
                              'GW Assigned\n(6th MI Census)',
                              'GW Completed\n(7th MI Census)', '% Completed']]
                _hdr_sty = [
                    ('BACKGROUND',  (0,0), (-1,0), rl_colors.HexColor('#D3D3D3')),
                    ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
                    ('ALIGN',       (0,0), (-1,0), 'CENTER'),
                    ('LINEBELOW',   (0,0), (-1,0), 0.8, rl_colors.HexColor('#9E9E9E')),
                    ('ROWHEIGHT',   (0,0), (-1,0), 22),
                    ('TOPPADDING',  (0,0), (-1,0), 3),
                    ('BOTTOMPADDING',(0,0),(-1,0), 3),
                ]

                # Build VAO groups as KeepTogether blocks
                _story_v = []
                _sno = 1
                _grand_a = _grand_c = 0

                _sty_vtitle = ParagraphStyle('vtitle2', fontSize=10,
                    fontName='Helvetica-Bold', alignment=TA_CENTER, leading=14,
                    spaceAfter=2, textColor=rl_colors.HexColor('#202124'))
                _sty_vgen = ParagraphStyle('vgen2', fontSize=7,
                    fontName='Helvetica', alignment=TA_CENTER, leading=9,
                    spaceAfter=4, textColor=rl_colors.HexColor('#5f6368'))

                # Page header
                _story_v += [
                    Paragraph(vill_title_line, _sty_vtitle),
                    Paragraph(f"Generated on: {ts}", _sty_vgen),
                ]

                # Global header row (repeated via Table repeatRows on each page)
                # We build ONE big table for the whole report with repeatRows=1
                _all_rows  = [_hdr_row[0]]
                _all_sty   = (list(_v_base) + _hdr_sty)
                _row_idx   = 1   # current table row index (0 = header)

                for _vao_name, _vao_grp in vil_fin.groupby("VAO_Name", sort=False):
                    _vao_a = int(_vao_grp["GW_Assigned"].sum())
                    _vao_c = int(_vao_grp["GW_Done"].sum())
                    _vao_p = (_vao_c / _vao_a) if _vao_a > 0 else (1.0 if _vao_c > 0 else 0.0)

                    # VAO header row (light blue)
                    _all_rows.append([
                        '', _vao_name,
                        f"[{len(_vao_grp)} villages]",
                        str(_vao_a), str(_vao_c),
                        f"{_vao_p*100:.1f}%"
                    ])
                    _all_sty += [
                        ('BACKGROUND',  (0, _row_idx), (-1, _row_idx), rl_colors.HexColor('#C9DAF8')),
                        ('FONTNAME',    (0, _row_idx), (-1, _row_idx), 'Helvetica-Bold'),
                        ('ALIGN',       (0, _row_idx), ( 0, _row_idx), 'CENTER'),
                        ('ALIGN',       (1, _row_idx), ( 2, _row_idx), 'LEFT'),
                        ('ALIGN',       (3, _row_idx), (-1, _row_idx), 'CENTER'),
                        ('LINEABOVE',   (0, _row_idx), (-1, _row_idx), 0.5, rl_colors.HexColor('#7BAFD4')),
                    ]
                    _row_idx += 1

                    # Village rows
                    for _, _vr in _vao_grp.iterrows():
                        _va = int(_vr["GW_Assigned"]); _vc = int(_vr["GW_Done"])
                        _vp = float(_vr["Pct_v"])
                        _good = _vp > 0.25 or (_va == 0 and _vc > 0)
                        _all_rows.append([
                            str(_sno), '',
                            str(_vr["Village"]),
                            str(_va), str(_vc),
                            f"{_vp*100:.1f}%"
                        ])
                        _alt_bg = rl_colors.white if _sno % 2 == 1 else rl_colors.HexColor('#F8F9FA')
                        _all_sty += [
                            ('BACKGROUND', (0, _row_idx), (-1, _row_idx), _alt_bg),
                            ('ALIGN',      (0, _row_idx), ( 0, _row_idx), 'CENTER'),
                            ('ALIGN',      (2, _row_idx), ( 2, _row_idx), 'LEFT'),
                            ('ALIGN',      (3, _row_idx), (-1, _row_idx), 'CENTER'),
                            ('BACKGROUND', (5, _row_idx), (5, _row_idx),
                             rl_colors.HexColor('#C6EFCE') if _good else rl_colors.HexColor('#FFC7CE')),
                            ('TEXTCOLOR',  (5, _row_idx), (5, _row_idx),
                             rl_colors.HexColor('#006100') if _good else rl_colors.HexColor('#9C0006')),
                            ('FONTNAME',   (5, _row_idx), (5, _row_idx), 'Helvetica-Bold'),
                        ]
                        _sno += 1; _row_idx += 1

                    # Sub-total row (yellow)
                    _all_rows.append(['', f"Sub-Total — {_vao_name}", '',
                                      str(_vao_a), str(_vao_c), f"{_vao_p*100:.1f}%"])
                    _all_sty += [
                        ('BACKGROUND', (0, _row_idx), (-1, _row_idx), rl_colors.HexColor('#FFF2CC')),
                        ('FONTNAME',   (0, _row_idx), (-1, _row_idx), 'Helvetica-Bold'),
                        ('ALIGN',      (0, _row_idx), ( 0, _row_idx), 'CENTER'),
                        ('ALIGN',      (1, _row_idx), ( 2, _row_idx), 'LEFT'),
                        ('ALIGN',      (3, _row_idx), (-1, _row_idx), 'CENTER'),
                        ('LINEABOVE',  (0, _row_idx), (-1, _row_idx), 0.4, rl_colors.HexColor('#C8A900')),
                        ('LINEBELOW',  (0, _row_idx), (-1, _row_idx), 0.6, rl_colors.HexColor('#C8A900')),
                    ]
                    _grand_a += _vao_a; _grand_c += _vao_c
                    _row_idx += 1

                # Grand total row (grey)
                _grand_p = (_grand_c / _grand_a) if _grand_a > 0 else 0.0
                _all_rows.append(['', 'Grand Total', '',
                                   str(_grand_a), str(_grand_c), f"{_grand_p*100:.1f}%"])
                _all_sty += [
                    ('BACKGROUND', (0, _row_idx), (-1, _row_idx), rl_colors.HexColor('#E8EAED')),
                    ('FONTNAME',   (0, _row_idx), (-1, _row_idx), 'Helvetica-Bold'),
                    ('ALIGN',      (0, _row_idx), ( 0, _row_idx), 'CENTER'),
                    ('ALIGN',      (1, _row_idx), ( 2, _row_idx), 'LEFT'),
                    ('ALIGN',      (3, _row_idx), (-1, _row_idx), 'CENTER'),
                    ('LINEABOVE',  (0, _row_idx), (-1, _row_idx), 1.0, rl_colors.HexColor('#5f6368')),
                ]

                _big_tbl = Table(_all_rows, colWidths=_vw, repeatRows=1)
                _big_tbl.setStyle(TableStyle(_all_sty))
                _story_v.append(_big_tbl)

                b_vill = io.BytesIO()
                SimpleDocTemplate(b_vill, pagesize=rl_landscape(A4),
                    leftMargin=_M, rightMargin=_M, topMargin=_M, bottomMargin=_M
                ).build(_story_v)
                b_vill.seek(0)
                del vil_fin, _all_rows, _all_sty, _story_v; gc.collect()
            except Exception as ve:
                logger.error("Village PDF generation failed: %s", ve, exc_info=True)
                b_vill = None

        # ── Status Card ──────────────────────────────────────────────
        card = [["Total No. of Villages",total_v],["No. of Completed Villages",comp_v],
                ["Villages work in progress",ip_v],["Villages work not started",ns_v],
                ["Villages mapped to enumerator",mapped],["Ground Water schedules submitted",gw_v],
                ["Surface Water schedules submitted",sw_v],["Water Body schedules submitted",wb_v],
                ["Villages submitted by enumerators",sub_v]]
        fig_c,axc=plt.subplots(figsize=(11.5,max(6.0,len(card)*0.8+2.5))); axc.axis("off")
        tbl=axc.table(cellText=[["  "+textwrap.fill(r[0],60),str(r[1])] for r in card],
                      colLabels=["Description","Count"],colWidths=[0.8,0.2],loc="center",bbox=[0,0,1,1])
        tbl.auto_set_font_size(False)
        hc = AppConfig.TALUK_COLORS.get(taluk,AppConfig.COLORS["primary"])
        for (r,c),cell in tbl.get_celld().items():
            cell.set_edgecolor(AppConfig.COLORS["neutral"]); cell.set_linewidth(1)
            if r==0: cell.set_facecolor(hc); cell.set_text_props(weight="bold",color="white",size=13); cell.set_height(0.08)
            else:
                cell.set_facecolor("white" if r%2==0 else AppConfig.COLORS["bg_secondary"])
                cell.set_text_props(size=12,color=AppConfig.COLORS["text"]); cell.set_height(0.09)
            if c==0: cell.set_text_props(ha="left")
            elif c==1: cell.set_text_props(ha="center")
        axc.set_title(f"{taluk} Status Report\n(Generated on: {ts})",fontweight="bold",fontsize=16,pad=20,color="black")
        b_card=io.BytesIO(); plt.savefig(b_card,format="png",dpi=80,bbox_inches="tight",pad_inches=0.1)
        b_card.seek(0); plt.close(fig_c)   # close immediately

        # ── Bar Chart ────────────────────────────────────────────────
        p=fin.sort_values("Completed",ascending=True).reset_index(drop=True)
        del fin; gc.collect()
        fig_g,ax=plt.subplots(figsize=(14,max(8,len(p)*0.55)))
        p["N"]=p["Name"].apply(lambda x: f"{x.split()[0]} {x.split()[1][0]}." if len(x.split())>1 else x)
        ys=np.arange(len(p))
        cols2=[AppConfig.COLORS["success"] if (pc>0.25 or (a==0 and c>0)) else AppConfig.COLORS["danger"]
               for pc,a,c in zip(p["Pct"],p["Assigned"],p["Completed"])]
        ax.barh(ys,p["Assigned"],color=AppConfig.COLORS["neutral"],label="Assigned",height=0.7)
        ax.barh(ys,p["Completed"],color=cols2,height=0.5)
        ax.invert_yaxis(); sns.despine(left=True,bottom=True)
        ax.xaxis.grid(True,linestyle="--",alpha=0.5,color="#dadce0")
        ax.set_yticks(ys); ax.set_yticklabels(p["N"],fontsize=12,fontweight="bold",color=AppConfig.COLORS["subtext"])
        mv=max(float(p["Assigned"].max()),1.0)
        for i,(a,c,pc) in enumerate(zip(p["Assigned"],p["Completed"],p["Pct"])):
            lbl=f"{int(c)} (100%)" if (a==0 and c>0) else f"{int(c)} ({pc*100:.1f}%)"
            ax.text(c+mv*0.01,i,lbl,va="center",weight="bold",size=11)
            ax.text(max(c+mv*0.01+len(lbl)*mv*0.017+mv*0.02,a+mv*0.02),i,f"{int(a)}",
                    va="center",ha="left",color=AppConfig.COLORS["subtext"],weight="bold",size=11)
        ax.margins(x=0.25); ax.set_ylim(-1,len(p)+2)
        ax.set_title("\n".join(textwrap.wrap(title_txt,90)),fontsize=14,weight="bold",pad=40,color=AppConfig.COLORS["text"])
        ax.set_xlabel("No of GW Schemes as per 6th MI Census upto 2018-19",fontsize=12,weight="bold",color=AppConfig.COLORS["subtext"])
        ax.annotate(f"GWS SUMMARY | Assigned: {int(tot_a):,} | Completed: {int(tot_c):,} | Progress: {prog*100:.2f}%",
                    xy=(0.5,1),xytext=(0,15),xycoords="axes fraction",textcoords="offset points",
                    ha="center",va="bottom",fontsize=12,weight="bold",color="white",
                    bbox=dict(boxstyle="round,pad=0.6",fc="black",ec="none",alpha=1.0))
        ax.legend(handles=[Patch(facecolor=AppConfig.COLORS["neutral"],label="Assigned"),
                            Patch(facecolor=AppConfig.COLORS["success"],label="Completed > 25%"),
                            Patch(facecolor=AppConfig.COLORS["danger"],label="Completed ≤ 25%")],
                  loc="lower right",fontsize=11,framealpha=0.9)
        b_g=io.BytesIO(); plt.tight_layout(); plt.savefig(b_g,format="png",dpi=80)
        b_g.seek(0); plt.close(fig_g)  # close immediately

        del p; gc.collect(); plt.close("all")
        logger.info("Report OK: %s", taluk)
        return {"x":b_xl,"v":b_vill,"c":b_card,"g":b_g,"metrics":metrics}

    except ValueError as e: raise RuntimeError(str(e)) from None
    except Exception as e:
        logger.error("generate_all_reports: %s\n%s", e, traceback.format_exc())
        raise RuntimeError(f"Report generation failed: {e}") from None
    finally:
        plt.close("all"); gc.collect()

# ──────────────────────────────────────────────────────────────────────
# 8. ADMIN EXCEL
# ──────────────────────────────────────────────────────────────────────
def build_admin_excel(df: pd.DataFrame, date_str: str) -> bytes:
    b = io.BytesIO()
    try:
        with pd.ExcelWriter(b, engine="xlsxwriter") as wr:
            df.to_excel(wr, index=False, startrow=3, sheet_name="Daily_Abstract")
            wb=wr.book; ws=wr.sheets["Daily_Abstract"]
            F=lambda **k: wb.add_format(k)
            ft=F(bold=True,font_size=14,align="center",valign="vcenter",text_wrap=True,border=1,bg_color="#1a73e8",font_color="white")
            fh=F(bold=True,border=1,align="center",valign="vcenter",bg_color="#E0E0E0",text_wrap=True,font_size=10)
            fb=F(border=1,align="center",valign="vcenter",font_size=10)
            fl=F(border=1,align="left",valign="vcenter",font_size=10)
            fta=F(bold=True,border=1,align="center",bg_color="#FFF2CC",font_size=10)
            ftl=F(bold=True,border=1,align="left",bg_color="#FFF2CC",font_size=10)
            fdp=F(bold=True,border=1,align="center",bg_color="#C6EFCE",font_color="#006100",font_size=10)
            fdn=F(border=1,align="center",bg_color="#FFC7CE",font_color="#9C0006",font_size=10)
            nc=len(df.columns)
            ws.merge_range(0,0,2,nc-1,f"7th Minor Irrigation Census — Mandya District Daily Abstract\nDate: {date_str}",ft)
            for ci,cn in enumerate(df.columns): ws.write(3,ci,cn,fh)
            for ri,row in enumerate(df.values):
                rn=4+ri; tot=str(row[0]).strip().lower() in ("total","grand total")
                for ci,v in enumerate(row):
                    cn=df.columns[ci]
                    if tot:   f=ftl if ci in (1,3) else fta
                    elif cn=="Difference":
                        try:  f=fdp if float(v)>=0 else fdn
                        except (ValueError, TypeError): f=fb
                    else: f=fl if ci in (1,3) else fb
                    ws.write(rn,ci,v,f)
            for ci,w in enumerate([6,10,10,14,12,12,14,12,14,14,14,14,12,12,12][:nc]):
                ws.set_column(ci,ci,w)
            ws.set_row(3,40)
    except Exception as e:
        logger.error("build_admin_excel: %s", e)
        return None
    b.seek(0); return b.read()

# ──────────────────────────────────────────────────────────────────────
# 9. CSS / FOOTER
# ──────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
html,body,[class*="css"]{{font-family:'Roboto',sans-serif;}}
#MainMenu,footer,header{{visibility:hidden!important;height:0!important;}}
[data-testid="stDecoration"],[data-testid="stFooter"],[data-testid="stStatusWidget"],.stDeployButton{{display:none!important;}}
.block-container{{padding-top:5rem!important;padding-bottom:10rem!important;max-width:1200px;}}
[data-testid="InputInstructions"]{{display:none!important;}}
.status-pill{{display:inline-flex;align-items:center;padding:.5rem 1rem;background:#e6f4ea;color:#137333;border-radius:999px;font-weight:500;border:1px solid #ceead6;}}
.section-header{{font-size:1.1rem;font-weight:600;color:{AppConfig.COLORS['primary']};margin-top:.5rem;text-transform:uppercase;}}
.section-sub{{font-size:.9rem;color:#5f6368;margin-top:.15rem;}}
@media(prefers-color-scheme:dark){{.section-sub{{color:#adb5bd;}}}}
[data-theme="dark"] .section-sub{{color:#adb5bd!important;}}
[data-theme="light"] .section-sub{{color:#5f6368!important;}}
.custom-footer{{position:fixed;left:0;bottom:0;width:100%;background:#000!important;color:#fff!important;text-align:center;padding:1.5rem 1rem 2.5rem;border-top:1px solid #333;z-index:2147483647!important;font-size:15px!important;line-height:1.6;}}
@media(max-width:640px){{.custom-footer{{font-size:13px!important;}}}}
</style>
<div class="custom-footer">Design &amp; Developed by <b>Gangadhar</b> | Statistical Inspector, Taluk Office Malavalli, Mandya</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# 10. ADMIN DASHBOARD
# ──────────────────────────────────────────────────────────────────────
def render_admin():
    st.markdown("## 🏛️ 7th Minor Irrigation Census — District Abstract")
    now = _ist_now()
    today_s = now.strftime("%Y-%m-%d")
    today_d = now.strftime("%d.%m.%Y")
    try:    sheet_url=st.secrets["sheets"]["master_sheet_url"]; use_gs=True
    except (KeyError, FileNotFoundError, Exception): sheet_url=None; use_gs=False

    # ── MASTER DATA MANAGEMENT (Admin only) ───────────────────────────
    with st.expander("📂 Manage Master Assignment Files (All Taluks)", expanded=False):
        if not use_gs:
            st.warning("⚠️ Google Sheets not configured. Add [sheets] master_sheet_url to Streamlit Secrets.")
        else:
            st.info("👆 Only District Admin can upload or update master assignment files. "
                    "Individual taluk officers have read-only access.")

            # Taluk selector — map display name → username key
            _admin_taluk_map = {v: k for k, v in AppConfig.USER_MAP.items() if k != "Mandya_Admin"}
            sel_taluk = st.selectbox(
                "Select Taluk to manage",
                options=sorted(_admin_taluk_map.keys()),
                key="admin_taluk_sel",
            )
            sel_user = _admin_taluk_map[sel_taluk]

            # Show current status for selected taluk
            cur = load_master_from_sheets(sel_user, sheet_url)
            if cur is not None:
                st.success(f"✅ {sel_taluk}: {len(cur)} rows currently in Google Sheets")
                action = "update"
            else:
                st.warning(f"⚠️ {sel_taluk}: No master data found in Google Sheets")
                action = "initialize"

            fa=st.file_uploader(
                f"Upload Master Assignment file for {sel_taluk} (Excel/CSV)",
                type=["xlsx","csv"], key=f"admin_master_{sel_user}",
            )
            if fa:
                ok, msg = validate_upload(fa)
                if not ok:
                    st.error(f"⚠️ {msg}")
                else:
                    btn_label = "💾 Update in Google Sheets" if action == "update" else "💾 Initialize Google Sheets"
                    if action == "update":
                        st.warning(f"⚠️ This will REPLACE existing master data for {sel_taluk}.")
                    if st.button(btn_label, type="primary", key="admin_master_save"):
                        with st.spinner(f"Uploading master file for {sel_taluk}…"):
                            _fa_bytes = fa.getvalue()
                            dfa2 = smart_load_dataframe(_fa_bytes, hashlib.md5(_fa_bytes).hexdigest())
                            if dfa2 is None:
                                st.error("❌ Cannot read the uploaded file.")
                            elif save_master_to_sheets(sel_user, dfa2, sheet_url):
                                st.success(f"✅ Master data for {sel_taluk} saved to Google Sheets ({len(dfa2)} rows)!")
                                st.cache_data.clear()
                            else:
                                st.error("❌ Failed to save. Check Google Sheet sharing settings.")

            if cur is not None:
                with st.expander(f"👁️ Preview current master data — {sel_taluk}", expanded=False):
                    st.dataframe(cur.head(20), use_container_width=True, hide_index=True)
                    st.caption(f"Showing first 20 of {len(cur)} rows")

    st.markdown("---")
    c1,c2,_=st.columns([2,2,4])
    with c1: prev=st.date_input("Previous Date",value=now.date()-timedelta(days=1))
    with c2: curr=st.date_input("Current Date",value=now.date())

    td=get_all_taluk_data(); pm=get_history_data(prev)
    rows=[]
    for i,t in enumerate(td):
        cg=t["gw"]; pg=pm.get(t["taluk"],0)
        rows.append({"Sl.No":i+1,"State":"KARNATAKA","District":"Mandya",
                     "Taluk":t["taluk"].replace(" Taluk",""),
                     "Total Villages":t["total_villages"],"Completed Villages":t["completed_v"],
                     "In Progress":t["in_progress"],"Not Started":t["not_started"],
                     "GW Submitted":t["gw"],"SW Submitted":t["sw"],"WB Submitted":t["wb"],
                     "Villages Submitted":t["submitted_v"],
                     f"GW {prev.strftime('%d.%m.%Y')}":pg,
                     f"GW {curr.strftime('%d.%m.%Y')}":cg,"Difference":cg-pg})
    if not rows: st.info("⏳ No taluk data yet."); return
    df=pd.DataFrame(rows)
    tr={"Sl.No":"Total","State":"","District":"","Taluk":"Grand Total"}
    for col in df.columns[4:]:
        try:    tr[col]=df[col].sum()
        except (TypeError, ValueError): tr[col]=""
    df=pd.concat([df,pd.DataFrame([tr])],ignore_index=True)
    st.dataframe(df,use_container_width=True,hide_index=True)

    tab=f"Abstract_{now.strftime('%d_%m_%Y')}"
    xb=build_admin_excel(df,today_d)
    c1,c2,c3=st.columns([1,1,2])
    with c1:
        if xb is None:
            st.error("❌ Excel generation failed. Check logs.")
        else:
          st.download_button("📥 Download Excel (Formatted)",xb,
                           f"Mandya_Abstract_{now.strftime('%d%m%Y')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True,type="primary")
    with c2:
        if use_gs:
            if st.button("☁️ Sync to Google Sheets",use_container_width=True,type="secondary"):
                with st.spinner("Syncing…"):
                    msg=sync_district_to_sheets(df,sheet_url,tab)
                    (st.success if "✅" in msg else st.error)(msg)
        else: st.caption("Configure [sheets] in Secrets.")
    with c3:
        if use_gs: st.caption(f"Sync → tab **`{tab}`** in master sheet.")

    st.markdown("---")
    st.markdown("#### 📊 Today's Report Status")
    sc=st.columns(len(_TALUK_ORDER))
    for i,tn in enumerate(_TALUK_ORDER):
        fp=os.path.join(_DATA_DIR,f"{tn.replace(' ','_')}.json")
        with sc[i]:
            color,label,time_str="#EA4335","❌ No Data",""
            if os.path.exists(fp):
                try:
                    with open(fp) as f: jd=json.load(f)
                    dt=datetime.fromisoformat(jd.get("timestamp","")).astimezone(timezone(timedelta(hours=5.5)))
                    same=dt.strftime("%Y-%m-%d")==today_s
                    color="#34A853" if same else "#FBBC04"
                    label="✅ Today" if same else f"🕒 {dt.strftime('%d %b')}"
                    time_str=dt.strftime("%I:%M %p")   # e.g. "03:45 PM" IST
                except Exception as e: logger.warning("Status card %s: %s",tn,e); color="#DADCE0"; label="⚠️"; time_str=""
            short=tn.replace(" Taluk","")
            time_html=(f"<br><span style='font-size:.65rem;color:#5f6368;font-weight:500'>{time_str}</span>"
                       if time_str else "")
            st.markdown(
                f"<div style='text-align:center;padding:8px 4px;border-radius:8px;"
                f"background:{color}20;border:1px solid {color};'>"
                f"<b style='font-size:.75rem;color:{color}'>{short}</b><br>"
                f"<span style='font-size:.7rem;color:#5f6368'>{label}</span>"
                f"{time_html}</div>",
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    if "logged_in" not in st.session_state: st.session_state["logged_in"]=False
    if st.session_state["logged_in"]: check_session_timeout()
    check_and_auto_sync()   # silent end-of-day sync every page interaction (23:45–23:59 IST)

    # ── Login ──────────────────────────────────────────────────────
    if not st.session_state["logged_in"]:
        _,col,_=st.columns([0.1,0.8,0.1])
        with col:
            img=get_base64_image("logo.png")
            if img: st.markdown(f'<div style="display:flex;justify-content:center;margin-bottom:1rem;"><img src="data:image/png;base64,{img}" width="160" style="border-radius:12px;"></div>',unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center'>7th Minor Irrigation Census</h2><p style='text-align:center;color:#5f6368'>Secure Progress Monitoring System</p>",unsafe_allow_html=True)
            if not check_login_attempts(): return
            with st.form("login_form"):
                usr=st.selectbox("Select Office",["Select…"]+AppConfig.AUTHORIZED_USERS)
                pwd=st.text_input("Password",type="password")
                if st.form_submit_button("Secure Login",type="primary",use_container_width=True):
                    if usr=="Select…": st.warning("Please select your office.")
                    elif hmac.compare_digest(pwd, get_password()) and usr in AppConfig.AUTHORIZED_USERS:
                        st.session_state.update({"logged_in":True,"user":usr,"last_active":time.time(),
                                                  "_login_tries":0,"show_update_master":False,"report_data":None})
                        logger.info("Login: %s", usr); st.rerun()
                    else:
                        record_failed_login()
                        rem=AppConfig.MAX_LOGIN_TRIES-st.session_state.get("_login_tries",0)
                        st.error(f"⛔ Incorrect password. {rem} attempt(s) remaining.")
        st.markdown("<div style='height:50vh'></div>",unsafe_allow_html=True); return

    user=st.session_state["user"]

    # ── Admin ──────────────────────────────────────────────────────
    if user=="Mandya_Admin":
        c1,c2=st.columns([.75,.25])
        with c1: st.markdown("<h3>👤 Administrator — Mandya District</h3>",unsafe_allow_html=True)
        with c2:
            if st.button("Log Out",type="secondary"): logger.info("Logout: %s",user); st.session_state.clear(); st.rerun()
        st.markdown("---"); render_admin(); return

    # ── Officer ────────────────────────────────────────────────────
    taluk=AppConfig.USER_MAP.get(user,"District")
    c1,c2=st.columns([.75,.25])
    with c1: st.markdown(f"<h3>📊 {taluk}</h3>",unsafe_allow_html=True)
    with c2:
        if st.button("Log Out",type="secondary"): logger.info("Logout: %s",user); st.session_state.clear(); st.rerun()
    st.markdown("<div style='margin-bottom:1.5rem;border-bottom:1px solid #dadce0;'></div>",unsafe_allow_html=True)

    try:    sheet_url=st.secrets["sheets"]["master_sheet_url"]; use_gs=bool(sheet_url)
    except (KeyError, FileNotFoundError, Exception): sheet_url=None; use_gs=False

    # ── Master data section ────────────────────────────────────────
    st.markdown('<div class="section-header">📂 Master Data Management</div>',unsafe_allow_html=True)
    with st.container():
        if use_gs:
            st.markdown('<div class="status-pill"><span style="margin-right:8px">☁️</span>Master data synced with Google Sheets</div>',unsafe_allow_html=True)
            st.markdown("<div style='margin-top:.75rem'></div>",unsafe_allow_html=True)
            dm=load_master_from_sheets(user,sheet_url)
            if dm is not None:
                st.success(f"✅ Loaded {len(dm)} rows from Google Sheets")
                # Officers get Refresh only — upload/update is Admin-only
                if st.button("🔄 Refresh",type="secondary"):
                    st.cache_data.clear(); st.rerun()
                st.caption("📌 Master data is managed by the District Admin. Contact admin to update.")
            else:
                st.warning("⚠️ Master data not yet loaded for your taluk.")
                st.info("📌 Please contact the District Admin (Mandya_Admin) to upload the master assignment file for your taluk.")
        else:
            st.info("💡 Google Sheets not configured — local file mode.")
            uf=os.path.join("user_data",user); os.makedirs(uf,exist_ok=True)
            pa=os.path.join(uf,"master_assignment"); saved=os.path.exists(pa)
            if saved and not st.session_state.get("update_mode",False):
                st.markdown('<div class="status-pill"><span style="margin-right:8px">✅</span>Master Assignment File Active</div>',unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:.5rem'></div>",unsafe_allow_html=True)
                if st.button("✏️ Update Master File",type="secondary"): st.session_state["update_mode"]=True; st.rerun()
            else:
                if saved:
                    if st.button("❌ Cancel Update"): st.session_state["update_mode"]=False; st.rerun()
                st.markdown("Upload **Master Assignment** file (Excel/CSV).")
                fl=st.file_uploader(" ",type=["xlsx","csv"],key="u_ml",label_visibility="collapsed")
                if fl:
                    ok,msg=validate_upload(fl)
                    if not ok: st.error(f"⚠️ {msg}")
                    elif save_file(fl,pa): st.session_state["update_mode"]=False; st.toast("Saved!"); st.rerun()

    # ── Daily reports section ──────────────────────────────────────
    st.markdown("<div style='margin:2rem 0;border-bottom:1px solid #dadce0'></div>",unsafe_allow_html=True)
    st.markdown('<div class="section-header">🚀 Daily Progress Reports</div>',unsafe_allow_html=True)
    if "report_data" not in st.session_state: st.session_state["report_data"]=None
    def clr(): st.session_state["report_data"]=None

    dfa=None; pal=None; ready=False
    if use_gs:
        dfa=load_master_from_sheets(user,sheet_url); ready=dfa is not None
    else:
        pal=os.path.join("user_data",user,"master_assignment"); ready=os.path.exists(pal)

    if not ready:
        st.warning("⚠️ Upload master assignment file first.")
    else:
        f3=st.file_uploader("Upload Task Monitoring File (CSV)",type=["csv"],on_change=clr,
                             help="Village counts auto-calculated from 'Present status of village schedule'.")
        if f3:
            ok,msg=validate_upload(f3)
            if not ok: st.error(f"⚠️ {msg}")
            else:
                if st.button("⚡ Generate Reports",type="primary",use_container_width=True):
                    st.session_state["report_data"]=None; gc.collect()
                    with st.spinner("Processing…"):
                        try:
                            if use_gs:
                                       if dfa is not None and not dfa.empty:
                                          da = dfa
                                       else:
                                           da = load_master_from_sheets(user, sheet_url)
                                       if da is None or da.empty:
                                          st.error("Master data unavailable.")
                                          st.stop()
                            else:
                                raw=open(pal,"rb").read()
                                da=smart_load_dataframe(raw,hashlib.md5(raw).hexdigest()); del raw; gc.collect()
                                if da is None: st.error("Master file unreadable."); st.stop()
                            mb=f3.getvalue()
                            dm2=smart_load_dataframe(mb,hashlib.md5(mb).hexdigest()); del mb; gc.collect()
                            if dm2 is None: st.error("Monitoring file unreadable."); st.stop()
                            res=generate_all_reports(da,dm2,taluk)
                            st.session_state["report_data"]=res
                        except RuntimeError as e: st.error(f"❌ {e}"); logger.error("Report: %s",e)
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {e}")
                            logger.error("Unexpected: %s\n%s",e,traceback.format_exc())

    if st.session_state.get("report_data"):
        d=st.session_state["report_data"]
        st.success("✅ Reports Generated"); st.markdown("---")
        # ── 1. Progress Graph ─────────────────────────────────────────
        c1,c2=st.columns([.7,.3])
        with c1:
            st.markdown('<p class="section-header">1. VAO wise Progress Graph</p>'
                        '<p class="section-sub">VAO-wise GW schedules — Assigned vs Completed.</p>',
                        unsafe_allow_html=True)
        with c2:
            st.download_button("📊 Download VAO wise Graph",d["g"],
                               "VAO_Progress_Graph.png","image/png",
                               use_container_width=True,type="primary")
        st.image(d["g"],use_container_width=True)
        st.markdown("<div style='margin:1.5rem 0;border-bottom:1px solid #f1f3f4'></div>",unsafe_allow_html=True)
        # ── 2. VAO-wise Summary Excel ─────────────────────────────────
        c1,c2=st.columns([.7,.3])
        with c1:
            st.markdown('<p class="section-header">2. VAO wise Summary</p>'
                        '<p class="section-sub">VAO-wise Assigned vs Completed GW schedules.</p>',
                        unsafe_allow_html=True)
        with c2:
            st.download_button("📥 Download VAO wise Progress Report",d["x"],
                               "VAO_Summary_Report.pdf",
                               mime="application/pdf",
                               use_container_width=True,type="primary")
        st.markdown("<div style='margin:1.5rem 0;border-bottom:1px solid #f1f3f4'></div>",unsafe_allow_html=True)
        # ── 3. Village-wise Report Excel ──────────────────────────────
        c1,c2=st.columns([.7,.3])
        with c1:
            st.markdown('<p class="section-header">3. VAO and Village wise Progress Report</p>'
                        '<p class="section-sub">VAO-wise village breakdown — Assigned vs Completed GW schedules.</p>',
                        unsafe_allow_html=True)
        with c2:
            if d.get("v"):
                st.download_button("📥 Download VAO and Village wise Progress Report",d["v"],
                                   "Village_Wise_Report.pdf",
                                   mime="application/pdf",
                                   use_container_width=True,type="primary")
            else:
                st.caption("⚠️ Village column not detected in files.")
        st.markdown("<div style='margin:1.5rem 0;border-bottom:1px solid #f1f3f4'></div>",unsafe_allow_html=True)
        # ── 4. Taluk Status Card ──────────────────────────────────────
        c1,c2=st.columns([.7,.3])
        with c1:
            st.markdown('<p class="section-header">4. Taluk Status Card</p>'
                        '<p class="section-sub">Summary card — optimised for sharing.</p>',
                        unsafe_allow_html=True)
        with c2:
            st.download_button("📋 Download Taluk Status",d["c"],
                               "Taluk_Status_Card.png","image/png",
                               use_container_width=True,type="primary")
        st.image(d["c"],width=600)

if __name__=="__main__":
    main()
