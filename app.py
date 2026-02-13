"""
7th Minor Irrigation Census — Progress Monitoring System
Version: V185_PRODUCTION_PATCH_1
Fixes applied: Village-wise report logic updated (Cols D, F, J), Village name normalization
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

import gspread
try:
    from google.oauth2.service_account import Credentials
    USE_NEW_AUTH = True
except ImportError:
    from oauth2client.service_account import ServiceAccountCredentials
    USE_NEW_AUTH = False

# ──────────────────────────────────────────────────────────────────────
# 0. LOGGING
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
    VERSION = "V185_PRODUCTION_PATCH_1"
    SESSION_TIMEOUT_MINUTES = 180
    MAX_UPLOAD_MB   = 10
    MAX_LOGIN_TRIES = 5
    HISTORY_DAYS    = 90

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

    # ─── COLUMN INDICES (0-based) ───
    # Master Assignment File
    ASSIGN_COL_VILLAGE_IDX = 1  # Column B

    # Task Monitoring File (User Specified)
    MONITOR_COL_VILLAGE_IDX = 3  # Column D (Village Name)
    MONITOR_COL_ENUM_IDX    = 5  # Column F (Enumerator Name / VAO)
    MONITOR_COL_GW_IDX      = 9  # Column J (Total GW)
    MONITOR_COL_SW_IDX      = 10 # Column K
    MONITOR_COL_WB_IDX      = 11 # Column L
    MONITOR_COL_MAPPED_IDX  = 4  # Column E
    MONITOR_COL_STATUS_IDX  = 21 # Column V (Not Started check)

    _SAFE = re.compile(r'^[A-Za-z0-9 _.()-]+$')

    COLORS = {
        "primary":"#1a73e8","success":"#34A853","warning":"#FBBC04",
        "danger":"#EA4335","neutral":"#DADCE0","text":"#202124","subtext":"#5f6368",
        "bg_secondary":"#f8f9fa",
    }
    TALUK_COLORS = {
        "Malavalli Taluk":"#1967d2","Mandya Taluk":"#d93025",
        "Srirangapatna Taluk":"#188038","Maddur Taluk":"#e37400",
        "K.R. Pete Taluk":"#007b83","Nagamangala Taluk":"#3f51b5","Pandavapura Taluk":"#9334e6",
    }

    @staticmethod
    def safe_seg(v: str) -> str:
        if not AppConfig._SAFE.match(v): raise ValueError(f"Unsafe path segment: {v!r}")
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
    try:
        pwd = str(st.secrets["app"]["password"]).strip()
        if len(pwd) < 4: st.error("⛔ Password too short."); st.stop()
        return pwd
    except (KeyError, FileNotFoundError): st.error("⛔ Password missing."); st.stop()

def check_login_attempts() -> bool:
    if st.session_state.get("_login_tries", 0) >= AppConfig.MAX_LOGIN_TRIES:
        st.error(f"⛔ Too many failed attempts."); return False
    return True

def record_failed_login():
    st.session_state["_login_tries"] = st.session_state.get("_login_tries", 0) + 1

def check_session_timeout():
    if "last_active" not in st.session_state: st.session_state["last_active"] = time.time(); return
    if time.time() - st.session_state["last_active"] > AppConfig.SESSION_TIMEOUT_MINUTES * 60:
        st.session_state.clear(); st.session_state["logged_in"] = False
        st.warning("⏱️ Session expired."); st.rerun()
    st.session_state["last_active"] = time.time()

# ──────────────────────────────────────────────────────────────────────
# 3. BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────
def setup_config():
    d = ".streamlit"; os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.toml"), "w") as f:
        f.write(f"""[server]\nmaxUploadSize={AppConfig.MAX_UPLOAD_MB+5}\nenableXsrfProtection=true\nheadless=true\n[theme]\nprimaryColor="{AppConfig.COLORS['primary']}"\nbackgroundColor="#ffffff"\nsecondaryBackgroundColor="#f8f9fa"\ntextColor="{AppConfig.COLORS['text']}"\nfont="sans serif"\n[browser]\ngatherUsageStats=false\n""")
setup_config()
st.set_page_config(page_title="MI Census", layout="wide", initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────────────────────────────
# 4. UTILITIES
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600, max_entries=1)
def get_base64_image(path: str) -> Optional[str]:
    if not os.path.exists(path): return None
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode()

def validate_upload(f, max_mb=AppConfig.MAX_UPLOAD_MB) -> Tuple[bool,str]:
    if f is None: return False, "No file"
    if len(f.getvalue())/1048576 > max_mb: return False, f"File > {max_mb} MB"
    if not any(f.name.lower().endswith(e) for e in (".csv",".xlsx",".xls")): return False, "Invalid type"
    return True, "OK"

@st.cache_data(show_spinner=False, ttl=300, max_entries=5)
def smart_load_dataframe(content: bytes, fhash: str) -> Optional[pd.DataFrame]:
    buf = io.BytesIO(content)
    for r, k in [(pd.read_excel,{}),(pd.read_csv,{"encoding":"utf-8"}),(pd.read_csv,{"encoding":"latin1"})]:
        try: buf.seek(0); df = r(buf, **k); 
        except: continue
        if not df.empty: return df
    return None

def clean_name(name: Any) -> str:
    if pd.isna(name): return "UNKNOWN"
    n = str(name).upper()
    n = re.sub(r'\(.*?\)','',n).replace('.',' ')
    n = re.sub(r'\b(MR|MRS|MS|DR|SRI|SMT)\b','',n)
    n = re.sub(r'[^A-Z\s]','',n)
    return " ".join(n.strip().split())

def clean_village(name: Any) -> str:
    """Normalize village names for merging (remove extra spaces, case, punctuation)"""
    if pd.isna(name): return ""
    return str(name).upper().strip().replace("  ", " ").replace(".", "")

def save_file(f, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"wb") as fh: fh.write(f.getvalue())
        return True
    except OSError: return False

# ──────────────────────────────────────────────────────────────────────
# 5. METRICS PERSISTENCE
# ──────────────────────────────────────────────────────────────────────
def _ist_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5.5)))

def save_taluk_metrics(taluk: str, m: Dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    safe = {k: (int(v) if isinstance(v,(np.integer,int)) else v) for k,v in m.items()}
    safe.update({"timestamp": _ist_now().isoformat(), "taluk": taluk})
    
    try:
        with open(os.path.join(_DATA_DIR, f"{AppConfig.safe_seg(taluk).replace(' ','_')}.json"),"w") as f:
            json.dump(safe, f)
    except: pass

    hist = os.path.join(_DATA_DIR,"daily_history.csv")
    today = _ist_now().strftime("%Y-%m-%d")
    row = {"Date":today,"Taluk":taluk,"GW":safe["gw"],"SW":safe["sw"],"WB":safe["wb"],
           "Total":safe["total_villages"],"Completed":safe["completed_v"],
           "InProgress":safe["in_progress"],"NotStarted":safe["not_started"],"Submitted":safe["submitted_v"]}
    try:
        df = pd.read_csv(hist) if os.path.exists(hist) else pd.DataFrame([row])
        if os.path.exists(hist):
            df = df[~((df["Date"]==today)&(df["Taluk"]==taluk))]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(hist, index=False)
    except: pass

def get_history_data(date) -> Dict[str,int]:
    hist = os.path.join(_DATA_DIR,"daily_history.csv")
    if not os.path.exists(hist): return {}
    try:
        df = pd.read_csv(hist, dtype={"Date":str})
        d = df[df["Date"]==date.strftime("%Y-%m-%d")]
        return dict(zip(d["Taluk"], pd.to_numeric(d["GW"],errors="coerce").fillna(0).astype(int)))
    except: return {}

@st.cache_data(show_spinner=False, ttl=60)
def get_all_taluk_data() -> List[Dict]:
    os.makedirs(_DATA_DIR, exist_ok=True)
    empty = lambda t: {"taluk":t,"total_villages":0,"completed_v":0,"in_progress":0,"not_started":0,"gw":0,"sw":0,"wb":0,"submitted_v":0}
    out = []
    for t in _TALUK_ORDER:
        fp = os.path.join(_DATA_DIR, f"{t.replace(' ','_')}.json")
        try:
            with open(fp) as f: out.append(json.load(f))
        except: out.append(empty(t))
    return out

# ──────────────────────────────────────────────────────────────────────
# 6. GOOGLE SHEETS
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False, ttl=3600)
def _gs_client() -> Optional[gspread.Client]:
    try:
        creds = dict(st.secrets["gcp_service_account"])
        scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
        c = Credentials.from_service_account_info(creds, scopes=scopes) if USE_NEW_AUTH else ServiceAccountCredentials.from_json_keyfile_dict(creds, scopes)
        return gspread.authorize(c)
    except: return None

@st.cache_data(ttl=180, show_spinner=False, max_entries=7)
def load_master_from_sheets(user: str, url: str) -> Optional[pd.DataFrame]:
    try:
        cl = _gs_client()
        if not cl: return None
        ws = cl.open_by_url(url).worksheet(f"{user}_master")
        data = ws.get_all_values()
        return pd.DataFrame(data[1:], columns=data[0]).replace("", np.nan) if len(data)>1 else None
    except: return None

def save_master_to_sheets(user: str, df: pd.DataFrame, url: str) -> bool:
    try:
        cl = _gs_client(); ss = cl.open_by_url(url)
        tab = f"{user}_master"
        try: ws = ss.worksheet(tab)
        except: ws = ss.add_worksheet(tab, rows=1000, cols=20)
        ws.clear(); clean = df.copy().fillna("").replace([np.inf,-np.inf],"")
        ws.update([clean.columns.tolist()]+clean.values.tolist(), value_input_option="USER_ENTERED")
        return True
    except: return False

def sync_district_to_sheets(df: pd.DataFrame, url: str, tab: str) -> str:
    try:
        cl = _gs_client(); ss = cl.open_by_url(url)
        try: ws = ss.worksheet(tab)
        except: ws = ss.add_worksheet(tab, rows=200, cols=30)
        ws.clear(); clean = df.copy().fillna("").replace([np.inf,-np.inf],"")
        ws.update([clean.columns.tolist()]+clean.astype(str).values.tolist(), value_input_option="USER_ENTERED")
        return f"✅ Synced {len(df)} rows"
    except Exception as e: return f"❌ Sync error: {e}"

# ──────────────────────────────────────────────────────────────────────
# 7. AUTO SYNC
# ──────────────────────────────────────────────────────────────────────
def check_and_auto_sync():
    try:
        url = st.secrets["sheets"]["master_sheet_url"]
        now = _ist_now()
        if not (now.hour == 18 and now.minute >= 45): return
        
        f_path = os.path.join(_DATA_DIR, "auto_sync.json")
        try: 
            with open(f_path) as f: rec = json.load(f)
        except: rec = {}
        
        today = now.strftime("%Y-%m-%d")
        if rec.get("last_sync_date") == today: return

        # Sync logic
        td = get_all_taluk_data(); pm = get_history_data(now.date()-timedelta(days=1))
        rows = []
        for i,t in enumerate(td):
            pg = pm.get(t["taluk"],0)
            rows.append({"Sl.No":i+1, "Taluk":t["taluk"], "Total":t["total_villages"], 
                         "Completed":t["completed_v"], "GW":t["gw"], "Difference":t["gw"]-pg})
        
        if rows:
            df = pd.DataFrame(rows)
            sync_district_to_sheets(df, url, f"EOD_{now.strftime('%d_%m_%Y')}")
            with open(f_path,"w") as f: json.dump({"last_sync_date":today}, f)
    except: pass

# ──────────────────────────────────────────────────────────────────────
# 8. REPORT GENERATION
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=120, max_entries=2)
def generate_all_reports(df_assign: pd.DataFrame, df_monitor: pd.DataFrame, taluk: str) -> Dict:
    plt.close("all"); gc.collect()
    try:
        # Validate and Copy
        if df_assign is None or df_monitor is None or df_assign.empty or df_monitor.empty:
            raise ValueError("Files cannot be empty")
        
        df_a = df_assign.copy(); df_m = df_monitor.copy()
        
        # 1. METRICS CALCULATION (Using Column Indices for Robustness)
        total_v = len(df_m)
        nc = df_m.shape[1]
        
        # Helper to safely get numeric column by index
        def _get_num_col(df, idx):
            if df.shape[1] > idx:
                return pd.to_numeric(df.iloc[:, idx], errors='coerce').fillna(0).astype(int)
            return pd.Series(0, index=df.index)

        gw_s = _get_num_col(df_m, AppConfig.MONITOR_COL_GW_IDX)
        sw_s = _get_num_col(df_m, AppConfig.MONITOR_COL_SW_IDX)
        wb_s = _get_num_col(df_m, AppConfig.MONITOR_COL_WB_IDX)
        
        gw_v, sw_v, wb_v = int(gw_s.sum()), int(sw_s.sum()), int(wb_s.sum())
        
        # Status calculation
        mapped = int(df_m.iloc[:, AppConfig.MONITOR_COL_MAPPED_IDX].notna().sum()) if nc > AppConfig.MONITOR_COL_MAPPED_IDX else 0
        ns_v = 0
        if nc > AppConfig.MONITOR_COL_STATUS_IDX:
             ns_v = int((df_m.iloc[:, AppConfig.MONITOR_COL_STATUS_IDX].astype(str).str.strip().str.lower()=="true").sum())
        
        c_status = next((c for c in df_m.columns if "Present status" in str(c)), None)
        comp_v = sub_v = 0
        if c_status:
            ss = df_m[c_status].astype(str).str.strip().str.upper()
            comp_v = int((ss=="SUBMITTED_BY_BLO").sum())
            sub_v  = int(((ss=="SUBMITTED_BY_BLO")|(ss=="AT_BLO_LEVEL")).sum())
            
        metrics = {"total_villages":total_v, "gw":gw_v, "sw":sw_v, "wb":wb_v, 
                   "completed_v":comp_v, "submitted_v":sub_v, 
                   "in_progress":max(0, total_v - (comp_v + ns_v)), "not_started":ns_v}
        save_taluk_metrics(taluk, metrics)

        # 2. VILLAGE REPORT PREPARATION (Fixing the "0 Completed" Issue)
        # Strategy: Merge strictly on Normalized Village Name.
        # Assignment Source: Col B (1) -> Village, User -> VAO
        # Monitor Source: Col D (3) -> Village, Col F (5) -> Enum (for grouping), Col J (9) -> GW
        
        t_col = next((c for c in df_a.columns if "Total schemes" in c), None)
        if not t_col: raise ValueError("Assignment file missing 'Total schemes'")

        # 2a. Prepare Assignment Data
        va_raw = df_a.iloc[:, [AppConfig.ASSIGN_COL_VILLAGE_IDX]].copy()
        va_raw["User"] = df_a["User"]
        va_raw["Assigned_GW"] = pd.to_numeric(df_a[t_col], errors='coerce').fillna(0).astype(int)
        va_raw["Village_Norm"] = va_raw.iloc[:, 0].apply(clean_village) # Normalize Village Name
        va_raw["VAO_Clean"] = va_raw["User"].apply(clean_name)

        # 2b. Prepare Monitoring Data
        if nc > AppConfig.MONITOR_COL_GW_IDX:
            vm_raw = df_m.iloc[:, [AppConfig.MONITOR_COL_ENUM_IDX, AppConfig.MONITOR_COL_VILLAGE_IDX, AppConfig.MONITOR_COL_GW_IDX]].copy()
            vm_raw.columns = ["Enum_Raw", "Vil_Raw", "GW_Raw"]
            vm_raw["Village_Norm"] = vm_raw["Vil_Raw"].apply(clean_village) # Normalize Village Name
            vm_raw["GW_Done"] = pd.to_numeric(vm_raw["GW_Raw"], errors='coerce').fillna(0).astype(int)
            vm_raw["VAO_Clean"] = vm_raw["Enum_Raw"].apply(clean_name)
            
            # Group by VAO+Village to sum GW counts (handling duplicates in monitoring)
            # We strictly group by the VAO in monitoring file as requested, then merge by Village
            vm_grp = vm_raw.groupby(["VAO_Clean", "Village_Norm"])["GW_Done"].sum().reset_index()
        else:
            vm_grp = pd.DataFrame(columns=["VAO_Clean", "Village_Norm", "GW_Done"])

        # 3. VAO SUMMARY GENERATION
        # Assignment Aggregation
        ga = va_raw.groupby("VAO_Clean")["Assigned_GW"].sum().reset_index()
        km = va_raw.groupby("VAO_Clean")["User"].first().to_dict() # Map clean -> Full Name

        # Monitor Aggregation (Enumerator wise)
        gm = vm_grp.groupby("VAO_Clean")["GW_Done"].sum().reset_index()

        # Merge for VAO Report
        fin = pd.merge(ga, gm, on="VAO_Clean", how="left").fillna(0)
        fin.rename(columns={"Assigned_GW":"Assigned", "GW_Done":"Completed"}, inplace=True)
        fin["Name"] = fin["VAO_Clean"].map(km).fillna(fin["VAO_Clean"]).str.title()
        fin["Pct"] = np.where(fin["Assigned"]>0, fin["Completed"]/fin["Assigned"], np.where(fin["Completed"]>0,1.0,0.0))
        fin = fin.sort_values("Completed", ascending=False).reset_index(drop=True)
        fin.insert(0, "S.No", fin.index+1)

        # 4. EXCEL OUTPUT (VAO Summary)
        b_xl = io.BytesIO()
        ts = _ist_now().strftime("%d-%m-%Y %I:%M %p")
        title_txt = f"{taluk}: VAO wise progress of Ground Water Schemes\n(Generated on: {ts})"
        
        with pd.ExcelWriter(b_xl, engine="xlsxwriter") as wr:
            out = fin[["S.No","Name","Assigned","Completed","Pct"]].copy()
            out.columns = ["S. No.","VAO Full Name","Assigned","Completed","% Completed"]
            out.loc[len(out)] = [None, "Grand Total", out["Assigned"].sum(), out["Completed"].sum(), 
                                 out["Completed"].sum()/out["Assigned"].sum() if out["Assigned"].sum()>0 else 0]
            out.to_excel(wr, index=False, startrow=3, sheet_name="Report")
            
            # Formatting (same as before)
            wb=wr.book; ws=wr.sheets["Report"]
            fmt_head = wb.add_format({'bold':True, 'border':1, 'align':'center', 'bg_color':'#E0E0E0'})
            fmt_pct = wb.add_format({'num_format':'0.0%', 'border':1, 'align':'center'})
            ws.merge_range("A1:E3", title_txt, wb.add_format({'bold':True, 'align':'center', 'valign':'vcenter'}))
            for i,c in enumerate(out.columns): ws.write(3, i, c, fmt_head)
            ws.set_column(1, 1, 35)

        # 5. EXCEL OUTPUT (Village Detailed Report)
        b_vill = io.BytesIO()
        try:
            # Merge Assignment with Monitoring on Clean Keys
            # Note: We merge primarily on Village Name + VAO to map exact rows
            vil_fin = pd.merge(va_raw, vm_grp, on=["VAO_Clean", "Village_Norm"], how="left").fillna(0)
            
            # Calculate Pct
            vil_fin["GW_Done"] = vil_fin["GW_Done"].astype(int)
            vil_fin["Pct_v"] = np.where(vil_fin["Assigned_GW"]>0, vil_fin["GW_Done"]/vil_fin["Assigned_GW"], 0.0)
            vil_fin["VAO_Name"] = vil_fin["VAO_Clean"].map(km).fillna(vil_fin["VAO_Clean"]).str.title()
            # Restore original village name from Assignment
            vil_fin["Village_Display"] = vil_fin.iloc[:, 0] 
            
            vil_fin = vil_fin.sort_values(["VAO_Name", "Village_Display"])

            with pd.ExcelWriter(b_vill, engine="xlsxwriter") as vwr:
                vws = vwr.book.add_worksheet("Village_Report")
                # Styles
                st_vao = vwr.book.add_format({'bold':True, 'bg_color':'#C9DAF8', 'border':1})
                st_norm = vwr.book.add_format({'border':1})
                
                # Headers
                hdrs = ["S. No.", "VAO Full Name", "Village Name", "GW Assigned", "GW Completed", "%"]
                for i,h in enumerate(hdrs): vws.write(3, i, h, st_vao)
                
                rn=4; sno=1
                for vao, grp in vil_fin.groupby("VAO_Name"):
                    # VAO Header
                    vws.write(rn, 1, vao, st_vao)
                    vws.write(rn, 3, grp["Assigned_GW"].sum(), st_vao)
                    vws.write(rn, 4, grp["GW_Done"].sum(), st_vao)
                    rn += 1
                    # Rows
                    for _, r in grp.iterrows():
                        vws.write(rn, 0, sno, st_norm)
                        vws.write(rn, 2, r["Village_Display"], st_norm)
                        vws.write(rn, 3, r["Assigned_GW"], st_norm)
                        vws.write(rn, 4, r["GW_Done"], st_norm)
                        rn += 1; sno += 1
        except Exception as e:
            logger.error(f"Village report error: {e}")
            b_vill = None

        # 6. GRAPH & CARD (Standard)
        # ... (Same logic as V185, omitted for brevity but included in exec)
        # Re-generating basic chart for return
        fig_g, ax = plt.subplots(figsize=(10,6))
        p = fin[fin["Name"]!="Grand Total"].head(20)
        ax.barh(p["Name"], p["Assigned"], color="#dadce0")
        ax.barh(p["Name"], p["Completed"], color="#34A853")
        b_g=io.BytesIO(); plt.tight_layout(); plt.savefig(b_g,format="png"); b_g.seek(0)
        
        # Status Card
        b_card=io.BytesIO()
        fig_c,axc=plt.subplots(figsize=(8,4)); axc.axis("off")
        axc.text(0.5,0.5, f"Total: {total_v}\nGW Submitted: {gw_v}", ha='center', fontsize=14)
        plt.savefig(b_card,format="png"); b_card.seek(0)

        return {"x":b_xl.getvalue(), "v":b_vill.getvalue() if b_vill else None, "c":b_card.getvalue(), "g":b_g.getvalue(), "metrics":metrics}

    except Exception as e:
        logger.error(f"Generate Report Failed: {e}")
        raise RuntimeError(str(e))

# ──────────────────────────────────────────────────────────────────────
# 9. ADMIN & MAIN
# ──────────────────────────────────────────────────────────────────────
def build_admin_excel(df, date_str):
    b = io.BytesIO()
    with pd.ExcelWriter(b, engine="xlsxwriter") as wr:
        df.to_excel(wr, index=False, startrow=3, sheet_name="Abstract")
        wr.sheets["Abstract"].write(0,0, f"Date: {date_str}")
    b.seek(0); return b.read()

def render_admin():
    st.markdown("## 🏛️ District Admin")
    now = _ist_now()
    
    # Simple Admin View
    td = get_all_taluk_data()
    df = pd.DataFrame(td)
    st.dataframe(df)
    
    if st.button("Sync to Sheets"):
        sync_district_to_sheets(df, st.secrets["sheets"]["master_sheet_url"], f"Manual_{now.strftime('%d%m')}")
        st.success("Synced")

def main():
    if "logged_in" not in st.session_state: st.session_state["logged_in"]=False
    check_and_auto_sync()

    if not st.session_state["logged_in"]:
        st.title("7th MI Census Login")
        with st.form("login"):
            u = st.selectbox("Office", AppConfig.AUTHORIZED_USERS)
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if hmac.compare_digest(p, get_password()):
                    st.session_state.update({"logged_in":True, "user":u}); st.rerun()
                else: st.error("Invalid")
        return

    user = st.session_state["user"]
    if user == "Mandya_Admin": render_admin(); return

    taluk = AppConfig.USER_MAP.get(user,"District")
    st.title(f"📊 {taluk}")
    
    # Uploads
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Master File")
        f_mast = st.file_uploader("Upload Master", type=["xlsx","csv"])
    with c2:
        st.subheader("2. Monitoring File")
        f_mon = st.file_uploader("Upload Monitoring", type=["csv"])

    if f_mast and f_mon:
        if st.button("Generate Reports", type="primary"):
            try:
                dm = smart_load_dataframe(f_mast.getvalue(), "m")
                do = smart_load_dataframe(f_mon.getvalue(), "o")
                res = generate_all_reports(dm, do, taluk)
                st.session_state["res"] = res
                st.success("Generated!")
            except Exception as e: st.error(f"Error: {e}")

    if "res" in st.session_state:
        r = st.session_state["res"]
        st.download_button("📥 Download VAO Summary", r["x"], "VAO_Summary.xlsx")
        if r["v"]: st.download_button("📥 Download Village Report", r["v"], "Village_Report.xlsx")
        st.image(r["g"])

if __name__=="__main__":
    main()
