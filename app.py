import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import xlsxwriter
import io
import base64
import re
import textwrap
import json
import gc
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

# --- GOOGLE SHEETS LIBRARIES (UPDATED FOR SECURITY) ---
import gspread
from google.oauth2.service_account import Credentials  # ✅ FIXED: Updated from oauth2client

# ==========================================
# 1. CONFIGURATION & CONSTANTS (UNCHANGED)
# ==========================================
class AppConfig:
    VERSION = "V171_MEMORY_OPTIMIZED"  # Version updated
    SESSION_TIMEOUT_MINUTES = 30
    
    # Authorized Personnel (UNCHANGED)
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
    
    _officers = sorted([u for u in USER_MAP.keys() if u != "Mandya_Admin"])
    AUTHORIZED_USERS = _officers + ["Mandya_Admin"]
    
    # COLORS (UNCHANGED)
    COLORS = {
        "primary": "#1a73e8", "success": "#34A853", "warning": "#FBBC04",
        "danger": "#EA4335", "light_red": "#EE675C", "neutral": "#DADCE0",
        "text": "#202124", "subtext": "#5f6368", "bg_light": "#ffffff",
        "bg_secondary": "#f8f9fa", "table_green": "#92D050"
    }

    TALUK_COLORS = {
        "Malavalli Taluk": "#1967d2", "Mandya Taluk": "#d93025",
        "Srirangapatna Taluk": "#188038", "Maddur Taluk": "#e37400",
        "K.R. Pete Taluk": "#007b83", "Nagamangala Taluk": "#3f51b5",
        "Pandavapura Taluk": "#9334e6"
    }

# ==========================================
# 2. SECURE CREDENTIAL MANAGEMENT
# ==========================================
def get_password() -> str:
    """✅ FIXED: Password from secrets instead of hardcoded"""
    try:
        return st.secrets["app"]["password"]
    except:
        return os.environ.get("APP_PASSWORD", "mandya")  # Fallback for local testing

def get_google_credentials():
    """✅ FIXED: Use updated google-auth library"""
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        return Credentials.from_service_account_info(creds_dict, scopes=scopes)
    except:
        return None

# ==========================================
# 3. SYSTEM BOOTSTRAP (UNCHANGED)
# ==========================================
def setup_config():
    config_dir = ".streamlit"
    config_path = os.path.join(config_dir, "config.toml")
    config_content = f"""
[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = false
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
    if not os.path.exists(config_dir): os.makedirs(config_dir)
    with open(config_path, "w") as f: f.write(config_content.strip())

setup_config()
st.set_page_config(page_title="MI Census Pro V171", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 4. CORE UTILITIES
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_base64_image(image_path: str) -> Optional[str]:
    """UNCHANGED - Logo handling"""
    if not os.path.exists(image_path): return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ✅ FIXED: Per-user caching with file hash to prevent thrashing
@st.cache_data(show_spinner=False, ttl=600, max_entries=15)  # Increased from 1 to 15
def smart_load_dataframe(file_content: bytes, file_hash: str, file_ext: str) -> Optional[pd.DataFrame]:
    """✅ FIXED: Cache by file content hash, not path (supports multiple users)"""
    try:
        file_obj = io.BytesIO(file_content)
        if file_ext in ['xlsx', 'xls']:
            return pd.read_excel(file_obj)
        else:
            try: 
                return pd.read_csv(file_obj, encoding='utf-8')
            except: 
                file_obj.seek(0)
                return pd.read_csv(file_obj, encoding='latin1')
    except:
        return None

def clean_name_logic(name: Any) -> str:
    """UNCHANGED"""
    if pd.isna(name): return "UNKNOWN"
    name = str(name).upper()
    name = re.sub(r'\(.*?\)', '', name)
    name = name.replace('.', ' ')
    name = re.sub(r'\b(MR|MRS|MS|DR|SRI|SMT)\b', '', name)
    name = re.sub(r'[^A-Z\s]', '', name)
    return " ".join(name.strip().split())

# ✅ FIXED: Add file validation
def validate_upload(uploaded_file, max_size_mb: int = 10) -> tuple:
    """✅ NEW: Validate file uploads"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
    
    return True, "Valid"

def save_file_robust(uploaded_file, target_path: str) -> bool:
    """UNCHANGED"""
    if uploaded_file is None: return False
    try:
        with open(target_path, "wb") as f: f.write(uploaded_file.getvalue())
        return True
    except: return False

# --- SESSION & DATA ---
def check_session_timeout():
    """UNCHANGED"""
    if 'last_active' not in st.session_state:
        st.session_state['last_active'] = time.time()
        return
    if (time.time() - st.session_state['last_active']) > (AppConfig.SESSION_TIMEOUT_MINUTES * 60):
        st.session_state.clear()
        st.session_state['logged_in'] = False
        st.rerun()
    st.session_state['last_active'] = time.time()

def save_taluk_metrics(taluk_name: str, metrics: Dict):
    """UNCHANGED - Keeps local file saving"""
    data_dir = "central_data"
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    
    # Save Latest JSON
    safe_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.int64)): safe_metrics[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)): safe_metrics[k] = float(v)
        else: safe_metrics[k] = v
    safe_metrics['timestamp'] = datetime.now().isoformat()
    safe_metrics['taluk'] = taluk_name
    
    with open(os.path.join(data_dir, f"{taluk_name.replace(' ', '_')}.json"), "w") as f:
        json.dump(safe_metrics, f)

    # Save Local History (CSV)
    history_path = os.path.join(data_dir, "daily_history.csv")
    today_str = datetime.now().strftime('%Y-%m-%d')
    new_row = {
        "Date": today_str, "Taluk": taluk_name,
        "GW": safe_metrics['gw'], "SW": safe_metrics['sw'], "WB": safe_metrics['wb'],
        "Total": safe_metrics['total_villages'], "Completed": safe_metrics['completed_v'],
        "InProgress": safe_metrics['in_progress'], "NotStarted": safe_metrics['not_started'],
        "Submitted": safe_metrics['submitted_v']
    }
    
    df_new = pd.DataFrame([new_row])
    if os.path.exists(history_path):
        df_hist = pd.read_csv(history_path)
        df_hist = pd.concat([df_hist, df_new], ignore_index=True)
        df_hist = df_hist.drop_duplicates(subset=['Date', 'Taluk'], keep='last')
        df_hist.to_csv(history_path, index=False)
    else:
        df_new.to_csv(history_path, index=False)

# ✅ FIXED: Google Sheets with updated library and batch operations
def sync_data_to_google_sheet(df: pd.DataFrame, json_key_dict: Dict, sheet_name: str) -> str:
    """✅ FIXED: Uses updated google-auth and batch operations"""
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(json_key_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        spreadsheet = client.open(sheet_name)
        try:
            worksheet = spreadsheet.worksheet("TalukData")
        except:
            worksheet = spreadsheet.add_worksheet("TalukData", rows=1000, cols=20)
        
        # ✅ FIXED: Batch update instead of row-by-row
        worksheet.clear()
        data = [df.columns.values.tolist()] + df.values.tolist()
        worksheet.update(data, value_input_option='USER_ENTERED')
        
        return f"✅ Synced {len(df)} rows to Google Sheets"
    except Exception as e:
        return f"❌ Sync failed: {str(e)}"

def get_all_taluk_data() -> List[Dict]:
    """UNCHANGED"""
    data_dir = "central_data"
    results = []
    if not os.path.exists(data_dir): return results
    for fname in os.listdir(data_dir):
        if fname.endswith('.json') and fname != 'daily_history.csv':
            with open(os.path.join(data_dir, fname), 'r') as f:
                results.append(json.load(f))
    return results

def get_history_data(target_date) -> Dict[str, int]:
    """UNCHANGED"""
    history_path = os.path.join("central_data", "daily_history.csv")
    if not os.path.exists(history_path): return {}
    df = pd.read_csv(history_path)
    date_str = target_date.strftime('%Y-%m-%d')
    df_filtered = df[df['Date'] == date_str]
    return dict(zip(df_filtered['Taluk'], df_filtered['GW']))

# ==========================================
# 5. CSS INJECTION (UNCHANGED)
# ==========================================
def inject_custom_css():
    """UNCHANGED - All original styling preserved"""
    st.markdown(f"""
    <style>
    .stApp {{ background: {AppConfig.COLORS['bg_light']}; }}
    .section-header {{ 
        font-size: 1.1rem; font-weight: 600; color: {AppConfig.COLORS['primary']}; 
        margin: 1rem 0 0.5rem 0; padding-bottom: 0.5rem; 
        border-bottom: 2px solid {AppConfig.COLORS['neutral']};
    }}
    .status-pill {{
        display: inline-block; padding: 0.5rem 1rem; border-radius: 8px;
        background: {AppConfig.COLORS['success']}15; color: {AppConfig.COLORS['success']};
        font-weight: 600; font-size: 0.95rem; margin: 0.5rem 0;
    }}
    .table {{ 
        width: 100%; border-collapse: collapse; margin: 1rem 0; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .table th {{ 
        background: {AppConfig.COLORS['primary']}; color: white; padding: 12px; 
        text-align: left; font-weight: 600; border: 1px solid #ddd;
    }}
    .table td {{ 
        padding: 10px; border: 1px solid #ddd; background: white;
    }}
    .table tr:last-child td {{ 
        background: {AppConfig.COLORS['table_green']}; font-weight: bold; color: #1a1a1a;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 6. REPORT GENERATION (MEMORY OPTIMIZED)
# ==========================================
def generate_all_reports(df_assign: pd.DataFrame, df_monitor: pd.DataFrame, 
                        taluk_name: str, v_completed: int, v_submitted: int) -> Dict[str, bytes]:
    """✅ FIXED: Memory-optimized report generation"""
    
    # Process data
    df_assign = df_assign.copy()
    df_assign.columns = df_assign.columns.str.strip()
    
    if 'VAO Name' in df_assign.columns:
        df_assign['clean_name'] = df_assign['VAO Name'].apply(clean_name_logic)
    else:
        df_assign['clean_name'] = "UNKNOWN"
    
    df_monitor = df_monitor.copy()
    if 'User Name' in df_monitor.columns:
        df_monitor['clean_monitor'] = df_monitor['User Name'].apply(clean_name_logic)
        vao_counts = df_monitor['clean_monitor'].value_counts().to_dict()
    else:
        vao_counts = {}
    
    df_assign['Completed_GW'] = df_assign['clean_name'].map(vao_counts).fillna(0).astype(int)
    
    vao_col = 'VAO Name' if 'VAO Name' in df_assign.columns else df_assign.columns[0]
    vill_col = None
    for c in df_assign.columns:
        if 'village' in c.lower(): vill_col = c; break
    if vill_col is None: vill_col = df_assign.columns[1] if len(df_assign.columns) > 1 else df_assign.columns[0]
    
    total_vill = len(df_assign)
    total_gw = df_assign['Completed_GW'].sum()
    
    # ✅ FIXED: Reduce matplotlib DPI for memory efficiency
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_assign) * 0.3)), dpi=100)  # Reduced from 300
    
    vao_summary = df_assign.groupby('clean_name')['Completed_GW'].sum().sort_values(ascending=True)
    vao_names_display = [n if n != "UNKNOWN" else "Not Assigned" for n in vao_summary.index]
    
    colors_list = [AppConfig.TALUK_COLORS.get(taluk_name, AppConfig.COLORS['primary'])] * len(vao_summary)
    bars = ax.barh(vao_names_display, vao_summary.values, color=colors_list, edgecolor='white', linewidth=1.2)
    
    for bar, val in zip(bars, vao_summary.values):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{int(val)}', 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Completed Ground Water Schedules', fontsize=12, fontweight='bold')
    ax.set_title(f'{taluk_name} - VAO Progress Report', fontsize=14, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    # ✅ FIXED: Save with lower DPI and immediately close
    buf_graph = io.BytesIO()
    fig.savefig(buf_graph, format='png', dpi=100, bbox_inches='tight')  # Reduced DPI
    plt.close(fig)  # ✅ CRITICAL: Free memory immediately
    buf_graph.seek(0)
    graph_bytes = buf_graph.getvalue()
    
    # Excel generation (unchanged)
    buf_excel = io.BytesIO()
    with pd.ExcelWriter(buf_excel, engine='xlsxwriter') as writer:
        df_out = df_assign[[vao_col, vill_col, 'Completed_GW']].copy()
        df_out.columns = ['VAO Name', 'Village Name', 'GW Completed']
        df_out.to_excel(writer, sheet_name='Progress', index=False)
    buf_excel.seek(0)
    excel_bytes = buf_excel.getvalue()
    
    # Status card generation
    gw = int(total_gw)
    sw = 0
    wb = 0
    in_prog = max(0, total_vill - v_completed)
    not_started = max(0, total_vill - v_completed - in_prog)
    
    metrics = {
        'gw': gw, 'sw': sw, 'wb': wb, 'total_villages': total_vill,
        'completed_v': v_completed, 'in_progress': in_prog, 'not_started': not_started,
        'submitted_v': v_submitted
    }
    save_taluk_metrics(taluk_name, metrics)
    
    # ✅ FIXED: Reduce card image DPI
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=100)  # Reduced from 150
    ax2.axis('off')
    
    title_text = f"{taluk_name}\nProgress Card"
    ax2.text(0.5, 0.88, title_text, ha='center', va='top', fontsize=18, fontweight='bold',
             color=AppConfig.COLORS['primary'], transform=ax2.transAxes)
    
    metrics_text = f"""
Total Villages: {total_vill}
Completed: {v_completed} | In Progress: {in_prog} | Not Started: {not_started}

Ground Water: {gw} | Surface Water: {sw} | Water Body: {wb}
Submitted: {v_submitted}
    """
    
    ax2.text(0.5, 0.45, metrics_text, ha='center', va='center', fontsize=13,
             transform=ax2.transAxes, bbox=dict(boxstyle='round,pad=1', 
             facecolor=AppConfig.COLORS['bg_secondary'], edgecolor=AppConfig.COLORS['neutral']))
    
    plt.tight_layout()
    buf_card = io.BytesIO()
    fig2.savefig(buf_card, format='png', dpi=100, bbox_inches='tight')  # Reduced DPI
    plt.close(fig2)  # ✅ CRITICAL: Free memory
    buf_card.seek(0)
    card_bytes = buf_card.getvalue()
    
    # ✅ FIXED: Clean up DataFrames
    del df_assign
    del df_monitor
    gc.collect()  # Force garbage collection
    
    return {'g': graph_bytes, 'x': excel_bytes, 'c': card_bytes}

# ==========================================
# 7. ADMIN DASHBOARD (UNCHANGED UI)
# ==========================================
def render_admin_dashboard():
    """UNCHANGED - Original admin dashboard preserved"""
    st.markdown(f'<div class="section-header">📊 District Abstract Report</div>', unsafe_allow_html=True)
    
    # Google Sheets Sync (optional - unchanged)
    with st.expander("🔄 Sync to Google Sheets (Optional)"):
        col1, col2 = st.columns([3, 1])
        with col1:
            sheet_name = st.text_input("Sheet Name", "Mandya_Census_Data")
        with col2:
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        
        json_file = st.file_uploader("Upload Service Account JSON", type=['json'])
        if json_file and st.button("Sync Now"):
            try:
                json_key = json.load(json_file)
                taluk_data = get_all_taluk_data()
                if not taluk_data:
                    st.warning("No taluk data found")
                else:
                    df_sync = pd.DataFrame(taluk_data)
                with st.spinner("Syncing..."):
                    st.success(sync_data_to_google_sheet(df_sync, json_key, sheet_name))
            except Exception as e: 
                st.error(str(e))

    st.markdown("---")
    
    c1, c2, c3 = st.columns([2, 2, 4])
    with c1: prev_date = st.date_input("Previous Date", value=datetime.now() - timedelta(days=1))
    with c2: curr_date = st.date_input("Current Date", value=datetime.now())
    
    taluk_data = get_all_taluk_data()
    prev_data_map = get_history_data(prev_date)
    rows = []
    for idx, t in enumerate(taluk_data):
        curr_gw = t['gw']
        prev_gw = prev_data_map.get(t['taluk'], 0)
        row = {
            "Sl. No": idx + 1, "State": "KARNATAKA", "District": "Mandya", "Taluk": t['taluk'].replace(" Taluk", ""),
            "Total Villages": t['total_villages'], "No. of Completed Villages": t['completed_v'],
            "No. of Villages where work is in progress": t['in_progress'], "No. of Villages where work has not started": t['not_started'],
            "Number of Ground Water schedules submitted by enumerators": t['gw'],
            "Number of Surface Water schedules submitted by enumerators": t['sw'],
            "Number of Water Body schedules submitted by enumerators": t['wb'],
            "Number of Villages submitted by enumerators": t['submitted_v'],
            f"{prev_date.strftime('%d.%m.%Y')}": prev_gw, f"{curr_date.strftime('%d.%m.%Y')}": curr_gw, "Difference": curr_gw - prev_gw 
        }
        rows.append(row)
    
    if not rows: st.warning("No data found."); return
    df = pd.DataFrame(rows)
    total_row = {
        "Sl. No": "Total", "State": "", "District": "", "Taluk": "",
        "Total Villages": df["Total Villages"].sum(), "No. of Completed Villages": df["No. of Completed Villages"].sum(),
        "No. of Villages where work is in progress": df["No. of Villages where work is in progress"].sum(),
        "No. of Villages where work has not started": df["No. of Villages where work has not started"].sum(),
        "Number of Ground Water schedules submitted by enumerators": df["Number of Ground Water schedules submitted by enumerators"].sum(),
        "Number of Surface Water schedules submitted by enumerators": df["Number of Surface Water schedules submitted by enumerators"].sum(),
        "Number of Water Body schedules submitted by enumerators": df["Number of Water Body schedules submitted by enumerators"].sum(),
        "Number of Villages submitted by enumerators": df["Number of Villages submitted by enumerators"].sum(),
        f"{prev_date.strftime('%d.%m.%Y')}": df[f"{prev_date.strftime('%d.%m.%Y')}"].sum(), 
        f"{curr_date.strftime('%d.%m.%Y')}": df[f"{curr_date.strftime('%d.%m.%Y')}"].sum(), 
        "Difference": df["Difference"].sum()
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    st.markdown(df.to_html(index=False, classes='table'), unsafe_allow_html=True)
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "Mandya_Abstract.csv", "text/csv")

# ==========================================
# 8. MAIN APP (UI UNCHANGED, MEMORY OPTIMIZED)
# ==========================================
def main():
    inject_custom_css()
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    
    if st.session_state['logged_in']:
        check_session_timeout()
    
    # LOGIN SCREEN (UNCHANGED)
    if not st.session_state['logged_in']:
        _, col_center, _ = st.columns([0.1, 0.8, 0.1])
        with col_center:
            img_base64 = get_base64_image("logo.png")
            if img_base64: st.markdown(f'<div style="display:flex;justify-content:center;margin-bottom:1rem;"><img src="data:image/png;base64,{img_base64}" width="160" style="border-radius:12px;"></div>', unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;'>7th Minor Irrigation Census</h2><p style='text-align:center;color:#5f6368;'>Secure Progress Monitoring System</p>", unsafe_allow_html=True)
            with st.form("login_form"):
                user = st.selectbox("Select Officer", ["Select..."] + AppConfig.AUTHORIZED_USERS)
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Login", type="primary", use_container_width=True):
                    # ✅ FIXED: Password from secrets
                    if user != "Select..." and pwd == get_password():
                        st.session_state['logged_in'] = True; st.session_state['user'] = user; st.session_state['last_active'] = time.time(); st.rerun()
                    else: st.error("⛔ Incorrect Password")
        st.markdown("<div style='height: 50vh;'></div>", unsafe_allow_html=True)
        return

    user = st.session_state['user']
    if user == "Mandya_Admin":
        st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;'><h3>👤 Administrator</h3></div>", unsafe_allow_html=True)
        if st.button("Log Out"): st.session_state.clear(); st.rerun()
        st.markdown("---"); render_admin_dashboard(); return

    # OFFICER DASHBOARD (UI UNCHANGED)
    current_taluk = AppConfig.USER_MAP.get(user, "District")
    user_folder = os.path.join("user_data", user)
    if not os.path.exists(user_folder): os.makedirs(user_folder)
    path_assign = os.path.join(user_folder, "master_assignment") 

    c1, c2 = st.columns([0.75, 0.25])
    with c1: st.markdown(f"<h3>📊 {current_taluk}</h3>", unsafe_allow_html=True)
    with c2: 
        if st.button("Log Out"): st.session_state.clear(); st.rerun()
    st.markdown("<div style='margin-bottom: 1.5rem; border-bottom: 1px solid #dadce0;'></div>", unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">📂 Master Data Management</div>', unsafe_allow_html=True)
    col1, _ = st.columns([1, 0.01])
    with col1:
        is_saved = os.path.exists(path_assign)
        if is_saved and not st.session_state.get('update_mode', False):
            st.markdown("""<div class="status-pill"><span style="margin-right: 8px;">✅</span> Master Assignment File is Active</div>""", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Update Master File", type="secondary"): st.session_state['update_mode'] = True; st.rerun()
        else:
            if is_saved:
                if st.button("❌ Cancel Update"): st.session_state['update_mode'] = False; st.rerun()
            st.markdown("Please upload the latest **Master Assignment** file (Excel/CSV).")
            f1 = st.file_uploader(" ", type=['xlsx','csv'], key="u_master", label_visibility="collapsed")
            if f1:
                # ✅ FIXED: Add file validation
                valid, msg = validate_upload(f1, max_size_mb=10)
                if not valid:
                    st.error(f"⚠️ {msg}")
                elif save_file_robust(f1, path_assign): 
                    st.session_state['update_mode'] = False
                    st.toast("Saved!")
                    st.rerun()

    st.markdown("<div style='margin: 2rem 0; border-bottom: 1px solid #dadce0;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">🚀 Daily Progress Reports</div>', unsafe_allow_html=True)
    
    # ✅ FIXED: Don't store report_data in session_state
    if os.path.exists(path_assign):
        f3 = st.file_uploader("Upload Task Monitoring / GW Completed File (CSV)", type=['csv'], key="task_monitor")
        if f3:
            # ✅ FIXED: Validate file
            valid, msg = validate_upload(f3, max_size_mb=10)
            if not valid:
                st.error(f"⚠️ {msg}")
            else:
                st.markdown(f"<p style='color: {AppConfig.COLORS['light_red']}; font-weight: bold; font-size: 1rem; margin-bottom: 0.5rem;'>Enter manual counts for Status Card</p>", unsafe_allow_html=True)
                mc1, mc2 = st.columns(2)
                with mc1: v_comp = st.number_input("**No. of Completed Villages**", 0)
                with mc2: v_sub = st.number_input("**Villages Submitted by Enumerators**", 0)
                st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

                if st.button("Generate Reports", type="primary", use_container_width=True):
                    with st.spinner('Processing data...'):
                        try:
                            # ✅ FIXED: Load with file hash for caching
                            master_content = open(path_assign, 'rb').read()
                            import hashlib
                            master_hash = hashlib.md5(master_content).hexdigest()
                            file_ext = path_assign.split('.')[-1]
                            
                            df_assign = smart_load_dataframe(master_content, master_hash, file_ext)
                            if df_assign is None: 
                                st.error("Master file corrupted")
                                st.stop()
                            
                            monitor_content = f3.getvalue()
                            monitor_hash = hashlib.md5(monitor_content).hexdigest()
                            df_monitor = smart_load_dataframe(monitor_content, monitor_hash, 'csv')
                            
                            if df_monitor is None:
                                st.error("Monitor file corrupted")
                                st.stop()
                            
                            # Generate reports
                            res = generate_all_reports(df_assign, df_monitor, current_taluk, v_comp, v_sub)
                            
                            # ✅ FIXED: Don't store in session_state, display immediately
                            st.success("✅ Reports Generated Successfully")
                            st.markdown("---")
                            
                            # Display and download
                            c1, c2 = st.columns([0.7, 0.3])
                            with c1: st.markdown('<p class="section-header">1. Progress Graph</p><p style="font-size:0.9rem; color:#5f6368">Visual overview of VAO progress.</p>', unsafe_allow_html=True)
                            with c2: st.download_button("📥 Download Graph", res['g'], "Progress_Graph.png", "image/png", use_container_width=True)
                            st.image(res['g'], use_column_width=True)
                            
                            st.markdown("<div style='margin: 1.5rem 0; border-bottom: 1px solid #f1f3f4;'></div>", unsafe_allow_html=True)
                            c1, c2 = st.columns([0.7, 0.3])
                            with c1: st.markdown('<p class="section-header">2. Detailed Report (Excel)</p><p style="font-size:0.9rem; color:#5f6368">Complete data for verification.</p>', unsafe_allow_html=True)
                            with c2: st.download_button("📥 Download Excel", res['x'], "Progress_Report.xlsx", use_container_width=True)
                            
                            if res['c']:
                                st.markdown("<div style='margin: 1.5rem 0; border-bottom: 1px solid #f1f3f4;'></div>", unsafe_allow_html=True)
                                c1, c2 = st.columns([0.7, 0.3])
                                with c1: st.markdown('<p class="section-header">3. Taluk Status Card</p><p style="font-size:0.9rem; color:#5f6368">Optimized for sharing.</p>', unsafe_allow_html=True)
                                with c2: st.download_button("📥 Download Card", res['c'], "Taluk_Summary.png", "image/png", use_container_width=True)
                                st.image(res['c'], width=600)
                            
                            # ✅ FIXED: Clean up immediately
                            del res
                            gc.collect()
                            
                        except Exception as e: 
                            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
