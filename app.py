import streamlit as st
import os
import pandas as pd
import numpy as np
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

# --- GOOGLE SHEETS LIBRARIES ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
class AppConfig:
    VERSION = "V165_UNRESTRICTED_UPLOAD"
    GLOBAL_PASSWORD = "mandya"
    SESSION_TIMEOUT_MINUTES = 30
    
    # Authorized Personnel
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
# 2. SYSTEM BOOTSTRAP
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
st.set_page_config(page_title="MI Census Pro V165", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 3. CORE UTILITIES
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_base64_image(image_path: str) -> Optional[str]:
    if not os.path.exists(image_path): return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

@st.cache_data(show_spinner=False, ttl=600)
def smart_load_dataframe(path: str, last_modified_time: float) -> Optional[pd.DataFrame]:
    if not os.path.exists(path): return None
    try: return pd.read_excel(path)
    except:
        try: return pd.read_csv(path, encoding='utf-8')
        except: return pd.read_csv(path, encoding='latin1')

def clean_name_logic(name: Any) -> str:
    if pd.isna(name): return "UNKNOWN"
    name = str(name).upper()
    name = re.sub(r'\(.*?\)', '', name)
    name = name.replace('.', ' ')
    name = re.sub(r'\b(MR|MRS|MS|DR|SRI|SMT)\b', '', name)
    name = re.sub(r'[^A-Z\s]', '', name)
    return " ".join(name.strip().split())

def save_file_robust(uploaded_file, target_path: str) -> bool:
    if uploaded_file is None: return False
    try:
        with open(target_path, "wb") as f: f.write(uploaded_file.getvalue())
        return True
    except: return False

# --- SESSION & DATA ---
def check_session_timeout():
    if 'last_active' not in st.session_state:
        st.session_state['last_active'] = time.time()
        return
    if (time.time() - st.session_state['last_active']) > (AppConfig.SESSION_TIMEOUT_MINUTES * 60):
        st.session_state.clear()
        st.session_state['logged_in'] = False
        st.rerun()
    st.session_state['last_active'] = time.time()

def save_taluk_metrics(taluk_name: str, metrics: Dict):
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
    try:
        if os.path.exists(history_path):
            df_hist = pd.read_csv(history_path)
            df_hist = df_hist[~((df_hist['Date'] == today_str) & (df_hist['Taluk'] == taluk_name))]
            df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df_hist = pd.DataFrame([new_row])
        df_hist.to_csv(history_path, index=False)
    except Exception as e: print(f"History Save Error: {e}")

def get_history_data(target_date) -> Dict[str, int]:
    history_path = os.path.join("central_data", "daily_history.csv")
    if not os.path.exists(history_path): return {}
    try:
        df = pd.read_csv(history_path)
        date_str = target_date.strftime('%Y-%m-%d')
        day_data = df[df['Date'] == date_str]
        return dict(zip(day_data['Taluk'], day_data['GW']))
    except: return {}

def get_all_taluk_data() -> List[Dict]:
    data_dir = "central_data"
    all_data = []
    taluk_order = ["K.R. Pete Taluk", "Maddur Taluk", "Malavalli Taluk", "Mandya Taluk", "Nagamangala Taluk", "Pandavapura Taluk", "Srirangapatna Taluk"]
    for t_name in taluk_order:
        f_path = os.path.join(data_dir, f"{t_name.replace(' ', '_')}.json")
        if os.path.exists(f_path):
            with open(f_path, "r") as f: all_data.append(json.load(f))
        else:
            all_data.append({"taluk": t_name, "total_villages": 0, "completed_v": 0, "in_progress": 0, "not_started": 0, "gw": 0, "sw": 0, "wb": 0, "submitted_v": 0})
    return all_data

# --- GOOGLE SHEET SYNC ENGINE (V165: UNRESTRICTED) ---
def sync_data_to_google_sheet(df: pd.DataFrame, json_key_dict: Dict, sheet_name: str) -> str:
    """Connects to G-Sheets and overwrites the data."""
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(json_key_dict, scope)
        client = gspread.authorize(creds)
        
        try:
            sheet = client.open(sheet_name).sheet1
        except gspread.SpreadsheetNotFound:
            return "❌ Sheet not found. Share it with the service account email."
            
        data = [df.columns.values.tolist()] + df.astype(str).values.tolist()
        sheet.clear()
        sheet.update(data)
        return f"✅ Successfully synced {len(df)} rows to '{sheet_name}'"
    except Exception as e:
        return f"❌ Sync Error: {str(e)}"

# ==========================================
# 4. REPORT GENERATION ENGINE
# ==========================================
@st.cache_data(show_spinner=False, ttl=300)
def generate_all_reports(df_assign: pd.DataFrame, df_monitor: pd.DataFrame, taluk_name: str, 
                         manual_completed_v: int, manual_submitted_v: int):
    try:
        df_assign.columns = df_assign.columns.str.strip()
        df_monitor.columns = df_monitor.columns.str.strip()
        num_cols = df_monitor.shape[1]
        
        col_gw = next((c for c in df_monitor.columns if 'Total schedules GW' in c), None)
        col_sw = next((c for c in df_monitor.columns if 'Total schedules SW' in c), None)
        col_wb = next((c for c in df_monitor.columns if 'Total schedules WB' in c), None)
        
        gw_series = pd.to_numeric(df_monitor[col_gw], errors='coerce').fillna(0) if col_gw else pd.to_numeric(df_monitor.iloc[:, 9], errors='coerce').fillna(0)
        sw_series = pd.to_numeric(df_monitor[col_sw], errors='coerce').fillna(0) if col_sw else pd.to_numeric(df_monitor.iloc[:, 10], errors='coerce').fillna(0)
        wb_series = pd.to_numeric(df_monitor[col_wb], errors='coerce').fillna(0) if col_wb else pd.to_numeric(df_monitor.iloc[:, 11], errors='coerce').fillna(0)
        
        df_monitor['Total schedules GW'] = gw_series
        map_val = df_monitor.iloc[:, 4].count() if num_cols > 4 else 0
        
        ip_val = 0; ns_val = 0
        if num_cols > 21:
            col_v = df_monitor.iloc[:, 21].astype(str).str.lower()
            ip_val = int(col_v[col_v == 'false'].count())
            ns_val = int(col_v[col_v == 'true'].count())

        metrics = {
            "total_villages": len(df_monitor), "mapped": int(map_val),
            "gw": int(gw_series.sum()), "sw": int(sw_series.sum()), "wb": int(wb_series.sum()),
            "completed_v": int(manual_completed_v), "submitted_v": int(manual_submitted_v),
            "in_progress": ip_val, "not_started": ns_val
        }
        save_taluk_metrics(taluk_name, metrics)

        df_assign['Clean_Key'] = df_assign['User'].apply(clean_name_logic)
        df_monitor['Clean_Key'] = df_monitor['Enu name'].apply(clean_name_logic)
        key_map = df_assign.groupby('Clean_Key')['User'].first().to_dict()

        t_col = next((c for c in df_assign.columns if 'Total schemes' in c), None)
        df_assign[t_col] = pd.to_numeric(df_assign[t_col], errors='coerce').fillna(0)
        grp_a = df_assign.groupby('Clean_Key')[t_col].sum().reset_index()
        grp_m = df_monitor.groupby('Clean_Key')['Total schedules GW'].sum().reset_index()

        del df_assign; del df_monitor; gc.collect()

        final = pd.merge(grp_a, grp_m, on='Clean_Key', how='left').fillna(0)
        final.rename(columns={t_col: 'Assigned', 'Total schedules GW': 'Completed'}, inplace=True)
        final['VAO Full Name'] = final['Clean_Key'].map(key_map).fillna(final['Clean_Key']).str.title()
        final['% Completed'] = np.where(final['Assigned'] > 0, final['Completed'] / final['Assigned'], np.where(final['Completed'] > 0, 1.0, 0.0))
        final = final.sort_values('Completed', ascending=False).reset_index(drop=True)
        final.insert(0, 'S. No.', final.index + 1)

        total_assigned = final['Assigned'].sum()
        total_completed = final['Completed'].sum()
        total_progress = (total_completed / total_assigned) if total_assigned > 0 else 0
        
        ts = (datetime.now(timezone.utc) + timedelta(hours=5.5)).strftime("%d-%m-%Y %I:%M %p")
        report_title = f"{taluk_name}: VAO wise progress of Ground Water Schemes (tube well) census wrt 6th Minor Irrigation Census upto 2018-19.\n(Generated on: {ts})"

        # EXCEL
        b_xl = io.BytesIO()
        with pd.ExcelWriter(b_xl, engine='xlsxwriter') as writer:
            out = final[['S. No.', 'VAO Full Name', 'Assigned', 'Completed', '% Completed']].copy()
            out.loc[len(out)] = [None, 'Grand Total', total_assigned, total_completed, total_progress]
            out.to_excel(writer, index=False, startrow=3, sheet_name='Report')
            wb = writer.book; ws = writer.sheets['Report']
            fmt_title = wb.add_format({'bold': True, 'font_size': 14, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True, 'border': 1, 'bg_color': '#D3D3D3'})
            fmt_header = wb.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#E0E0E0', 'text_wrap': True})
            fmt_body = wb.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
            fmt_green = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'border': 1, 'num_format': '0.0%', 'align': 'center'})
            fmt_red = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'border': 1, 'num_format': '0.0%', 'align': 'center'})
            fmt_total_pct = wb.add_format({'bold': True, 'border': 1, 'align': 'center', 'bg_color': '#F2F2F2', 'num_format': '0.0%'})
            fmt_total = wb.add_format({'bold': True, 'border': 1, 'align': 'center', 'bg_color': '#F2F2F2'})
            ws.merge_range('A1:E3', report_title, fmt_title)
            for col_idx, col_name in enumerate(out.columns): ws.write(3, col_idx, col_name, fmt_header)
            for r_idx, row in enumerate(out.values):
                row_num = 4 + r_idx
                is_last = (r_idx == len(out) - 1)
                for c_idx, val in enumerate(row):
                    if is_last: ws.write(row_num, c_idx, val, fmt_total_pct if c_idx == 4 else fmt_total)
                    else:
                        if c_idx == 4: ws.write(row_num, c_idx, val, fmt_green if (val > 0.1 or (row[2]==0 and row[3]>0)) else fmt_red)
                        else: ws.write(row_num, c_idx, val, fmt_body)
            ws.set_column(0, 0, 8); ws.set_column(1, 1, 35); ws.set_column(2, 4, 15)
        b_xl.seek(0)

        # CARD
        plt.rcParams['font.family'] = 'sans-serif'; plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'sans-serif']
        card_data = [
            ["Total No. of Villages", metrics['total_villages']], ["No. of Completed Villages", metrics['completed_v']],
            ["No. of Villages work in progress", metrics['in_progress']], ["No. of Villages work not started", metrics['not_started']],
            ["Villages mapped to enumerator", metrics['mapped']], ["Ground Water schedules submitted", metrics['gw']],
            ["Surface Water schedules submitted", metrics['sw']], ["Water Body schedules submitted", metrics['wb']],
            ["Villages submitted by enumerators", metrics['submitted_v']]
        ]
        fh = max(6, len(card_data) * 0.8 + 2.5)
        fig_c, axc = plt.subplots(figsize=(11.5, fh)); axc.axis('off')
        tbl = axc.table(cellText=[["  "+textwrap.fill(r[0], 60), str(r[1])] for r in card_data], colLabels=["Description", "Count"], colWidths=[0.8, 0.2], loc='center', bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        header_color = AppConfig.TALUK_COLORS.get(taluk_name, AppConfig.COLORS['primary'])
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(AppConfig.COLORS['neutral']); cell.set_linewidth(1)
            if r == 0: cell.set_facecolor(header_color); cell.set_text_props(weight='bold', color='white', size=13); cell.set_height(0.08)
            else: cell.set_facecolor('white' if r % 2 == 0 else AppConfig.COLORS['bg_secondary']); cell.set_text_props(size=12, color=AppConfig.COLORS['text']); cell.set_height(0.09)
            if c == 0: cell.set_text_props(ha='left'); 
            elif c == 1: cell.set_text_props(ha='center')
        axc.set_title(f"{taluk_name} Status Report\n(Generated on: {ts})", fontweight='bold', fontsize=16, pad=20, color='black')
        b_card = io.BytesIO(); plt.savefig(b_card, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1); b_card.seek(0); plt.close(fig_c)

        # GRAPH
        p = final.sort_values('Completed', ascending=True).reset_index(drop=True)
        fig_g, ax = plt.subplots(figsize=(16, max(10, len(p)*0.6)))
        p['N'] = p['VAO Full Name'].apply(lambda x: f"{x.split()[0]} {x.split()[1][0]}." if len(x.split())>1 else x.split()[0])
        ys = np.arange(len(p))
        cols = [AppConfig.COLORS['success'] if (x>0.1 or (a==0 and c>0)) else AppConfig.COLORS['danger'] for x,a,c in zip(p['% Completed'], p['Assigned'], p['Completed'])]
        ax.barh(ys, p['Assigned'], color=AppConfig.COLORS['neutral'], label='Assigned', height=0.7)
        ax.barh(ys, p['Completed'], color=cols, height=0.5)
        ax.invert_yaxis(); sns.despine(left=True, bottom=True); ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#dadce0')
        ax.set_yticks(ys); ax.set_yticklabels(p['N'], fontsize=12, fontweight='bold', color=AppConfig.COLORS['subtext'])
        mv = max(p['Assigned']) if len(p)>0 else 1
        for i, (a, c, pc) in enumerate(zip(p['Assigned'], p['Completed'], p['% Completed'])):
            ctx = f"{int(c)} (100%)" if (a==0 and c>0) else f"{int(c)} ({pc*100:.1f}%)"
            ax.text(c+(mv*0.01), i, ctx, va='center', weight='bold', size=11)
            est_w = len(ctx)*(mv*0.017); end_pos = c+(mv*0.01)+est_w; def_pos = a+(mv*0.02); final_pos = max(end_pos+(mv*0.02), def_pos)
            ax.text(final_pos, i, f"{int(a)}", va='center', ha='left', color=AppConfig.COLORS['subtext'], weight='bold', size=11)
        ax.margins(x=0.25); ax.set_ylim(-1, len(p)+2)
        wrapped_title = "\n".join(textwrap.wrap(report_title, width=90))
        ax.set_title(wrapped_title, fontsize=14, weight='bold', pad=40, color=AppConfig.COLORS['text'])
        ax.set_xlabel("No of GW Schemes as per 6th MI Census upto 2018-19", fontsize=12, weight='bold', color=AppConfig.COLORS['subtext'])
        summary_text = f"GWS SUMMARY | Assigned: {int(total_assigned):,} | Completed: {int(total_completed):,} | Progress: {total_progress*100:.2f}%"
        ax.annotate(summary_text, xy=(0.5, 1), xytext=(0, 15), xycoords='axes fraction', textcoords='offset points',
                    ha='center', va='bottom', fontsize=12, weight='bold', color='white', bbox=dict(boxstyle="round,pad=0.6", fc="black", ec="none", alpha=1.0))
        leg = [Patch(facecolor=AppConfig.COLORS['neutral'], label='Assigned'), Patch(facecolor=AppConfig.COLORS['success'], label='Completed > 10%'), Patch(facecolor=AppConfig.COLORS['danger'], label='Completed ≤ 10%')]
        ax.legend(handles=leg, loc='lower right', fontsize=11, framealpha=0.9)
        b_grph = io.BytesIO(); plt.tight_layout(); plt.savefig(b_grph, format='png', dpi=100); b_grph.seek(0); plt.close(fig_g)
        
        del final; del p; del out; gc.collect()
        return {'x': b_xl, 'c': b_card, 'g': b_grph}
    except Exception as e: raise RuntimeError(str(e))

# ==========================================
# 5. UI COMPONENTS
# ==========================================
def inject_custom_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Roboto', sans-serif; }}
    #MainMenu, footer, header {{visibility: hidden !important; height: 0px !important;}}
    [data-testid="stDecoration"] {{display: none !important;}}
    [data-testid="stFooter"] {{display: none !important;}}
    .stDeployButton {{display: none !important;}}
    [data-testid="stStatusWidget"] {{display: none !important;}}
    
    .block-container {{ padding-top: 6rem !important; padding-bottom: 12rem !important; max-width: 1200px; }}
    [data-testid="InputInstructions"] {{ display: none !important; }}
    .status-pill {{ display: inline-flex; align-items: center; padding: 0.5rem 1rem; background-color: #e6f4ea; color: #137333; border-radius: 999px; font-weight: 500; border: 1px solid #ceead6; }}
    .section-header {{ font-size: 1.1rem; font-weight: 600; color: {AppConfig.COLORS['primary']}; margin-top: 0.5rem; text-transform: uppercase; }}
    
    .custom-footer {{ 
        position: fixed; left: 0; bottom: 0; width: 100%; 
        background-color: #000000 !important; color: #ffffff !important; 
        text-align: center; 
        padding: 1.5rem 1rem 2.5rem 1rem; 
        border-top: 1px solid #333; 
        z-index: 2147483647 !important; 
        font-size: 15px !important; 
        line-height: 1.6;
    }}
    .mobile-break {{ display: inline; }}
    @media (max-width: 640px) {{ 
        .custom-footer {{ font-size: 13px !important; }} 
        .mobile-break {{ display: block; margin-top: 4px; }}
    }}
    </style>
    <div class="custom-footer">
        Design & Developed by <b>Gangadhar</b> | Statistical Inspector, 
        <span class="mobile-break">Taluk Office Malavalli, Mandya</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 6. ADMIN DASHBOARD
# ==========================================
def render_admin_dashboard():
    st.markdown("## 🏛️ 7th Minor Irrigation Census Progress Report")
    
    # --- V165: GOOGLE SHEET SYNC UI (UNRESTRICTED) ---
    with st.expander("☁️ Cloud Sync (Save to Google Sheets)", expanded=False):
        st.markdown("**Instructions:** Upload your Service Account JSON file and enter the Google Sheet name to sync the District Abstract.")
        c_up, c_in = st.columns([1, 1])
        with c_up: 
            # REMOVED type=['json'] so you can pick any file
            key_file = st.file_uploader("1. Service Account Key (JSON)", help="Upload your key file here.")
        with c_in: 
            sheet_name = st.text_input("2. Sheet Name")
        
        if key_file and sheet_name and st.button("🚀 Sync Now", type="primary"):
            try:
                json_key = json.load(key_file)
                # Recalculate Dashboard Data for Sync
                taluk_data_sync = get_all_taluk_data()
                prev_date_sync = datetime.now() - timedelta(days=1)
                prev_data_map_sync = get_history_data(prev_date_sync)
                
                rows_sync = []
                for idx, t in enumerate(taluk_data_sync):
                    prev_gw = prev_data_map_sync.get(t['taluk'], 0)
                    rows_sync.append({
                        "Sl. No": idx + 1, "Taluk": t['taluk'].replace(" Taluk", ""),
                        "Total Villages": t['total_villages'], "Completed": t['completed_v'],
                        "In Progress": t['in_progress'], "Not Started": t['not_started'],
                        "GW Submitted": t['gw'], "SW Submitted": t['sw'], "WB Submitted": t['wb'],
                        "Villages Submitted": t['submitted_v'],
                        "Previous GW": prev_gw, "Current GW": t['gw'], "Difference": t['gw'] - prev_gw
                    })
                
                df_sync = pd.DataFrame(rows_sync)
                # Add Total
                total_sync = {
                    "Sl. No": "Total", "Taluk": "",
                    "Total Villages": df_sync["Total Villages"].sum(), "Completed": df_sync["Completed"].sum(),
                    "In Progress": df_sync["In Progress"].sum(), "Not Started": df_sync["Not Started"].sum(),
                    "GW Submitted": df_sync["GW Submitted"].sum(), "SW Submitted": df_sync["SW Submitted"].sum(),
                    "WB Submitted": df_sync["WB Submitted"].sum(), "Villages Submitted": df_sync["Villages Submitted"].sum(),
                    "Previous GW": df_sync["Previous GW"].sum(), "Current GW": df_sync["Current GW"].sum(), "Difference": df_sync["Difference"].sum()
                }
                df_sync = pd.concat([df_sync, pd.DataFrame([total_sync])], ignore_index=True)
                
                with st.spinner("Syncing..."):
                    st.success(sync_data_to_google_sheet(df_sync, json_key, sheet_name))
            except Exception as e: st.error(str(e))

    st.markdown("---")
    
    # Regular Dashboard
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
# 7. MAIN APP
# ==========================================
def main():
    inject_custom_css()
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    
    if st.session_state['logged_in']:
        check_session_timeout()
    
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
                    if user != "Select..." and pwd == AppConfig.GLOBAL_PASSWORD:
                        st.session_state['logged_in'] = True; st.session_state['user'] = user; st.session_state['last_active'] = time.time(); st.rerun()
                    else: st.error("⛔ Incorrect Password")
        st.markdown("<div style='height: 50vh;'></div>", unsafe_allow_html=True)
        return

    user = st.session_state['user']
    if user == "Mandya_Admin":
        st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;'><h3>👤 Administrator</h3></div>", unsafe_allow_html=True)
        if st.button("Log Out"): st.session_state.clear(); st.rerun()
        st.markdown("---"); render_admin_dashboard(); return

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
            if f1 and save_file_robust(f1, path_assign): st.session_state['update_mode'] = False; st.toast("Saved!"); st.rerun()

    st.markdown("<div style='margin: 2rem 0; border-bottom: 1px solid #dadce0;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">🚀 Daily Progress Reports</div>', unsafe_allow_html=True)
    
    if 'report_data' not in st.session_state: st.session_state['report_data'] = None
    def clear_report_cache(): st.session_state['report_data'] = None

    if os.path.exists(path_assign):
        f3 = st.file_uploader("Upload Today's Task Monitoring File (CSV)", type=['csv'], on_change=clear_report_cache)
        if f3:
            st.markdown(f"<p style='color: {AppConfig.COLORS['light_red']}; font-weight: bold; font-size: 1rem; margin-bottom: 0.5rem;'>Enter manual counts for Status Card</p>", unsafe_allow_html=True)
            mc1, mc2 = st.columns(2)
            with mc1: v_comp = st.number_input("**No. of Completed Villages**", 0)
            with mc2: v_sub = st.number_input("**Villages Submitted by Enumerators**", 0)
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

            if st.button("Generate Reports", type="primary", use_container_width=True):
                with st.spinner('Processing data...'):
                    try:
                        m_time = os.path.getmtime(path_assign)
                        df_assign = smart_load_dataframe(path_assign, m_time)
                        if df_assign is None: st.error("Master file corrupted"); st.stop()
                        try: df_monitor = pd.read_csv(f3)
                        except: f3.seek(0); df_monitor = pd.read_csv(f3, encoding='latin1')
                        res = generate_all_reports(df_assign, df_monitor, current_taluk, v_comp, v_sub)
                        st.session_state['report_data'] = res
                        del df_assign; del df_monitor; gc.collect()
                    except Exception as e: st.error(f"Error: {str(e)}")

    if st.session_state.get('report_data'):
        data = st.session_state['report_data']
        st.success("✅ Reports Generated Successfully")
        st.markdown("---")
        c1, c2 = st.columns([0.7, 0.3])
        with c1: st.markdown('<p class="section-header">1. Progress Graph</p><p style="font-size:0.9rem; color:#5f6368">Visual overview of VAO progress.</p>', unsafe_allow_html=True)
        with c2: st.download_button("📥 Download Graph", data['g'], "Progress_Graph.png", "image/png", use_container_width=True)
        st.image(data['g'], use_column_width=True)
        st.markdown("<div style='margin: 1.5rem 0; border-bottom: 1px solid #f1f3f4;'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([0.7, 0.3])
        with c1: st.markdown('<p class="section-header">2. Detailed Report (Excel)</p><p style="font-size:0.9rem; color:#5f6368">Complete data for verification.</p>', unsafe_allow_html=True)
        with c2: st.download_button("📥 Download Excel", data['x'], "Progress_Report.xlsx", use_container_width=True)
        st.markdown("<div style='margin: 1.5rem 0; border-bottom: 1px solid #f1f3f4;'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([0.7, 0.3])
        with c1: st.markdown('<p class="section-header">3. Taluk Status Card</p><p style="font-size:0.9rem; color:#5f6368">Optimized for sharing.</p>', unsafe_allow_html=True)
        with c2: st.download_button("📥 Download Card", data['c'], "Taluk_Summary.png", "image/png", use_container_width=True)
        st.image(data['c'], width=600)

if __name__ == "__main__":
    main()
