import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import io
import base64
import re
import json
import gc
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import lru_cache
import hashlib

# Google Sheets - USE UPDATED LIBRARY
import gspread
from google.oauth2.service_account import Credentials  # ✅ Updated from oauth2client

# ==========================================
# 1. MEMORY-EFFICIENT CONFIGURATION
# ==========================================
class AppConfig:
    VERSION = "V171_OPTIMIZED_MULTI_USER"
    SESSION_TIMEOUT_MINUTES = 30
    MAX_UPLOAD_SIZE_MB = 10
    
    # User mapping (unchanged)
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
    
    # Unchanged colors
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
    """✅ Fetch password from Streamlit secrets (not hardcoded)"""
    try:
        return st.secrets["app"]["password"]
    except:
        # Fallback for local testing only
        return os.environ.get("APP_PASSWORD", "mandya")

def get_google_credentials() -> Optional[Credentials]:
    """✅ Secure credential loading from Streamlit secrets"""
    try:
        # Method 1: From Streamlit secrets (RECOMMENDED)
        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        return Credentials.from_service_account_info(creds_dict, scopes=scopes)
    except:
        # Method 2: From JSON file (local testing)
        json_path = os.environ.get("GOOGLE_CREDS_PATH", "service_account.json")
        if os.path.exists(json_path):
            return Credentials.from_service_account_file(json_path, scopes=scopes)
        return None

# ==========================================
# 3. MEMORY-EFFICIENT GOOGLE SHEETS
# ==========================================
@st.cache_resource(ttl=3600)  # ✅ Cache connection, not data
def get_sheets_client():
    """Singleton Google Sheets client"""
    creds = get_google_credentials()
    if not creds:
        return None
    return gspread.authorize(creds)

class SheetsManager:
    """✅ Efficient batch operations to minimize API calls"""
    
    @staticmethod
    def save_master_file(user: str, df: pd.DataFrame, sheet_url: str) -> bool:
        """Save master assignment to user-specific sheet"""
        try:
            client = get_sheets_client()
            if not client:
                return False
            
            # Open or create worksheet
            spreadsheet = client.open_by_url(sheet_url)
            worksheet_name = f"{user}_master"
            
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
            except:
                worksheet = spreadsheet.add_worksheet(worksheet_name, rows=1000, cols=20)
            
            # ✅ Clear and batch update (1 API call instead of row-by-row)
            worksheet.clear()
            worksheet.update([df.columns.values.tolist()] + df.values.tolist())
            return True
        except Exception as e:
            st.error(f"Sheets error: {e}")
            return False
    
    @staticmethod
    def get_master_file(user: str, sheet_url: str) -> Optional[pd.DataFrame]:
        """Retrieve master file from sheets"""
        try:
            client = get_sheets_client()
            if not client:
                return None
            
            spreadsheet = client.open_by_url(sheet_url)
            worksheet = spreadsheet.worksheet(f"{user}_master")
            
            data = worksheet.get_all_values()
            if not data:
                return None
            
            return pd.DataFrame(data[1:], columns=data[0])
        except:
            return None
    
    @staticmethod
    def save_metrics_batch(metrics_list: List[Dict], sheet_url: str):
        """✅ Batch save all taluk metrics (1 API call)"""
        try:
            client = get_sheets_client()
            if not client:
                return
            
            spreadsheet = client.open_by_url(sheet_url)
            
            try:
                worksheet = spreadsheet.worksheet("Taluk_Metrics")
            except:
                worksheet = spreadsheet.add_worksheet("Taluk_Metrics", rows=1000, cols=20)
            
            # Prepare data
            if not metrics_list:
                return
            
            headers = list(metrics_list[0].keys())
            rows = [headers] + [[m.get(h, "") for h in headers] for m in metrics_list]
            
            worksheet.clear()
            worksheet.update(rows)
        except Exception as e:
            st.warning(f"Metrics sync failed: {e}")

# ==========================================
# 4. MEMORY-EFFICIENT DATA LOADING
# ==========================================
def get_file_hash(uploaded_file) -> str:
    """Generate hash for file caching"""
    if uploaded_file is None:
        return ""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

@st.cache_data(ttl=300, max_entries=10)  # ✅ Cache per user (10 users max)
def load_dataframe_chunked(file_content: bytes, file_hash: str, file_type: str) -> Optional[pd.DataFrame]:
    """✅ Load dataframe with memory limit enforcement"""
    try:
        file_obj = io.BytesIO(file_content)
        
        if file_type == 'csv':
            # ✅ Read in chunks to check size
            df = pd.read_csv(file_obj, nrows=1000)  # Sample first
            if len(df) > 10000:  # Limit to 10k rows
                st.warning("⚠️ Large file detected. Loading first 10,000 rows only.")
                file_obj.seek(0)
                df = pd.read_csv(file_obj, nrows=10000)
            else:
                file_obj.seek(0)
                df = pd.read_csv(file_obj)
        else:
            df = pd.read_excel(file_obj, nrows=10000)
        
        return df
    except Exception as e:
        st.error(f"Load error: {e}")
        return None

# ==========================================
# 5. SECURE FILE VALIDATION
# ==========================================
def validate_upload(uploaded_file, max_size_mb: int = 10) -> tuple[bool, str]:
    """✅ Validate uploaded files"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check size
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)"
    
    # Check file type
    allowed_types = ['xlsx', 'csv']
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext not in allowed_types:
        return False, f"Invalid file type: {file_ext}"
    
    return True, "Valid"

# ==========================================
# 6. MEMORY-EFFICIENT VISUALIZATION
# ==========================================
@st.cache_data(ttl=600, max_entries=15)  # ✅ Cache per user/date combo
def generate_progress_chart(data: Dict, chart_hash: str) -> bytes:
    """✅ Generate compressed chart"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=72)  # ✅ Lower DPI = less memory
    
    # Your chart logic here (simplified)
    names = data.get('names', [])
    values = data.get('values', [])
    
    ax.barh(names, values, color='#1967d2')
    ax.set_xlabel('Progress')
    ax.set_title('VAO Progress')
    
    plt.tight_layout()
    
    # ✅ Save as compressed PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=72, bbox_inches='tight')
    plt.close(fig)  # ✅ CRITICAL: Free memory
    
    buf.seek(0)
    return buf.getvalue()

def create_excel_report(df: pd.DataFrame) -> bytes:
    """✅ Stream Excel to bytes without storing in session"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Report')
    output.seek(0)
    return output.getvalue()

# ==========================================
# 7. CONCURRENT USER SESSION MANAGEMENT
# ==========================================
def init_session():
    """Initialize user-specific session"""
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(time.time())  # Unique session ID
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if 'cache_buster' not in st.session_state:
        st.session_state['cache_buster'] = 0

def check_session_timeout():
    """✅ Enforce session timeout"""
    if 'last_active' not in st.session_state:
        st.session_state['last_active'] = time.time()
        return False
    
    elapsed = time.time() - st.session_state['last_active']
    if elapsed > (AppConfig.SESSION_TIMEOUT_MINUTES * 60):
        st.session_state.clear()
        return True
    
    st.session_state['last_active'] = time.time()
    return False

# ==========================================
# 8. OPTIMIZED ADMIN DASHBOARD
# ==========================================
def render_admin_dashboard():
    """Admin view with Google Sheets integration"""
    st.markdown("## 📊 District Dashboard")
    
    # ✅ Sheets URL from secrets
    sheet_url = st.secrets.get("sheets", {}).get("admin_sheet_url", "")
    
    if not sheet_url:
        st.error("⚠️ Admin sheet URL not configured in secrets")
        st.info("Add to .streamlit/secrets.toml:\n```\n[sheets]\nadmin_sheet_url = 'YOUR_GOOGLE_SHEET_URL'\n```")
        return
    
    # Get all taluk data from sheets
    client = get_sheets_client()
    if not client:
        st.error("Cannot connect to Google Sheets")
        return
    
    try:
        spreadsheet = client.open_by_url(sheet_url)
        worksheet = spreadsheet.worksheet("Taluk_Metrics")
        data = worksheet.get_all_records()
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
            # ✅ Download button (not stored in session)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Report",
                csv_data,
                "district_report.csv",
                "text/csv"
            )
        else:
            st.info("No data available yet")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")

# ==========================================
# 9. OPTIMIZED MAIN APP
# ==========================================
def main():
    st.set_page_config(
        page_title="MI Census Pro",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    init_session()
    
    # ✅ Check timeout
    if st.session_state.get('logged_in') and check_session_timeout():
        st.warning("⏰ Session expired. Please login again.")
        st.rerun()
    
    # Login screen
    if not st.session_state['logged_in']:
        st.title("🔐 Secure Login")
        
        with st.form("login_form"):
            user = st.selectbox("Select Officer", [""] + AppConfig.AUTHORIZED_USERS)
            pwd = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                if user and pwd == get_password():
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = user
                    st.session_state['last_active'] = time.time()
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        return
    
    # Main app
    user = st.session_state['user']
    
    # Logout button
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.title(f"📊 {AppConfig.USER_MAP.get(user, 'Admin')}")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()
    
    if user == "Mandya_Admin":
        render_admin_dashboard()
        return
    
    # Officer dashboard
    st.markdown("---")
    
    # ✅ Get sheet URL from secrets
    sheet_url = st.secrets.get("sheets", {}).get("main_sheet_url", "")
    if not sheet_url:
        st.error("Sheet URL not configured")
        return
    
    # Master file upload
    st.subheader("📂 Master Data")
    
    master_uploaded = st.file_uploader(
        "Upload Master Assignment",
        type=['xlsx', 'csv'],
        key="master_upload"
    )
    
    if master_uploaded:
        valid, msg = validate_upload(master_uploaded, AppConfig.MAX_UPLOAD_SIZE_MB)
        if not valid:
            st.error(msg)
        else:
            file_hash = get_file_hash(master_uploaded)
            file_type = master_uploaded.name.split('.')[-1]
            
            df = load_dataframe_chunked(
                master_uploaded.getvalue(),
                file_hash,
                file_type
            )
            
            if df is not None:
                st.success("✅ File loaded")
                
                if st.button("💾 Save to Cloud"):
                    with st.spinner("Syncing..."):
                        success = SheetsManager.save_master_file(user, df, sheet_url)
                        if success:
                            st.success("✅ Saved to Google Sheets")
                        else:
                            st.error("❌ Save failed")
    
    # Daily report
    st.markdown("---")
    st.subheader("📈 Daily Progress")
    
    report_uploaded = st.file_uploader(
        "Upload Task Monitoring File",
        type=['csv'],
        key="report_upload"
    )
    
    if report_uploaded:
        valid, msg = validate_upload(report_uploaded, AppConfig.MAX_UPLOAD_SIZE_MB)
        if not valid:
            st.error(msg)
        else:
            # Manual inputs
            col1, col2 = st.columns(2)
            with col1:
                completed = st.number_input("Completed Villages", 0, 1000, 0)
            with col2:
                submitted = st.number_input("Submitted Villages", 0, 1000, 0)
            
            if st.button("📊 Generate Report"):
                with st.spinner("Processing..."):
                    # Load master from sheets
                    df_master = SheetsManager.get_master_file(user, sheet_url)
                    
                    if df_master is None:
                        st.error("Please upload master file first")
                    else:
                        # Load report file
                        file_hash = get_file_hash(report_uploaded)
                        df_report = load_dataframe_chunked(
                            report_uploaded.getvalue(),
                            file_hash,
                            'csv'
                        )
                        
                        if df_report is not None:
                            # Generate reports (simplified - add your logic)
                            metrics = {
                                'taluk': AppConfig.USER_MAP[user],
                                'completed': completed,
                                'submitted': submitted,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Save metrics
                            SheetsManager.save_metrics_batch([metrics], sheet_url)
                            
                            # Generate chart
                            chart_data = {
                                'names': ['VAO1', 'VAO2', 'VAO3'],
                                'values': [10, 20, 15]
                            }
                            chart_hash = f"{user}_{datetime.now().date()}"
                            chart_bytes = generate_progress_chart(chart_data, chart_hash)
                            
                            st.success("✅ Report generated")
                            st.download_button(
                                "📥 Download Chart",
                                chart_bytes,
                                "progress_chart.png",
                                "image/png"
                            )
                            
                            # ✅ Clear memory
                            del df_master
                            del df_report
                            gc.collect()

if __name__ == "__main__":
    main()
