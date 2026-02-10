"""
MI Census Pro - Streamlit Cloud Free Tier Optimized
Version: V200_FREECLOUD_OPTIMIZED
Author: Gangadhar (Enhanced by AI)
Organization: Taluk Office Malavalli, Mandya

OPTIMIZATIONS FOR FREE TIER:
- Aggressive cache reduction (TTL: 30-60s, max 3 entries)
- Lazy loading for heavy operations
- Session state for temporary storage (no cache pressure)
- Image preview mode (low DPI) + optional full quality download
- Chunked processing for large files
- Eliminated redundant computations
- Memory-efficient dataframe operations
- Deferred heavy calculations
"""

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
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. ENTERPRISE CONFIGURATION
# ==========================================
class AppConfig:
    VERSION = "V200_FREECLOUD_OPTIMIZED"
    GLOBAL_PASSWORD = "mandya"
    
    # 🔴 REDUCED: Much lower for free tier
    MAX_CACHE_SIZE = 32  # Reduced from 128 MB
    IMAGE_DPI_GRAPH = 60  # Reduced from 80 (preview)
    IMAGE_DPI_CARD = 72   # Reduced from 100 (preview)
    IMAGE_DPI_FULL = 150  # For optional high-quality download
    COMPRESSION_LEVEL = 9  # Maximum PNG compression
    CHUNK_SIZE = 512 * 1024  # Smaller chunks (512KB)
    
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
    
    # Google Brand Palette
    COLORS = {
        "primary": "#1a73e8",
        "success": "#34A853",
        "warning": "#FBBC04",
        "danger": "#EA4335",
        "light_red": "#EE675C",
        "neutral": "#DADCE0",
        "text": "#202124",
        "subtext": "#5f6368",
        "bg_light": "#ffffff",
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

# ==========================================
# 2. OPTIMIZED SYSTEM BOOTSTRAP
# ==========================================
def setup_config():
    """Minimal config optimized for free tier"""
    config_dir = ".streamlit"
    config_path = os.path.join(config_dir, "config.toml")
    config_content = f"""
[server]
maxUploadSize = 100
maxMessageSize = 100
enableCORS = false
enableXsrfProtection = true
headless = true
runOnSave = false
fileWatcherType = "none"
timeout = 1800

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[theme]
primaryColor = "{AppConfig.COLORS['primary']}"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "{AppConfig.COLORS['text']}"
font = "sans serif"

[client]
toolbarMode = "minimal"
showErrorDetails = false
showSidebarNavigation = false

[logger]
level = "error"

[cache]
maxMegabytes = 32
ttlSeconds = 30
"""
    os.makedirs(config_dir, exist_ok=True)
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(config_content.strip())

setup_config()
st.set_page_config(
    page_title="MI Census Pro V200",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 3. SESSION STATE MANAGEMENT (KEY OPTIMIZATION)
# ==========================================
def init_session_state():
    """Initialize session state - no cache pressure"""
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = {}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    if 'file_hashes' not in st.session_state:
        st.session_state.file_hashes = {}

init_session_state()

# ==========================================
# 4. ULTRA-OPTIMIZED UTILITIES
# ==========================================

@st.cache_data(ttl=30, max_entries=2, show_spinner=False)
def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash - very short cache"""
    if not os.path.exists(file_path):
        return "none"
    
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):  # Smaller chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return str(os.path.getmtime(file_path))

@lru_cache(maxsize=500)  # Much smaller
def clean_name_logic(name: Any) -> str:
    """Cached name cleaning for performance"""
    if pd.isna(name):
        return "UNKNOWN"
    
    name = str(name).upper()
    name = re.sub(r'\(.*?\)', '', name)
    name = name.replace('.', ' ')
    name = re.sub(r'\b(MR|MRS|MS|DR|SRI|SMT)\b', '', name)
    name = re.sub(r'[^A-Z\s]', '', name)
    return " ".join(name.strip().split())

def save_file_robust(uploaded_file, target_path: str) -> bool:
    """Robust file saving with chunking"""
    if uploaded_file is None:
        return False
    
    try:
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        
        file_size = uploaded_file.size
        
        with open(target_path, "wb") as f:
            uploaded_file.seek(0)
            while True:
                chunk = uploaded_file.read(AppConfig.CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
        
        return True
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        return False

def compress_image_aggressive(img_buffer: io.BytesIO) -> io.BytesIO:
    """Aggressive PNG compression for free tier"""
    try:
        from PIL import Image
        
        img_buffer.seek(0)
        img = Image.open(img_buffer)
        
        # Reduce image size for display
        max_size = (800, 600)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        compressed = io.BytesIO()
        img.save(
            compressed, 
            format='PNG', 
            optimize=True, 
            compress_level=9
        )
        compressed.seek(0)
        return compressed
    except Exception:
        img_buffer.seek(0)
        return img_buffer

# ==========================================
# 5. DATA PERSISTENCE (LIGHTWEIGHT)
# ==========================================

def save_taluk_metrics(taluk_name: str, metrics: Dict):
    """Lightweight metrics saving"""
    data_dir = "central_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Convert numpy types
    safe_metrics = {
        k: (int(v) if isinstance(v, (np.integer, np.int64)) else 
            float(v) if isinstance(v, (np.floating, np.float64)) else v)
        for k, v in metrics.items()
    }
    
    safe_metrics['timestamp'] = datetime.now().isoformat()
    safe_metrics['taluk'] = taluk_name
    
    file_path = os.path.join(data_dir, f"{taluk_name.replace(' ', '_')}.json")
    temp_path = file_path + ".tmp"
    
    try:
        with open(temp_path, "w") as f:
            json.dump(safe_metrics, f)  # No pretty print to save bytes
        os.replace(temp_path, file_path)
        update_history(taluk_name, safe_metrics)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def update_history(taluk_name: str, metrics: Dict):
    """Update daily history - optimized"""
    history_path = os.path.join("central_data", "daily_history.csv")
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    new_row = {
        "Date": today_str,
        "Taluk": taluk_name,
        "GW": metrics.get('gw', 0),
        "SW": metrics.get('sw', 0),
        "WB": metrics.get('wb', 0),
        "Total": metrics.get('total_villages', 0),
        "Completed": metrics.get('completed_v', 0),
        "InProgress": metrics.get('in_progress', 0),
        "NotStarted": metrics.get('not_started', 0),
        "Submitted": metrics.get('submitted_v', 0)
    }
    
    try:
        if os.path.exists(history_path):
            df_hist = pd.read_csv(history_path, dtype=str)  # Read as string first
            df_hist = df_hist[~((df_hist['Date'] == today_str) & (df_hist['Taluk'] == taluk_name))]
            df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df_hist = pd.DataFrame([new_row])
        
        temp_path = history_path + ".tmp"
        df_hist.to_csv(temp_path, index=False)
        os.replace(temp_path, history_path)
    except Exception:
        pass  # Silently fail if history update fails

@st.cache_data(ttl=60, max_entries=1, show_spinner=False)  # Very short cache
def get_all_taluk_data() -> List[Dict]:
    """Cached taluk data with minimal entries"""
    data_dir = "central_data"
    taluk_order = [
        "K.R. Pete Taluk",
        "Maddur Taluk",
        "Malavalli Taluk",
        "Mandya Taluk",
        "Nagamangala Taluk",
        "Pandavapura Taluk",
        "Srirangapatna Taluk"
    ]
    
    all_data = []
    for t_name in taluk_order:
        f_path = os.path.join(data_dir, f"{t_name.replace(' ', '_')}.json")
        
        if os.path.exists(f_path):
            try:
                with open(f_path, "r") as f:
                    all_data.append(json.load(f))
            except Exception:
                all_data.append({
                    "taluk": t_name,
                    "total_villages": 0,
                    "completed_v": 0,
                    "in_progress": 0,
                    "not_started": 0,
                    "gw": 0,
                    "sw": 0,
                    "wb": 0,
                    "submitted_v": 0
                })
        else:
            all_data.append({
                "taluk": t_name,
                "total_villages": 0,
                "completed_v": 0,
                "in_progress": 0,
                "not_started": 0,
                "gw": 0,
                "sw": 0,
                "wb": 0,
                "submitted_v": 0
            })
    
    return all_data

# ==========================================
# 6. OPTIMIZED REPORT GENERATION
# ==========================================

@st.cache_data(ttl=60, max_entries=1, show_spinner=False)  # Ultra-minimal caching
def generate_all_reports(
    df_assign: pd.DataFrame,
    df_monitor: pd.DataFrame,
    taluk_name: str,
    manual_completed_v: int,
    manual_submitted_v: int
):
    """
    High-performance report generation
    OPTIMIZATIONS:
    - Vectorized pandas operations
    - Minimal dataframe copies
    - Lazy image generation
    - Memory-efficient calculations
    """
    
    try:
        # Clean column names
        df_assign.columns = df_assign.columns.str.strip()
        df_monitor.columns = df_monitor.columns.str.strip()
        
        # Extract metrics (vectorized)
        gw_series = pd.to_numeric(
            df_monitor.get('Total schedules GW', pd.Series([0]*len(df_monitor))),
            errors='coerce'
        ).fillna(0)
        
        sw_series = pd.to_numeric(
            df_monitor.get('Total schedules SW', pd.Series([0]*len(df_monitor))),
            errors='coerce'
        ).fillna(0)
        
        wb_series = pd.to_numeric(
            df_monitor.get('Total schedules WB', pd.Series([0]*len(df_monitor))),
            errors='coerce'
        ).fillna(0)
        
        # Create metrics
        metrics = {
            "total_villages": len(df_monitor),
            "mapped": int(df_monitor.iloc[:, 4].notna().sum() if df_monitor.shape[1] > 4 else 0),
            "gw": int(gw_series.sum()),
            "sw": int(sw_series.sum()),
            "wb": int(wb_series.sum()),
            "completed_v": int(manual_completed_v),
            "submitted_v": int(manual_submitted_v),
            "in_progress": 0,
            "not_started": 0
        }
        
        save_taluk_metrics(taluk_name, metrics)
        
        # Process assignment data - minimal copies
        df_assign['Clean_Key'] = df_assign['User'].apply(clean_name_logic)
        df_monitor['Clean_Key'] = df_monitor['Enu name'].apply(clean_name_logic)
        
        key_map = df_assign.groupby('Clean_Key', as_index=False)['User'].first()
        key_map = dict(zip(key_map['Clean_Key'], key_map['User']))
        
        # Find scheme column
        scheme_col = next((c for c in df_assign.columns if 'Total schemes' in c), None)
        if not scheme_col:
            scheme_col = df_assign.columns[-1]
        
        df_assign[scheme_col] = pd.to_numeric(df_assign[scheme_col], errors='coerce').fillna(0)
        df_monitor['Total schedules GW'] = gw_series
        
        # Aggregate (vectorized)
        assigned = df_assign.groupby('Clean_Key')[scheme_col].sum()
        completed = df_monitor.groupby('Clean_Key')['Total schedules GW'].sum()
        
        # Merge
        final = pd.DataFrame({
            'Clean_Key': assigned.index,
            'Assigned': assigned.values,
            'Completed': completed.reindex(assigned.index, fill_value=0).values
        })
        
        final['VAO Full Name'] = final['Clean_Key'].map(key_map).fillna(final['Clean_Key']).str.title()
        final['% Completed'] = np.where(
            final['Assigned'] > 0,
            final['Completed'] / final['Assigned'],
            0.0
        )
        
        final = final.sort_values('Completed', ascending=False).reset_index(drop=True)
        final.insert(0, 'S. No.', final.index + 1)
        
        # Totals
        total_assigned = final['Assigned'].sum()
        total_completed = final['Completed'].sum()
        total_progress = (total_completed / total_assigned) if total_assigned > 0 else 0
        
        # Timestamp
        ts = (datetime.now(timezone.utc) + timedelta(hours=5.5)).strftime("%d-%m-%Y %I:%M %p")
        report_title = f"{taluk_name}: VAO wise progress (Generated: {ts})"
        
        # Generate reports
        excel_data = generate_excel_report(final, report_title, total_assigned, total_completed, total_progress)
        
        return {
            'excel': excel_data,
            'metrics': metrics,
            'final_df': final,
            'title': report_title,
            'total_assigned': total_assigned,
            'total_completed': total_completed,
            'total_progress': total_progress
        }
        
    except Exception as e:
        raise RuntimeError(f"Report generation failed: {str(e)}")

def generate_excel_report(final: pd.DataFrame, title: str, total_assigned: float, 
                          total_completed: float, total_progress: float) -> io.BytesIO:
    """Generate optimized Excel report"""
    b_xl = io.BytesIO()
    
    with pd.ExcelWriter(b_xl, engine='xlsxwriter') as writer:
        out = final[['S. No.', 'VAO Full Name', 'Assigned', 'Completed', '% Completed']].copy()
        out.loc[len(out)] = [None, 'Grand Total', total_assigned, total_completed, total_progress]
        
        out.to_excel(writer, index=False, startrow=2, sheet_name='Report')
        
        wb = writer.book
        ws = writer.sheets['Report']
        
        # Minimal formatting
        fmt_title = wb.add_format({
            'bold': True,
            'font_size': 12,
            'align': 'center',
            'bg_color': '#D3D3D3'
        })
        
        fmt_header = wb.add_format({
            'bold': True,
            'border': 1,
            'align': 'center',
            'bg_color': '#E0E0E0'
        })
        
        fmt_green = wb.add_format({
            'bg_color': '#C6EFCE',
            'num_format': '0.0%',
            'border': 1
        })
        
        # Write title
        ws.merge_range('A1:E1', title, fmt_title)
        
        # Apply header formatting
        for col_num, value in enumerate(['S. No.', 'VAO Full Name', 'Assigned', 'Completed', '% Completed']):
            ws.write(1, col_num, value, fmt_header)
        
        # Set column widths
        ws.set_column('A:A', 6)
        ws.set_column('B:B', 25)
        ws.set_column('C:C', 12)
        ws.set_column('D:D', 12)
        ws.set_column('E:E', 14)
        
        return b_xl

def generate_status_card_display(taluk_name: str, metrics: Dict, ts: str) -> str:
    """Generate HTML card for display (no image generation)"""
    total = metrics['total_villages']
    completed = metrics['completed_v']
    pending = total - completed
    completion_pct = (completed / total * 100) if total > 0 else 0
    
    html_card = f"""
    <div style="
        border: 2px solid {AppConfig.COLORS['primary']};
        border-radius: 10px;
        padding: 20px;
        background: #f8f9fa;
        text-align: center;
    ">
        <h2 style="color: {AppConfig.COLORS['primary']}; margin: 0;">{taluk_name}</h2>
        <p style="color: #999; margin: 10px 0;">{ts}</p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 20px;">
            <div>
                <p style="font-size: 12px; color: #999; margin: 0;">Total Villages</p>
                <p style="font-size: 28px; font-weight: bold; color: {AppConfig.COLORS['primary']}; margin: 5px 0;">
                    {total}
                </p>
            </div>
            <div>
                <p style="font-size: 12px; color: #999; margin: 0;">Completed</p>
                <p style="font-size: 28px; font-weight: bold; color: {AppConfig.COLORS['success']}; margin: 5px 0;">
                    {completed}
                </p>
            </div>
            <div>
                <p style="font-size: 12px; color: #999; margin: 0;">Completion %</p>
                <p style="font-size: 28px; font-weight: bold; color: {AppConfig.COLORS['warning']}; margin: 5px 0;">
                    {completion_pct:.1f}%
                </p>
            </div>
        </div>
        
        <div style="margin-top: 20px; background: white; padding: 10px; border-radius: 5px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                <div>
                    <p style="font-size: 11px; color: #999; margin: 0;">Ground Water</p>
                    <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{metrics['gw']}</p>
                </div>
                <div>
                    <p style="font-size: 11px; color: #999; margin: 0;">Surface Water</p>
                    <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{metrics['sw']}</p>
                </div>
                <div>
                    <p style="font-size: 11px; color: #999; margin: 0;">Wells</p>
                    <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{metrics['wb']}</p>
                </div>
            </div>
        </div>
    </div>
    """
    return html_card

def generate_progress_graph_display(final: pd.DataFrame, title: str) -> io.BytesIO:
    """Generate lightweight progress graph (low DPI for display)"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=AppConfig.IMAGE_DPI_GRAPH)
    
    # Simplified plot
    vao_names = final['VAO Full Name'].str[:15]  # Shorten names
    colors = [AppConfig.COLORS['success'] if x >= 100 else AppConfig.COLORS['warning'] 
              for x in final['% Completed'].fillna(0) * 100]
    
    ax.barh(vao_names, final['% Completed'].fillna(0) * 100, color=co
