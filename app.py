"""
MI Census Pro - Streamlit Cloud Edition (Free Tier Optimized)
Version: V200_FREECLOUD_FIXED
Author: Enhanced by AI
Optimized for: Streamlit Cloud Free Tier (zero 403 errors)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import io
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIG
# ========================================
class AppConfig:
    VERSION = "V200_FREECLOUD_FIXED"
    PASSWORD = "mandya"
    
    USERS = {
        "Chethan_NGM": "Nagamangala Taluk",
        "Gangadhar_MLV": "Malavalli Taluk",
        "Nagarjun_KRP": "K.R. Pete Taluk",
        "Prashanth_SRP": "Srirangapatna Taluk",
        "Purushottam_PDV": "Pandavapura Taluk",
        "Siddaraju_MDY": "Mandya Taluk",
        "Sunil_MDR": "Maddur Taluk",
        "Mandya_Admin": "District Admin"
    }
    
    COLORS = {
        "primary": "#1a73e8",
        "success": "#34A853",
        "warning": "#FBBC04",
        "danger": "#EA4335",
    }
    
    TALUKS = [
        "K.R. Pete Taluk",
        "Maddur Taluk",
        "Malavalli Taluk",
        "Mandya Taluk",
        "Nagamangala Taluk",
        "Pandavapura Taluk",
        "Srirangapatna Taluk"
    ]

# ========================================
# SETUP
# ========================================
def setup_streamlit_config():
    """Create optimal Streamlit config"""
    config_dir = ".streamlit"
    config_path = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_path):
        os.makedirs(config_dir, exist_ok=True)
        config = """
[server]
maxUploadSize = 100
maxMessageSize = 100
enableCORS = false
enableXsrfProtection = true
headless = true

[theme]
primaryColor = "#1a73e8"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"

[client]
toolbarMode = "minimal"
showErrorDetails = false

[logger]
level = "error"

[cache]
maxMegabytes = 32
ttlSeconds = 30
"""
        with open(config_path, 'w') as f:
            f.write(config)

setup_streamlit_config()

st.set_page_config(
    page_title="MI Census Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================================
# SESSION STATE
# ========================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.report_data = None

# ========================================
# UTILITIES
# ========================================
def save_taluk_data(taluk: str, data: Dict):
    """Save taluk metrics"""
    os.makedirs("central_data", exist_ok=True)
    filepath = f"central_data/{taluk.replace(' ', '_')}.json"
    
    safe_data = {
        k: (int(v) if isinstance(v, (np.integer, np.int64)) else
            float(v) if isinstance(v, (np.floating, np.float64)) else v)
        for k, v in data.items()
    }
    safe_data['timestamp'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(safe_data, f)

def load_taluk_data(taluk: str) -> Dict:
    """Load taluk metrics"""
    filepath = f"central_data/{taluk.replace(' ', '_')}.json"
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    return {
        "taluk": taluk,
        "total_villages": 0,
        "completed": 0,
        "gw": 0,
        "sw": 0,
        "wb": 0
    }

@st.cache_data(ttl=60, max_entries=1)
def process_report(df_assign, df_monitor, taluk_name, completed, submitted):
    """Process report data"""
    try:
        df_assign.columns = df_assign.columns.str.strip()
        df_monitor.columns = df_monitor.columns.str.strip()
        
        # Extract metrics
        total = len(df_monitor)
        
        gw = pd.to_numeric(
            df_monitor.get('Total schedules GW', pd.Series([0]*total)),
            errors='coerce'
        ).fillna(0).sum()
        
        sw = pd.to_numeric(
            df_monitor.get('Total schedules SW', pd.Series([0]*total)),
            errors='coerce'
        ).fillna(0).sum()
        
        wb = pd.to_numeric(
            df_monitor.get('Total schedules WB', pd.Series([0]*total)),
            errors='coerce'
        ).fillna(0).sum()
        
        metrics = {
            "total_villages": int(total),
            "completed": int(completed),
            "submitted": int(submitted),
            "gw": int(gw),
            "sw": int(sw),
            "wb": int(wb)
        }
        
        return metrics, df_monitor
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None

def generate_excel_report(df_data: pd.DataFrame, metrics: Dict, taluk_name: str) -> io.BytesIO:
    """Generate Excel report"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Total Villages', 'GW Schemes', 'SW Schemes', 'Wells', 'Completed', 'Submitted'],
                'Value': [
                    metrics['total_villages'],
                    metrics['gw'],
                    metrics['sw'],
                    metrics['wb'],
                    metrics['completed'],
                    metrics['submitted']
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Data sheet
            df_data.to_excel(writer, sheet_name='Data', index=False)
            
            # Format
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D3D3D3',
                'border': 1,
                'align': 'center'
            })
            
            summary_sheet = writer.sheets['Summary']
            for col_num, value in enumerate(summary_data['Metric']):
                summary_sheet.write(0, col_num, value, header_format)
        
        output.seek(0)
        return output
        
    except Exception as e:
        st.error(f"Excel generation error: {str(e)}")
        return None

def create_status_card_html(taluk_name: str, metrics: Dict) -> str:
    """Create HTML status card"""
    total = metrics['total_villages']
    completed = metrics['completed']
    pct = (completed / total * 100) if total > 0 else 0
    
    html = f"""
    <div style="
        border: 3px solid {AppConfig.COLORS['primary']};
        border-radius: 10px;
        padding: 20px;
        background: #f8f9fa;
        margin: 10px 0;
    ">
        <h2 style="color: {AppConfig.COLORS['primary']}; margin: 0;">{taluk_name}</h2>
        <hr style="border: none; border-top: 2px solid {AppConfig.COLORS['primary']};">
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div style="text-align: center;">
                <p style="color: #666; font-size: 12px; margin: 0;">Total Villages</p>
                <p style="font-size: 32px; font-weight: bold; color: {AppConfig.COLORS['primary']}; margin: 10px 0;">{total}</p>
            </div>
            <div style="text-align: center;">
                <p style="color: #666; font-size: 12px; margin: 0;">Completed</p>
                <p style="font-size: 32px; font-weight: bold; color: {AppConfig.COLORS['success']}; margin: 10px 0;">{completed}</p>
            </div>
            <div style="text-align: center;">
                <p style="color: #666; font-size: 12px; margin: 0;">Completion %</p>
                <p style="font-size: 32px; font-weight: bold; color: {AppConfig.COLORS['warning']}; margin: 10px 0;">{pct:.1f}%</p>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px;">
            <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                <p style="color: #666; font-size: 11px; margin: 0;">GW Schemes</p>
                <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{metrics['gw']}</p>
            </div>
            <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                <p style="color: #666; font-size: 11px; margin: 0;">SW Schemes</p>
                <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{metrics['sw']}</p>
            </div>
            <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                <p style="color: #666; font-size: 11px; margin: 0;">Wells</p>
                <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{metrics['wb']}</p>
            </div>
        </div>
    </div>
    """
    return html

def create_progress_chart(df_data: pd.DataFrame, title: str) -> io.BytesIO:
    """Create progress chart"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=70)
        
        if len(df_data) > 0:
            sample = df_data.head(10)
            values = [1.0] * len(sample)
            colors = [AppConfig.COLORS['success']] * len(sample)
            
            ax.barh(range(len(sample)), values, color=colors)
            ax.set_yticks(range(len(sample)))
            ax.set_yticklabels([f"Entry {i+1}" for i in range(len(sample))])
            ax.set_xlabel('Progress', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.2)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=70, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        st.error(f"Chart generation error: {str(e)}")
        return None

# ========================================
# MAIN APP
# ========================================
def main():
    st.markdown("# 🏛️ MI Census Pro")
    st.markdown("---")
    
    # Sidebar Login
    with st.sidebar:
        st.write("### 🔐 Login")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            user = st.selectbox(
                "Officer:",
                list(AppConfig.USERS.keys()),
                label_visibility="collapsed"
            )
        
        with col2:
            password = st.text_input(
                "Password:",
                type="password",
                label_visibility="collapsed"
            )
        
        if st.button("Login", key="login_btn"):
            if password == AppConfig.PASSWORD:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.success(f"✅ Welcome, {user}!")
                st.rerun()
            else:
                st.error("❌ Invalid password")
        
        if st.session_state.authenticated:
            st.write(f"**User:** {st.session_state.user}")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.rerun()
    
    # Check Authentication
    if not st.session_state.authenticated:
        st.warning("⚠️ Please login to continue")
        return
    
    # Main Content
    tab1, tab2, tab3 = st.tabs(["📊 Report", "📈 Dashboard", "⚙️ Admin"])
    
    # TAB 1: REPORT
    with tab1:
        st.write("### 📊 Generate Report")
        
        col1, col2 = st.columns(2)
        with col1:
            taluk = st.selectbox("Select Taluk:", AppConfig.TALUKS)
        with col2:
            st.write("")  # Spacing
        
        assign_file = st.file_uploader("📁 Master File (Assignment):", type=["xlsx", "csv"], key="assign")
        monitor_file = st.file_uploader("📁 Monitor File (Data):", type=["xlsx", "csv"], key="monitor")
        
        col1, col2 = st.columns(2)
        with col1:
            completed = st.number_input("Completed Villages:", min_value=0, step=1)
        with col2:
            submitted = st.number_input("Submitted Villages:", min_value=0, step=1)
        
        if st.button("🚀 Generate Report", key="gen_report"):
            if assign_file and monitor_file:
                with st.spinner("⏳ Processing..."):
                    try:
                        # Load files
                        if assign_file.name.endswith('.xlsx'):
                            df_assign = pd.read_excel(assign_file)
                        else:
                            df_assign = pd.read_csv(assign_file)
                        
                        if monitor_file.name.endswith('.xlsx'):
                            df_monitor = pd.read_excel(monitor_file)
                        else:
                            df_monitor = pd.read_csv(monitor_file)
                        
                        # Process
                        metrics, df_data = process_report(df_assign, df_monitor, taluk, int(completed), int(submitted))
                        
                        if metrics:
                            # Save data
                            save_taluk_data(taluk, metrics)
                            
                            # Display results
                            st.success("✅ Report generated!")
                            
                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Villages", metrics['total_villages'])
                            with col2:
                                st.metric("GW Schemes", metrics['gw'])
                            with col3:
                                st.metric("SW Schemes", metrics['sw'])
                            with col4:
                                st.metric("Wells", metrics['wb'])
                            
                            # Status card
                            st.markdown(create_status_card_html(taluk, metrics), unsafe_allow_html=True)
                            
                            # Chart
                            chart_buf = create_progress_chart(df_data, f"{taluk} - Data Summary")
                            if chart_buf:
                                st.image(chart_buf, use_column_width=True)
                            
                            # Excel download
                            excel_buf = generate_excel_report(df_data, metrics, taluk)
                            if excel_buf:
                                st.download_button(
                                    "📥 Download Excel Report",
                                    excel_buf.getvalue(),
                                    f"{taluk}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please upload both files")
    
    # TAB 2: DASHBOARD
    with tab2:
        st.write("### 📈 District Dashboard")
        
        # Load all taluk data
        all_data = []
        for taluk in AppConfig.TALUKS:
            data = load_taluk_data(taluk)
            all_data.append(data)
        
        if all_data:
            df_dashboard = pd.DataFrame(all_data)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Taluks", len(df_dashboard))
            with col2:
                total_villages = df_dashboard['total_villages'].sum()
                st.metric("Total Villages", int(total_villages))
            with col3:
                total_completed = df_dashboard['completed'].sum()
                st.metric("Total Completed", int(total_completed))
            with col4:
                if total_villages > 0:
                    pct = (total_completed / total_villages * 100)
                    st.metric("Overall %", f"{pct:.1f}%")
                else:
                    st.metric("Overall %", "0%")
            
            # Table
            st.write("#### Taluk-wise Summary")
            display_cols = ['taluk', 'total_villages', 'completed', 'gw', 'sw', 'wb']
            st.dataframe(
                df_dashboard[display_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("ℹ️ No data available. Generate reports to see dashboard.")
    
    # TAB 3: ADMIN
    with tab3:
        st.write("### ⚙️ Admin Functions")
        
        if st.session_state.user == "Mandya_Admin":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Clear Cache"):
                    st.cache_data.clear()
                    st.success("✅ Cache cleared")
            
            with col2:
                if st.button("📥 Export All Data"):
                    all_data = []
                    for taluk in AppConfig.TALUKS:
                        data = load_taluk_data(taluk)
                        all_data.append(data)
                    
                    df_export = pd.DataFrame(all_data)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        "📊 Download CSV",
                        csv,
                        f"census_export_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )
            
            st.info(f"**Version:** {AppConfig.VERSION}")
            st.info(f"**Server Time:** {datetime.now(timezone.utc) + timedelta(hours=5.5)}")
        else:
            st.warning("⚠️ Admin access only")

if __name__ == "__main__":
    main()
