# Home.py
import streamlit as st
import pandas as pd
import io
from cricket_rules import FORMAT_CONFIG

REQUIRED_COLS = [
    "BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", 
    "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", 
    "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", 
    "InterceptionZ", "InterceptionY", "Over"
]

st.set_page_config(layout="wide", page_title="VR Story Assistant", page_icon="hawk-logo.png")

# --- Custom CSS ---
st.markdown("<style>section[data-testid='stSidebar'] { width: 220px !important; }</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛠️ App Administration")
    if st.button("🔄 Clear App Cache & Reload"):
        st.cache_data.clear() 
        st.success("Cache Cleared Successfully!")
        st.rerun()

@st.cache_data(ttl=900, show_spinner="Processing raw performance dataset...")
def load_and_validate_csv(file_bytes):
    df_raw = pd.read_csv(io.BytesIO(file_bytes))
        
    missing_cols = [col for col in REQUIRED_COLS if col not in df_raw.columns]
    if missing_cols:
        return {"success": False, "error": f"Missing required columns: {', '.join(missing_cols)}"}
    
    # === GLOBAL FIX: FORCE UPPERCASE ON ALL NAMES ===
    if "BatsmanName" in df_raw.columns:
        df_raw["BatsmanName"] = df_raw["BatsmanName"].astype(str).str.upper()
    if "BowlerName" in df_raw.columns:
        df_raw["BowlerName"] = df_raw["BowlerName"].astype(str).str.upper()
    if "BattingTeam" in df_raw.columns:
        df_raw["BattingTeam"] = df_raw["BattingTeam"].astype(str).str.upper()
        
    return {"success": True, "df": df_raw}

def process_upload(uploaded_file):
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            result = load_and_validate_csv(file_bytes)
            
            if not result["success"]:
                st.error(result["error"])
                st.session_state.pop('data_df', None)
                return False
            else:
                st.session_state['data_df'] = result["df"]
                st.session_state['file_name'] = uploaded_file.name
                st.success(f"Data uploaded successfully! File: {uploaded_file.name}. Please select a dashboard from the sidebar.")
                return True
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return False
    return False

if 'store' not in st.session_state:
    st.session_state['store'] = {
        'data_df': None,
        'format': None,
        'file_name': None
    }

# ==========================================
# ==========================================
# MAIN PAGE UI
# ==========================================
st.title("🦅 Hawkeye Master Hub")

# 1. MANDATORY FORMAT SELECTION
st.markdown("### Step 1: Select Format")

# Change the widget key name to a temporary widget key
selected_format = st.selectbox(
    "Choose the Match Format & Ruleset:",
    options=list(FORMAT_CONFIG.keys()),
    key="format_widget_temp" 
)

# 🔥 FIX: Copy the value to a permanent, non-widget key so it survives page changes!
st.session_state['global_format_selection'] = selected_format

# Show what ball type was registered by the "Brain"
ball_type = FORMAT_CONFIG[selected_format]["ball_type"]
st.info(f"Loaded rules for **{selected_format}** ({ball_type} Ball Metrics Active).")

# 2. DATA UPLOAD
st.markdown("### Step 2: Upload Data")
uploaded_file = st.file_uploader("Upload your tracking CSV file here", type=["csv"], key="main_uploader")

if uploaded_file is not None:
    if 'data_df' not in st.session_state or uploaded_file.name != st.session_state.get('file_name'):
        process_upload(uploaded_file)

# 3. SPEED UNIT PREFERENCE
st.markdown("---")
st.radio("Select Speed Metric Display Unit:", options=["KPH", "MPH"], index=0, horizontal=True, key="speed_unit_preference")

# DATA PREVIEW
if 'data_df' in st.session_state:
    st.write(f"**Total Deliveries Loaded:** {len(st.session_state['data_df']):,}")
