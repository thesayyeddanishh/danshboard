"""
Consolidated VR Story Assistant
Run from this folder:
  streamlit run Home.py
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
from io import StringIO

from cricket_config import FORMAT_KEYS, FORMAT_LABELS, resolve_format
from sql_loader import DumpReadError, load_dashboard_df

REQUIRED_COLS = [
    "Innings",
    "Over",
    "Ball",
    "BatsmanName",
    "BowlerName",
    "BowlingTeam",
    "BattingTeam",
    "IsBowlerRightHanded",
    "IsBatsmanRightHanded",
    "Wicket",
    "Runs",
    "ReleaseY",
    "ReleaseZ",
    "ReleaseSpeed",
    "Deviation",
    "Swing",
    "DeliveryType",
    "StumpsY",
    "StumpsZ",
    "CreaseY",
    "CreaseZ",
    "LandingX",
    "LandingY",
    "BounceX",
    "BounceY",
    "InterceptionX",
    "InterceptionZ",
    "InterceptionY",
    "Ground",
    "Tour",
    "Year",
    "Match"
]

st.set_page_config(
    page_title="VR DANshboard", 
    page_icon="hawk-logo.png",
    layout="wide"
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 200px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def process_upload(uploaded_file, source: str = "csv") -> bool:
    if uploaded_file is None:
        return False
    try:
        if source == "sql":
            with st.spinner("Reading SQL dump..."):
                df_raw = load_dashboard_df(uploaded_file.getvalue())
        else:
            data = uploaded_file.getvalue().decode("utf-8")
            df_raw = pd.read_csv(StringIO(data))
        if not all(col in df_raw.columns for col in REQUIRED_COLS):
            missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
            label = "SQL dump" if source == "sql" else "CSV file"
            st.error(f"The {label} is missing required columns: {', '.join(missing)}")
            st.session_state.pop("data_df", None)
            st.session_state.pop("file_name", None)
            return False
        st.session_state["data_df"] = df_raw
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["data_source"] = source
        if "initial_load_complete" in st.session_state:
            st.success(f"Data successfully replaced!\n Total deliveries loaded: {len(df_raw):,}.")
        else:
            st.session_state["initial_load_complete"] = True
            st.success(
                f"Data uploaded successfully! File: {uploaded_file.name}. "
                "Use the sidebar pages: Batters, Pacers, Spinners, Leaderboard."
            )
        return True
    except DumpReadError as e:
        st.error(str(e))
        st.session_state.pop("data_df", None)
        st.session_state.pop("file_name", None)
        return False
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.session_state.pop("data_df", None)
        st.session_state.pop("file_name", None)
        return False

st.title("VR Story Assistant")
fmt_labels = [FORMAT_LABELS[k] for k in FORMAT_KEYS]
default_key = st.session_state.get("cricket_format", "men_t20i")
default_idx = FORMAT_KEYS.index(default_key) if default_key in FORMAT_KEYS else 0
c1, c2 = st.columns([3, 1])
with c1:
    choice = st.selectbox("Format", fmt_labels, index=default_idx, key="format_select_main")
with c2:
    _saved_unit = st.session_state.get("speed_unit", "kph")
    _unit_default = "MPH" if _saved_unit == "mph" else "KPH"
    unit_choice = st.segmented_control(
        "Speed Unit", ["KPH", "MPH"], default=_unit_default, key="speed_unit_select"
    )
st.session_state["cricket_format"] = FORMAT_KEYS[fmt_labels.index(choice)]
st.session_state["speed_unit"] = (unit_choice or _unit_default).lower()
_cfg = resolve_format(st.session_state["cricket_format"])

st.subheader("2. Upload CSV or SQL dump")
source_choice = st.radio("Data source", ["CSV", "SQL dump"], horizontal=True, key="source_select")
source_key = "sql" if source_choice == "SQL dump" else "csv"

if source_key == "csv":
    with st.expander("Required CSV columns when exporting from CDS"):
        st.markdown(
            """
* **General:** `Innings`, `Over`, `Ball`, `BowlerName`, `BatsmanName`, `BowlingTeam`, `BattingTeam`
* **Handedness:** `IsBowlerRightHanded`, `IsBatsmanRightHanded`
* **Results:** `Wicket`, `Runs`
* **Coordinates:** `Bounce X Y`, `Interception X Y Z`, `Crease Y Z`, `Stumps Y Z`, `Release Y Z`, `Landing X Y`
* **Delivery:** `DeliveryType`, `ReleaseSpeed`, `Deviation`, `Swing`
* **Match:** `Ground`, `Tour`, `Year`, `Match`
            """
        )
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"], key="main_uploader")
else:
    with st.expander("SQL dump requirements"):
        st.markdown(
            """
Upload a PostgreSQL dump of the Hawk-Eye `cricketdatabase` made with `pg_dump -Fc`
(custom format). All the required dashboard columns are built automatically from the
database tables, so the result is identical to a CSV export. Plain-text `.sql` files are not supported.
            """
        )
    uploaded_file = st.file_uploader(
        "Upload your SQL dump here", type=["sql", "dump", "backup"], key="sql_uploader"
    )

if uploaded_file is not None:
    if "data_df" not in st.session_state or uploaded_file.name != st.session_state.get("file_name"):
        process_upload(uploaded_file, source_key)
      
st.write("Wait for the next line to turn blue after uploading")

if "data_df" in st.session_state:
    df_loaded = st.session_state["data_df"]
    _src_label = "SQL dump" if st.session_state.get("data_source") == "sql" else "CSV"
    st.info(f"{_src_label} is loaded. Open **BATTERS**, **PACERS**, **SPINNERS**, or **LEADERBOARD** from the sidebar.")
    st.write(f"Total deliveries: {len(df_loaded):,}")
    if st.checkbox("Show first 5 rows"):
        st.write(f"**File:** `{st.session_state.get('file_name', 'N/A')}`")
        st.dataframe(df_loaded.head())

import streamlit.components.v1 as components

st.markdown("""
    <script>
        // We set up an observer to watch for title changes and force it back
        const targetNode = document.querySelector('title');
        const observer = new MutationObserver(function(mutations) {
            if (document.title !== "VR DANshboard") {
                document.title = "VR DANshboard";
            }
        });
        observer.observe(targetNode, { childList: true, subtree: true, characterData: true });
        // Set it initially
        document.title = "VR DANshboard";
    </script>
""", unsafe_allow_html=True)
