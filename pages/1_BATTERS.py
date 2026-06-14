import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys
import os

# Ensure the app can find the brain file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cricket_rules import FORMAT_CONFIG

# ==========================================
# 1. INITIAL STATE CHECKS & BRAIN LOADING
# ==========================================
if 'data_df' not in st.session_state or 'global_format_selection' not in st.session_state:
    st.warning("⚠️ No dataset detected. Please go to the **Home Page**, select your format, and upload your CSV first.")
    st.stop()

# Load Data and Settings
df_raw = st.session_state['data_df'].copy()
active_format = st.session_state['global_format_selection']
speed_unit = st.session_state.get('speed_unit_preference', 'KPH')

# Fetch the specific rules for this format from the Brain
rules = FORMAT_CONFIG[active_format]
BALL_TYPE = rules["ball_type"]
SEAM_BINS = rules["seam_bins"]
SPIN_BINS = rules["spin_bins"]

st.markdown(f"## 🏏 Batters Analysis ({active_format})")
st.markdown(f"**Metrics Mode:** `{BALL_TYPE} Ball` | **Speed Unit:** `{speed_unit}`")

# ==========================================
# 2. DYNAMIC MULTI-SELECT FILTERS
# ==========================================
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

# Multi-select functions (defaulting to all if empty)
with filter_col1:
    tours = df_raw["Tour"].dropna().unique().tolist()
    sel_tours = st.multiselect("Tour(s)", options=tours, default=tours)

with filter_col2:
    matches = df_raw[df_raw["Tour"].isin(sel_tours)]["Match"].dropna().unique().tolist() if sel_tours else df_raw["Match"].dropna().unique().tolist()
    sel_matches = st.multiselect("Match(es)", options=matches, default=matches)

with filter_col3:
    teams = df_raw["BattingTeam"].dropna().unique().tolist()
    sel_teams = st.multiselect("Batting Team(s)", options=teams, default=teams)

with filter_col4:
    batsmen = df_raw[df_raw["BattingTeam"].isin(sel_teams)]["BatsmanName"].dropna().unique().tolist() if sel_teams else df_raw["BatsmanName"].dropna().unique().tolist()
    sel_batsmen = st.multiselect("Batsman Name(s)", options=batsmen)

# Apply Filters
df_filtered = df_raw.copy()
if sel_tours: df_filtered = df_filtered[df_filtered["Tour"].isin(sel_tours)]
if sel_matches: df_filtered = df_filtered[df_filtered["Match"].isin(sel_matches)]
if sel_teams: df_filtered = df_filtered[df_filtered["BattingTeam"].isin(sel_teams)]
if sel_batsmen: df_filtered = df_filtered[df_filtered["BatsmanName"].isin(sel_batsmen)]

df_seam = df_filtered[df_filtered["DeliveryType"] == "Seam"]
df_spin = df_filtered[df_filtered["DeliveryType"] == "Spin"]

# ==========================================
# 3. DYNAMIC CHARTING FUNCTIONS
# ==========================================

# --- CHART 2: CREASE BEEHIVE (Adapts to White/Red Ball) ---
def create_crease_beehive(df_in, delivery_type, ball_type):
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No data for Analysis", ha='center', va='center')
        ax.axis('off')
        return fig

    wickets = df_in[df_in["Wicket"] == True]
    non_wickets_all = df_in[df_in["Wicket"] == False]
    boundaries = non_wickets_all[(non_wickets_all["Runs"] == 4) | (non_wickets_all["Runs"] == 6)]
    regular_balls = non_wickets_all[(non_wickets_all["Runs"] != 4) & (non_wickets_all["Runs"] != 6)]
    
    df_lateral = df_in.copy()
    is_rhb = df_in["IsBatsmanRightHanded"].iloc[0] if not df_in.empty and "IsBatsmanRightHanded" in df_in.columns else True

    def assign_lateral_zone(row):
        y = row["CreaseY"]
        if row["IsBatsmanRightHanded"] == True:
            if y > 0.18: return "LEG"
            elif y >= -0.18: return "STUMPS"
            elif y > -0.65: return "OUTSIDE OFF"
            else: return "WAY OUTSIDE OFF"
        else:
            if y > 0.65: return "WAY OUTSIDE OFF"
            elif y > 0.18: return "OUTSIDE OFF"
            elif y >= -0.18: return "STUMPS"
            else: return "LEG"
            
    df_lateral["LateralZone"] = df_lateral.apply(assign_lateral_zone, axis=1)
    
    summary = df_lateral.groupby("LateralZone").agg(
        Runs=("Runs", "sum"), Wickets=("Wicket", lambda x: int((x == True).sum())), Balls=("Wicket", "count")
    )
    
    ordered_zones = ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]
    summary = summary.reindex(ordered_zones).fillna(0)
    summary["Avg"] = np.where(summary["Wickets"] > 0, summary["Runs"] / summary["Wickets"], 0)
    
    # White Ball specific calc
    if ball_type == "White":
        summary["SR"] = np.where(summary["Balls"] > 0, (summary["Runs"] / summary["Balls"]) * 100, 0)

    if not is_rhb:
        summary = summary.iloc[::-1]

    fig = plt.figure(figsize=(7, 5)) 
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.005) 
    ax_bh = fig.add_subplot(gs[0, 0])
    ax_boxes = fig.add_subplot(gs[1, 0])
    fig.patch.set_facecolor('white')

    # Traces
    ax_bh.scatter(regular_balls["CreaseY"], regular_balls["CreaseZ"], s=40, c='lightgrey', edgecolor='white', linewidths=1.0, alpha=0.95)
    ax_bh.scatter(boundaries["CreaseY"], boundaries["CreaseZ"], s=80, c='royalblue', edgecolor='white', linewidths=1.0, alpha=0.95)
    ax_bh.scatter(wickets["CreaseY"], wickets["CreaseZ"], s=80, c='red', edgecolor='white', linewidths=1.0, alpha=0.95)

    ax_bh.axvline(x=-0.18, color="grey", linestyle="--", linewidth=0.5) 
    ax_bh.axvline(x=0.18, color="grey", linestyle="--", linewidth=0.5)
    ax_bh.axvline(x=0, color="grey", linestyle="--", linewidth=0.5) 
    ax_bh.axhline(y=0.78, color="grey", linestyle="-", linewidth=0.5)
    
    ax_bh.set_xlim([-2, 2]); ax_bh.set_ylim([0, 2])
    ax_bh.set_aspect('equal', adjustable='box')
    ax_bh.axis('off')

    # Lateral Boxes
    num_regions = len(ordered_zones)
    box_width = 1 / num_regions
    left = 0
    
    # Coloring metric depends on ball type
    color_metric = summary["SR"] if ball_type == "White" else summary["Avg"]
    max_val = color_metric.max() if color_metric.max() > 0 else (200 if ball_type == "White" else 100)
    norm = mcolors.Normalize(vmin=0, vmax=max_val)
    cmap = plt.get_cmap('Wistia')

    for index, row in summary.iterrows():
        runs = int(row["Runs"])
        outs = int(row["Wickets"])
        avg = row["Avg"]
        color_val = row["SR"] if ball_type == "White" else row["Avg"]

        color = cmap(norm(color_val)) if color_val > 0 else 'white'
        r, g, b, a = mcolors.to_rgba(color)
        luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text_color = 'white' if luminosity < 0.5 else 'black'

        ax_boxes.add_patch(patches.Rectangle((left, 0), box_width, 1, edgecolor="black", facecolor=color, linewidth=1))

        # Dynamic Text
        label_top = f"{runs} R, {outs} W"
        if ball_type == "White":
            label_bottom = f"{avg:.1f} Avg, {row['SR']:.0f} SR"
        else:
            label_bottom = f"{avg:.1f} Avg"

        ax_boxes.text(left + box_width / 2, 1.05, index, ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')
        ax_boxes.text(left + box_width / 2, 0.65, label_top, ha='center', va='center', fontsize=8, fontweight='bold', color=text_color)
        ax_boxes.text(left + box_width / 2, 0.35, label_bottom, ha='center', va='center', fontsize=8, fontweight='bold', color=text_color)

        left += box_width

    ax_boxes.set_xlim(0, 1); ax_boxes.set_ylim(-0.1, 1.3)
    ax_boxes.axis('off')
    return fig

# --- CHART 3: PITCH MAP (Reads dynamic bins from Brain) ---
def create_pitch_map(df_in, custom_bins):
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(3,4.7))
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        ax.axis('off')
        return fig

    pitch_wickets = df_in[df_in["Wicket"] == True]
    pitch_non_wickets = df_in[df_in["Wicket"] == False]
    
    fig, ax = plt.subplots(figsize=(3,4.7))
    ax.set_facecolor('white')

    boundary_y_values = sorted([v[0] for v in custom_bins.values() if v[0] > -4.0], reverse=True)
    for y_val in boundary_y_values:
        ax.axhline(y=y_val, color="lightgrey", linewidth=1.0, linestyle="--")

    for length_name, bounds in custom_bins.items():
        mid_y = (bounds[0] + bounds[1]) / 2
        ax.text(x=-1.45, y=mid_y, s=length_name.upper(), ha='left', va='center', fontsize=8, color="grey", fontweight='bold')
    
    ax.scatter(pitch_non_wickets["BounceY"], pitch_non_wickets["BounceX"], s=60, c='#D3D3D3', edgecolor='white', alpha=0.9)
    ax.scatter(pitch_wickets["BounceY"], pitch_wickets["BounceX"], s=90, c='red', edgecolor='white', alpha=0.95)

    ax.axvline(x=-0.18, color="#777777", linestyle="--", linewidth=1)
    ax.axvline(x=0.18, color="#777777", linestyle="--", linewidth=1)
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([16.0, -5.0]) # Extended slightly for Women's full toss bounds
    ax.set_xticks([]); ax.set_yticks([])
    
    for spine_name in ['left', 'top', 'bottom','right']:
        ax.spines[spine_name].set_visible(True)
    return fig

# --- CHART 3b: PITCH LENGTH BARS (Dynamic Metrics & Bins) ---
def create_pitch_Length_bars(df_in, custom_bins, ball_type):
    FIG_SIZE = (3, 4.7) 
    if df_in.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.axis('off')
        return fig

    ordered_keys = list(custom_bins.keys())

    def assign_pitch_Length(x):
        for length_name, bounds in custom_bins.items():
            if bounds[0] <= x < bounds[1]: return length_name
        return None

    df_pitch = df_in.copy()
    df_pitch["PitchLength"] = df_pitch["BounceX"].apply(assign_pitch_Length)
    
    df_summary = df_pitch.groupby("PitchLength").agg(
        Runs=("Runs", "sum"),  
        Dismissals=("Wicket", lambda x: int((x == True).sum())), 
        Balls=("Wicket", "count")
    ).reset_index().set_index("PitchLength").reindex(ordered_keys).fillna(0)

    df_summary["Average"] = np.where(df_summary["Dismissals"] > 0, df_summary["Runs"] / df_summary["Dismissals"], 0)
    
    # Brain check: Determine which metrics to plot
    if ball_type == "White":
        df_summary["StrikeRate"] = np.where(df_summary["Balls"] > 0, (df_summary["Runs"] / df_summary["Balls"]) * 100, 0)
        metrics = ["StrikeRate", "Average", "Runs"]
        titles = ["Batting Strike Rate", "Batting Average", "Runs"]
    else:
        metrics = ["Average", "Runs", "Dismissals"]
        titles = ["Batting Average", "Runs", "Dismissals"]

    categories = df_summary.index.tolist()[::-1]
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE, sharey=True) 
    plt.subplots_adjust(hspace=0.8) 

    for i, ax in enumerate(axes):
        metric = metrics[i]
        title = titles[i]
        values = df_summary[metric].values[::-1] 
        max_val = df_summary[metric].max() * 1.2 if df_summary[metric].max() > 0 else (100 if metric != "Dismissals" else 10)
        
        ax.set_xlim(0, max_val)
        ax.barh(categories, values, height=0.49, color='#ff5000', zorder=3, alpha=0.9)
        
        for j, (cat, val) in enumerate(zip(categories, values)):
            label = f"{int(val)}" if metric in ["Dismissals", "Runs"] else f"{val:.0f}"
            ax.text(val, j, label, ha='left', va='center', fontsize=9, fontweight='bold', color='black', zorder=4)

        ax.set_title(title, fontsize=10, fontweight='bold', pad=0, loc='left')
        ax.set_facecolor('white')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', length=0)

        if i == 2:
            ax.set_yticks(np.arange(len(categories)), labels=[c.upper() for c in categories], fontsize=8)
        else:
            ax.set_yticks(np.arange(len(categories)), labels=[''] * len(categories))
            
        ax.xaxis.grid(False); ax.yaxis.grid(False); ax.set_xticks([]) 
        for spine_name in ['left', 'right', 'top', 'bottom']: ax.spines[spine_name].set_visible(True)
            
    return fig

# ==========================================
# 4. RENDER LAYOUT
# ==========================================
col1, col2 = st.columns(2)
    
with col1:
    st.markdown("### v SEAM")
    st.markdown("###### CREASE BEEHIVE v SEAM")
    st.pyplot(create_crease_beehive(df_seam, "Seam", BALL_TYPE), use_container_width=True)
    
    pitch_col, pitch_bars = st.columns(2)
    with pitch_col:
        st.markdown("###### PITCHMAP v SEAM")
        st.pyplot(create_pitch_map(df_seam, SEAM_BINS), use_container_width=True)  
    with pitch_bars:
        st.markdown("###### ")
        st.pyplot(create_pitch_Length_bars(df_seam, SEAM_BINS, BALL_TYPE), use_container_width=True)   

with col2:
    st.markdown("### v SPIN")
    st.markdown("###### CREASE BEEHIVE v SPIN")
    st.pyplot(create_crease_beehive(df_spin, "Spin", BALL_TYPE), use_container_width=True)
 
    pitch_col, pitch_bars = st.columns(2)
    with pitch_col:
        st.markdown("###### PITCHMAP v SPIN")
        st.pyplot(create_pitch_map(df_spin, SPIN_BINS), use_container_width=True)  
    with pitch_bars:
        st.markdown("###### ")
        st.pyplot(create_pitch_Length_bars(df_spin, SPIN_BINS, BALL_TYPE), use_container_width=True)
