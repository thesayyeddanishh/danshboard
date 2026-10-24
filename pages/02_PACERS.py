import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from io import StringIO
import base64
import matplotlib.patheffects as pe
from matplotlib import cm, colors, patches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from cricket_config import (
    resolve_format,
    get_pitch_bins as pitch_bins_for_format,
    ordered_seam_keys,
    seam_speed_group,
    pacer_effectiveness_seam_order,
    format_banner_caps,
    FORMAT_BANNER_STYLE,
    pitch_map_figsize,
    pitch_bin_percentages,
    add_crease_lateral_zone_background,
)

# --- CHART 3: PITCHMAP (BOUNCE LOCATION) ---
def create_pacer_pitch_map(df_in):
    cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
    PITCH_BINS = pitch_bins_for_format("Seam", cfg)

    if df_in.empty:
        fw, fh = pitch_map_figsize(cfg, width=3.0)
        fig, ax = plt.subplots(figsize=(fw, fh))
        ax.text(0.5, 0.5, "No data for Pacer Pitch Map", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    nw = df_in[df_in["Wicket"] == False].copy()
    runs = pd.to_numeric(nw.get("Runs", pd.Series(0, index=nw.index)), errors="coerce").fillna(0)
    is_boundary = runs.isin([4, 6])
    pitch_boundaries = nw[is_boundary]
    pitch_others = nw[~is_boundary]
    pitch_wickets = df_in[df_in["Wicket"] == True]

    fig, ax = plt.subplots(figsize=(3.5, 7))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    boundary_y_values = sorted([v[0] for v in PITCH_BINS.values() if v[0] > -4.0], reverse=True)
    for y_val in boundary_y_values:
        ax.axhline(y=y_val, color="lightgrey", linewidth=1.0, linestyle="--")

    pct_by_bin = pitch_bin_percentages(df_in, PITCH_BINS, bounce_col="BounceX")
    y_off = 0.2
    for length, bounds in PITCH_BINS.items():
        mid_y = (bounds[0] + bounds[1]) / 2
        ax.text(
            x=-1.45,
            y=mid_y + y_off,
            s=str(length).upper(),
            ha="left",
            va="center",
            fontsize=8,
            color="black",
            fontweight="bold",
        )
        p = int(pct_by_bin.get(length, 0))
        ax.text(
            x=-1.45,
            y=mid_y - y_off,
            s=f"{p}%",
            ha="left",
            va="center",
            fontsize=16,
            color="black",
            fontweight="bold",
        )

    s_nw, s_b, s_w = ((60, 75, 90) if cfg.is_test else (100, 100, 150))
    if not pitch_others.empty:
        ax.scatter(
            pitch_others["BounceY"],
            pitch_others["BounceX"],
            s=s_nw,
            c="#D3D3D3",
            edgecolor="white",
            linewidths=1.0,
            alpha=0.9,
            label="Others",
        )
    if not pitch_boundaries.empty:
        ax.scatter(
            pitch_boundaries["BounceY"],
            pitch_boundaries["BounceX"],
            s=s_b,
            c="royalblue",
            edgecolor="white",
            linewidths=0.5,
            alpha=0.75,
            label="Boundaries",
        )
    if not pitch_wickets.empty:
        ax.scatter(
            pitch_wickets["BounceY"],
            pitch_wickets["BounceX"],
            s=s_w,
            c="red",
            edgecolor="white",
            linewidths=0.5,
            alpha=1,
            label="Wicket",
        )

    ax.axvline(x=-0.115, color="#777777", linestyle="-", linewidth=0.5)
    ax.axvline(x=0.115, color="#777777", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="#777777", linestyle="-", linewidth=0.5)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([16.0, -4.0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    spine_color = "black"
    spine_width = 0.5
    for spine_name in ["left", "top", "bottom", "right"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color(spine_color)
        ax.spines[spine_name].set_linewidth(spine_width)

    plt.tight_layout()
    return fig

# --- CHART 3b: PITCH LENGTH METRICS (BOWLER FOCUS) ---
# --- Helper function for Pitch Bins (Hardcoded for Seam) ---
def get_pacer_pitch_bins():
    cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
    return pitch_bins_for_format("Seam", cfg)
def create_pacer_pitch_length_bars(df_in):
    cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
    FIG_SIZE = pitch_map_figsize(cfg, width=3.0)

    if df_in.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Data for Pacer Pitch Length Comparison", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    PITCH_BINS_DICT = get_pacer_pitch_bins()
    ordered_keys = ordered_seam_keys(cfg)

    def assign_pitch_length(x):
        for length, bounds in PITCH_BINS_DICT.items():
            if bounds[0] <= x < bounds[1]:
                return length
        return None

    df_pitch = df_in.copy()
    df_pitch["PitchLength"] = df_pitch["BounceX"].apply(assign_pitch_length)

    df_summary = (
        df_pitch.groupby("PitchLength")
        .agg(
            Runs=("Runs", "sum"),
            Wickets=("Wicket", lambda x: (x == True).sum()),
            Balls=("Wicket", "count"),
        )
        .reset_index()
        .set_index("PitchLength")
        .reindex(ordered_keys)
        .fillna(0)
    )

    df_summary["Economy"] = df_summary.apply(
        lambda row: (row["Runs"] / row["Balls"] * 6) if row["Balls"] > 0 else 0.0, axis=1
    )
    df_summary["Avg"] = df_summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else row["Runs"], axis=1
    )
    df_summary["SR"] = df_summary.apply(
        lambda row: row["Balls"] / row["Wickets"] if row["Wickets"] > 0 else row["Balls"], axis=1
    )

    categories = df_summary.index.tolist()[::-1]
    fig, axes = plt.subplots(3, 1, figsize=(3.5,7), sharey=True)
    plt.subplots_adjust(hspace=10)

    if cfg.is_test:
        metrics = ["Wickets", "Avg", "SR"]
        titles = ["Wickets", "Bowling Average", "Bowling Strike Rate"]
        max_w = df_summary["Wickets"].max() * 1.2 if df_summary["Wickets"].max() > 0 else 10
        max_avg = df_summary["Avg"].max() * 1.2 if df_summary["Avg"].max() > 0 else 50
        max_sr = df_summary["SR"].max() * 1.2 if df_summary["SR"].max() > 0 else 40
        xlim_limits = {"Wickets": (0, max_w), "Avg": (0, max_avg), "SR": (0, max_sr)}
    else:
        metrics = ["Wickets", "Economy", "Avg"]
        titles = ["Wickets", "Economy", "Bowling Average"]
        max_w = df_summary["Wickets"].max() * 1.2 if df_summary["Wickets"].max() > 0 else 10
        max_eco = df_summary["Economy"].max() * 1.2 if df_summary["Economy"].max() > 0 else 12
        max_avg = df_summary["Avg"].max() * 1.2 if df_summary["Avg"].max() > 0 else 50
        xlim_limits = {"Wickets": (0, max_w), "Economy": (0, max_eco), "Avg": (0, max_avg)}

    for i, ax in enumerate(axes):
        metric = metrics[i]
        title = titles[i]
        values = df_summary[metric].values[::-1]
        ax.barh(categories, values, height=0.49, color="#ff5000", zorder=3, alpha=0.9)

        if cfg.is_test:
            ax.text(
                0,
                1.05,
                title,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="bottom",
                ha="left",
                color="black",
            )
        else:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=0, loc="left")

        for j, (cat, val) in enumerate(zip(categories, values)):
            if metric == "Wickets":
                label = f"{int(val)}"
            elif metric == "Economy":
                label = f"{val:.1f}"
            else:
                label = f"{val:.1f}"

            ax.text(
                val,
                j,
                label,
                ha="left",
                va="center",
                fontsize=11 if cfg.is_test else 9,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="White", alpha=0.8, edgecolor="none", pad=2 if cfg.is_test else 2),
                zorder=4,
            )

        ax.set_facecolor("white")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", length=0)

        if i == 2:
            ax.set_yticks(
                np.arange(len(categories)),
                labels=[c.upper() for c in categories],
                fontsize=8 if cfg.is_test else 8,
            )
        else:
            ax.set_yticks(np.arange(len(categories)), labels=[""] * len(categories))

        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.set_xticks([])
        if cfg.is_test:
            ax.set_xlim(0, xlim_limits[metric][1] * 1.15)
        else:
            ax.set_xlim(0, xlim_limits[metric][1])

        spine_color = "lightgray"
        for spine_name in ["left", "right", "top", "bottom"]:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color(spine_color)
            ax.spines[spine_name].set_linewidth(1.0)

    plt.tight_layout(pad=0.5)
    return fig
    
# =========================================================
# Chart 1: CREASE BEEHIVE 
# ========================================================
def create_pacer_crease_beehive(df_in, handedness_label): # Renamed function and parameter
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(7, 6)); 
        ax.text(0.5, 0.5, f"No data for Analysis ({handedness_label})", ha='center', va='center', fontsize=12); 
        ax.axis('off'); 
        return fig

    # --- Data Filtering ---
    wickets = df_in[df_in["Wicket"] == True]
    non_wickets_all = df_in[df_in["Wicket"] == False]
    boundaries = non_wickets_all[(non_wickets_all["Runs"] == 4) | (non_wickets_all["Runs"] == 6)]
    regular_balls = non_wickets_all[(non_wickets_all["Runs"] != 4) & (non_wickets_all["Runs"] != 6)]
    
    # --- Lateral Zone Data Prep (Chart 2b) ---
    df_lateral = df_in.copy()
    is_rhb = df_in["IsBatsmanRightHanded"].iloc[0] if not df_in.empty and "IsBatsmanRightHanded" in df_in.columns else True 

    def assign_lateral_zone(row):
        y = row["CreaseY"]
        z = row["CreaseZ"]

        if z > 1.85: return None
        
        if row["IsBatsmanRightHanded"] == True:
            if 0.115 < y <= 1.75: return "LEG"
            elif -0.115 < y <= 0.115: return "STUMPS"
            elif -0.415 < y <= -0.115: return "OUTSIDE OFF"
            elif -1.245 < y <= -0.415: return "WAY OUTSIDE OFF"
        else :
            if -1.75 <= y < -0.115: return "LEG"
            elif -0.115 <= y < 0.115: return "STUMPS"
            elif 0.115 <= y < 0.415: return "OUTSIDE OFF"
            elif 0.415 <= y < 1.245: return "WAY OUTSIDE OFF"
        return None
            
    df_lateral["LateralZone"] = df_lateral.apply(assign_lateral_zone, axis=1)
    
    summary = (
        df_lateral.groupby("LateralZone").agg(
            Runs=("Runs", "sum"), Wickets=("Wicket", lambda x: (x == True).sum()), Balls=("Wicket", "count")
        )
    )
    
    # 1. Define standard zone order (WOO to LEG)
    ordered_zones_base = ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]
    ordered_zones = ordered_zones_base if is_rhb else ordered_zones_base[::-1]
    summary = summary.reindex(ordered_zones).fillna(0)
    summary["Economy"] = summary.apply(lambda row: (row["Runs"] / row["Balls"]) * 6 if row["Balls"] > 0 else np.nan, axis=1)

    # 3. HANDEDNESS AWARE REVERSAL: Reverse order for LHB
    if not is_rhb:
        # Reverses the DataFrame for LHB (LEG, STUMPS, OUTSIDE OFF, WAY OUTSIDE OFF)
        summary = summary.iloc[::-1]
    
    # -----------------------------------------------------------
    # --- 1. SETUP SUBPLOTS ---
    fig = plt.figure(figsize=(7, 6)) 
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.01) 
    ax_bh = fig.add_subplot(gs[0, 0])      
    ax_boxes = fig.add_subplot(gs[1, 0])   
    fig.patch.set_facecolor('white')
    

    # -----------------------------------------------------------
    ## --- 2. CHART 2a: CREASE BEEHIVE (ax_bh) ---
    is_rhb_val = bool(df_in["IsBatsmanRightHanded"].iloc[0]) if not df_in.empty else True
    add_crease_lateral_zone_background(ax_bh, is_rhb=is_rhb, zorder=0)

    # --- Traces ---
    ax_bh.scatter(
        regular_balls["CreaseY"],
        regular_balls["CreaseZ"],
        s=40,
        c="lightgrey",
        edgecolor="white",
        linewidths=0.5,
        alpha=1,
        label="Regular Ball",
        zorder=3,
    )
    ax_bh.scatter(
        boundaries["CreaseY"],
        boundaries["CreaseZ"],
        s=55,
        c="royalblue",
        edgecolor="white",
        linewidths=0.0,
        alpha=0.95,
        label="Boundary",
        zorder=4,
    )
    ax_bh.scatter(
        wickets["CreaseY"],
        wickets["CreaseZ"],
        s=70,
        c="red",
        edgecolor="white",
        linewidths=0.0,
        alpha=0.95,
        label="Wicket",
        zorder=5,
    )

    # --- Reference Lines ---
    ax_bh.axvline(x=-0.115, color="grey", linestyle="--", linewidth=0.5, zorder=2)
    ax_bh.axvline(x=0.115, color="grey", linestyle="--", linewidth=0.5, zorder=2)
    ax_bh.axvline(x=0, color="grey", linestyle="--", linewidth=0.5, zorder=2)
    ax_bh.axhline(y=0.746, color="grey", linestyle="-", linewidth=0.25, zorder=2)

    # --- Annotation ---
    ax_bh.text(-1.5, 0.746, "Stump line", ha="left", va="bottom", fontsize=8, color="grey", transform=ax_bh.transData)

    # --- Formatting ---
    ax_bh.set_xlim([-1.5, 1.5])
    ax_bh.set_ylim([-0.25, 2])
    ax_bh.set_aspect('equal', adjustable='datalim')
    ax_bh.set_xticks([]); ax_bh.set_yticks([]); ax_bh.grid(False)
    for spine in ax_bh.spines.values():
        spine.set_visible(False)
    ax_bh.set_facecolor('white')
    
    # -----------------------------------------------------------
    ## --- 3. CHART 2b: LATERAL PERFORMANCE BOXES (ax_boxes) --
    num_regions = len(ordered_zones)
    box_width = 1 / num_regions
    box_height = 0.4 
    left = 0
    
    # Color Normalization
    eco_values = summary["Economy"].dropna()
    norm = mcolors.Normalize(vmin=3, vmax=9) 
    cmap = plt.get_cmap('Wistia')
    # cmap = colormaps['Wistia']

    for index, row in summary.iterrows():
        eco = row["Economy"]
        wkts = int(row["Wickets"]) # Ensure wkts is defined here from the row
        balls = int(row["Balls"])
        
        # Color based on Economy (handling NaN)
        color = cmap(norm(eco)) if not np.isnan(eco) else (1, 1, 1, 1) # White if no balls bowled
        
        # Draw the Rectangle
        ax_boxes.add_patch(
            patches.Rectangle((left, 0), box_width, box_height, 
                              edgecolor="black", facecolor=color, linewidth=0.4)
        )
        
        # Label 1: Zone Name
        ax_boxes.text(left + box_width / 2, box_height + 0.1, 
                      index, ha='center', va='bottom', fontsize=7, color='black')
        
        # Contrast logic for text
        text_color = 'black'
        if balls > 0:
            r, g, b, a = color
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'
        
        # Label 2: Wickets and Economy (Replaces Average)
        label_wkts_eco = f"{wkts}W - Eco {eco:.1f}" if not np.isnan(eco) else "No Data"
        ax_boxes.text(left + box_width / 2, box_height * 0.5, 
                      label_wkts_eco,
                      ha='center', va='center', fontsize=9, fontweight='bold', color=text_color)
        
        left += box_width

    # Formatting
    ax_boxes.set_xlim(0, 1)
    ax_boxes.set_ylim(0, box_height + 0.3) 
    ax_boxes.axis('off')
    for spine in ax_boxes.spines.values():
        spine.set_visible(False)
    ax_boxes.set_facecolor('white')

    # -----------------------------------------------------------
    ## --- 4. DRAW SINGLE COMPACT BORDER AROUND THE ENTIRE FIGURE ---
    
    plt.tight_layout(pad=0.2)
    
    PADDING = 0.008

    bh_bbox = ax_bh.get_position()
    box_bbox = ax_boxes.get_position()
    
    x0_orig = min(bh_bbox.x0, box_bbox.x0)
    y0_orig = box_bbox.y0
    x1_orig = max(bh_bbox.x1, box_bbox.x1)
    y1_orig = bh_bbox.y1
    
    x0_pad = x0_orig - PADDING
    y0_pad = y0_orig - PADDING
    
    width_pad = (x1_orig - x0_orig) + (2 * PADDING)
    height_pad = (y1_orig - y0_orig) + (2 * PADDING)

    border_rect = patches.Rectangle(
        (x0_pad, y0_pad), 
        width_pad, 
        height_pad, 
        facecolor='none', 
        edgecolor='black', 
        linewidth=0.5, 
        transform=fig.transFigure, 
        clip_on=False
    )

    fig.patches.append(border_rect)

    return fig

# Chart 5 Bowler Release Map
def create_pacer_release_analysis(df_in, handedness_label): 
    FIG_SIZE = (3, 3) # Increased height for both charts

    if df_in.empty or "ReleaseY" not in df_in.columns or "ReleaseZ" not in df_in.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, f"No data for Release Analysis vs. {handedness_label}", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # --- 1. Calculate Lateral Release Performance (LEFT vs RIGHT) ---
    df_temp = df_in.copy()
    
    # Categorize based on ReleaseY sign
    df_temp["ReleaseCategory"] = np.where(
        df_temp["ReleaseY"] < 0, "LEFT (<0)", 
        np.where(df_temp["ReleaseY"] > 0, "RIGHT (>0)", "CENTER (=0)")
    )
    
    df_temp = df_temp[df_temp["ReleaseCategory"] != "CENTER (=0)"]
    
    # Calculation functions
    def calculate_ba(row):
        # Use np.nan as a flag for "N/A"
        return row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else np.nan

    def calculate_sr(row):
        # Strike Rate = Balls per Wicket (normalized by 6 for Cricket SR)
        return (row["Balls"] / row["Wickets"]) if row["Wickets"] > 0 else np.nan
        
    summary = df_temp.groupby("ReleaseCategory").agg(
        Wickets=("Wicket", lambda x: (x == True).sum()),
        Runs=("Runs", "sum"),
        Balls=("Wicket", "count")
    )

    # Ensure both categories are present for consistent plotting
    summary = summary.reindex(["LEFT (<0)", "RIGHT (>0)"]).fillna(0)
    
    summary["BA"] = summary.apply(calculate_ba, axis=1)
    summary["SR"] = summary.apply(calculate_sr, axis=1)

    # Formatting helper
    def format_metric(value, is_wickets=False):
        if is_wickets:
            return f"{int(value)}"
        if np.isnan(value) or value == np.inf:
            return "N/A"
        return f"{value:.1f}"

    left = summary.loc["LEFT (<0)"]
    right = summary.loc["RIGHT (>0)"]

    # --- 2. Setup Figure and GridSpec ---
    fig = plt.figure(figsize=FIG_SIZE, facecolor='white')
    gs = GridSpec(2, 1, figure=fig, height_ratios=[4, 1], hspace=0.1)
    
    ax_map = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[1, 0])

    # --- 3. Plot Release Zone Map (ax_map) ---
    
    release_wickets = df_in[df_in["Wicket"] == True]
    release_non_wickets = df_in[df_in["Wicket"] == False]
    
    # Non-Wickets (light grey)
    ax_map.scatter(
        release_non_wickets["ReleaseY"], release_non_wickets["ReleaseZ"], 
        s=10, color='#D3D3D3', alpha=1.0, linewidths=0.5, label="No Wicket"
    )

    # Wickets (red)
    ax_map.scatter(
        release_wickets["ReleaseY"], release_wickets["ReleaseZ"], 
        s=10, color='red', alpha=1.0, linewidths=1.0, label="Wicket", zorder=5
    )
    
    # Add Stump Lines
    stump_lines = [-0.115, 0, 0.115]
    for y_val in stump_lines:
        ax_map.axvline(x=y_val, color="#777777", linestyle="-", linewidth=0.5)
    
    # Formatting Map
    ax_map.set_xlim(-1.5, 1.5)
    ax_map.set_ylim(0.5, 2.5)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_facecolor('white')
    ax_map.grid(True)

    
    # Hide all map spines
    for spine in ax_map.spines.values():
        spine.set_visible(False)
        
    # --- 4. Draw Lateral Metrics Table (ax_metrics) ---
    
    # Hide all metrics spines/ticks/labels
    ax_metrics.axis('off')
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(-0.5, 1)

    # Titles
    # Metric Labels (Left Alignment for labels)
    ax_metrics.text(0.05, 1, "W:", ha='right', va='center', fontsize=5, fontweight='bold')
    ax_metrics.text(0.05, 0.5, "Avg:", ha='right', va='center', fontsize=5, fontweight='bold')
    ax_metrics.text(0.05, 0, "SR:", ha='right', va='center', fontsize=5, fontweight='bold')

    # LEFT Values
    ax_metrics.text(0.2, 1, format_metric(left["Wickets"], is_wickets=True), ha='center', va='center', fontsize=8, color='red', fontweight='bold')
    ax_metrics.text(0.2, 0.5, format_metric(left["BA"]), ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    ax_metrics.text(0.2, 0, format_metric(left["SR"]), ha='center', va='center', fontsize=8, color='black', fontweight='bold')

    # RIGHT Values
    ax_metrics.text(0.9, 1, format_metric(right["Wickets"], is_wickets=True), ha='center', va='center', fontsize=8, color='red', fontweight='bold')
    ax_metrics.text(0.9, 0.5, format_metric(right["BA"]), ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    ax_metrics.text(0.9, 0, format_metric(right["SR"]), ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    
    # --- 5. Add Sharp Border to Figure ---
    plt.tight_layout(pad=0.1)
    
    # Create and add a custom Rectangle patch for sharp border
    ax_bbox = ax_map.get_position()
    # Calculate padding relative to figure size
    padding_x = 0.001 * FIG_SIZE[0] / fig.get_size_inches()[0] 
    padding_y = 0.001 * FIG_SIZE[1] / fig.get_size_inches()[1] 
    
    border_rect = patches.Rectangle(
        (0.05, 0.13), 
        0.9, 
        0.8, 
        facecolor='none',
        edgecolor='black',
        linewidth=0.5,
        transform=fig.transFigure,
        clip_on=False,
        joinstyle='miter' # Ensures sharp corners
    )
    fig.add_artist(border_rect)

    return fig

# --- CHART 4: RELEASE SPEED DISTRIBUTION ---
def create_pacer_release_speed_distribution(df_in, handedness_label):
    FIG_SIZE = (5, 4)

    if df_in.empty or "ReleaseSpeed" not in df_in.columns or df_in["ReleaseSpeed"].empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, f"No Data or Missing 'ReleaseSpeed' for {handedness_label}", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # 1. Prepare Data and Determine Histogram Parameters
    speeds = df_in["ReleaseSpeed"].dropna().values
    speeds = speeds[(speeds >= 50) & (speeds <= 155)]

    total_balls = len(speeds)
    
    if total_balls == 0:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Deliveries Found", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Calculate the range to ensure fixed bin width of 5 km/h
    min_speed = np.floor(speeds.min() / 5) * 5
    max_speed = np.ceil(speeds.max() / 5) * 5
    
    # Generate bins with a fixed width of 5 km/h
    bin_width = 5
    bins = np.arange(min_speed, max_speed + bin_width, bin_width)
    
    # Calculate histogram counts and edges
    counts, bin_edges = np.histogram(speeds, bins=bins)
    
    # 2. Process Data for Plotting & Filtering
    raw_percentages = (counts / total_balls) * 100
    # Filter out bins with less than 5 balls
    MIN_BALLS = 5
    valid_counts = []
    valid_bin_labels = []
    
    for i in range(len(counts)):
        if raw_percentages[i] >= 1.0:
            lower = int(bin_edges[i])
            upper = int(bin_edges[i+1])
            label = f"{lower}-{upper}"
            
            valid_counts.append(counts[i])
            valid_bin_labels.append(label)

    if not valid_counts:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, f"No Bins Meet the {MIN_BALLS}-Ball Minimum Filter", ha='center', va='center', fontsize=10)
        ax.axis('off')
        return fig
        
    # Calculate percentages for valid bins only
    valid_percentages = (np.array(valid_counts) / total_balls) * 100
    
    # Reverse order for horizontal bar chart (fastest speeds typically at the top)
    plot_percentages = valid_percentages[::-1]
    plot_labels = valid_bin_labels[::-1]
    plot_counts = valid_counts[::-1]
    
    # 3. Chart Generation (Horizontal Bar / Histogram)
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    
    y_pos = np.arange(len(plot_labels))
    
    ax.barh(
        y_pos,
        plot_percentages,
        color='#ff5000', # Single, uniform color
        height=0.6
    )

    
            # Add percentage labels
    for i, pct in enumerate(plot_percentages):
        count = plot_counts[i]
        # Display percentage (e.g., 25%)
        label_text = f'{pct:.0f}%'
    
        # Placement logic: Always outside the bar
        # 1. x_pos: Start the text slightly after the end of the bar (pct)
        x_pos = pct + 0.5 
        # 2. ha: Align the start of the text to x_pos
        ha = 'left' 
        text_color = 'black'
    
        ax.text(
        x_pos, 
        i, 
        label_text, 
        ha=ha, va='center', fontsize=12, color=text_color, fontweight='bold'
    )
        


    # 4. Formatting
    
    # Set Y-axis labels
    ax.set_yticks(y_pos, labels=plot_labels, fontsize=10)
    
    # Set X-axis limit slightly higher than the max percentage for clean labels
    max_pct = np.max(plot_percentages) if len(plot_percentages) > 0 else 0
    ax.set_xlim(0, max(max_pct * 1.1, 10)) 
    
    # Hide axis ticks/labels
    ax.set_xticklabels([])
    ax.set_xticks([])
    
    # Remove all spines 
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # --- ADDING SHARP BORDER ---
    # We create a custom Rectangle patch with 'miter' joinstyle and add it to the figure.
    # Get the bounding box of the axes in figure coordinates
    ax_bbox = ax.get_position()
    
    # Calculate padding based on figure dimensions to ensure a consistent border
    # Use 0.01 for x and y to give a small padding
    padding_x = 0.2* FIG_SIZE[0] / fig.get_size_inches()[0] # Scale padding based on total figure width
    padding_y = 0.01 * FIG_SIZE[1] / fig.get_size_inches()[1] # Scale padding based on total figure height

    border_rect = patches.Rectangle(
        (ax_bbox.x0 - padding_x, ax_bbox.y0 - padding_y), # Start (x,y)
        ax_bbox.width + 2 * padding_x,                    # Width
        ax_bbox.height + 2 * padding_y,                   # Height
        facecolor='none',
        edgecolor='black',
        linewidth=0.5,
        transform=fig.transFigure, # Use figure coordinates
        clip_on=False,             # Ensure it's not clipped
        joinstyle='miter'          # THIS ENSURES SHARP CORNERS
    )
    fig.add_artist(border_rect) # Add the custom rectangle to the figure

    return fig

# Chart 11: Speed Effectiveness
def create_pacer_speed_effectiveness_3col(df_in, handedness_label):
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        ax.axis('off')
        return fig

    # 1. Define Speed Groups
    cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
    unit = st.session_state.get("speed_unit", "kph")

    def assign_speed_group(speed):
        return seam_speed_group(speed, cfg, unit)

    df_temp = df_in.copy()
    df_temp["ReleaseSpeed"] = pd.to_numeric(df_temp["ReleaseSpeed"], errors='coerce')
    df_temp = df_temp.dropna(subset=["ReleaseSpeed"])
    df_temp["SpeedGroup"] = df_temp["ReleaseSpeed"].apply(assign_speed_group)

    # 2. Aggregate Data (Updated to include Wickets)
    ordered_groups = pacer_effectiveness_seam_order(cfg, unit)
    summary = df_temp.groupby("SpeedGroup").agg(
        Runs=("Runs", "sum"), 
        Balls=("Runs", "count"),
        Wickets=("Wicket", lambda x: (x == True).sum()) # Added for SR and Avg
    ).reindex(ordered_groups).fillna(0)
    
    # --- UPDATED: White Ball Metrics (Eco, SR, Avg) ---
    summary["Eco"] = (summary["Runs"] / summary["Balls"] * 6).fillna(0)
    
    # Bowling Strike Rate: Balls per Wicket
    summary["SR"] = summary.apply(
        lambda row: row["Balls"] / row["Wickets"] if row["Wickets"] > 0 else row["Balls"], axis=1
    )
    
    # Bowling Average: Runs per Wicket
    summary["Avg"] = summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else row["Runs"], axis=1
    )

    # 3. Plotting (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 2.89), sharey=True)
    plt.subplots_adjust(wspace=0.3)

    y = np.arange(len(ordered_groups))
    height = 0.5
    color_pacer = "#ff5000"

    # --- Column 1: Economy ---
    ax1.barh(y, summary["Eco"], color=color_pacer, edgecolor="white", height=height)
    ax1.set_title("Economy", fontsize=14, fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(ordered_groups, fontsize=13, fontweight="bold")
    for i, v in enumerate(summary["Eco"]):
        ax1.text(v + 0.2, i, f"{v:.1f}", va="center", fontweight="bold", fontsize=12)

    # --- Column 2: Bowling Strike Rate (SR) / Balls ---
    ax2.barh(y, summary["SR"], color=color_pacer, edgecolor="white", height=height)
    ax2.set_title("SR", fontsize=14, fontweight="bold")
    for i, (idx, row) in enumerate(summary.iterrows()):
        val = row["SR"]
        label = f"{int(row['Balls'])} B" if row["Wickets"] == 0 else f"{val:.1f}"
        ax2.text(val + 1, i, label, va="center", fontweight="bold", fontsize=12)

    # --- Column 3: Bowling Average (Avg) / Runs ---
    ax3.barh(y, summary["Avg"], color=color_pacer, edgecolor="white", height=height)
    ax3.set_title("Avg", fontsize=14, fontweight="bold")
    for i, (idx, row) in enumerate(summary.iterrows()):
        val = row["Avg"]
        label = f"{int(row['Runs'])} R" if row["Wickets"] == 0 else f"{val:.1f}"
        ax3.text(val + 1, i, label, va="center", fontweight="bold", fontsize=12)

    # Formatting
    for ax in [ax1, ax2, ax3]:
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.invert_yaxis()

    plt.tight_layout(pad=0.35)
    fig.canvas.draw()
    p0 = ax1.get_position()
    p2 = ax3.get_position()
    pad_x = 0.012
    pad_y = 0.02
    border_rect = patches.Rectangle(
        (p0.x0 - pad_x, p0.y0 - pad_y),
        (p2.x1 - p0.x0) + 2 * pad_x,
        (p0.y1 - p0.y0) + 2 * pad_y,
        facecolor="none",
        edgecolor="black",
        linewidth=0.6,
        transform=fig.transFigure,
        clip_on=False,
    )
    

    return fig
    
### Chart: SCORING AREAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. DEFINE THE UTILITY FUNCTIONS FIRST
def calculate_scoring_wagon(row):
    """Calculates the scoring area based on LandingX/Y coordinates."""
    LX = row.get("LandingX")
    LY = row.get("LandingY")
    RH = row.get("IsBatsmanRightHanded")
    if RH is None or LX is None or LY is None or row.get("Runs", 0) == 0: 
        return None
    
    def atan_safe(numerator, denominator): 
        return np.arctan(numerator / denominator) if denominator != 0 else np.nan 
    
    if RH == True: 
        if LX <= 0 and LY > 0: return "FINE LEG"
        elif LX <= 0 and LY <= 0: return "THIRD MAN"
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "COVER"
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG OFF" 
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "SQUARE LEG"
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG ON"
    elif RH == False: 
        if LX <= 0 and LY > 0: return "THIRD MAN"
        elif LX <= 0 and LY <= 0: return "FINE LEG"
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "SQUARE LEG"
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG ON"
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "COVER"
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG OFF"
    return None

def calculate_scoring_angle(area):
    if area in ["FINE LEG", "THIRD MAN"]: return 90
    elif area in ["COVER", "SQUARE LEG", "LONG OFF", "LONG ON"]: return 45
    return 0

# 2. NOW DEFINE THE MAIN CHART FUNCTION
def create_pacer_wagon_wheel(df_in):
    # Standard size for single wagon wheel
    FIG_SIZE = (5, 3)

    if df_in.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # 1. Calculate Scoring Areas
    df_in["ScoringWagon"] = df_in.apply(calculate_scoring_wagon, axis=1)
    df_in["FixedAngle"] = df_in["ScoringWagon"].apply(calculate_scoring_angle)
    
    # 2. Aggregate Data (Total Runs)
    summary = df_in.groupby("ScoringWagon").agg(
        TotalRuns=("Runs", "sum"), 
        FixedAngle=("FixedAngle", 'first')
    ).reset_index().dropna(subset=["ScoringWagon"])

    # Handedness Check
    is_rhb = df_in["IsBatsmanRightHanded"].mode().iloc[0] if not df_in["IsBatsmanRightHanded"].empty else True
    all_areas = ["FINE LEG", "SQUARE LEG", "LONG ON", "LONG OFF", "COVER", "THIRD MAN"] if is_rhb else \
                ["THIRD MAN", "COVER", "LONG OFF", "LONG ON", "SQUARE LEG", "FINE LEG"]
    
    template = pd.DataFrame({"ScoringWagon": all_areas, "FixedAngle": [calculate_scoring_angle(a) for a in all_areas]})
    summary = template.merge(summary.drop(columns=["FixedAngle"]), on="ScoringWagon", how="left").fillna(0)
    
    # --- CHANGE: Calculate Run Percentage instead of SR ---
    total_runs_overall = summary["TotalRuns"].sum()
    if total_runs_overall > 0:
        summary["RunPct"] = (summary["TotalRuns"] / total_runs_overall * 100)
    else:
        summary["RunPct"] = 0
        
    summary['Rank'] = summary['RunPct'].rank(method='dense', ascending=False)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor('white')
    
    angles = summary["FixedAngle"].tolist()
    pct_values = summary["RunPct"].tolist()
    # Highlight the area with the highest percentage of runs
    colors = ['#ff5000' if r == 1 and v > 0 else 'white' for r, v in zip(summary['Rank'], pct_values)]

    # Pie Chart
    wedges, _ = ax.pie(angles, colors=colors, wedgeprops={"width": 1, "edgecolor": "black", "linewidth": 0.8}, 
                       startangle=90, counterclock=False)

    # Labels
    for i, wedge in enumerate(wedges):
        val = pct_values[i]
        if val > 0:
            angle = (wedge.theta2 + wedge.theta1) / 2.
            x = 0.65 * np.cos(np.deg2rad(angle))
            y = 0.65 * np.sin(np.deg2rad(angle))
            
            # Contrast for #1 Rank
            t_color = 'white' if colors[i] == '#ff5000' else 'black'
            # Display as percentage (e.g., 25%)
            ax.text(x, y, f"{val:.0f}%", ha='center', va='center', fontsize=12, fontweight='bold', color=t_color)
            
    ax.axis('equal')
    return fig


# PAGE SETUP LAYOUT

st.set_page_config(
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

# 1. CRITICAL: GET DATA AND CHECK FOR AVAILABILITY
if 'data_df' not in st.session_state:
    st.error("Please go back to the **Home** page and upload the data first to begin the analysis.")
    st.stop()
    
df_raw = st.session_state['data_df']
_cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))

col_title_space, col_format_banner, col_legend = st.columns([2.5, 2.5, 5])

with col_title_space:
    st.title("PACERS")

with col_format_banner:
    st.markdown(
        f'<div style="margin-top: 38px; text-align: left; width: 100%;"><span style="{FORMAT_BANNER_STYLE}">{format_banner_caps(_cfg)}</span></div>',
        unsafe_allow_html=True,
    )

with col_legend:
    legend_markdown = """
    <p style='font-size: 20px; margin-top: 40px; text-align: right;'>
        <span style='color: red; font-size: 25px;'>&#9679;</span> Wickets &nbsp;&nbsp;&nbsp;
        <span style='color: royalblue; font-size: 25px;'>&#9679;</span> Boundaries &nbsp;&nbsp;&nbsp;
        <span style='color: lightgrey; font-size: 25px;'>&#9679;</span> Others
    </p>
    """
    st.markdown(legend_markdown, unsafe_allow_html=True)
    
# Ensure columns exist before attempting to convert them
if "BatsmanName" in df_raw.columns:
    df_raw["BatsmanName"] = df_raw["BatsmanName"].astype(str).str.upper()
if "BowlerName" in df_raw.columns:
    # Assuming 'BowlerName' is used elsewhere, convert it here too for consistency
    df_raw["BowlerName"] = df_raw["BowlerName"].astype(str).str.upper()


# 2. BASE FILTER: ONLY SEAM DELIVERIES
df_seam_base = df_raw[df_raw["DeliveryType"] == "Seam"]


# --- Prepare Initial Filter Options ---
def handle_all_selection(key):
    """Ensures 'All' behaves cleanly with specific options."""
    selected = st.session_state[key]
    if len(selected) == 0:
        st.session_state[key] = ["All"]
    elif "All" in selected and len(selected) > 1:
        if selected[-1] == "All":
            st.session_state[key] = ["All"]
        else:
            st.session_state[key] = [x for x in selected if x != "All"]

def _multiselect_is_all(sel):
    return sel is None or len(sel) == 0 or "All" in sel


all_teams = ["All"] + sorted(df_seam_base["BowlingTeam"].dropna().unique().tolist())

row1 = st.columns(4)
with row1[0]:
    bowl_team_sel = st.multiselect(
        "Bowling Team", 
        all_teams, 
        default=["All"],
        key="team_filter",
        on_change=handle_all_selection,
        args=("team_filter",)
    )

# 2. Dynamically filter batsmen options based on chosen team
if _multiselect_is_all(bowl_team_sel):
    df_bat_opts = df_seam_base
else:
    teams_only = [t for t in bowl_team_sel if t != "All"]
    df_bat_opts = df_seam_base[df_seam_base["BowlingTeam"].isin(teams_only)]

# This defines the variable that was throwing the NameError
bowler_options = ["All"] + sorted(df_bat_opts["BowlerName"].dropna().unique().tolist())

with row1[1]:
    bowler_sel = st.multiselect(
        "Bowler Name", 
        bowler_options, 
        default=["All"],
        key="batter_filter",
        on_change=handle_all_selection,
        args=("batter_filter",)
    )

# 3. Safely locate remaining data columns
year_col = next((c for c in df_raw.columns if c.strip().lower() == "year"), None)
ground_col = next((c for c in df_raw.columns if c.strip().lower() == "ground"), None)
tour_col = next((c for c in df_raw.columns if c.strip().lower() == "tour"), None)
match_col = next((c for c in df_raw.columns if c.strip().lower() == "match"), None)

with row1[2]:
    if year_col:
        year_vals = sorted(df_raw[year_col].dropna().unique().astype(int).astype(str).tolist())
        selected_years = st.multiselect(
            "Year", 
            ["All"] + year_vals, 
            default=["All"],
            key="year_filter",
            on_change=handle_all_selection,
            args=("year_filter",)
        )
    else:
        selected_years = ["All"]
        st.info("Year N/A")

with row1[3]:
    if ground_col:
        venue_vals = sorted(df_raw[ground_col].dropna().unique().tolist())
        selected_venues = st.multiselect(
            "Venue", 
            ["All"] + venue_vals, 
            default=["All"],
            key="venue_filter",
            on_change=handle_all_selection,
            args=("venue_filter",)
        )
    else:
        selected_venues = ["All"]
        st.info("Venue N/A")

row2 = st.columns(2)

with row2[0]:
    if tour_col:
        tour_vals = sorted(df_raw[tour_col].dropna().astype(str).unique().tolist())
        selected_tours = st.multiselect(
            "Tour", 
            ["All"] + tour_vals, 
            default=["All"],
            key="tour_filter",
            on_change=handle_all_selection,
            args=("tour_filter",)
        )
    else:
        selected_tours = ["All"]
        st.caption("Tour N/A")

with row2[1]:
    if match_col:
        match_vals = sorted(df_raw[match_col].dropna().astype(str).unique().tolist())
        selected_matches = st.multiselect(
            "Match", 
            ["All"] + match_vals, 
            default=["All"],
            key="match_filter",
            on_change=handle_all_selection,
            args=("match_filter",)
        )
    else:
        selected_matches = ["All"]
        st.caption("Match N/A")


# 4. Apply Filters to the Base seam Data
df_filtered = df_seam_base.copy()

if not _multiselect_is_all(bowl_team_sel):
        teams_only = [t for t in bowl_team_sel if t != "All"]
        df_filtered = df_filtered[df_filtered["BowlingTeam"].isin(teams_only)]

if not _multiselect_is_all(bowler_sel):
        bowls = [b for b in bowler_sel if b != "All"]
        df_filtered = df_filtered[df_filtered["BowlerName"].isin(bowls)]

yc = next((c for c in df_filtered.columns if c.strip().lower() == "year"), None)
if yc and not _multiselect_is_all(selected_years):
        years = [int(y) for y in selected_years if y != "All"]
        df_filtered = df_filtered[df_filtered[yc].astype(int).isin(years)]

gc = next((c for c in df_filtered.columns if c.strip().lower() == "ground"), None)
if gc and not _multiselect_is_all(selected_venues):
        venues = [v for v in selected_venues if v != "All"]
        df_filtered = df_filtered[df_filtered[gc].isin(venues)]

tc = next((c for c in df_filtered.columns if c.strip().lower() == "tour"), None)
if tc and not _multiselect_is_all(selected_tours):
        tours = [t for t in selected_tours if t != "All"]
        df_filtered = df_filtered[df_filtered[tc].astype(str).isin(tours)]

mc = next((c for c in df_filtered.columns if c.strip().lower() == "match"), None)
if mc and not _multiselect_is_all(selected_matches):
        matches = [m for m in selected_matches if m != "All"]
        df_filtered = df_filtered[df_filtered[mc].astype(str).isin(matches)]


if _multiselect_is_all(bowler_sel):
    heading_text = "ALL"
else:
    names = [b for b in bowler_sel if b != "All"]
    heading_text = " + ".join(names) if names else "ALL"
# Use st.markdown to inject HTML, setting the text color directly
st.markdown(
    f"<h3 style='color: #ff5000;'><b>{heading_text}</b></h3>",
    unsafe_allow_html=True
)


# =========================================================
# 5. SPLIT AND DISPLAY CHARTS (RHB vs LHB) 🏏
# =========================================================

# Check for the required column to split the data
if "IsBatsmanRightHanded" not in df_filtered.columns:
    st.error("Cannot split data by handedness: 'IsBatsmanRightHanded' column is missing.")
    st.stop()

# --- Data Split ---
# True is Right-Handed (RHB), False is Left-Handed (LHB)
df_rhb = df_filtered[df_filtered["IsBatsmanRightHanded"] == True]
df_lhb = df_filtered[df_filtered["IsBatsmanRightHanded"] == False]


# --- Display Layout ---
col_rhb, col_lhb = st.columns(2)

# === LEFT COLUMN: AGAINST RIGHT-HANDED BATSMEN (RHB) ===
with col_rhb:
    st.markdown("###  v RIGHT-HAND BAT")

    # Chart 3: PITCHMAP
    pitch_map_col, run_pct_col = st.columns([1, 1]) 
    with pitch_map_col:
        st.markdown("###### PITCHMAP v RHB")
        st.pyplot(create_pacer_pitch_map(df_rhb), use_container_width=True)    
    with run_pct_col:
        st.markdown("##### ")
        st.pyplot(create_pacer_pitch_length_bars(df_rhb), use_container_width=True)
        
    # Chart 1a: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE v RHB")
    st.pyplot(create_pacer_crease_beehive(df_rhb, "RHB"), use_container_width=True)

    # Chart 1b: Lateral Performance Boxes (Bowling Avg)
    # st.pyplot(create_pacer_lateral_performance_boxes(df_rhb, "RHB"), use_container_width=True)
    

    # Chart 4/5: RELEASE
    st.markdown("###### RELEASE SPEED v RHB")
    st.pyplot(create_pacer_release_speed_distribution(df_rhb, "RHB"), use_container_width=True)
        
    st.markdown("###### RELEASE v RHB")
    st.pyplot(create_pacer_release_analysis(df_rhb, "RHB"), use_container_width=True)
    
    # Chart 16: Speed Distribution - Death Overs
    st.markdown("###### SPEED DISTRIBUTION METRICS")
    st.pyplot(create_pacer_speed_effectiveness_3col(df_rhb, "RHB"), use_container_width=True)
    
    # Chart 17: Scoring Areas
    st.markdown("###### SCORING AREAS of RHB ")
    st.pyplot(create_pacer_wagon_wheel(df_rhb), use_container_width=True)

# === RIGHT COLUMN: AGAINST LEFT-HANDED BATSMEN (LHB) ===
with col_lhb:
    st.markdown("###  v LEFT-HAND BAT")

    # Chart 3: PITCHMAP
    pitch_map_col, run_pct_col = st.columns([1, 1]) 
    with pitch_map_col:
        st.markdown("###### PITCHMAP v LHB")
        st.pyplot(create_pacer_pitch_map(df_lhb), use_container_width=True)    
    with run_pct_col:
        st.markdown("##### ")
        st.pyplot(create_pacer_pitch_length_bars(df_lhb), use_container_width=True)
        
    # Chart 1a: Crease Beehive (using the new local function)
    st.markdown("###### CREASE BEEHIVE v LHB")
    st.pyplot(create_pacer_crease_beehive(df_lhb, "LHB"), use_container_width=True)

    # Chart 1b: Lateral Performance Boxes (Bowling Avg)
    # st.pyplot(create_pacer_lateral_performance_boxes(df_lhb, "LHB"), use_container_width=True)

    # Chart 4/5: RELEASE
    st.markdown("###### RELEASE SPEED v LHB")
    st.pyplot(create_pacer_release_speed_distribution(df_lhb, "LHB"), use_container_width=True)
    
    st.markdown("###### RELEASE v LHB")
    st.pyplot(create_pacer_release_analysis(df_lhb, "LHB"), use_container_width=True)
        
    # Chart 16: Speed Distribution - Death Overs
    st.markdown("###### SPEED DISTRIBUTION METRICS")
    st.pyplot(create_pacer_speed_effectiveness_3col(df_lhb, "LHB"), use_container_width=True)
    
    # Chart 17: Scoring Areas
    st.markdown("###### SCORING AREAS of LHB ")
    st.pyplot(create_pacer_wagon_wheel(df_lhb), use_container_width=True)
