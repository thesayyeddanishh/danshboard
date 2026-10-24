import matplotlib
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
from matplotlib.backends.backend_pdf import PdfPages
import io, zipfile

from cricket_config import (
    resolve_format,
    get_pitch_bins as pitch_bins_for_format,
    ordered_seam_keys,
    ordered_spin_keys,
    seam_speed_group,
    seam_speed_ordered_groups,
    spin_speed_group,
    spin_speed_ordered_groups,
    format_banner_caps,
    FORMAT_BANNER_STYLE,
    pitch_map_figsize,
    pitch_bin_percentages,
    add_crease_lateral_zone_background,
)

# --- 1. GLOBAL UTILITY FUNCTIONS ---

# Required columns check
REQUIRED_COLS = [
    "BatsmanName", "DeliveryType", "Wicket", "StumpsY", "StumpsZ", 
    "BattingTeam", "CreaseY", "CreaseZ", "Runs", "IsBatsmanRightHanded", 
    "LandingX", "LandingY", "BounceX", "BounceY", "InterceptionX", 
    "InterceptionZ", "InterceptionY", "Over"
]
# Function to encode Matplotlib figure to image for Streamlit
def fig_to_image(fig):
    return fig

# Chart 2: CREASE BEEHIVE
def create_crease_beehive(df_in, delivery_type):
    if df_in.empty:
        fig, ax = plt.subplots(figsize=(7, 6)); 
        ax.text(0.5, 0.5, "No data for Analysis", ha='center', va='center', fontsize=12); 
        ax.axis('off'); 
        return fig
    
    _cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))

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
    
    # 2. Define standard zone order (RHB: Left to Right == WOO to LEG)
    ordered_zones = ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"]
    summary = summary.reindex(ordered_zones).fillna(0)
    summary["Avg Runs/Wicket"] = summary.apply(lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else np.nan, axis=1)
    summary["SR"] = summary.apply(lambda row: (row["Runs"] / row["Balls"]) * 100 if row["Balls"] > 0 else np.nan, axis=1)

    # 3. HANDEDNESS AWARE REVERSAL: Reverse order for LHB
    if not is_rhb:
        # Reverses the DataFrame for LHB (LEG, STUMPS, OUTSIDE OFF, WAY OUTSIDE OFF)
        summary = summary.iloc[::-1]

    # -----------------------------------------------------------
    # --- 1. SETUP SUBPLOTS (Increased Figure Width) ---

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.01)
    ax_bh = fig.add_subplot(gs[0, 0])      # Top subplot (Beehive)
    ax_boxes = fig.add_subplot(gs[1, 0])   # Bottom subplot (Lateral Boxes)
    fig.patch.set_facecolor('white')

    # -----------------------------------------------------------
    ## --- 2. CHART 2a: CREASE BEEHIVE (ax_bh) ---
    is_rhb_val = bool(df_seam["IsBatsmanRightHanded"].iloc[0]) if not df_seam.empty else True
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
    ax_bh.text(-1.5, 0.78, "Stump line", ha="left", va="bottom", fontsize=8, color="grey", transform=ax_bh.transData)

    # --- Formatting ---
    ax_bh.set_xlim([-1.5, 1.5])
    ax_bh.set_ylim([-0.25, 2])
    ax_bh.set_aspect('equal', adjustable='datalim')
    ax_bh.set_xticks([]); ax_bh.set_yticks([]); ax_bh.grid(False)
    for spine in ax_bh.spines.values():
        spine.set_visible(False)
    ax_bh.set_facecolor('white')

## --- CHART 2b: LATERAL PERFORMANCE BOXES (ax_boxes) ---
    
    # 1. PRE-CALCULATION: Ensure Average exists before the loop
    summary = summary.copy()
    # Calculate Average: Runs / Wickets. Fill NaNs with 0.
    summary["Avg"] = (summary["Runs"] / summary["Wickets"]).fillna(0)
    
    num_regions = len(ordered_zones)
    box_width = 1 / num_regions
    box_height = 1  
    left = 0
    
    # 2. COLOR NORMALIZATION (Test: heatmap by batting average; white-ball: by SR)
    cmap = plt.get_cmap("Wistia")
    if _cfg.is_test:
        avg_values = summary["Avg"].replace([np.inf, -np.inf], np.nan)
        avg_max = avg_values.max() if avg_values.max() > 0 else 100
        norm = mcolors.Normalize(vmin=0, vmax=avg_max)
    else:
        sr_values = summary["SR"].replace([np.inf, -np.inf], np.nan)
        sr_max = sr_values.max() if sr_values.max() > 0 else 200
        norm = mcolors.Normalize(vmin=0, vmax=sr_max)

    # 3. DRAWING THE BOXES
    for index, row in summary.iterrows():
        runs = int(row["Runs"])
        outs = int(row["Wickets"])
        avg = row["Avg"]
        sr = row["SR"]

        if _cfg.is_test:
            if np.isnan(avg) or avg == np.inf:
                color = "white"
                text_color = "black"
                avg_display = "0.0"
            else:
                color = cmap(norm(avg))
                avg_display = f"{avg:.1f}" if outs > 0 else "-"
                r, g, b, a = color
                luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "white" if luminosity < 0.5 else "black"
            label_bottom = f"{avg_display} Avg"
        else:
            if np.isnan(sr) or sr == np.inf:
                color = "white"
                text_color = "black"
                sr_display = "0"
                avg_display = "0.0"
            else:
                color = cmap(norm(sr))
                sr_display = f"{sr:.0f}"
                avg_display = f"{avg:.1f}" if outs > 0 else "-"
                r, g, b, a = color
                luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "white" if luminosity < 0.5 else "black"
            label_bottom = f"{avg_display} Avg, {sr_display} SR"

        ax_boxes.add_patch(
            patches.Rectangle(
                (left, 0),
                box_width,
                box_height,
                edgecolor="black",
                facecolor=color,
                linewidth=1,
            )
        )

        ax_boxes.text(
            left + box_width / 2,
            box_height + 0.05,
            index,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="black",
        )

        label_top = f"{runs} R, {outs} W"

        ax_boxes.text(
            left + box_width / 2,
            box_height * 0.65,
            label_top,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=text_color,
        )

        ax_boxes.text(
            left + box_width / 2,
            box_height * 0.35,
            label_bottom,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=text_color,
        )

        left += box_width

    # 4. FINAL FORMATTING
    ax_boxes.set_xlim(0, 1)
    ax_boxes.set_ylim(-0.1, box_height + 0.3)
    ax_boxes.axis('off')

    plt.tight_layout(pad=0.2)

    # BORDER LOGIC
    PADDING = 0.008
    bh_bbox = ax_bh.get_position()
    box_bbox = ax_boxes.get_position()

    x0_orig = min(bh_bbox.x0, box_bbox.x0)
    y0_orig = box_bbox.y0
    x1_orig = max(bh_bbox.x1, box_bbox.x1)
    y1_orig = bh_bbox.y1

    border_rect = patches.Rectangle(
        (x0_orig - PADDING, y0_orig - PADDING), 
        (x1_orig - x0_orig) + (2 * PADDING), 
        (y1_orig - y0_orig) + (2 * PADDING),  
        facecolor='none', edgecolor='black', linewidth=0.5, 
        transform=fig.transFigure, clip_on=False
    )
        
    fig.patches.append(border_rect)

    return fig


# --- CHART 3: PITCHMAP ---
# --- Helper function for Pitch Bins (Centralized) ---
def get_pitch_bins(delivery_type):
    _cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
    return pitch_bins_for_format(delivery_type, _cfg)

# --- CHART 3: PITCH MAP (BOUNCE LOCATION) ---
def create_pitch_map(df_in, delivery_type):
    _cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
    pw = 3.0 if delivery_type == "Seam" else 4.0
    fig_w, fig_h = pitch_map_figsize(_cfg, width=pw)

    if df_in.empty:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.text(0.5, 0.5, f"No data for Pitch Map ({delivery_type})", ha="center", va="center", fontsize=12)
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

    PITCH_BINS = get_pitch_bins(delivery_type)

    boundary_y_values = sorted([v[0] for v in PITCH_BINS.values() if v[0] > -4.0], reverse=True)
    for y_val in boundary_y_values:
        ax.axhline(y=y_val, color="lightgrey", linewidth=1.0, linestyle="--")

    pct_by_bin = pitch_bin_percentages(df_in, PITCH_BINS, bounce_col="BounceX")
    y_off = 0.2 if _cfg.is_test or delivery_type == "Seam" else 0.22
    for Length, bounds in PITCH_BINS.items():
        mid_y = (bounds[0] + bounds[1]) / 2
        ax.text(
            x=-1.45,
            y=mid_y + y_off,
            s=str(Length).upper(),
            ha="left",
            va="center",
            fontsize=8,
            color="black",
            fontweight="bold",
        )
        p = int(pct_by_bin.get(Length, 0))
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

    # Draw order: Others (bottom) → Boundaries → Wickets (top)
    if not pitch_others.empty:
        ax.scatter(
            pitch_others["BounceY"],
            pitch_others["BounceX"],
            s=60,
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
            s=75,
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
            s=90,
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


def _create_pitch_Length_bars_test(df_in, delivery_type):
    """Red-ball layout: Runs / Dismissals / Batting average by length."""
    _cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
    FIG_SIZE = pitch_map_figsize(_cfg, width=3.0)

    if df_in.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Data for Pitch Length Comparison", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    PITCH_BINS_DICT = pitch_bins_for_format(delivery_type, _cfg)

    if delivery_type == "Seam":
        ordered_keys = ordered_seam_keys(_cfg)
    elif delivery_type == "Spin":
        ordered_keys = ordered_spin_keys(_cfg)
    else:
        fig, ax = plt.subplots(figsize=(3.5, 7))
        ax.text(0.5, 0.5, "Invalid Delivery Type", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    def assign_pitch_Length(x):
        for length, bounds in PITCH_BINS_DICT.items():
            if bounds[0] <= x < bounds[1]:
                return length
        return None

    df_pitch = df_in.copy()
    df_pitch["PitchLength"] = df_pitch["BounceX"].apply(assign_pitch_Length)

    df_summary = (
        df_pitch.groupby("PitchLength")
        .agg(
            Runs=("Runs", "sum"),
            Dismissals=("Wicket", lambda x: int((x == True).sum())),
        )
        .reset_index()
        .set_index("PitchLength")
        .reindex(ordered_keys)
        .fillna(0)
    )

    df_summary["Average"] = np.where(
        df_summary["Dismissals"] > 0, df_summary["Runs"] / df_summary["Dismissals"], 0
    )

    for col in ["Average", "Runs", "Dismissals"]:
        df_summary[col] = df_summary[col].replace([np.inf, -np.inf], 0).fillna(0)

    categories = df_summary.index.tolist()[::-1]
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 7), sharey=True)

    metrics = ["Runs", "Dismissals", "Average"]
    titles = ["Runs", "Dismissals", "Batting Average"]

    def get_limit(val):
        return val * 1.2 if val > 0 else 10

    max_runs = get_limit(df_summary["Runs"].max())
    max_dismissals = get_limit(df_summary["Dismissals"].max())
    max_avg = get_limit(df_summary["Average"].max())

    xlim_limits = {
        "Runs": (0, max_runs),
        "Dismissals": (0, max_dismissals),
        "Average": (0, max_avg),
    }

    for i, ax in enumerate(axes):
        metric = metrics[i]
        title = titles[i]
        values = df_summary[metric].values[::-1]

        ax.barh(categories, values, height=0.49, color="#ff5000", zorder=3, alpha=0.9)

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

        for j, (cat, val) in enumerate(zip(categories, values)):
            if metric in ["Dismissals", "Runs"]:
                label = f"{int(float(val))}"
            else:
                label = f"{val:.1f}"

            ax.text(
                val,
                j,
                label,
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="White", alpha=0.8, edgecolor="none", pad=2),
                zorder=4,
            )

        ax.set_facecolor("white")
        ax.tick_params(axis="y", length=0)

        if i == 2:
            ax.set_yticks(
                np.arange(len(categories)),
                labels=[c.upper() for c in categories],
                fontsize=8,
                fontweight="normal",
            )
        else:
            ax.set_yticks(np.arange(len(categories)), labels=[""] * len(categories))

        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.set_xticks([])
        ax.set_xlim(0, xlim_limits[metric][1] * 1.15)

        spine_color = "lightgray"
        for spine_name in ["left", "right", "top", "bottom"]:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color(spine_color)
            ax.spines[spine_name].set_linewidth(1.0)

    plt.subplots_adjust(top=0.9, hspace=0.4)
    return fig


# --- CHART 3b: PITCH Length RUN % (EQUAL SIZED BOXES) ---
def create_pitch_Length_bars(df_in, delivery_type):
    """
    White-ball: SR / Average / Runs by pitch length. Test cricket delegates to
    `_create_pitch_Length_bars_test`.
    """
    _cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
    if _cfg.is_test:
        return _create_pitch_Length_bars_test(df_in, delivery_type)

    FIG_SIZE = pitch_map_figsize(_cfg, width=3.0)

    if df_in.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Data for Pitch Length Comparison", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    PITCH_BINS_DICT = pitch_bins_for_format(delivery_type, _cfg)

    if delivery_type == "Seam":
        ordered_keys = ordered_seam_keys(_cfg)
    elif delivery_type == "Spin":
        ordered_keys = ordered_spin_keys(_cfg)
    else:
        fig, ax = plt.subplots(figsize=(3.5, 7))
        ax.text(0.5, 0.5, "Invalid Delivery Type", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    def assign_pitch_Length(x):
        for length, bounds in PITCH_BINS_DICT.items():
            if bounds[0] <= x < bounds[1]:
                return length
        return None

    df_pitch = df_in.copy()
    df_pitch["PitchLength"] = df_pitch["BounceX"].apply(assign_pitch_Length)

    df_summary = (
        df_pitch.groupby("PitchLength")
        .agg(
            Runs=("Runs", "sum"),
            Wickets=("Wicket", lambda x: (x == True).sum()),
            Balls=("Wicket", "count"),
            Boundaries=("Runs", lambda x: ((x == 4) | (x == 6)).sum()),
        )
        .reset_index()
        .set_index("PitchLength")
        .reindex(ordered_keys)
        .fillna(0)
    )

    df_summary["StrikeRate"] = df_summary.apply(
        lambda row: (row["Runs"] / row["Balls"]) * 100 if row["Balls"] > 0 else 0, axis=1
    )

    df_summary["Average"] = df_summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else 0, axis=1
    )

    df_summary["Dismissals"] = df_summary["Wickets"]

    categories = df_summary.index.tolist()[::-1]

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 7), sharey=True)
    plt.subplots_adjust(hspace=10)

    metrics = ["Dismissals", "Average", "StrikeRate"]
    titles = ["Dismissals", "Batting Average", "Batting Strike Rate"]

    max_dismissals = df_summary["Dismissals"].max() * 1.2 if df_summary["Dismissals"].max() > 0 else 100
    max_avg = df_summary["Average"].max() * 1.2 if df_summary["Average"].max() > 0 else 100
    max_sr = df_summary["StrikeRate"].max() * 1.2 if df_summary["StrikeRate"].max() > 0 else 300

    xlim_limits = {
        "Dismissals": (0, max_dismissals),
        "Average": (0, max_avg),
        "StrikeRate": (0, max_sr),
    }

    for i, ax in enumerate(axes):
        metric = metrics[i]
        title = titles[i]
        values = df_summary[metric].values[::-1]

        ax.set_xlim(xlim_limits[metric])

        ax.barh(categories, values, height=0.49, color="#ff5000", zorder=3, alpha=0.9)

        for j, (cat, val) in enumerate(zip(categories, values)):
            if metric == "Dismissals":
                label = f"{int(val)}"
            elif metric == "StrikeRate":
                label = f"{val:.0f}"
            else:
                label = f"{val:.1f}"

            ax.text(
                val,
                j,
                label,
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="White", alpha=0.8, edgecolor="none", pad=2),
                zorder=4,
            )

        ax.set_title(title, fontsize=10, fontweight="bold", pad=0, loc="left")
        ax.set_facecolor("white")

        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", length=0)

        if i == 2:
            ax.set_yticks(np.arange(len(categories)), labels=[c.upper() for c in categories], fontsize=8)
        else:
            ax.set_yticks(np.arange(len(categories)), labels=[""] * len(categories))

        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

        ax.set_xticks([])
        ax.set_xlim(0, xlim_limits[metric][1])

        spine_color = "lightgray"
        spine_width = 1.0
        for spine_name in ["left", "right", "top", "bottom"]:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color(spine_color)
            ax.spines[spine_name].set_linewidth(spine_width)
    plt.tight_layout(pad=0.5)
    return fig
    
  
# --- CHART 4a: INTERCEPTION SIDE-ON --- (Wide View)
# --- Helper function for Interception Bins ---

def get_interception_bins():
    """Defines the bins for the Crease Width Split chart."""
    return {
        "0m-1m": [0, 1],
        "1m-2m": [1, 2],
        "2m-3m": [2, 3],
        "3m+": [3, 10]  # Assuming max possible value is < 100
    }

def create_interception_side_on(df_in, delivery_type):
    # Define Figure Size (slightly narrower and taller for the vertical stack)
    FIG_WIDTH = 7
    FIG_HEIGHT = 5
    FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)

    if df_in.empty or df_in["InterceptionX"].isnull().all():
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No Data for Combined Interception Analysis", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # --- SETUP GRID FOR TWO ROWS ---
    # Top: Scatter Plot (Larger) | Bottom: Bar Chart (Smaller)
    fig = plt.figure(figsize=FIG_SIZE)
    # Ratio: 80% for scatter plot, 20% for bar chart
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1) 
    
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    
    fig.patch.set_facecolor('white')

    # ----------------------------------------------------------------------
    ## --- PART 1: CHART 4a - INTERCEPTION SIDE-ON SCATTER (ax_scatter) ---
    # ----------------------------------------------------------------------
    df_interception = df_in[df_in["InterceptionX"] > -999].copy()    
    df_interception["ColorType"] = "Other"
    df_interception.loc[df_interception["Wicket"] == True, "ColorType"] = "Wicket"
    df_interception.loc[df_interception["Runs"].isin([4, 6]), "ColorType"] = "Boundary"
    # Define color_map inline as it's needed for the loop
    color_map = {"Wicket": "red", "Boundary": "royalblue", "Other": "white"}
    
    # 1. Plot Data (Layered for correct border visibility)
    
    # Plot "Other" (White with Grey Border)
    df_other = df_interception[df_interception["ColorType"] == "Other"]
    # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on X-axis ===
    ax_scatter.scatter(
        df_other["InterceptionX"] + 10, df_other["InterceptionZ"], 
        color='#D3D3D3', edgecolors='white', linewidths=0.3, s=40, label="Other", zorder=2
    )
    
    # Plot "Wicket" and "Boundary" (Solid colors)
    for ctype in ["Boundary", "Wicket"]:
        df_slice = df_interception[df_interception["ColorType"] == ctype]
        # === USING PROVIDED LOGIC: PLOT (InterceptionX + 10) on X-axis ===
        ax_scatter.scatter(
            df_slice["InterceptionX"] + 10, df_slice["InterceptionZ"], 
            color=color_map[ctype],edgecolors='white', linewidths=0.3, s=65, label=ctype, zorder=3
        )

    # 2. Draw Vertical Dashed Lines with Labels (FIXED LINES: 0.0, 1.25, 2.0, 3.0)
    line_specs = {
        0.0: "Stumps",
        1.000: "1m",
        2.000: "2m",     
        3.000: "3m" 
    }
    
    for x_val, label in line_specs.items():
        ax_scatter.axvline(x=x_val, color='lightgrey', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)     
        ax_scatter.text(x_val, 1.75, label.split(':')[-1].strip(), ha='center', va='center', fontsize=8, color='grey', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=1)
        
    ax_scatter.axhline(y=0.746, color="grey", linestyle="--", linewidth=0.5,zorder=1)
    ax_scatter.text(0.1, 0.746, "Stump Height", ha='left', va='bottom', fontsize=7, color="grey", transform=ax_scatter.transData)
    
    # Set Y limit as fixed
    y_limit = 2
    
    # Set X limit based on delivery type
    if delivery_type == "Seam":
        x_limit_max = 3.5
    elif delivery_type == "Spin":
        x_limit_max = 4.5
    else:
        x_limit_max = 3.5 
        
    x_limit_min = -0.2
    
    ax_scatter.set_xlim(x_limit_min, x_limit_max) 
    ax_scatter.set_ylim(0, y_limit) 
    # ... (Rest of the styling remains the same)
    ax_scatter.tick_params(axis='y', which='both', labelleft=False, left=False); ax_scatter.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['left'].set_visible(False)
    ax_scatter.spines['bottom'].set_visible(False)

# ----------------------------------------------------------------------
    ## --- PART 2: CHART 4b - CREASE WIDTH SPLIT BARS (ax_bar) ---
    # ----------------------------------------------------------------------
    
    # 1. Data Preparation
    INTERCEPTION_BINS = get_interception_bins()
    ordered_keys = ["0m-1m", "1m-2m", "2m-3m", "3m+"]  # Order: Close to Wide
    COLORMAP = 'Wistia'
    
    def assign_crease_width(x):
        for width, bounds in INTERCEPTION_BINS.items():
            if bounds[0] <= x < bounds[1]: return width
        return None

    df_crease = df_in.copy()
    df_crease["CreaseWidth"] = (df_crease["InterceptionX"] + 10).apply(assign_crease_width)
    
    df_summary = df_crease.groupby("CreaseWidth").agg(
        Runs=("Runs", "sum"), 
        Wickets=("Wicket", lambda x: (x == True).sum()), 
        Balls=("Wicket", "count")
    ).reset_index().set_index("CreaseWidth").reindex(ordered_keys).fillna(0)
    
    # NEW: Calculate Strike Rate (SR)
    df_summary["SR"] = df_summary.apply(
        lambda row: (row["Runs"] / row["Balls"]) * 100 if row["Balls"] > 0 else np.nan, axis=1
    )
    # Calculate Average (Avg)
    df_summary["Avg"] = df_summary.apply(
        lambda row: row["Runs"] / row["Wickets"] if row["Wickets"] > 0 else row["Runs"], axis=1
    )
    
    # 2. Plotting Equal Boxes
    num_boxes = len(ordered_keys)
    box_width = 1.0 / num_boxes 
    left = 0.0
    box_height = 0.8

    # Normalization changed to Strike Rate
    max_sr_val = df_summary["SR"].replace([np.inf, -np.inf], np.nan).max()
    max_sr = max_sr_val if max_sr_val > 0 else 200 # Default max for scaling
    
    norm = mcolors.Normalize(vmin=0, vmax=max_sr)
    cmap = plt.get_cmap(COLORMAP)
    
    for index, row in df_summary.iterrows():
        runs = int(row["Runs"])
        wickets = int(row["Wickets"])
        sr = row["SR"]
        avg = row["Avg"]
        
        # --- CONDITIONAL STYLING LOGIC ---
        if sr == 0:
            sr_display = '0'
            avg_display = '0.0'
            color = 'white'
            text_color = 'black'
        else:
            sr_display = f"{sr:.0f}"
            avg_display = f"{avg:.1f}"
            color = cmap(norm(sr)) 
            
            # Contrast logic for text
            r, g, b, a = color
            luminosity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminosity < 0.5 else 'black'
            
        # Draw the box  
        ax_bar.barh(
            y=0.5,             
            width=box_width,
            height=0.6,          
            left=left,         
            color=color,
            edgecolor='black',
            linewidth=0.4
        )
        
        # --- UPDATED TEXT: Multi-line Format ---
        # Line 1: Runs and Wickets
        label_top = f"{runs} Runs, {wickets}W"
        # Line 2: Avg and SR
        label_bottom = f"{avg_display} Avg, {sr_display} SR"
    
        center_x = left + box_width / 2
    
        # Position Line 1 (Upper half of the colored box)
        ax_bar.text(center_x, 0.62, label_top, ha='center', va='center', 
                fontsize=8, fontweight='bold', color=text_color)
    
        # Position Line 2 (Lower half of the colored box)
        ax_bar.text(center_x, 0.38, label_bottom, ha='center', va='center', 
                fontsize=8, fontweight='bold', color=text_color)
    
        # Crease Width Label (Top of the box)
        ax_bar.text(center_x, 0.82, index, ha='center', va='bottom', fontsize=9, color='black')

        left += box_width

    # 3. Styling for Bar Chart
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1) 
    ax_bar.axis('off')


    # ----------------------------------------------------------------------
    ## --- PART 3: DRAW SINGLE COMPACT BORDER ---
    # ----------------------------------------------------------------------
    
    plt.tight_layout(pad=0.2) 
    
    PADDING = 0.005 

    # Get the bounding box of the top (scatter) and bottom (bar) charts
    scatter_bbox = ax_scatter.get_position()
    bar_bbox = ax_bar.get_position() 
    # Determine the total bounds (figure coordinates)
    x0_orig = scatter_bbox.x0         
    y0_orig = bar_bbox.y0  
    x1_orig = scatter_bbox.x1     
    y1_orig = scatter_bbox.y1         
    
    # Apply Padding
    x0_pad = x0_orig - PADDING
    y0_pad = y0_orig - PADDING
    
    width_pad = (x1_orig - x0_orig) + (2 * PADDING)
    height_pad = (y1_orig - y0_orig) + (2 * PADDING)

    # Draw the custom Rectangle 
    border_rect = patches.Rectangle(
        (x0_pad-0.008, y0_pad+0.02), 
        width_pad+0.017, 
        height_pad,  
        facecolor='none', 
        edgecolor='black', 
        linewidth=0.5, 
        transform=fig.transFigure, 
        clip_on=False
    )

    fig.patches.append(border_rect)

    return fig




# --- Helper Functions for Chart 6 ---
def calculate_scoring_wagon(row):
    """Calculates the scoring area based on LandingX/Y coordinates and handedness."""
    LX = row.get("LandingX")
    LY = row.get("LandingY")
    RH = row.get("IsBatsmanRightHanded")
    
    if RH is None or LX is None or LY is None or row.get("Runs", 0) == 0: 
        return None
    
    def atan_safe(numerator, denominator): 
        return np.arctan(numerator / denominator) if denominator != 0 else np.nan 
    
    # Right Handed Batsman Logic
    if RH == True: 
        if LX <= 0 and LY > 0: return "FINE LEG"
        elif LX <= 0 and LY <= 0: return "THIRD MAN"
        elif LX > 0 and LY < 0:
            if atan_safe(LY, LX) < np.pi / -4: return "COVER"
            elif atan_safe(LX, LY) <= np.pi / -4: return "LONG OFF" 
        elif LX > 0 and LY >= 0:
            if atan_safe(LY, LX) >= np.pi / 4: return "SQUARE LEG"
            elif atan_safe(LY, LX) <= np.pi / 4: return "LONG ON"
            
    # Left Handed Batsman Logic
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
    """Defines the fixed angle size for each wedge."""
    if area in ["FINE LEG", "THIRD MAN"]: return 90
    elif area in ["COVER", "SQUARE LEG", "LONG OFF", "LONG ON"]: return 45
    return 0

# --- Main Combined Function (Chart 6) ---
def create_wagon_wheel(df_in, delivery_type):
    FIG_WIDTH = 6
    FIG_HEIGHT = 4
    FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)

    if df_in.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, f"No Data for {delivery_type} Analysis", ha='center', va='center', fontsize=8)
        ax.axis('off')
        return fig

    # CRITICAL FIX: Initialize the figure and axes to avoid NameError
    fig, (ax_wagon) = plt.subplots(1, 1, figsize=FIG_SIZE)
    plt.subplots_adjust(hspace=0.3)

    try:
        # 1. Data Preparation (Must be indented 8 spaces from the left)
        df_wagon = df_in.copy()
        df_wagon["ScoringWagon"] = df_wagon.apply(calculate_scoring_wagon, axis=1)
        df_wagon["FixedAngle"] = df_wagon["ScoringWagon"].apply(calculate_scoring_angle)
        
        summary_with_shots = df_wagon.groupby("ScoringWagon").agg(
            TotalRuns=("Runs", "sum"), 
            Balls=("Runs", "count"),
            FixedAngle=("FixedAngle", 'first')
        ).reset_index().dropna(subset=["ScoringWagon"])
        
        # 2. Handedness & Area Logic
        handedness_mode = df_in["IsBatsmanRightHanded"].dropna().mode()
        is_right_handed = handedness_mode.iloc[0] if not handedness_mode.empty else True
        
        all_areas = ["FINE LEG", "SQUARE LEG", "LONG ON", "LONG OFF", "COVER", "THIRD MAN"] if is_right_handed else ["THIRD MAN", "COVER", "LONG OFF", "LONG ON", "SQUARE LEG", "FINE LEG"]
            
        template_df = pd.DataFrame({
            "ScoringWagon": all_areas, 
            "FixedAngle": [calculate_scoring_angle(area) for area in all_areas]
        })

        wagon_summary = template_df.merge(summary_with_shots.drop(columns=["FixedAngle"], errors='ignore'), on="ScoringWagon", how="left").fillna(0)
        wagon_summary["RunPercentage"] = (wagon_summary["TotalRuns"] / wagon_summary["TotalRuns"].sum() * 100).fillna(0)
        
        # 3. Plot Part 1: Wagon Wheel (ax_wagon)
        angles = wagon_summary["FixedAngle"].tolist()
        wagon_summary['Rank'] = wagon_summary['RunPercentage'].rank(method='dense', ascending=False)
        colors = ['#ff5000' if (r == 1 and p > 0) else 'white' for r, p in zip(wagon_summary['Rank'], wagon_summary['RunPercentage'])]

        wedges, texts, autotexts = ax_wagon.pie(
            angles, 
            colors=colors, 
            wedgeprops={"width": 1, "edgecolor": "black"}, 
            startangle=90, 
            counterclock=False, 
            autopct='%1.0f%%', # Changed from '' to '%1.0f%%'
            pctdistance=0.6
        )

        # 4. Styling Text with Contrast Logic
        for i, autotext in enumerate(autotexts):
            if wagon_summary["RunPercentage"].iloc[i] > 0:
                autotext.set_text(f'{wagon_summary["RunPercentage"].iloc[i]:.0f}%')
                autotext.set_fontsize(15); autotext.set_fontweight('bold')
                rgb = mcolors.to_rgb(colors[i])
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                autotext.set_color('white' if lum < 0.5 else 'black')
            else:
                autotext.set_text('')

        ax_wagon.axis('equal')

    except Exception as e:
        # If ax_wagon is already defined, we can use it to show the error
        ax_wagon.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
        ax_wagon.axis('off')

    return fig        

    # === CRITICAL FIX: CENTERING PERCENTAGE LABELS AND STYLING ===
    # --- Inside the autotext loop ---
    for i, autotext in enumerate(autotexts):
            if i >= len(run_percentages): 
                break
            
            percent = run_percentages[i]
            
            if percent > 0:
                # 1. Set text and alignment
                autotext.set_text(f'{percent:.0f}%')
                autotext.set_horizontalalignment('center')
                autotext.set_verticalalignment('center')
                
                # 2. Set styling (Font size and weight)
                autotext.set_fontsize(15)
                autotext.set_fontweight('bold')
                
                # 3. Dynamic contrast: Determine if text should be white or black
                color_rgb = mcolors.to_rgb(colors[i])
                luminosity = 0.2126 * color_rgb[0] + 0.7152 * color_rgb[1] + 0.0722 * color_rgb[2]
                
                # If background is dark (luminosity < 0.5), use white text
                if luminosity < 0.5:
                    autotext.set_color('white')
                else:
                    autotext.set_color('black')
            else:
                # Hide text for 0% slices
                autotext.set_text('')
    ax_wagon.axis('equal');

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_speed_metrics_bar(df_in, delivery_type):

    if df_in.empty:
        return None

    _cfg = resolve_format(
        st.session_state.get("cricket_format", "men_t20i")
    )
    _unit = st.session_state.get("speed_unit", "kph")

    def assign_speed_group(speed):
        if delivery_type == "Seam":
            return seam_speed_group(speed, _cfg, _unit)
        return spin_speed_group(speed, _cfg, _unit)

    # -----------------------------
    # Prepare Data
    # -----------------------------
    df_temp = df_in.copy()

    df_temp["ReleaseSpeed"] = pd.to_numeric(
        df_temp["ReleaseSpeed"],
        errors="coerce"
    )

    df_temp = df_temp.dropna(subset=["ReleaseSpeed"])

    df_temp["SpeedGroup"] = (
        df_temp["ReleaseSpeed"]
        .apply(assign_speed_group)
    )

    if delivery_type == "Seam":
        ordered_groups = seam_speed_ordered_groups(_cfg, _unit)
    else:
        ordered_groups = spin_speed_ordered_groups(_cfg, _unit)

    # -----------------------------
    # Aggregate
    # -----------------------------
    summary = (
        df_temp.groupby("SpeedGroup")
        .agg(
            Runs=("Runs", "sum"),
            Balls=("Runs", "count"),
            Dismissals=("Wicket", "sum")
        )
        .reindex(ordered_groups)
        .fillna(0)
    )

    summary["SR"] = (
        summary["Runs"] /
        summary["Balls"] * 100
    ).fillna(0)

    summary["Avg"] = (
        summary["Runs"] /
        summary["Dismissals"]
    )

    summary.loc[
        np.isinf(summary["Avg"]),
        "Avg"
    ] = summary["Runs"]

    summary["Avg"] = summary["Avg"].fillna(0)

    # -----------------------------
    # Plot
    # -----------------------------
    metrics = ["Runs", "Dismissals", "Avg", "SR"]

    fig = plt.figure(
        figsize=(7, 9),
        facecolor="white"
    )

    # Outer border around entire figure
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(2)

    gs = fig.add_gridspec(
        nrows=4,
        ncols=1,
        hspace=0.35
    )

    for idx, metric in enumerate(metrics):

        ax = fig.add_subplot(gs[idx])

        values = summary[metric].values
        groups = summary.index.tolist()

        y = np.arange(len(groups))

        # Orange bars
        ax.barh(
            y,
            values,
            height=0.55,
            color="#ff5000",
            edgecolor="#ff5000"
        )

        max_val = max(values) if max(values) > 0 else 1

        # Value labels
        for ypos, val in zip(y, values):

            if metric in ["Avg", "SR"]:
                label = f"{val:.0f}"
            else:
                label = f"{int(val)}"

            ax.text(
                val + max_val * 0.04,
                ypos,
                label,
                va="center",
                ha="left",
                fontsize=14,
                fontweight="bold"
            )

        # Speed-group labels
        ax.set_yticks(y)
        ax.set_yticklabels(
            groups,
            fontsize=16
        )

        ax.invert_yaxis()
    
        # Metric label (Runs, Dismissals, Avg, SR)
        ax.text(
            -0.10,
            0.5,
            metric,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="center",
            ha="right"
        )
    
        # Remove x-axis clutter
        ax.set_xticks([])
        ax.tick_params(
            axis="y",
            length=0,
            pad=2
        )
    
        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    
        # Keep left spine
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["left"].set_position(("data", 0))
    
        ax.set_xlim(
            -max_val * 0.15,
            max_val * 1.35
        )
    
    # No chart title
    # No fig.suptitle()
    
    fig.tight_layout(
        rect=[0.05, 0.03, 0.98, 0.98]
    )
    
    return fig
    
#----------------------------------------
# PAGE LAYOUT SETUP
#----------------------------------------

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

# =========================================================
# 💥 1. CRITICAL: GET DATA FROM SESSION STATE
# This check ensures the page cannot run without data uploaded via Home.py
# =========================================================
if 'data_df' not in st.session_state:
    st.error("Please go back to the **Home** page and upload the data first to begin the analysis.")
    # Stop execution of the rest of the script if data is missing
    st.stop()


# Retrieve the full raw DataFrame
df_raw = st.session_state['data_df']
_cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
# Header: wide title | file | format (right) | legend
col_title_space, col_format_banner, col_legend = st.columns([2.5, 2.5, 5])

with col_title_space:
    st.title("BATTERS")

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
# NOTE: BattingTeam is often case-sensitive, but converting Batsman/Bowler is key here.


# =========================================================
# 🌟 FILTERS 🌟 (multiselect; "All" = no filter)
# =========================================================

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


# 1. Setup choices for Batting Team
all_teams = ["All"] + sorted(df_raw["BattingTeam"].dropna().unique().tolist())

row1 = st.columns(4)

with row1[0]:
    bat_team_sel = st.multiselect(
        "Batting Team", 
        all_teams, 
        default=["All"],
        key="team_filter",
        on_change=handle_all_selection,
        args=("team_filter",)
    )

# 2. Dynamically filter batsmen options based on chosen team
if _multiselect_is_all(bat_team_sel):
    df_bat_opts = df_raw
else:
    teams_only = [t for t in bat_team_sel if t != "All"]
    df_bat_opts = df_raw[df_raw["BattingTeam"].isin(teams_only)]

# This defines the variable that was throwing the NameError
batsmen_options = ["All"] + sorted(df_bat_opts["BatsmanName"].dropna().unique().tolist())

with row1[1]:
    batsman_sel = st.multiselect(
        "Batsman Name", 
        batsmen_options, 
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
        year_vals = sorted(pd.to_numeric(df_raw[year_col], errors='coerce').dropna().astype(int).astype(str).unique().tolist())
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

# =========================================================

def apply_filters(df):
    df_filtered = df.copy()

    if not _multiselect_is_all(bat_team_sel):
        teams_only = [t for t in bat_team_sel if t != "All"]
        df_filtered = df_filtered[df_filtered["BattingTeam"].isin(teams_only)]

    if not _multiselect_is_all(batsman_sel):
        bats = [b for b in batsman_sel if b != "All"]
        df_filtered = df_filtered[df_filtered["BatsmanName"].isin(bats)]

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

    return df_filtered

# Separate by delivery type BEFORE filtering to save a little processing, then apply filters
df_seam_base = df_raw[df_raw["DeliveryType"] == "Seam"]
df_spin_base = df_raw[df_raw["DeliveryType"] == "Spin"]

# Apply filters
df_seam = apply_filters(df_seam_base)
df_spin = apply_filters(df_spin_base)
    
if _multiselect_is_all(batsman_sel):
    heading_text = "ALL"
else:
    names = [b for b in batsman_sel if b != "All"]
    heading_text = " + ".join(names) if names else "ALL"
# Use st.markdown to inject HTML, setting the text color directly
st.markdown(
    f"<h3 style='color: #ff5000;'><b>{heading_text}</b></h3>",
    unsafe_allow_html=True
)

# --- 4. DISPLAY CHARTS IN TWO COLUMNS (SEAM vs. SPIN) ---
col1, col2 = st.columns(2)
    
# --- LEFT COLUMN: SEAM ANALYSIS ---
with col1:
    st.markdown("### v SEAM")
    
    # Row 2: Crease Beehive Scatter
    st.markdown("###### CREASE BEEHIVE v SEAM")
    st.pyplot(create_crease_beehive(df_seam, "Seam"), use_container_width=True)
    
    # Row 4: Pitch Map and Vertical Run % Bar (Side-by-Side)
    pitch_col, pitch_bars = st.columns(2)
    with pitch_col:
        st.markdown("###### PITCHMAP v SEAM")
        st.pyplot(create_pitch_map(df_seam, "Seam"), use_container_width=True)  
    with pitch_bars:
        st.markdown("###### ")
        st.pyplot(create_pitch_Length_bars(df_seam, "Seam"), use_container_width=True)   

    # Row 5: Interception Side-On (Wide View)
    st.markdown("###### INTERCEPTION SIDE-VIEW v SEAM")
    st.pyplot(create_interception_side_on(df_seam, "Seam"), use_container_width=True)

    # Row 7: Interception and Scoring Areas (Side-by-Side)
    st.markdown("###### SCORING AREAS v SEAM ")
    st.pyplot(create_wagon_wheel(df_seam, "Seam"), use_container_width=True)
    
    # --- Row 9 & 10 RELEASE SPEED ANALYSIS---
    st.markdown("###### METRICS BY RELEASE SPEED v SEAM")
    st.pyplot(create_speed_metrics_bar(df_seam, "Seam"), use_container_width=True)
        
# --- RIGHT COLUMN: SPIN ANALYSIS ---
with col2:
    st.markdown("### v SPIN")
    
    # Row 2: Crease Beehive Scatter
    st.markdown("###### CREASE BEEHIVE v SPIN")
    st.pyplot(create_crease_beehive(df_spin, "Spin"), use_container_width=True)
 
    # Row 4: Pitch Map and Vertical Run % Bar (Side-by-Side)
    pitch_col, pitch_bars = st.columns(2)
    with pitch_col:
        st.markdown("###### PITCHMAP v SPIN")
        st.pyplot(create_pitch_map(df_spin, "Spin"), use_container_width=True)  
    with pitch_bars:
        st.markdown("###### ")
        st.pyplot(create_pitch_Length_bars(df_spin, "Spin"), use_container_width=True)    
    
    # Row 5: Interception Side-On (Wide View)
    st.markdown("###### INTERCEPTION SIDE-VIEW v SPIN")
    st.pyplot(create_interception_side_on(df_spin, "Spin"), use_container_width=True)

    # Row 7: Scoring Areas (Side-by-Side)
    st.markdown("###### SCORING AREAS v SPIN")
    st.pyplot(create_wagon_wheel(df_spin,'SPIN'), use_container_width=True)
    
    # --- Row 9 & 10 RELEASE SPEED ANALYSIS---
    st.markdown("###### METRICS BY RELEASE SPEED v SPIN")
    st.pyplot(create_speed_metrics_bar(df_spin, "Spin"), use_container_width=True)
    
