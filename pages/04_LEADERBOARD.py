import streamlit as st
import pandas as pd
import numpy as np

from cricket_config import (
    resolve_format,
    match_phase_options,
    apply_match_phase_filter,
    leaderboard_batter_length_options,
    leaderboard_batter_pace_options,
    filter_batter_length,
    filter_batter_pace,
    filter_pacer_length,
    filter_pacer_pace,
    pacer_view_types,
    pacer_length_filter_options,
    spinner_view_types,
    batter_sr_or_avg_label,
    PACERS_METRIC_VIEW_TYPES,
    SPINNERS_METRIC_VIEW_TYPES,
    hitting_stumps_mask,
    format_banner_caps,
    FORMAT_BANNER_STYLE,
)


def rank_slice(
    df: pd.DataFrame,
    by: str,
    *,
    higher_is_better: bool,
    bottom: bool,
    n: int = 10,
) -> pd.DataFrame:
    """Top N = best first; Bottom N = worst first (flip sort direction)."""
    if df is None or df.empty:
        return df
    asc = not higher_is_better
    if bottom:
        asc = not asc
    return df.sort_values(by=by, ascending=asc, na_position="last").head(n)


st.set_page_config(layout="wide", page_title="Leaderboard")

# --- ADVANCED DASHBOARD THEME CSS ---
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 200px !important;
        }
        /* Global Header Typography - Safe Layout to prevent clipping */
        h1 {
            font-weight: 800 !important;
            color: #1E293B !important;
            letter-spacing: -0.5px;
            margin-top: 0px !important;
            padding-top: 0px !important;
            margin-bottom: 5px !important;
        }
        h2, h3 {
            font-weight: 700 !important;
            color: #334155 !important;
            margin-top: 0px !important;
            padding-top: 0px !important;
        }
        
        /* Table Styling Overrides */
        th[data-testid="stTableHeadCell"] {
            text-align: center !important;
            background-color: #F8FAFC !important;
            color: #475569 !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            font-size: 0.85rem !important;
            border-bottom: 2px solid #E2E8F0 !important;
        }
        th {
            text-align: center !important;
        }
        div[data-testid="stTableHeadCellContent"] {
            justify-content: center !important;
            text-align: center !important;
        }

        /* Compact Layout Overrides for the Control Panel Column */
        div[data-testid="column"]:nth-of-type(2) label p {
            font-size: 0.825rem !important;
            font-weight: 600 !important;
            margin-bottom: -2px !important; /* Pulls widgets closer to labels */
        }
        
        div[data-testid="column"]:nth-of-type(2) div[data-baseweb="select"] > div {
            min-height: 32px !important; /* Keeps dropdown inputs compact */
            height: 32px !important;
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }

        div[data-testid="column"]:nth-of-type(2) div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            min-height: 32px !important;
            height: 32px !important;
        }

        /* Tighten default vertical element spacing gaps in the controls layout */
        div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"] > div {
            gap: 0.4rem !important; 
        }
        
        /* Subtitle formatting for active filters */
        .filter-caption {
            color: #64748B;
            font-size: 0.9rem;
            margin-top: -10px !important;
            margin-bottom: 20px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

_cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
_col_title_lb, _col_fmt_lb = st.columns([2.5, 1.2])
with _col_title_lb:
    st.title("LEADERBOARD")
with _col_fmt_lb:
    st.markdown(
        f'<div style="margin-top: 28px; text-align: right; width: 100%;"><span style="{FORMAT_BANNER_STYLE}">{format_banner_caps(_cfg)}</span></div>',
        unsafe_allow_html=True,
    )
st.write("---")

# --- 1. SESSION STATE DATA CHECK ---
if 'data_df' not in st.session_state or st.session_state['data_df'] is None:
    st.error("No Data Found! Please navigate back to the HOME page and upload your CSV tracking asset first.")
else:
    # Safely extract and copy the dataframe
    df_raw = st.session_state['data_df'].copy()
    df_raw.columns = df_raw.columns.str.strip()
    cfg = _cfg
    unit = st.session_state.get("speed_unit", "kph")
    
    # Clean and explicitly prepare data types for accurate filtering
    df_raw["ReleaseSpeed"] = pd.to_numeric(df_raw["ReleaseSpeed"], errors="coerce")
    df_raw["BounceX"] = pd.to_numeric(df_raw["BounceX"], errors="coerce")
    df_raw["Deviation"] = pd.to_numeric(df_raw["Deviation"], errors="coerce")
    if "Swing" in df_raw.columns:
        df_raw["Swing"] = pd.to_numeric(df_raw["Swing"], errors="coerce")
    df_raw["Wicket"] = df_raw["Wicket"].astype(bool)
    df_raw["Runs"] = pd.to_numeric(df_raw["Runs"], errors="coerce").fillna(0)
    
    # Ensure Over column is parsed cleanly as numbers for matching boundary rules
    if "Over" in df_raw.columns:
        df_raw["Over"] = pd.to_numeric(df_raw["Over"], errors="coerce")

    # ------------------------------------------------------------------
    # --- SPLIT SCREEN LAYOUT DESIGN ---
    # ------------------------------------------------------------------
    main_display_col, filter_panel_col = st.columns([5.5, 2.5], gap="large")

    # Objects to hold our conditional data slice
    df_filtered = pd.DataFrame()
    filter_label = ""

    # ==================================================================
    # STEP 2: RENDER CONTROLS IN THE RIGHT FILTER PANEL (SIDEBAR LOOK)
    # ==================================================================
    with filter_panel_col:
        st.subheader("⚙️ Control Panel")
        
        # Filter 1: Player Role Selection
        f1 = st.selectbox("Select Player Role", ["BATTERS", "PACERS", "SPINNERS"])

        f_rank = st.selectbox("Leaderboard rank", ["Top 10", "Bottom 10"])
        want_bottom = f_rank == "Bottom 10"
        
        # Shared Filter 2: Match Phase Filter (Overs) — T20I/ODI; hidden phases for Test
        phase_opts = match_phase_options(cfg)
        f_overs = st.selectbox("Match Phase (Overs)", phase_opts)
        df_raw = apply_match_phase_filter(df_raw, f_overs, cfg)

        # Conditional Filters based on Selected Role
        if f1 == "BATTERS":
            df_raw = df_raw[df_raw["DeliveryType"] == "Seam"]
            f2 = st.selectbox(batter_sr_or_avg_label(cfg), ["LENGTH", "PACE"])

            if f2 == "LENGTH":
                length_opts = leaderboard_batter_length_options(cfg)
                f3 = st.selectbox("Select Length", length_opts)
                filter_label = f3

                if f3 == "All":
                    df_filtered = df_raw.copy()
                else:
                    df_filtered = filter_batter_length(df_raw, f3, cfg)

            elif f2 == "PACE":
                hi, lo = leaderboard_batter_pace_options(cfg, unit)
                f3 = st.selectbox("Select Pace", [hi, lo])
                filter_label = f"Pace ({f3} {unit})"
                df_filtered = filter_batter_pace(df_raw, f3, cfg, unit)

            min_balls = st.number_input("Minimum balls faced", min_value=1, value=10, step=1)

        elif f1 == "PACERS":
            df_role_base = df_raw[df_raw["DeliveryType"].str.lower() == "seam"] if "DeliveryType" in df_raw.columns else df_raw.copy()

            f2 = st.selectbox("View Type", pacer_view_types(cfg))

            if f2 == "All":
                df_filtered = df_role_base.copy()
                filter_label = "All Lengths"
            elif f2 in ["Economy By Length", "% by Lengths", "Bowling Strike Rate By Length"]:
                plen_opts = pacer_length_filter_options(cfg)
                f3 = st.selectbox("Select Length", plen_opts)
                filter_label = f3
                df_filtered = filter_pacer_length(df_role_base, f3, cfg)

            elif f2 in ["Economy by Pace", "% Balls by Pace", "Bowling Average by Pace"]:
                hi, lo = leaderboard_batter_pace_options(cfg, unit)
                f3 = st.selectbox("Select Pace Range", [hi, lo])
                filter_label = f"Pace ({f3} {unit})"
                df_filtered = filter_pacer_pace(df_role_base, f3, cfg, unit)

            elif f2 in PACERS_METRIC_VIEW_TYPES:
                df_filtered = df_role_base.copy()
                filter_label = f2

            min_balls = st.number_input("Minimum balls bowled", min_value=1, value=10, step=1)

        elif f1 == "SPINNERS":
            df_role_base = df_raw[df_raw["DeliveryType"].str.lower() == "spin"] if "DeliveryType" in df_raw.columns else df_raw.copy()
            
            # "All" added as the top default option
            f2 = st.selectbox("View Type", spinner_view_types(cfg))
            
            if f2 == "All":
                df_filtered = df_role_base.copy()
                filter_label = "All Lengths"
            elif f2 in ["Economy By Length", "% by Lengths", "Bowling Strike Rate By Length"]:
                f3 = st.selectbox(
                    "Select Length", 
                    ["OVERPITCHED", "FULL", "GOOD", "SHORT"]
                )
                filter_label = f3
                
                if f3 == "OVERPITCHED":
                    df_filtered = df_role_base[df_role_base["BounceX"] <= 2.8]
                elif f3 == "FULL":
                    df_filtered = df_role_base[(df_role_base["BounceX"] > 2.8) & (df_role_base["BounceX"] <= 4.4)]
                elif f3 == "GOOD":
                    df_filtered = df_role_base[(df_role_base["BounceX"] > 4.4) & (df_role_base["BounceX"] <= 6.2)]
                elif f3 == "SHORT":
                    df_filtered = df_role_base[df_role_base["BounceX"] > 6.2]
                    
            elif f2 == "Turn Direction":
                f3 = st.selectbox("Select Ball Turn Direction", ["Turn Left", "No Turn", "Turn Right"])
                filter_label = f3
                
                if f3 == "Turn Left":
                    df_filtered = df_role_base[df_role_base["Deviation"] < -0.1]
                elif f3 == "No Turn":
                    df_filtered = df_role_base[(df_role_base["Deviation"] >= -0.1) & (df_role_base["Deviation"] <= 0.1)]
                elif f3 == "Turn Right":
                    df_filtered = df_role_base[df_role_base["Deviation"] > 0.1]

            elif f2 in SPINNERS_METRIC_VIEW_TYPES:
                df_filtered = df_role_base.copy()
                filter_label = f2

            min_balls = st.number_input("Minimum balls bowled", min_value=1, value=10, step=1)

    # ==================================================================
    # STEP 3: EXECUTE CALCULATIONS & RENDER LEADERBOARDS ON LEFT SIDE
    # ==================================================================
    with main_display_col:
        rank_label = "Bottom 10" if want_bottom else "Top 10"

        # --- RENDER ENGINE: BATTERS ---
        if f1 == "BATTERS":
            if not df_filtered.empty:
                if cfg.is_test:
                    leaderboard = df_filtered.groupby("BatsmanName").agg(
                        Runs=("Runs", "sum"),
                        Balls_Faced=("Ball", "size"),
                        Dismissals=("Wicket", lambda x: sorted(x).count(True)),
                    ).reset_index()
                    leaderboard = leaderboard[leaderboard["Balls_Faced"] >= min_balls]
                    leaderboard = leaderboard[leaderboard["Dismissals"] > 0]
                    if not leaderboard.empty:
                        leaderboard["Average"] = leaderboard["Runs"] / leaderboard["Dismissals"]
                        leaderboard = rank_slice(
                            leaderboard, "Average", higher_is_better=True, bottom=want_bottom
                        )
                        leaderboard["Average"] = leaderboard["Average"].round(1)
                        leaderboard["Runs"] = leaderboard["Runs"].astype(int)
                        leaderboard.columns = ["BATSMAN NAME", "RUNS", "BALLS FACED", "DISMISSALS", "AVERAGE"]
                        phase_note = "" if cfg.is_test else f"Phase: <b>{f_overs}</b> | "
                        st.subheader(f"{rank_label} Batters by Average vs {filter_label}")
                        st.markdown(
                            f'<div class="filter-caption">Applied Filters: {phase_note}Minimum <b>{min_balls} Balls Faced</b></div>',
                            unsafe_allow_html=True,
                        )
                        column_configuration = {
                            "BATSMAN NAME": st.column_config.TextColumn(width=200),
                            "RUNS": st.column_config.NumberColumn(alignment="center", width=75),
                            "BALLS FACED": st.column_config.NumberColumn(alignment="center", width=75),
                            "DISMISSALS": st.column_config.NumberColumn(alignment="center", width=75),
                            "AVERAGE": st.column_config.NumberColumn(alignment="center", width=100),
                        }
                        st.dataframe(
                            leaderboard.set_index("BATSMAN NAME"),
                            use_container_width=True,
                            column_config=column_configuration,
                        )
                    else:
                        st.info(
                            f"No batters found matching the minimum requirement threshold of {min_balls} balls faced."
                        )
                else:
                    batter_col = "Batter" if "Batter" in df_filtered.columns else "BatsmanName"

                    leaderboard = df_filtered.groupby(batter_col).agg(
                        Runs=("Runs", "sum"),
                        Balls_Faced=("Ball", "size"),
                        Dismissals=("Wicket", lambda x: sorted(x).count(True)),
                    ).reset_index()

                    leaderboard = leaderboard[leaderboard["Balls_Faced"] >= min_balls]

                    if not leaderboard.empty:
                        leaderboard["Strike Rate"] = (leaderboard["Runs"] / leaderboard["Balls_Faced"]) * 100
                        leaderboard = rank_slice(
                            leaderboard, "Strike Rate", higher_is_better=True, bottom=want_bottom
                        )

                        leaderboard["Strike Rate"] = leaderboard["Strike Rate"].round(1)
                        leaderboard["Runs"] = leaderboard["Runs"].astype(int)

                        leaderboard.columns = ["Batter", "Runs", "Balls faced", "Dismissals", "Strike Rate"]

                        st.subheader(f"{rank_label} Batters by Strike Rate vs {filter_label}")
                        st.markdown(
                            f'<div class="filter-caption">Applied Filters: Phase: <b>{f_overs}</b> | Minimum Requirement: <b>{min_balls} Balls Faced</b></div>',
                            unsafe_allow_html=True,
                        )

                        column_configuration = {
                            "Batter": st.column_config.TextColumn(width=200),
                            "Runs": st.column_config.NumberColumn(alignment="center", width=75),
                            "Balls faced": st.column_config.NumberColumn(alignment="center", width=75),
                            "Dismissals": st.column_config.NumberColumn(alignment="center", width=75),
                            "Strike Rate": st.column_config.NumberColumn(alignment="center", width=100),
                        }

                        st.dataframe(
                            leaderboard.set_index("Batter"),
                            use_container_width=True,
                            column_config=column_configuration,
                        )
                    else:
                        st.info(
                            f"No batters found matching the minimum requirement threshold of {min_balls} balls faced."
                        )
            else:
                st.info("No delivery metrics recorded in the raw data matching this custom query scenario.")

        # --- RENDER ENGINE: PACERS ---
        elif f1 == "PACERS":
            if "BowlerName" in df_raw.columns:
                df_bowler_totals = df_role_base.groupby("BowlerName").agg(Total_Balls=("Runs", "count")).reset_index()

                if f2 in PACERS_METRIC_VIEW_TYPES:
                    if df_filtered.empty:
                        st.info("No delivery metrics recorded for the current filter.")
                    else:
                        d = df_filtered.copy()
                        d["_hit"] = hitting_stumps_mask(d).astype(float)
                        if "ReleaseSpeed" in d.columns:
                            d["_rs"] = pd.to_numeric(d["ReleaseSpeed"], errors="coerce")
                        else:
                            d["_rs"] = np.nan
                        if "Swing" in d.columns:
                            d["_sw"] = pd.to_numeric(d["Swing"], errors="coerce")
                        else:
                            d["_sw"] = np.nan
                        if "Deviation" in d.columns:
                            d["_dev"] = pd.to_numeric(d["Deviation"], errors="coerce")
                        else:
                            d["_dev"] = np.nan
                        if "BounceX" in d.columns:
                            d["_bx"] = pd.to_numeric(d["BounceX"], errors="coerce")
                        else:
                            d["_bx"] = np.nan

                        if f2 == "Avg Speed":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_rs", "mean"))
                        elif f2 == "Avg Swing":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_sw", "mean"))
                        elif f2 == "Avg Seam":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_dev", "mean"))
                        elif f2 == "Avg Length":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_bx", "mean"))
                        elif f2 == "Hitting Stumps %":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_hit", "mean"))
                            g["Val"] = g["Val"] * 100.0
                        else:
                            g = pd.DataFrame(columns=["BowlerName", "Balls", "Val"])

                        g = g.reset_index().rename(columns={"BowlerName": "BOWLER NAME"})
                        g = g[g["Balls"] >= min_balls]
                        if f2 in ("Avg Swing", "Avg Seam"):
                            g["Val"] = g["Val"].abs()
                        if g.empty:
                            st.info(
                                f"No pacers found matching the minimum requirement threshold of {min_balls} balls bowled."
                            )
                        else:
                            g = rank_slice(g, "Val", higher_is_better=True, bottom=want_bottom)
                            g["Val"] = g["Val"].round(1)
                            g.columns = ["BOWLER NAME", "BALLS", f2.upper()]
                            st.subheader(f"{rank_label} Pacers — {f2}")
                            phase_line = "" if cfg.is_test else f'Phase: <b>{f_overs}</b> | '
                            st.markdown(
                                f'<div class="filter-caption">Applied Filters: {phase_line}Minimum Requirement: <b>{min_balls} Balls Bowled</b></div>',
                                unsafe_allow_html=True,
                            )
                            mcol = f2.upper()
                            st.dataframe(
                                g.set_index("BOWLER NAME"),
                                use_container_width=True,
                                column_config={
                                    "BALLS": st.column_config.NumberColumn(alignment="center"),
                                    mcol: st.column_config.NumberColumn(alignment="center"),
                                },
                            )
                elif not df_filtered.empty:
                    leaderboard = df_filtered.groupby("BowlerName").agg(
                        Runs_Conceded=("Runs", "sum"),
                        Balls_Bowled=("Runs", "count"),
                        Wickets=("Wicket", lambda x: sorted(x).count(True))
                    ).reset_index()

                    leaderboard = leaderboard.merge(df_bowler_totals, on="BowlerName", how="left")
                    leaderboard = leaderboard[leaderboard["Balls_Bowled"] >= min_balls]

                    if not leaderboard.empty:
                        if cfg.is_test:
                            leaderboard = leaderboard[leaderboard["Wickets"] > 0]
                            leaderboard["Bowling Strike Rate"] = (
                                leaderboard["Balls_Bowled"] / leaderboard["Wickets"]
                            )
                            leaderboard["Bowling Strike Rate"] = leaderboard["Bowling Strike Rate"].round(2)

                            if f2 == "All":
                                leaderboard = rank_slice(
                                    leaderboard, "Wickets", higher_is_better=True, bottom=want_bottom
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Bowling Strike Rate"]
                                col_titles = ["BOWLER NAME", "BALLS", "WICKETS", "BOWLING STRIKE RATE"]
                                pct_col_name = None
                            elif f2 == "% by Lengths":
                                leaderboard["% of Length"] = (
                                    leaderboard["Balls_Bowled"] / leaderboard["Total_Balls"]
                                ) * 100
                                leaderboard["% of Length"] = leaderboard["% of Length"].round(1)
                                leaderboard = rank_slice(
                                    leaderboard, "% of Length", higher_is_better=True, bottom=want_bottom
                                )
                                final_cols = [
                                    "BowlerName",
                                    "Balls_Bowled",
                                    "Wickets",
                                    "Bowling Strike Rate",
                                    "% of Length",
                                ]
                                col_titles = [
                                    "BOWLER NAME",
                                    "BALLS",
                                    "WICKETS",
                                    "BOWLING STRIKE RATE",
                                    "% OF LENGTH",
                                ]
                                pct_col_name = "% OF LENGTH"
                            elif f2 == "% Balls by Pace":
                                leaderboard["% of Pace Context"] = (
                                    leaderboard["Balls_Bowled"] / leaderboard["Total_Balls"]
                                ) * 100
                                leaderboard["% of Pace Context"] = leaderboard["% of Pace Context"].round(1)
                                leaderboard = rank_slice(
                                    leaderboard,
                                    "% of Pace Context",
                                    higher_is_better=True,
                                    bottom=want_bottom,
                                )
                                final_cols = [
                                    "BowlerName",
                                    "Balls_Bowled",
                                    "Wickets",
                                    "Bowling Strike Rate",
                                    "% of Pace Context",
                                ]
                                col_titles = [
                                    "BOWLER NAME",
                                    "BALLS",
                                    "WICKETS",
                                    "BOWLING STRIKE RATE",
                                    "% OF PACE",
                                ]
                                pct_col_name = "% OF PACE"
                            else:
                                leaderboard = rank_slice(
                                    leaderboard,
                                    "Bowling Strike Rate",
                                    higher_is_better=False,
                                    bottom=want_bottom,
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Bowling Strike Rate"]
                                col_titles = ["BOWLER NAME", "BALLS", "WICKETS", "BOWLING STRIKE RATE"]
                                pct_col_name = None
                        else:
                            leaderboard["Economy"] = (
                                leaderboard["Runs_Conceded"] / leaderboard["Balls_Bowled"]
                            ) * 6
                            leaderboard["Economy"] = leaderboard["Economy"].round(2)

                            if f2 == "All":
                                leaderboard = rank_slice(
                                    leaderboard, "Wickets", higher_is_better=True, bottom=want_bottom
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Economy"]
                                col_titles = ["Bowler", "Balls", "Wickets", "Economy"]
                                pct_col_name = None
                            elif f2 == "% by Lengths":
                                leaderboard["% of Length"] = (
                                    leaderboard["Balls_Bowled"] / leaderboard["Total_Balls"]
                                ) * 100
                                leaderboard["% of Length"] = leaderboard["% of Length"].round(1)
                                leaderboard = rank_slice(
                                    leaderboard, "% of Length", higher_is_better=True, bottom=want_bottom
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Economy", "% of Length"]
                                col_titles = ["Bowler", "Balls", "Wickets", "Economy", "% of Length"]
                                pct_col_name = "% of Length"
                            elif f2 == "% Balls by Pace":
                                leaderboard["% of Pace Context"] = (
                                    leaderboard["Balls_Bowled"] / leaderboard["Total_Balls"]
                                ) * 100
                                leaderboard["% of Pace Context"] = leaderboard["% of Pace Context"].round(1)
                                leaderboard = rank_slice(
                                    leaderboard,
                                    "% of Pace Context",
                                    higher_is_better=True,
                                    bottom=want_bottom,
                                )
                                final_cols = [
                                    "BowlerName",
                                    "Balls_Bowled",
                                    "Wickets",
                                    "Economy",
                                    "% of Pace Context",
                                ]
                                col_titles = ["Bowler", "Balls", "Wickets", "Economy", "% of Pace"]
                                pct_col_name = "% of Pace"
                            else:
                                leaderboard = rank_slice(
                                    leaderboard, "Economy", higher_is_better=False, bottom=want_bottom
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Economy"]
                                col_titles = ["Bowler", "Balls", "Wickets", "Economy"]
                                pct_col_name = None

                        leaderboard = leaderboard[final_cols]
                        leaderboard.columns = col_titles

                        title_suffix = "by Wickets" if f2 == "All" else f"vs {filter_label} ({f2})"
                        st.subheader(f"{rank_label} Pacers' Performance {title_suffix}")
                        phase_line = (
                            ""
                            if cfg.is_test
                            else f'Phase: <b>{f_overs}</b> | '
                        )
                        st.markdown(
                            f'<div class="filter-caption">Applied Filters: {phase_line}Minimum Requirement: <b>{min_balls} Balls Bowled</b></div>',
                            unsafe_allow_html=True,
                        )

                        idx_col = col_titles[0]
                        column_configuration = {
                            idx_col: st.column_config.TextColumn(width=200),
                            col_titles[1]: st.column_config.NumberColumn(alignment="center", width=75),
                            col_titles[2]: st.column_config.NumberColumn(alignment="center", width=75),
                            col_titles[3]: st.column_config.NumberColumn(alignment="center", width=85),
                        }
                        if pct_col_name:
                            column_configuration[pct_col_name] = st.column_config.NumberColumn(
                                alignment="center", width=100, format="%.1f%%"
                            )

                        st.dataframe(
                            leaderboard.set_index(idx_col), use_container_width=True, column_config=column_configuration
                        )
                    else:
                        st.info(f"No pacers found matching the minimum requirement threshold of {min_balls} balls bowled.")
                else:
                    st.info("No delivery metrics recorded in the raw data matching this custom query scenario.")
            else:
                st.error("Column tracking identifier 'BowlerName' missing in uploaded sheet format structure.")

        # --- RENDER ENGINE: SPINNERS ---
        elif f1 == "SPINNERS":
            if "BowlerName" in df_raw.columns:
                df_bowler_totals = df_role_base.groupby("BowlerName").agg(Total_Balls=("Runs", "count")).reset_index()

                if f2 in SPINNERS_METRIC_VIEW_TYPES:
                    if df_filtered.empty:
                        st.info("No delivery metrics recorded for the current filter.")
                    else:
                        d = df_filtered.copy()
                        d["_hit"] = hitting_stumps_mask(d).astype(float)
                        if "ReleaseSpeed" in d.columns:
                            d["_rs"] = pd.to_numeric(d["ReleaseSpeed"], errors="coerce")
                        else:
                            d["_rs"] = np.nan
                        if "Swing" in d.columns:
                            d["_sw"] = pd.to_numeric(d["Swing"], errors="coerce")
                        else:
                            d["_sw"] = np.nan
                        if "Deviation" in d.columns:
                            d["_dev"] = pd.to_numeric(d["Deviation"], errors="coerce")
                        else:
                            d["_dev"] = np.nan
                        if "BounceX" in d.columns:
                            d["_bx"] = pd.to_numeric(d["BounceX"], errors="coerce")
                        else:
                            d["_bx"] = np.nan

                        if f2 == "Avg Speed":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_rs", "mean"))
                        elif f2 == "Avg Drift":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_sw", "mean"))
                        elif f2 == "Avg Turn":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_dev", "mean"))
                        elif f2 == "Avg Length":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_bx", "mean"))
                        elif f2 == "Hitting Stumps %":
                            g = d.groupby("BowlerName").agg(Balls=("Runs", "count"), Val=("_hit", "mean"))
                            g["Val"] = g["Val"] * 100.0
                        else:
                            g = pd.DataFrame()

                        g = g.reset_index().rename(columns={"BowlerName": "BOWLER NAME"})
                        g = g[g["Balls"] >= min_balls]
                        if f2 in ("Avg Drift", "Avg Turn"):
                            g["Val"] = g["Val"].abs()
                        if g.empty:
                            st.info(
                                f"No spinners found matching the minimum requirement threshold of {min_balls} balls bowled."
                            )
                        else:
                            g = rank_slice(g, "Val", higher_is_better=True, bottom=want_bottom)
                            g["Val"] = g["Val"].round(1)
                            g.columns = ["BOWLER NAME", "BALLS", f2.upper()]
                            st.subheader(f"{rank_label} Spinners — {f2}")
                            phase_line = "" if cfg.is_test else f'Phase: <b>{f_overs}</b> | '
                            st.markdown(
                                f'<div class="filter-caption">Applied Filters: {phase_line}Minimum Requirement: <b>{min_balls} Balls Bowled</b></div>',
                                unsafe_allow_html=True,
                            )
                            mcol = f2.upper()
                            st.dataframe(
                                g.set_index("BOWLER NAME"),
                                use_container_width=True,
                                column_config={
                                    "BALLS": st.column_config.NumberColumn(alignment="center"),
                                    mcol: st.column_config.NumberColumn(alignment="center"),
                                },
                            )
                elif not df_filtered.empty:
                    leaderboard = df_filtered.groupby("BowlerName").agg(
                        Runs_Conceded=("Runs", "sum"),
                        Balls_Bowled=("Runs", "count"),
                        Wickets=("Wicket", lambda x: sorted(x).count(True))
                    ).reset_index()

                    leaderboard = leaderboard.merge(df_bowler_totals, on="BowlerName", how="left")
                    leaderboard = leaderboard[leaderboard["Balls_Bowled"] >= min_balls]

                    if not leaderboard.empty:
                        if cfg.is_test:
                            leaderboard["Bowling Strike Rate"] = np.where(
                                leaderboard["Wickets"] > 0,
                                leaderboard["Balls_Bowled"] / leaderboard["Wickets"],
                                np.nan,
                            )
                            leaderboard["Bowling Strike Rate"] = leaderboard["Bowling Strike Rate"].round(2)

                            if f2 == "All":
                                leaderboard = rank_slice(
                                    leaderboard, "Wickets", higher_is_better=True, bottom=want_bottom
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Bowling Strike Rate"]
                                col_titles = ["BOWLER NAME", "BALLS", "WICKETS", "BOWLING STRIKE RATE"]
                                pct_col_name = None
                            elif f2 in ["% by Lengths", "Turn Direction"]:
                                leaderboard["% Metric"] = (
                                    leaderboard["Balls_Bowled"] / leaderboard["Total_Balls"]
                                ) * 100
                                leaderboard["% Metric"] = leaderboard["% Metric"].round(1)
                                leaderboard = rank_slice(
                                    leaderboard, "% Metric", higher_is_better=True, bottom=want_bottom
                                )
                                final_cols = [
                                    "BowlerName",
                                    "Balls_Bowled",
                                    "Wickets",
                                    "Bowling Strike Rate",
                                    "% Metric",
                                ]
                                col_titles = [
                                    "BOWLER NAME",
                                    "BALLS",
                                    "WICKETS",
                                    "BOWLING STRIKE RATE",
                                    "% METRIC",
                                ]
                                pct_col_name = "% METRIC"
                            else:
                                leaderboard = rank_slice(
                                    leaderboard,
                                    "Bowling Strike Rate",
                                    higher_is_better=False,
                                    bottom=want_bottom,
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Bowling Strike Rate"]
                                col_titles = ["BOWLER NAME", "BALLS", "WICKETS", "BOWLING STRIKE RATE"]
                                pct_col_name = None
                        else:
                            leaderboard["Economy"] = (
                                leaderboard["Runs_Conceded"] / leaderboard["Balls_Bowled"]
                            ) * 6
                            leaderboard["Economy"] = leaderboard["Economy"].round(2)

                            if f2 == "All":
                                leaderboard = rank_slice(
                                    leaderboard, "Wickets", higher_is_better=True, bottom=want_bottom
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Economy"]
                                col_titles = ["Bowler", "Balls", "Wickets", "Economy"]
                                pct_col_name = None
                            elif f2 in ["% by Lengths", "Turn Direction"]:
                                leaderboard["% Metric"] = (
                                    leaderboard["Balls_Bowled"] / leaderboard["Total_Balls"]
                                ) * 100
                                leaderboard["% Metric"] = leaderboard["% Metric"].round(1)
                                leaderboard = rank_slice(
                                    leaderboard, "% Metric", higher_is_better=True, bottom=want_bottom
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Economy", "% Metric"]
                                col_titles = ["Bowler", "Balls", "Wickets", "Economy", "% Metric"]
                                pct_col_name = "% Metric"
                            else:
                                leaderboard = rank_slice(
                                    leaderboard, "Economy", higher_is_better=False, bottom=want_bottom
                                )
                                final_cols = ["BowlerName", "Balls_Bowled", "Wickets", "Economy"]
                                col_titles = ["Bowler", "Balls", "Wickets", "Economy"]
                                pct_col_name = None

                        leaderboard = leaderboard[final_cols]
                        leaderboard.columns = col_titles

                        title_suffix = "by Wickets" if f2 == "All" else f"vs {filter_label} ({f2})"
                        st.subheader(f"{rank_label} Spinners Performance {title_suffix}")
                        phase_line = "" if cfg.is_test else f'Phase: <b>{f_overs}</b> | '
                        st.markdown(
                            f'<div class="filter-caption">Applied Filters: {phase_line}Minimum Requirement: <b>{min_balls} Balls Bowled</b></div>',
                            unsafe_allow_html=True,
                        )

                        idx_col = col_titles[0]
                        column_configuration = {
                            idx_col: st.column_config.TextColumn(width=200),
                            col_titles[1]: st.column_config.NumberColumn(alignment="center", width=75),
                            col_titles[2]: st.column_config.NumberColumn(alignment="center", width=75),
                            col_titles[3]: st.column_config.NumberColumn(alignment="center", width=85),
                        }
                        if pct_col_name:
                            column_configuration[pct_col_name] = st.column_config.NumberColumn(
                                alignment="center", width=100, format="%.1f%%"
                            )

                        st.dataframe(
                            leaderboard.set_index(idx_col), use_container_width=True, column_config=column_configuration
                        )
                    else:
                        st.info(f"No spinners found matching the minimum requirement threshold of {min_balls} balls bowled.")
                else:
                    st.info("No delivery metrics recorded in the raw data matching this custom query scenario.")
            else:
                st.error("Column tracking identifier 'BowlerName' missing in uploaded sheet format structure.")
