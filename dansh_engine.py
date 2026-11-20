from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cricket_config import (
    apply_match_phase_filter,
    get_pitch_bins,
    hitting_stumps_mask,
    match_phase_options,
    resolve_format,
    seam_speed_group,
    spin_speed_group,
)


REQUIRED = {
    "Innings", "Over", "Ball", "BowlerName", "BatsmanName", "BowlingTeam",
    "BattingTeam", "IsBowlerRightHanded", "IsBatsmanRightHanded", "Wicket",
    "Runs", "BounceX", "BounceY", "InterceptionX", "InterceptionY", "InterceptionZ",
    "CreaseY", "CreaseZ", "StumpsY", "StumpsZ", "ReleaseY", "ReleaseZ",
    "LandingX", "LandingY", "ReleaseSpeed", "Deviation", "DeliveryType", "Swing",
    "Ground", "Tour", "Year", "Match",
}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = d.columns.astype(str).str.strip()

    for c in [
        "Runs", "ReleaseSpeed", "BounceX", "BounceY", "InterceptionX", "InterceptionY",
        "InterceptionZ", "CreaseY", "CreaseZ", "StumpsY", "StumpsZ", "ReleaseY",
        "ReleaseZ", "LandingX", "LandingY", "Deviation", "Swing", "Over", "Ball", "Year",
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "Wicket" in d.columns:
        if d["Wicket"].dtype == object:
            d["Wicket"] = d["Wicket"].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
            ).fillna(False)
        else:
            d["Wicket"] = d["Wicket"].astype(bool)

    # Hawk-Eye names are authoritative; case differences are not different players.
    for c in ["BowlerName", "BatsmanName", "BowlingTeam", "BattingTeam", "Ground", "Tour"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()
            if c in {"BowlerName", "BatsmanName"}:
                d[c] = d[c].str.upper()

    if "DeliveryType" in d.columns:
        d["DeliveryType"] = d["DeliveryType"].astype(str).str.strip().str.title()
        # User rule: Medium is Seam.
        d.loc[d["DeliveryType"].str.lower().eq("medium"), "DeliveryType"] = "Seam"

    return d


def add_derived_columns(df: pd.DataFrame, cfg, speed_unit: str = "kph") -> pd.DataFrame:
    d = df.copy()

    # The user-defined analytical validity window for pitch length.
    bx = pd.to_numeric(d["BounceX"], errors="coerce")
    d["_BounceX_Valid"] = bx.between(-10, 18, inclusive="both")

    def classify_pitch(row):
        typ = row.get("DeliveryType")
        x = row.get("BounceX")
        if pd.isna(x) or not (-10 <= float(x) <= 18):
            return np.nan
        bins = get_pitch_bins(typ, cfg)
        for name, bounds in bins.items():
            lo, hi = bounds
            if float(x) >= lo and float(x) < hi:
                return name
        return np.nan

    d["PitchLength"] = d.apply(classify_pitch, axis=1)

    # Match the existing cricket_config phase implementation exactly.
    phase_values = []
    for over in d["Over"]:
        if pd.isna(over) or cfg.is_test:
            phase_values.append("All")
        elif cfg.is_odi:
            if over < 11:
                phase_values.append("Powerplay (1-10)")
            elif over < 41:
                phase_values.append("Middle (11-40)")
            else:
                phase_values.append("Death (41-50)")
        else:
            if over < 7:
                phase_values.append("Powerplay (1-6)")
            elif over < 17:
                phase_values.append("Middle (7-16)")
            else:
                phase_values.append("Death (17-20)")
    d["Phase"] = phase_values

    d["ReleaseSide"] = np.select(
        [d["ReleaseY"] < 0, d["ReleaseY"] > 0],
        ["LEFT", "RIGHT"],
        default="CENTER",
    )

    d["StumpChannel"] = np.where(hitting_stumps_mask(d), "HITTING", "MISSING")

    # Existing dashboard crease-zone geometry, adjusted for batter handedness.
    cy = pd.to_numeric(d["CreaseY"], errors="coerce")
    rh = d["IsBatsmanRightHanded"].astype(bool)
    m = np.where(rh, 1.0, -1.0)
    lateral = cy * m
    d["CreaseZone"] = np.select(
        [
            lateral < -0.415,
            (lateral >= -0.415) & (lateral < -0.115),
            (lateral >= -0.115) & (lateral <= 0.115),
            lateral > 0.115,
        ],
        ["WAY OUTSIDE OFF", "OUTSIDE OFF", "STUMPS", "LEG"],
        default="UNKNOWN",
    )

    # Speed validity per user's global analytical rule.
    rs = pd.to_numeric(d["ReleaseSpeed"], errors="coerce")
    d["_SpeedValid"] = rs.between(40, 150, inclusive="both")
    d["_MovementValid"] = pd.to_numeric(d["Deviation"], errors="coerce").between(-15, 15, inclusive="both")
    d["_SwingValid"] = pd.to_numeric(d["Swing"], errors="coerce").between(-15, 15, inclusive="both")

    def speed_group(row):
        if not row.get("_SpeedValid", False):
            return np.nan
        s = row.get("ReleaseSpeed")
        if pd.isna(s):
            return np.nan
        typ = str(row.get("DeliveryType", "")).lower()
        return (spin_speed_group(float(s), cfg, speed_unit) if typ == "spin"
                else seam_speed_group(float(s), cfg, speed_unit))

    d["SpeedGroup"] = d.apply(speed_group, axis=1)
    return d


def filter_plan(df: pd.DataFrame, plan: Dict[str, Any], cfg, speed_unit: str) -> pd.DataFrame:
    d = df.copy()
    f = plan.get("filters") or {}

    def apply_exact(col: str, key: str):
        nonlocal d
        val = f.get(key)
        if val is None or val == "" or val == "All":
            return
        if col not in d.columns:
            return
        if isinstance(val, list):
            vals = [str(x).strip().upper() for x in val]
            d = d[d[col].astype(str).str.upper().isin(vals)]
        else:
            target = str(val).strip().upper()
            d = d[d[col].astype(str).str.upper() == target]

    for col, key in [
        ("BowlerName", "bowler"), ("BatsmanName", "batter"),
        ("BowlingTeam", "bowling_team"), ("BattingTeam", "batting_team"),
        ("Ground", "ground"), ("Tour", "tour"), ("Match", "match"),
    ]:
        apply_exact(col, key)

    year = f.get("year")
    if year not in (None, "", "All") and "Year" in d.columns:
        years = year if isinstance(year, list) else [year]
        d = d[d["Year"].isin(pd.to_numeric(years, errors="coerce"))]

    delivery = f.get("delivery_type")
    if delivery not in (None, "", "All") and "DeliveryType" in d.columns:
        delivery = "Seam" if str(delivery).lower() == "medium" else str(delivery).title()
        d = d[d["DeliveryType"] == delivery]

    phase = f.get("phase")
    if phase not in (None, "", "All"):
        d = apply_match_phase_filter(d, str(phase), cfg)

    hand = f.get("batter_hand")
    if hand not in (None, "", "All"):
        want = str(hand).lower()
        if "IsBatsmanRightHanded" in d.columns:
            d = d[d["IsBatsmanRightHanded"] == (want in {"right", "r", "rhb", "true"})]

    hand = f.get("bowler_hand")
    if hand not in (None, "", "All"):
        want = str(hand).lower()
        if "IsBowlerRightHanded" in d.columns:
            d = d[d["IsBowlerRightHanded"] == (want in {"right", "r", "rhb", "true"})]

    pitch = f.get("pitch_length")
    if pitch not in (None, "", "All"):
        d = d[d["PitchLength"].astype(str).str.upper() == str(pitch).upper()]

    speed = f.get("speed_group")
    if speed not in (None, "", "All"):
        d = d[d["SpeedGroup"].astype(str) == str(speed)]

    release_side = f.get("release_side")
    if release_side not in (None, "", "All"):
        d = d[d["ReleaseSide"].astype(str).str.upper() == str(release_side).upper()]

    stump = f.get("stump_channel")
    if stump not in (None, "", "All"):
        d = d[d["StumpChannel"].astype(str).str.upper() == str(stump).upper()]

    return d


def metrics(df: pd.DataFrame, role: str = "auto") -> Dict[str, Any]:
    if df is None or df.empty:
        return {"deliveries": 0, "runs": 0, "wickets": 0, "balls": 0}

    balls = int(len(df))
    runs = float(pd.to_numeric(df["Runs"], errors="coerce").fillna(0).sum())
    wickets = int(df["Wicket"].astype(bool).sum())
    out = {"deliveries": balls, "runs": runs, "wickets": wickets, "balls": balls}

    # Existing leaderboard/page formulas.
    out["economy"] = (runs / balls) * 6 if balls else None
    out["bowling_average"] = runs / wickets if wickets else None
    out["bowling_strike_rate"] = balls / wickets if wickets else None
    out["batting_strike_rate"] = (runs / balls) * 100 if balls else None
    out["batting_average"] = runs / wickets if wickets else None
    out["dot_pct"] = float((df["Runs"] == 0).mean() * 100) if balls else None
    out["boundary_pct"] = float(df["Runs"].isin([4, 6]).mean() * 100) if balls else None
    out["fours"] = int((df["Runs"] == 4).sum())
    out["sixes"] = int((df["Runs"] == 6).sum())
    out["hitting_stumps_pct"] = float((df["StumpChannel"] == "HITTING").mean() * 100) if balls else None

    if "ReleaseSpeed" in df:
        valid = pd.to_numeric(df["ReleaseSpeed"], errors="coerce")
        valid = valid[valid.between(40, 150)]
        out["avg_speed"] = float(valid.mean()) if len(valid) else None
    if "Swing" in df:
        v = pd.to_numeric(df["Swing"], errors="coerce")
        v = v[v.between(-15, 15)]
        out["avg_swing"] = float(v.mean()) if len(v) else None
    if "Deviation" in df:
        v = pd.to_numeric(df["Deviation"], errors="coerce")
        v = v[v.between(-15, 15)]
        out["avg_deviation"] = float(v.mean()) if len(v) else None
        out["avg_movement_abs"] = float(v.abs().mean()) if len(v) else None
    if "BounceX" in df:
        v = pd.to_numeric(df["BounceX"], errors="coerce")
        v = v[v.between(-10, 18)]
        out["avg_length"] = float(v.mean()) if len(v) else None
    return out


def group_analysis(df: pd.DataFrame, group_by: str, role: str, top_n: int = 10) -> pd.DataFrame:
    mapping = {
        "pitch_length": "PitchLength", "phase": "Phase", "speed_group": "SpeedGroup",
        "release_side": "ReleaseSide", "crease_zone": "CreaseZone", "stump_channel": "StumpChannel",
        "batter_hand": "IsBatsmanRightHanded", "bowler_hand": "IsBowlerRightHanded",
        "ground": "Ground", "bowler": "BowlerName", "batter": "BatsmanName",
        "batting_team": "BattingTeam", "bowling_team": "BowlingTeam", "year": "Year",
    }
    col = mapping.get(group_by, group_by)
    if col not in df.columns:
        return pd.DataFrame()

    g = df.groupby(col, dropna=False).agg(
        Balls=("Runs", "size"), Runs=("Runs", "sum"), Wickets=("Wicket", "sum")
    ).reset_index()
    g["Economy"] = np.where(g["Balls"] > 0, g["Runs"] / g["Balls"] * 6, np.nan)
    g["Bowling Average"] = np.where(g["Wickets"] > 0, g["Runs"] / g["Wickets"], np.nan)
    g["Bowling SR"] = np.where(g["Wickets"] > 0, g["Balls"] / g["Wickets"], np.nan)
    g["Batting SR"] = np.where(g["Balls"] > 0, g["Runs"] / g["Balls"] * 100, np.nan)
    g["Batting Average"] = np.where(g["Wickets"] > 0, g["Runs"] / g["Wickets"], np.nan)
    dot = df.groupby(col)["Runs"].apply(lambda s: (s == 0).mean() * 100)
    g["Dot %"] = g[col].map(dot)
    return g.sort_values("Balls", ascending=False).head(top_n)



def matchup_analysis(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.groupby(["BowlerName", "BatsmanName"], dropna=False).agg(
        Balls=("Runs", "size"), Runs=("Runs", "sum"), Wickets=("Wicket", "sum")
    ).reset_index()
    g["Economy"] = np.where(g["Balls"] > 0, g["Runs"] / g["Balls"] * 6, np.nan)
    g["Batting SR"] = np.where(g["Balls"] > 0, g["Runs"] / g["Balls"] * 100, np.nan)
    g["Bowling Average"] = np.where(g["Wickets"] > 0, g["Runs"] / g["Wickets"], np.nan)
    g["Bowling SR"] = np.where(g["Wickets"] > 0, g["Balls"] / g["Wickets"], np.nan)
    return g.sort_values(["Wickets", "Balls", "Runs"], ascending=[False, False, True]).head(top_n)

def resolve_subject_name(query: str, values: List[str]) -> Optional[str]:
    q = query.upper()
    exact = [v for v in values if str(v).upper() == q]
    if exact:
        return exact[0]
    candidates = [v for v in values if str(v).upper() in q]
    if candidates:
        return sorted(candidates, key=lambda x: len(str(x)), reverse=True)[0]
    # token overlap for modest spelling variation, without fuzzy rewriting names.
    q_tokens = set(re.findall(r"[A-Z0-9]+", q))
    best = None
    best_score = 0
    for v in values:
        toks = set(re.findall(r"[A-Z0-9]+", str(v).upper()))
        score = len(q_tokens & toks)
        if score > best_score:
            best, best_score = v, score
    return best if best_score >= 2 else None


def confidence_meter(n: int) -> Tuple[str, int]:
    # A transparent sample-size heuristic, not statistical significance.
    if n < 10:
        return "VERY LOW — very less sample size", 15
    if n < 25:
        return "LOW", 35
    if n < 50:
        return "MODERATE", 55
    if n < 100:
        return "GOOD", 72
    if n < 250:
        return "HIGH", 85
    return "VERY HIGH", 95
