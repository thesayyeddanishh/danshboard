"""
Shared format configuration for consolidated cricket dashboards.

Session state:
  st.session_state["cricket_format"] — one of FORMAT_KEYS (set on Home page).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

FORMAT_KEYS: Tuple[str, ...] = (
    "men_t20i",
    "women_t20i",
    "men_odi",
    "women_odi",
    "men_test",
    "men_test_aus",
)

FORMAT_LABELS: Dict[str, str] = {
    "men_t20i": "Men's T20",
    "women_t20i": "Women's T20",
    "men_odi": "Men's ODI",
    "women_odi": "Women's ODI",
    "men_test": "Men's Red Ball",
    "men_test_aus": "Men's Red Ball - AUS",
}


@dataclass(frozen=True)
class FormatConfig:
    key: str
    label: str
    is_test: bool
    is_womens: bool
    is_odi: bool
    sidebar_title: str
    sidebar_sub: str


def resolve_format(key: Optional[str]) -> FormatConfig:
    k = (key or "men_t20i").strip()
    if k not in FORMAT_KEYS:
        k = "men_t20i"

    is_test = k in ("men_test", "men_test_aus")
    is_womens = k.startswith("women_")
    is_odi = k.endswith("_odi")
    
    if is_test:
        title = "Men's (AUS)" if k == "men_test_aus" else "Men's"
        sub = "Red Ball (Test — AUS bins)" if k == "men_test_aus" else "Red Ball (Test)"
    elif is_womens:
        title, sub = ("Women's", "ODI") if is_odi else ("Women's", "T20I")
    else:
        title, sub = ("Men's", "ODI") if is_odi else ("Men's", "T20I")
    return FormatConfig(
        key=k,
        label=FORMAT_LABELS[k],
        is_test=is_test,
        is_womens=is_womens,
        is_odi=is_odi,
        sidebar_title=title,
        sidebar_sub=sub,
    )


# --- Match phase (Leaderboard & any Over-based filters) ---
# Over column is treated as 0-based over index (consistent with existing T20 filters).

PHASE_ALL = "All"


def match_phase_options(cfg: FormatConfig) -> List[str]:
    if cfg.is_test:
        return [PHASE_ALL]
    if cfg.is_odi:
        return [
            PHASE_ALL,
            "Powerplay (1-10)",
            "Middle (11-40)",
            "Death (41-50)",
        ]
    return [
        PHASE_ALL,
        "Powerplay (1-6)",
        "Middle (7-16)",
        "Death (17-20)",
    ]


def apply_match_phase_filter(df: pd.DataFrame, phase: str, cfg: FormatConfig) -> pd.DataFrame:
    if df is None or df.empty or "Over" not in df.columns:
        return df
    if phase == PHASE_ALL or cfg.is_test:
        return df
    over = pd.to_numeric(df["Over"], errors="coerce")
    d = df.copy()
    d["_over_num"] = over
    d = d.dropna(subset=["_over_num"])
    if cfg.is_odi:
        if phase == "Powerplay (1-10)":
            d = d[d["_over_num"] < 11]
        elif phase == "Middle (11-40)":
            d = d[(d["_over_num"] >= 11) & (d["_over_num"] < 41)]
        elif phase == "Death (41-50)":
            d = d[(d["_over_num"] >= 41) & (d["_over_num"] < 50)]
    else:
        if phase == "Powerplay (1-6)":
            d = d[d["_over_num"] < 7]
        elif phase == "Middle (7-16)":
            d = d[(d["_over_num"] >= 7) & (d["_over_num"] < 17)]
        elif phase == "Death (17-20)":
            d = d[d["_over_num"] >= 17]
    return d.drop(columns=["_over_num"])


# --- Pitch bins (Seam / Spin) for charts ---

def seam_pitch_bins_men_t20() -> Dict[str, List[float]]:
    return {
        "Full Toss": [-4, 0.5],
        "Yorker": [0.5, 2.5],
        "The Slot": [2.5, 5.8],
        "Length": [5.8, 8],
        "Short": [8, 10],
        "Bouncer": [10, 16],
    }


def seam_pitch_bins_women_t20() -> Dict[str, List[float]]:
    # Women's white-ball: short band runs through 16 m (no separate "Bouncer" bucket)
    return {
        "Full Toss": [-5, 0.9],
        "Yorker": [0.9, 2.8],
        "The Slot": [2.8, 5.5],
        "Length": [5.5, 8],
        "Short": [8, 16],
    }


def seam_pitch_bins_test() -> Dict[str, List[float]]:
    return {
        "Full": [-2, 5.8],
        "Length": [5.8, 8],
        "Short": [8, 10],
        "Bouncer": [10, 16],
    }

def seam_pitch_bins_test_aus() -> Dict[str, List[float]]:
    return {
        "Full": [-2, 5],
        "Length": [5, 7],
        "Short": [7, 10],
        "Bouncer": [10, 16],
    }


def spin_pitch_bins_men_t20() -> Dict[str, List[float]]:
    return {
        "OP": [-2, 2.8],
        "Full": [2.8, 4.4],
        "Good": [4.4, 6.2],
        "Short": [6.2, 15.0],
    }


def spin_pitch_bins_women_t20() -> Dict[str, List[float]]:
    return {
        "OP": [-2, 2.5],
        "Full": [2.5, 4],
        "Good": [4, 5.7],
        "Short": [5.7, 12],
    }


def spin_pitch_bins_test() -> Dict[str, List[float]]:
    return {
        "OP": [-2, 2.8],
        "Full": [2.8, 4.4],
        "Good": [4.4, 6.2],
        "Short": [6.2, 15.0],
    }

def get_pitch_bins(delivery_type: str, cfg: FormatConfig) -> Dict[str, List[float]]:
    # --- New Logic for AUS ---
    if cfg.key == "men_test_aus":
        if delivery_type == "Seam":
            return seam_pitch_bins_test_aus()
        if delivery_type == "Spin":
            return spin_pitch_bins_test()
        return {}

    if cfg.is_test:
        if delivery_type == "Seam":
            return seam_pitch_bins_test()
        if delivery_type == "Spin":
            return spin_pitch_bins_test()
        return {}
    if cfg.is_womens:
        if delivery_type == "Seam":
            return seam_pitch_bins_women_t20()
        if delivery_type == "Spin":
            return spin_pitch_bins_women_t20()
        return {}
    if delivery_type == "Seam":
        return seam_pitch_bins_men_t20()
    if delivery_type == "Spin":
        return spin_pitch_bins_men_t20()
    return {}


def ordered_seam_keys(cfg: FormatConfig) -> List[str]:
    if cfg.is_test:
        return ["Full", "Length", "Short", "Bouncer"]
    if cfg.is_womens:
        return ["Full Toss", "Yorker", "The Slot", "Length", "Short"]
    return ["Full Toss", "Yorker", "The Slot", "Length", "Short", "Bouncer"]


def ordered_spin_keys(cfg: FormatConfig) -> List[str]:
    return ["OP", "Full", "Good", "Short"]


# --- Speed buckets (Batters / Pacers charts) ---

def _normalize_speed_unit(unit: Optional[str]) -> str:
    return "mph" if (unit or "kph").strip().lower() == "mph" else "kph"


def _seam_speed_bounds(cfg: FormatConfig, unit: str) -> Tuple[int, int]:
    """(lo, hi) thresholds for seam/pace speed groups per gender and unit."""
    u = _normalize_speed_unit(unit)
    if cfg.is_womens:
        return (65, 75) if u == "mph" else (105, 118)
    return (75, 85) if u == "mph" else (125, 140)


def _spin_speed_bounds(cfg: FormatConfig, unit: str) -> Tuple[int, int]:
    """(lo, hi) thresholds for spin speed groups per gender and unit."""
    u = _normalize_speed_unit(unit)
    if cfg.is_womens:
        return (43, 52) if u == "mph" else (70, 85)
    return (52, 60) if u == "mph" else (85, 95)


def seam_speed_group(speed: float, cfg: FormatConfig, unit: str = "kph") -> str:
    lo, hi = _seam_speed_bounds(cfg, unit)
    if speed < lo:
        return f"<{lo}"
    if lo <= speed <= hi:
        return f"{lo}-{hi}"
    return f"{hi}+"


def seam_speed_ordered_groups(cfg: FormatConfig, unit: str = "kph") -> List[str]:
    lo, hi = _seam_speed_bounds(cfg, unit)
    return [f"{hi}+", f"{lo}-{hi}", f"<{lo}"]


def spin_speed_group(speed: float, cfg: FormatConfig, unit: str = "kph") -> str:
    lo, hi = _spin_speed_bounds(cfg, unit)
    if speed < lo:
        return f"<{lo}"
    if lo <= speed <= hi:
        return f"{lo}-{hi}"
    return f"{hi}+"


def spin_speed_ordered_groups(cfg: FormatConfig, unit: str = "kph") -> List[str]:
    lo, hi = _spin_speed_bounds(cfg, unit)
    return [f"{hi}+", f"{lo}-{hi}", f"<{lo}"]


def pacer_effectiveness_seam_order(cfg: FormatConfig, unit: str = "kph") -> List[str]:
    """Y-axis order for pacer 3-col speed chart (matches original apps)."""
    lo, hi = _seam_speed_bounds(cfg, unit)
    return [f"<{lo}", f"{lo}-{hi}", f"{hi}+"]


# --- Leaderboard length / pace thresholds ---

def leaderboard_batter_length_options(cfg: FormatConfig) -> List[str]:
    if cfg.is_test:
        return ["All", "FULL", "LENGTH", "SHORT", "BOUNCER"]
    base = ["All", "FULL TOSS", "YORKER", "THE SLOT", "LENGTH", "SHORT"]
    if cfg.is_womens:
        return base
    return base + ["BOUNCER"]


def leaderboard_batter_pace_options(cfg: FormatConfig, unit: str = "kph") -> Tuple[str, str]:
    lo, hi = _seam_speed_bounds(cfg, unit)
    return f"Above {hi}", f"Below {lo}"


def leaderboard_pacer_pace_range_labels(cfg: FormatConfig, unit: str = "kph") -> Tuple[str, str]:
    return leaderboard_batter_pace_options(cfg, unit)


def filter_batter_length(df: pd.DataFrame, f3: str, cfg: FormatConfig) -> pd.DataFrame:
    """Leaderboard: seam length bucket for batters (BounceX thresholds)."""
    if f3 == "All":
        return df.copy()
    bx = df["BounceX"]
    # Men's Red Ball (AUS) must come before generic is_test — both have is_test=True.
    if cfg.key == "men_test_aus":
        if f3 == "FULL":
            return df[bx < 5.0].copy()
        if f3 == "LENGTH":
            return df[(bx >= 5.0) & (bx < 7.0)].copy()
        if f3 == "SHORT":
            return df[(bx >= 7.0) & (bx < 10.0)].copy()
        if f3 == "BOUNCER":
            return df[bx >= 10.0].copy()
        return df.copy()

    if cfg.is_test:
        if f3 == "FULL":
            return df[bx < 5.8].copy()
        if f3 == "LENGTH":
            return df[(bx >= 5.8) & (bx < 8.0)].copy()
        if f3 == "SHORT":
            return df[(bx >= 8.0) & (bx < 10.0)].copy()
        if f3 == "BOUNCER":
            return df[bx >= 10.0].copy()
        return df.copy()

    if cfg.is_womens:
        if f3 == "FULL TOSS":
            return df[bx < 0.9].copy()
        if f3 == "YORKER":
            return df[(bx >= 0.9) & (bx < 2.8)].copy()
        if f3 == "THE SLOT":
            return df[(bx >= 2.8) & (bx < 5.5)].copy()
        if f3 == "LENGTH":
            return df[(bx >= 5.5) & (bx < 8.0)].copy()
        if f3 == "SHORT":
            return df[(bx >= 8.0) & (bx < 16.0)].copy()
        return df.copy()
    if f3 == "FULL TOSS":
        return df[bx < 0.5].copy()
    if f3 == "YORKER":
        return df[(bx >= 0.5) & (bx < 2.5)].copy()
    if f3 == "THE SLOT":
        return df[(bx >= 2.5) & (bx < 5.8)].copy()
    if f3 == "LENGTH":
        return df[(bx >= 5.8) & (bx < 8.0)].copy()
    if f3 == "SHORT":
        return df[(bx >= 8.0) & (bx < 10.0)].copy()
    if f3 == "BOUNCER":
        return df[bx >= 10.0].copy()
    return df.copy()


def filter_pacer_length(df: pd.DataFrame, f3: str, cfg: FormatConfig) -> pd.DataFrame:
    """Leaderboard: seam length for pacers (includes All for Test)."""
    if f3 == "All":
        return df.copy()
    bx = df["BounceX"]
    if cfg.is_test:
        return filter_batter_length(df, f3, cfg)
    if cfg.is_womens:
        if f3 == "FULL TOSS":
            return df[bx < 0.9].copy()
        if f3 == "YORKER":
            return df[(bx >= 0.9) & (bx < 2.8)].copy()
        if f3 == "THE SLOT":
            return df[(bx >= 2.8) & (bx < 5.5)].copy()
        if f3 == "LENGTH":
            return df[(bx >= 5.5) & (bx < 8.0)].copy()
        if f3 == "SHORT":
            return df[(bx >= 8.0) & (bx < 16.0)].copy()
        return df.copy()
    if f3 == "FULL TOSS":
        return df[bx < 0.5].copy()
    if f3 == "YORKER":
        return df[(bx >= 0.5) & (bx < 2.5)].copy()
    if f3 == "THE SLOT":
        return df[(bx >= 2.5) & (bx < 5.8)].copy()
    if f3 == "LENGTH":
        return df[(bx >= 5.8) & (bx < 8.0)].copy()
    if f3 == "SHORT":
        return df[(bx >= 8.0) & (bx < 10.0)].copy()
    if f3 == "BOUNCER":
        return df[bx >= 10.0].copy()
    return df.copy()


def filter_batter_pace(df: pd.DataFrame, f3: str, cfg: FormatConfig, unit: str = "kph") -> pd.DataFrame:
    lo, hi = _seam_speed_bounds(cfg, unit)
    hi_label, lo_label = f"Above {hi}", f"Below {lo}"
    rs = df["ReleaseSpeed"]
    if f3 == hi_label:
        return df[rs > hi].copy()
    if f3 == lo_label:
        return df[rs < lo].copy()
    return df.copy()


def filter_pacer_pace(df: pd.DataFrame, f3: str, cfg: FormatConfig, unit: str = "kph") -> pd.DataFrame:
    return filter_batter_pace(df, f3, cfg, unit)


PACERS_METRIC_VIEW_TYPES: Tuple[str, ...] = (
    "Avg Speed",
    "Avg Swing",
    "Avg Seam",
    "Avg Length",
    "Hitting Stumps %",
)

SPINNERS_METRIC_VIEW_TYPES: Tuple[str, ...] = (
    "Avg Speed",
    "Avg Drift",
    "Avg Turn",
    "Avg Length",
    "Hitting Stumps %",
)


def pacer_view_types(cfg: FormatConfig) -> List[str]:
    base = (
        [
            "All",
            "Bowling Strike Rate By Length",
            "% by Lengths",
            "Bowling Average by Pace",
            "% Balls by Pace",
        ]
        if cfg.is_test
        else ["All", "Economy By Length", "% by Lengths", "Economy by Pace", "% Balls by Pace"]
    )
    return list(base) + list(PACERS_METRIC_VIEW_TYPES)


def batter_sr_or_avg_label(cfg: FormatConfig) -> str:
    return "Avg by Length / Pace" if cfg.is_test else "SR by Length / Pace"


def pacer_length_filter_options(cfg: FormatConfig) -> List[str]:
    if cfg.is_test:
        return ["All", "FULL", "LENGTH", "SHORT", "BOUNCER"]
    base = ["FULL TOSS", "YORKER", "THE SLOT", "LENGTH", "SHORT"]
    if cfg.is_womens:
        return base
    return base + ["BOUNCER"]


def spinner_view_types(cfg: FormatConfig) -> List[str]:
    base = (
        ["All", "Bowling Strike Rate By Length", "% by Lengths", "Turn Direction"]
        if cfg.is_test
        else ["All", "Economy By Length", "% by Lengths", "Turn Direction"]
    )
    return base + list(SPINNERS_METRIC_VIEW_TYPES)


# --- Page header / pitch-map helpers ---

def format_banner_caps(cfg: FormatConfig) -> str:
    """Bold caps line for active format (e.g. Men's T20)."""
    return cfg.label.upper()


# HTML/CSS for orange caps banner (right-aligned; +2pt vs base 1.05rem)
FORMAT_BANNER_STYLE = "color:#ff6600;font-size:calc(1.05rem + 2pt);font-weight:700;"


def pitch_map_figsize(cfg: FormatConfig, width: float = 3.0) -> Tuple[float, float]:
    """Match pitch scatter height to pitch-length bar charts (test vs white-ball)."""
    h = 7.0 if cfg.is_test else 7
    return (width, h)


def add_crease_lateral_zone_background(ax, is_rhb: bool, zorder: int = 0) -> None:
    m = 1 if is_rhb else -1
    ax.axvspan(-1.245 * m, -0.415 * m, facecolor="#ffffcc", alpha=0.25, zorder=zorder, linewidth=0)
    ax.axvspan(-0.415 * m, -0.115 * m, facecolor="#ffcccc", alpha=0.25, zorder=zorder, linewidth=0)
    ax.axvspan(-0.115 * m, 0.115 * m, facecolor="#ccffcc", alpha=0.25, zorder=zorder, linewidth=0)
    ax.axvspan(0.115 * m, 0.6 * m, facecolor="#cce5ff", alpha=0.25, zorder=zorder, linewidth=0)


def pitch_bin_percentages(df: pd.DataFrame, pitch_bins: Dict[str, List[float]], bounce_col: str = "BounceX") -> Dict[str, float]:
    """Percent of rows in each bin (same half-open bounds as pitch charts)."""
    if df is None or df.empty or not pitch_bins:
        return {k: 0.0 for k in pitch_bins}
    bx = pd.to_numeric(df[bounce_col], errors="coerce")
    valid = bx.notna()
    total = int(valid.sum())
    if total == 0:
        return {k: 0.0 for k in pitch_bins}
    out: Dict[str, float] = {}
    for name, bounds in pitch_bins.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        cnt = int(((bx >= lo) & (bx < hi) & valid).sum())
        out[name] = round(100.0 * cnt / total, 0)
    return out


def hitting_stumps_mask(df: pd.DataFrame) -> pd.Series:
    """Same geometry as Spinners 'Hitting/Missing' chart (stump channel)."""
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    need = {"StumpsY", "StumpsZ"}
    if not need.issubset(df.columns):
        return pd.Series(False, index=df.index)
    sy = pd.to_numeric(df["StumpsY"], errors="coerce")
    sz = pd.to_numeric(df["StumpsZ"], errors="coerce")
    return (
        (sy >= -0.1503)
        & (sy <= 0.1503)
        & (sz >= 0)
        & (sz <= 0.746)
    ).fillna(False)
