from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from cricket_config import FORMAT_LABELS, FORMAT_KEYS, resolve_format, format_banner_caps
from dansh_engine import (
    add_derived_columns,
    confidence_meter,
    filter_plan,
    group_analysis,
    matchup_analysis,
    metrics,
    normalize_dataframe,
    resolve_subject_name,
)

st.set_page_config(page_title="DANsh AI", page_icon="hawk-logo.png", layout="wide")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { width: 200px !important; }
    .dansh-title { font-size: 2.25rem; font-weight: 850; letter-spacing: -1px; margin-bottom: 0; }
    .dansh-sub { color:#64748B; margin-top:-5px; margin-bottom:20px; }
    .ai-card { background:#0f172a; color:#f8fafc; padding:18px 20px; border-radius:14px; }
    .metric-card { border:1px solid #E2E8F0; border-radius:12px; padding:12px; background:#fff; }
    .orange { color:#ff6600; font-weight:800; }
    </style>
    """, unsafe_allow_html=True,
)


def xai_key() -> str:
    return os.getenv("XAI_API_KEY", st.secrets.get("XAI_API_KEY", "") if hasattr(st, "secrets") else "")


def extract_response_text(payload: Dict[str, Any]) -> str:
    if payload.get("output_text"):
        return payload["output_text"]
    chunks = []
    for item in payload.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                chunks.append(c["text"])
    return "\n".join(chunks).strip()


def call_grok(prompt: str, api_key: str, *, model: str = "grok-4.7") -> str:
    """Call xAI Responses API with a canonical OpenResponses payload.

    The API currently accepts both string and array input, but the role/content
    form is the most explicit and gives much better diagnostics when a request
    is rejected. We also surface xAI's actual error body instead of only the
    generic HTTP status.
    """
    url = "https://api.x.ai/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_output_tokens": 5000,
        "store": False,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=180)
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach xAI Responses API: {e}") from e

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:2000]
        raise RuntimeError(f"xAI Responses API returned HTTP {r.status_code}: {detail}")

    try:
        body = r.json()
    except Exception as e:
        raise RuntimeError(f"xAI returned a non-JSON response: {r.text[:1000]}") from e

    text = extract_response_text(body)
    if not text:
        raise RuntimeError(f"xAI returned no text output. Response: {body}")
    return text


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("Grok did not return valid JSON.")
        return json.loads(m.group(0))


def dataset_catalog(df: pd.DataFrame) -> Dict[str, Any]:
    def vals(c, n=250):
        if c not in df.columns:
            return []
        return sorted(df[c].dropna().astype(str).unique().tolist())[:n]
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "players_bowlers": vals("BowlerName"),
        "players_batters": vals("BatsmanName"),
        "grounds": vals("Ground"),
        "batting_teams": vals("BattingTeam"),
        "bowling_teams": vals("BowlingTeam"),
        "tours": vals("Tour"),
        "matches": vals("Match"),
        "years": vals("Year"),
        "delivery_types": vals("DeliveryType"),
    }


def build_planner_prompt(question: str, catalog: Dict[str, Any], cfg, speed_unit: str) -> str:
    return f"""
You are DANsh, a cricket analytics query planner. You DO NOT calculate statistics.
You translate the analyst's natural-language question into a deterministic analysis plan for Python.
Never invent player/team/ground names. Use only values supplied in the catalog when selecting entities.
The existing dashboard formulas are the source of truth. DeliveryType Medium has already been normalised to Seam.
BounceX analytical validity is -10 to +18 metres. ReleaseSpeed analytical validity is 40 to 150.
InterceptionX is measured from the CENTER OF THE PITCH. InterceptionY is from the middle-stump line.
Interception -999.9 means no contact. Swing is pre-bounce; Deviation is post-bounce. Deviation analytical range is -15 to +15 degrees.

CURRENT FORMAT: {cfg.label}
SPEED UNIT: {speed_unit}

Return ONLY JSON with these keys:
analysis_type, role, filters, metric, group_by, compare_to_baseline, visualization, top_n, min_sample, user_needs_clarification.

analysis_type should be one of:
metric, strengths, matchup, ground, swot, comparison, trend, discovery, key_matchups, distribution.
role should be one of: batter, bowler, both, auto.
filters is an object with optional keys: bowler, batter, bowling_team, batting_team, ground, tour, match, year, phase, delivery_type, batter_hand, bowler_hand, pitch_length, speed_group, release_side, stump_channel.
metric may be: economy, bowling_average, bowling_strike_rate, batting_average, batting_strike_rate, wickets, runs, deliveries, dot_pct, boundary_pct, avg_speed, avg_swing, avg_deviation, avg_length, hitting_stumps_pct.
group_by may be one of: pitch_length, phase, speed_group, release_side, crease_zone, stump_channel, batter_hand, bowler_hand, ground, bowler, batter, batting_team, bowling_team, year, or null.
visualization may be: table, bar, line, scatter, pitch_map, none.
compare_to_baseline is true when the question asks whether something is better/worse/stronger than normal.

CATALOG:
{json.dumps(catalog, ensure_ascii=False)}

QUESTION:
{question}
"""


def fallback_plan(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    q = question.lower()
    role = "bowler" if any(x in q for x in ["bowler", "bowling", "bowled", "pace", "seam bowler", "spinner"]) else "batter" if any(x in q for x in ["batter", "batting", "batsman"]) else "auto"
    analysis_type = "matchup" if " vs " in q or "versus" in q or "matchup" in q else "ground" if "ground" in q or "venue" in q else "strengths" if "strength" in q else "swot" if "swot" in q else "metric"
    metric = "wickets" if "wicket" in q else "economy" if "economy" in q else "batting_strike_rate" if "strike rate" in q and role == "batter" else "bowling_strike_rate" if "strike rate" in q else "runs"
    return {
        "analysis_type": analysis_type, "role": role, "filters": {}, "metric": metric,
        "group_by": "pitch_length" if "length" in q else "phase" if "phase" in q or "overs" in q else None,
        "compare_to_baseline": "better" in q or "worse" in q or "strength" in q,
        "visualization": "bar" if "length" in q or "phase" in q else "table",
        "top_n": 10, "min_sample": 10, "user_needs_clarification": False,
    }


def run_analysis(df: pd.DataFrame, plan: Dict[str, Any], cfg, speed_unit: str) -> Dict[str, Any]:
    d = filter_plan(df, plan, cfg, speed_unit)
    role = str(plan.get("role", "auto"))
    if role == "bowler":
        role_label = "Bowler"
    elif role == "batter":
        role_label = "Batter"
    else:
        role_label = "Analysis"

    result: Dict[str, Any] = {"rows": len(d), "role": role_label, "metrics": metrics(d, role), "data": None, "baseline": None, "raw_for_map": d}
    group_by = plan.get("group_by")
    if plan.get("analysis_type") in {"matchup", "key_matchups"} and not group_by:
        result["data"] = matchup_analysis(d, int(plan.get("top_n") or 15))
    elif group_by:
        g = group_analysis(d, group_by, role, int(plan.get("top_n") or 10))
        result["data"] = g
    else:
        result["data"] = pd.DataFrame([result["metrics"]])

    # Entity baseline: compare filtered result to the broader entity when possible.
    f = plan.get("filters") or {}
    if plan.get("compare_to_baseline"):
        base = df
        if role == "bowler" and f.get("bowler"):
            base = base[base["BowlerName"].str.upper() == str(f["bowler"]).upper()]
        elif role == "batter" and f.get("batter"):
            base = base[base["BatsmanName"].str.upper() == str(f["batter"]).upper()]
        result["baseline"] = metrics(base, role)

    conf_label, conf_score = confidence_meter(len(d))
    result["confidence"] = {"label": conf_label, "score": conf_score, "sample": len(d)}
    return result


def result_for_grok(question: str, plan: Dict[str, Any], result: Dict[str, Any]) -> str:
    clean = dict(result)
    if isinstance(clean.get("data"), pd.DataFrame):
        clean["data"] = clean["data"].head(20).to_dict(orient="records")
    return json.dumps({"question": question, "plan": plan, "result": clean}, default=str)


def visualise(result: Dict[str, Any], plan: Dict[str, Any]):
    if plan.get("visualization") == "pitch_map":
        raw = result.get("raw_for_map")
        if isinstance(raw, pd.DataFrame) and {"BounceX", "BounceY"}.issubset(raw.columns):
            plot_df = raw.copy()
            plot_df["Outcome"] = plot_df["Wicket"].map({True: "Wicket", False: "No Wicket"})
            fig = px.scatter(plot_df, x="BounceY", y="BounceX", color="Outcome", hover_data=[c for c in ["BowlerName", "BatsmanName", "Runs", "ReleaseSpeed", "Deviation"] if c in plot_df.columns])
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=500, xaxis_title="BounceY — line (m)", yaxis_title="BounceX — length (m)")
            st.plotly_chart(fig, use_container_width=True)
            return
    data = result.get("data")
    if not isinstance(data, pd.DataFrame) or data.empty:
        return
    vis = plan.get("visualization", "table")
    if vis == "pitch_map" and {"BounceX", "BounceY"}.issubset(data.columns):
        fig = px.scatter(data, x="BounceY", y="BounceX", color="Wicket", hover_data=[c for c in ["BowlerName", "BatsmanName", "Runs"] if c in data.columns])
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=500, xaxis_title="BounceY — line (m)", yaxis_title="BounceX — length (m)")
        st.plotly_chart(fig, use_container_width=True)
    elif vis == "bar":
        label = data.columns[0]
        metric = plan.get("metric")
        preferred = {
            "economy": "Economy", "bowling_average": "Bowling Average", "bowling_strike_rate": "Bowling SR",
            "batting_average": "Batting Average", "batting_strike_rate": "Batting SR", "wickets": "Wickets",
            "runs": "Runs", "deliveries": "Balls", "dot_pct": "Dot %"
        }.get(metric)
        y = preferred if preferred in data.columns else ("Wickets" if "Wickets" in data.columns else data.columns[-1])
        fig = px.bar(data, x=label, y=y, text_auto=True)
        fig.update_layout(height=390, margin=dict(l=20,r=20,t=35,b=20))
        st.plotly_chart(fig, use_container_width=True)
    elif vis == "line" and "Year" in data.columns:
        y = plan.get("metric") or "Runs"
        ymap = {"runs":"Runs","wickets":"Wickets","economy":"Economy","batting_strike_rate":"Batting SR"}
        y = ymap.get(y, y)
        if y in data.columns:
            fig = px.line(data, x="Year", y=y, markers=True)
            st.plotly_chart(fig, use_container_width=True)


# ---------------- PAGE ----------------
if "data_df" not in st.session_state or st.session_state["data_df"] is None:
    st.error("Please upload the Hawk-Eye CSV from HOME before opening DANsh AI.")
    st.stop()

raw = normalize_dataframe(st.session_state["data_df"])
cfg = resolve_format(st.session_state.get("cricket_format", "men_t20i"))
speed_unit = st.session_state.get("speed_unit", "kph")
df = add_derived_columns(raw, cfg, speed_unit)

c1, c2 = st.columns([3.2, 1.2])
with c1:
    st.markdown('<div class="dansh-title">DANsh AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="dansh-sub">Hawk-Eye Cricket Intelligence · Grok plans · Python calculates · Grok interprets</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div style="text-align:right;margin-top:15px"><span class="orange">{format_banner_caps(cfg)}</span><br>{len(df):,} deliveries</div>', unsafe_allow_html=True)

api_key = xai_key()
with st.sidebar:
    st.header("DANsh Controls")
    api_override = st.text_input("xAI API key", value="", type="password", help="Optional. Prefer XAI_API_KEY in Streamlit secrets/environment.")
    if api_override.strip():
        api_key = api_override.strip()
    model = st.selectbox("Grok model", ["grok-4.7"], index=0)
    min_sample = st.number_input("Low-sample warning threshold", 1, 500, 10)
    st.caption("Numerical calculations are performed locally from the uploaded dataset. Grok is used for planning and interpretation.")

examples = [
    "What are this bowler's strengths?",
    "How does Batter X perform against spin?",
    "Show Bowler X vs Batter Y.",
    "What are the statistics at this ground?",
    "Give me a SWOT of this batter.",
    "What are the key matchups before the game?",
    "Find something interesting in the data.",
]

qcol, bcol = st.columns([6, 1])
with qcol:
    question = st.text_area("Ask DANsh", placeholder="e.g. How does this bowler perform against left-handers in the death overs?", height=90)
with bcol:
    st.write("")
    st.write("")
    ask = st.button("ANALYSE", type="primary", use_container_width=True)

with st.expander("Example questions", expanded=False):
    st.write(" · ".join(examples))

if ask and question.strip():
    with st.spinner("DANsh is planning the analysis…"):
        try:
            if api_key:
                planner_text = call_grok(build_planner_prompt(question, dataset_catalog(df), cfg, speed_unit), api_key, model=model)
                plan = parse_json(planner_text)
            else:
                plan = fallback_plan(question, df)
        except Exception as e:
            st.warning(f"Grok planning failed, so DANsh used the local fallback planner: {e}")
            plan = fallback_plan(question, df)

        # Resolve explicit names deterministically against the actual dataset.
        filters = plan.setdefault("filters", {})
        if filters.get("bowler"):
            filters["bowler"] = resolve_subject_name(str(filters["bowler"]), df["BowlerName"].unique().tolist()) or filters["bowler"]
        if filters.get("batter"):
            filters["batter"] = resolve_subject_name(str(filters["batter"]), df["BatsmanName"].unique().tolist()) or filters["batter"]

        result = run_analysis(df, plan, cfg, speed_unit)

    if result["rows"] == 0:
        st.error("No deliveries matched the requested analysis. Try widening the filters or specifying the format/context.")
        st.json(plan)
        st.stop()

    st.markdown("### Analysis")
    if api_key:
        with st.spinner("Grok is interpreting the calculated evidence…"):
            try:
                interpretation_prompt = f"""
You are DANsh, a senior cricket analyst. Python has already calculated the evidence below.
Do not recalculate or invent numbers. Explain what the evidence means.
Separate observed facts from analytical inference. Give a confidence meter and mention sample size.
If the sample is small, explicitly say so. For tactical recommendations, make them evidence-led and clearly label them as recommendations.
Use concise professional cricket-analyst language. The user asked:
{question}

CALCULATED EVIDENCE:
{result_for_grok(question, plan, result)}
"""
                answer = call_grok(interpretation_prompt, api_key, model=model)
                st.markdown('<div class="ai-card">' + answer.replace("\n", "<br>") + '</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Grok interpretation failed: {e}")
                st.info("The deterministic calculation is still shown below.")
    else:
        st.info("No xAI API key configured. Showing deterministic analysis; add XAI_API_KEY to enable Grok planning and interpretation.")

    m = result["metrics"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("DELIVERIES", f"{result['rows']:,}")
    k2.metric("RUNS", f"{m.get('runs',0):,.0f}")
    k3.metric("WICKETS", f"{m.get('wickets',0):,}")
    k4.metric("ECONOMY", "—" if m.get('economy') is None else f"{m['economy']:.2f}")
    k5.metric("CONFIDENCE", f"{result['confidence']['score']}%")

    if result["confidence"]["sample"] < int(min_sample):
        st.warning(f"Very less sample size: {result['confidence']['sample']:,} deliveries. Treat the conclusion cautiously.")

    st.markdown("#### Calculation plan")
    st.json(plan)

    if isinstance(result.get("data"), pd.DataFrame):
        st.markdown("#### Evidence")
        st.dataframe(result["data"], use_container_width=True, hide_index=True)
        visualise(result, plan)

    with st.expander("Data quality / analytical rules", expanded=False):
        st.write({
            "BounceX filter": "-10 to +18 m",
            "ReleaseSpeed filter": "40 to 150",
            "Deviation filter": "-15 to +15 degrees",
            "Interception -999.9": "No contact",
            "Medium": "Treated as Seam",
            "InterceptionX reference": "Center of pitch",
            "Pitch bins": "Inherited from cricket_config.py",
            "Phase logic": "Inherited from cricket_config.py",
            "Metric formulas": "Matched to existing dashboard formulas where implemented",
        })
else:
    st.markdown("### Start with a question")
    st.info("DANsh will turn the question into a structured plan, calculate the answer deterministically from Hawk-Eye data, then ask Grok to interpret the evidence.")
