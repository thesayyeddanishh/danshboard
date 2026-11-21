from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

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


def provider_key(provider: str) -> str:
    env_name = "XAI_API_KEY" if provider == "Grok / xAI" else "GROQ_API_KEY"
    try:
        secret_value = st.secrets.get(env_name, "") if hasattr(st, "secrets") else ""
    except Exception:
        secret_value = ""
    return os.getenv(env_name, secret_value)


def extract_response_text(payload: Dict[str, Any]) -> str:
    if payload.get("output_text"):
        return payload["output_text"]
    chunks = []
    for item in payload.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in {"output_text", "text"} and c.get("text"):
                chunks.append(c["text"])
    return "\n".join(chunks).strip()


def call_ai(prompt: str, api_key: str, *, provider: str, model: str) -> str:
    """Call either xAI/Grok or Groq through their OpenAI-compatible Responses APIs."""
    if provider == "Grok / xAI":
        url = "https://api.x.ai/v1/responses"
        provider_label = "xAI/Grok"
    else:
        url = "https://api.groq.com/openai/v1/responses"
        provider_label = "Groq"

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
        "max_output_tokens": 1800,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=180)
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach {provider_label} Responses API: {e}") from e

    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:2000]
        raise RuntimeError(f"{provider_label} Responses API returned HTTP {r.status_code}: {detail}")

    try:
        body = r.json()
    except Exception as e:
        raise RuntimeError(f"{provider_label} returned a non-JSON response: {r.text[:1000]}") from e

    text = extract_response_text(body)
    if not text:
        raise RuntimeError(f"{provider_label} returned no text output. Response: {body}")
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


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def _matching_names(question: str, values: List[str], limit: int = 8) -> List[str]:
    """Return only names that are explicitly present in the user's question."""
    q = _norm_text(question)
    unique = sorted({_norm_text(v) for v in values if str(v).strip()}, key=lambda x: (-len(x), x))
    return [v for v in unique if v and v in q][:limit]


def dataset_catalog(df: pd.DataFrame, question: str = "") -> Dict[str, Any]:
    """Build a compact planner catalog. Never send the full player/match catalog to the LLM."""
    def vals(c, n=25):
        if c not in df.columns:
            return []
        out = sorted({_norm_text(v) for v in df[c].dropna().tolist() if str(v).strip()})
        return out[:n]

    batters = df["BatsmanName"].dropna().astype(str).tolist() if "BatsmanName" in df.columns else []
    bowlers = df["BowlerName"].dropna().astype(str).tolist() if "BowlerName" in df.columns else []

    return {
        "rows": int(len(df)),
        "columns": [c for c in df.columns if c not in {"BounceX", "BounceY", "InterceptionX", "InterceptionY", "InterceptionZ", "CreaseY", "CreaseZ", "StumpsY", "StumpsZ", "ReleaseY", "ReleaseZ", "LandingX", "LandingY"}],
        "explicit_batter_candidates": _matching_names(question, batters),
        "explicit_bowler_candidates": _matching_names(question, bowlers),
        "grounds": vals("Ground", 10),
        "batting_teams": vals("BattingTeam", 15),
        "bowling_teams": vals("BowlingTeam", 15),
        "tours": vals("Tour", 10),
        "years": vals("Year", 10),
        "delivery_types": vals("DeliveryType", 5),
    }


def infer_explicit_filters(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Extract unambiguous constraints from the question without relying on an LLM."""
    q = _norm_text(question)
    filters: Dict[str, Any] = {}

    # Year: only accept years that actually exist in the loaded dataset.
    available_years = set(pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).tolist()) if "Year" in df.columns else set()
    year_hits = [int(y) for y in re.findall(r"\b(20\d{2})\b", q) if int(y) in available_years]
    if year_hits:
        filters["year"] = year_hits[-1]

    # Delivery type: Medium is analytically Seam by the user's established rule.
    if re.search(r"\b(SEAM|SEAMER|PACE|PACERS|FAST BOWLING)\b", q):
        filters["delivery_type"] = "Seam"
    elif re.search(r"\b(SPIN|SPINNER|SPINNERS)\b", q):
        filters["delivery_type"] = "Spin"

    batters = df["BatsmanName"].dropna().astype(str).unique().tolist() if "BatsmanName" in df.columns else []
    bowlers = df["BowlerName"].dropna().astype(str).unique().tolist() if "BowlerName" in df.columns else []
    batter_hits = _matching_names(question, batters, limit=5)
    bowler_hits = _matching_names(question, bowlers, limit=5)

    # If the same name is present in both lists, keep the explicit role wording if available.
    batter_set = set(batter_hits)
    bowler_set = set(bowler_hits)
    role_words_bowler = bool(re.search(r"\b(BOWLER|BOWLING|BOWLED|SEAM BOWLER|SPINNER)\b", q))
    role_words_batter = bool(re.search(r"\b(BATTER|BATSMAN|BATTING|BAT)\b", q))

    if batter_hits and bowler_hits:
        # Prefer distinct names where the query contains a matchup connector.
        if re.search(r"\b(VS|VERSUS|AGAINST|MATCHUP|V)\b", q):
            if len(batter_hits) == 1 and len(bowler_hits) >= 1:
                filters["batter"] = batter_hits[0]
                # Avoid assigning the same player as both unless there is no alternative.
                other = next((n for n in bowler_hits if n != batter_hits[0]), None)
                if other:
                    filters["bowler"] = other
            elif len(bowler_hits) == 1 and len(batter_hits) >= 1:
                filters["bowler"] = bowler_hits[0]
                other = next((n for n in batter_hits if n != bowler_hits[0]), None)
                if other:
                    filters["batter"] = other
    elif batter_hits and not role_words_bowler:
        filters["batter"] = batter_hits[0]
    elif bowler_hits and role_words_bowler and not role_words_batter:
        filters["bowler"] = bowler_hits[0]
    elif batter_hits:
        filters["batter"] = batter_hits[0]

    return filters


def build_planner_prompt(question: str, catalog: Dict[str, Any], cfg, speed_unit: str) -> str:
    # Keep this prompt deliberately small: Groq's current on-demand TPM limit can be 8k.
    return f"""You are DANsh, a cricket analytics query planner. Do NOT calculate statistics.
Translate the user's question into a deterministic Python analysis plan.
Use only entities in the compact catalog below. If the question contains an explicit player/year/delivery type, preserve those filters exactly.
Existing dashboard formulas are the source of truth. Medium is Seam. BounceX validity is -10 to +18 m. ReleaseSpeed validity is 40 to 150. InterceptionX is measured from the CENTER OF THE PITCH. Interception -999.9 means no contact. Deviation validity is -15 to +15 degrees.

FORMAT: {cfg.label}
SPEED UNIT: {speed_unit}

Return ONLY JSON with keys: analysis_type, role, filters, metric, group_by, compare_to_baseline, visualization, top_n, min_sample, user_needs_clarification.
analysis_type: metric | strengths | matchup | ground | swot | comparison | trend | discovery | key_matchups | distribution
role: batter | bowler | both | auto
filters may contain: bowler, batter, bowling_team, batting_team, ground, tour, match, year, phase, delivery_type, batter_hand, bowler_hand, pitch_length, speed_group, release_side, stump_channel
group_by: pitch_length | phase | speed_group | release_side | crease_zone | stump_channel | batter_hand | bowler_hand | ground | bowler | batter | batting_team | bowling_team | year | null
visualization: table | bar | line | scatter | pitch_map | none

COMPACT CATALOG:
{json.dumps(catalog, ensure_ascii=False, separators=(',', ':'))}

QUESTION:
{question}
"""


def fallback_plan(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    q = question.lower()
    filters = infer_explicit_filters(question, df)
    has_batter = bool(filters.get("batter"))
    has_bowler = bool(filters.get("bowler"))
    role = "both" if has_batter and has_bowler else "batter" if has_batter else "bowler" if has_bowler else ("bowler" if any(x in q for x in ["bowler", "bowling", "bowled", "seam bowler", "spinner"]) else "batter" if any(x in q for x in ["batter", "batting", "batsman"]) else "auto")
    analysis_type = "matchup" if has_batter and has_bowler or " vs " in q or "versus" in q or "matchup" in q else "ground" if "ground" in q or "venue" in q else "strengths" if "strength" in q else "swot" if "swot" in q else "metric"
    metric = "wickets" if "wicket" in q else "economy" if "economy" in q else "batting_strike_rate" if "strike rate" in q and role == "batter" else "bowling_strike_rate" if "strike rate" in q else "runs"
    return {
        "analysis_type": analysis_type,
        "role": role,
        "filters": filters,
        "metric": metric,
        "group_by": "pitch_length" if "length" in q else "phase" if "phase" in q or "overs" in q else None,
        "compare_to_baseline": "better" in q or "worse" in q or "strength" in q,
        "visualization": "bar" if "length" in q or "phase" in q else "table",
        "top_n": 10,
        "min_sample": 10,
        "user_needs_clarification": False,
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
    st.markdown('<div class="dansh-sub">Hawk-Eye Cricket Intelligence · AI plans · Python calculates · AI interprets</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div style="text-align:right;margin-top:15px"><span class="orange">{format_banner_caps(cfg)}</span><br>{len(df):,} deliveries</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("DANsh Controls")
    provider = st.selectbox(
        "AI provider",
        ["Grok / xAI", "Groq"],
        index=0,
        help="Choose which provider DANsh uses for planning and interpretation. Python still performs all numerical calculations locally.",
    )

    if provider == "Grok / xAI":
        key_env = "XAI_API_KEY"
        model_options = ["grok-4.7"]
        key_label = "xAI API key"
        key_help = "Optional. Prefer XAI_API_KEY in Streamlit secrets/environment."
    else:
        key_env = "GROQ_API_KEY"
        model_options = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        key_label = "Groq API key"
        key_help = "Optional. Prefer GROQ_API_KEY in Streamlit secrets/environment."

    api_key = provider_key(provider)
    api_override = st.text_input(key_label, value="", type="password", help=key_help)
    if api_override.strip():
        api_key = api_override.strip()

    model = st.selectbox("Model", model_options, index=0)
    min_sample = st.number_input("Low-sample warning threshold", 1, 500, 10)
    if api_key:
        st.success(f"{provider} connected")
    else:
        st.info(f"Add {key_env} to enable {provider}.")
    st.caption("Grok/Groq plans and interprets. Python performs the deterministic cricket calculations locally.")

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
                planner_catalog = dataset_catalog(df, question)
                planner_text = call_ai(build_planner_prompt(question, planner_catalog, cfg, speed_unit), api_key, provider=provider, model=model)
                plan = parse_json(planner_text)
            else:
                plan = fallback_plan(question, df)
        except Exception as e:
            st.warning(f"AI planning failed, so DANsh used the local fallback planner: {e}")
            plan = fallback_plan(question, df)

        # Deterministic guardrail: explicit constraints in the user's question always win.
        # This prevents an AI failure or incomplete plan from silently reverting to the full dataset.
        filters = plan.setdefault("filters", {})
        explicit_filters = infer_explicit_filters(question, df)
        for key, value in explicit_filters.items():
            filters[key] = value

        if filters.get("bowler"):
            filters["bowler"] = resolve_subject_name(str(filters["bowler"]), df["BowlerName"].unique().tolist()) or filters["bowler"]
        if filters.get("batter"):
            filters["batter"] = resolve_subject_name(str(filters["batter"]), df["BatsmanName"].unique().tolist()) or filters["batter"]

        # Keep the role consistent with explicit player constraints.
        if filters.get("batter") and filters.get("bowler"):
            plan["role"] = "both"
        elif filters.get("batter"):
            plan["role"] = "batter"
        elif filters.get("bowler"):
            plan["role"] = "bowler"

        result = run_analysis(df, plan, cfg, speed_unit)

    if result["rows"] == 0:
        st.error("No deliveries matched the requested analysis. Try widening the filters or specifying the format/context.")
        st.json(plan)
        st.stop()

    st.markdown("### Analysis")
    if api_key:
        with st.spinner(f"{provider} is interpreting the calculated evidence…"):
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
                answer = call_ai(interpretation_prompt, api_key, provider=provider, model=model)
                st.markdown('<div class="ai-card">' + answer.replace("\n", "<br>") + '</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"AI interpretation failed: {e}")
                st.info("The deterministic calculation is still shown below.")
    else:
        st.info("No AI API key configured. Showing deterministic analysis; add the selected provider key to enable AI planning and interpretation.")

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
    st.info("DANsh will turn the question into a structured plan, calculate the answer deterministically from Hawk-Eye data, then ask the selected AI provider to interpret the evidence.")
