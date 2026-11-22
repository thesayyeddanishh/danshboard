# DANsh AI — Page 5

DANsh is the Hawk-Eye cricket intelligence layer for the Streamlit dashboard.

## AI providers

DANsh supports both providers from the Page 5 sidebar:

- **Grok / xAI** — `grok-4.7`, using xAI's Responses API.
- **Groq** — `openai/gpt-oss-120b` or `openai/gpt-oss-20b`, using Groq's OpenAI-compatible Responses API.

Configure the provider key in Streamlit secrets or environment variables:

```toml
XAI_API_KEY = "your-xai-key"
GROQ_API_KEY = "your-groq-key"
```

You can keep both keys configured and switch providers from the DANsh sidebar without changing code.

## Architecture

**Selected AI provider plans → Python calculates → selected AI provider interprets**

All critical cricket calculations are performed locally from the uploaded Hawk-Eye dataset. The AI provider is not trusted to calculate the numerical results.

## Locked Hawk-Eye rules

- BounceX analytical range: `-10` to `+18` metres.
- BounceX = `0` at the base of middle stump at the batting end.
- InterceptionX is measured from the **center of the pitch**.
- InterceptionY is relative to the middle-stump line.
- Interception `-999.9` means no bat/pad/body contact.
- ReleaseSpeed analytical range: `40` to `150`.
- Swing = pre-bounce lateral movement.
- Deviation = post-bounce lateral movement; analytical range `-15` to `+15` degrees.
- Medium deliveries are treated as Seam.
- Existing pitch bins, phase logic and dashboard formulas are inherited from the project's analytical configuration.


## v4 request-size protection

The planner no longer sends the full player/match catalog to the AI provider. It sends only compact metadata plus question-matched entity candidates. Explicit player, year, team, ground and Seam/Spin filters are also extracted deterministically before analysis. This prevents a failed AI planning request from silently analysing the entire dataset.

Planner and interpretation output budgets are kept separate to reduce provider TPM pressure.


## DANsh v5 planner safeguards

- The planner no longer sends the complete player/match catalog to Grok or Groq.
- Explicit player, year, and Seam/Spin constraints are extracted deterministically from the user question.
- Those explicit constraints override an incomplete AI plan.
- If AI planning fails, the local fallback still applies the detected filters instead of analysing the full dataset.
- This prevents queries such as `Abhishek Sharma v Seam in 2025` from silently reverting to all deliveries.
- The planner response budget is capped separately from the evidence interpretation call.


## DANsh v6 UI / ranking improvements
- Ranking questions such as Top-N batting/bowling metrics are compiled deterministically before any AI call.
- Explicit RHB/LHB and pitch-length terms are enforced locally.
- Rankings are sorted by the requested metric, not by delivery count.
- Ranking results use a horizontal bar chart plus a readable evidence table.
- Raw delivery rows are never sent to the AI interpretation step; only compact calculated evidence is sent.
- The result page shows applied filters, ranking direction, sample size, visualisation, evidence, then AI interpretation.


## DANsh v8 deployment note

For the metric-aware ranking engine, deploy these two files together:
- `pages/05_DANSH_AI.py`
- `dansh_engine.py`

Do not mix the v8 Page 5 with an older `dansh_engine.py`. v8 also treats an explicitly requested year as a hard constraint, so a request for a season not present in the dataset correctly returns no matching deliveries instead of silently dropping the year filter.
