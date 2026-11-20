Dashboard Link: https://danshboard.streamlit.app/


## DANsh AI (Page 5)

Page 5 adds a natural-language Hawk-Eye analysis layer. The architecture is:

**Grok plans → Python calculates → Grok interprets.**

The page uses the existing `cricket_config.py` format and pitch rules and includes the agreed Hawk-Eye semantics such as BounceX filtering, Medium → Seam, interception `-999.9` as no contact, and InterceptionX measured from the center of the pitch.

### xAI API key

Set `XAI_API_KEY` as an environment variable or Streamlit secret. The page uses the xAI Responses API with `grok-4.7`. Without a key, a deterministic local fallback planner still lets the page demonstrate the calculation engine.

### xAI API note
DANsh uses the canonical xAI Responses API role/content input format. If xAI rejects a request, the page now shows the actual API error payload instead of only `400 Client Error`.
