# AI Company Discovery Agent (Prototype)

A stopgap prototype that answers natural-language business discovery queries
(e.g. *"Find pharmaceutical companies in Powai"* or *"Give me companies like
Tata AutoComp"*) using an LLM for reasoning/discovery, and deterministic
Python code for CIN (Corporate Identification Number) verification. No
embeddings, vector databases, RAG, SQL, or ranking models are used, by
design.

## How it works

1. You type a natural-language query (CLI or Streamlit UI).
2. `discovery.py` sends it to the configured LLM provider (Gemini by default).
3. The LLM reasons about which real companies satisfy the request and returns
   structured JSON: company name, detailed reasoning, and a confidence level.
4. For each company, `discovery.py` calls `cin_lookup.py` for a
   **deterministic, LLM-free** CIN lookup, sequentially.
5. `discovery.py` returns a plain list of `CompanyResult` objects — the
   frontend (CLI or Streamlit) decides how to display them.

All business logic lives in `discovery.py`, which has no UI-specific code.
This means the CLI, the Streamlit UI, and any future frontend (a FastAPI
service, a React app, etc.) all call the exact same `discover()` function
and get the exact same data back.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy the example environment file and fill in your API key(s):

   ```bash
   cp .env.example .env
   ```

   At minimum, set `GEMINI_API_KEY` (default provider). To use OpenAI
   instead, set `LLM_PROVIDER=OPENAI` and `OPENAI_API_KEY`.

3. Run the app:

   **Streamlit UI (primary):**

   ```bash
   streamlit run ui.py
   ```

   This opens in your browser automatically. Type a query into the search
   field and submit — results appear as cards, each with the company name,
   a confidence badge, CIN (or "Not found"), and the reasoning behind the
   match. Past queries in the current session appear in the sidebar; click
   one to re-run it (it always fetches a fresh LLM response, never a cached
   one).

   **Command line:**

   ```bash
   python app.py
   ```

   > Note: `web_app.py` (a Flask front end) also exists in this repo from an
   > earlier iteration but is not part of the current architecture — the
   > Streamlit UI is the supported frontend going forward. It still runs
   > standalone if you want it (`python web_app.py`), but it hasn't been
   > wired up to `discovery.py` and won't reflect changes made there.

4. Type a query, e.g.:

   ```
   > Find industrial valve manufacturers in Maharashtra
   ```

   Type `quit` to exit.

## Switching LLM providers

Switching providers is purely a configuration change — no code edits needed.
In `.env`, set:

```
LLM_PROVIDER=GEMINI    # or OPENAI
```

and supply the corresponding API key. `llm_provider.py` contains the
abstraction layer (`LLMProvider` base class with `GeminiProvider` and
`OpenAIProvider` implementations); adding a new provider means adding one
more subclass there.

## Important limitations (read before relying on this prototype)

**Company discovery accuracy depends entirely on the LLM's knowledge.**
This prototype does not run its own web-search or retrieval pipeline for
company discovery — per the design brief, that is intentionally deferred to
a future version. That means:

- If you leave `GEMINI_USE_SEARCH=false` (the default) or use OpenAI, the
  model answers from what it learned during training, which can be outdated
  or incomplete. Treat every claim as **provisional**, especially specific
  figures like turnover.
- If you set `GEMINI_USE_SEARCH=true` (and your Gemini API tier supports it),
  Gemini will use its built-in Google Search grounding tool to ground answers
  in live search results. This is a model-side tool call, not a retrieval
  pipeline you're hosting, so it doesn't conflict with the "no RAG" design
  constraint — but it does meaningfully improve factual grounding and is
  recommended if available.
- The system prompt explicitly instructs the model to say when it's unsure
  rather than guess, but no LLM output should be treated as verified fact
  without independent confirmation.

**CIN verification is structurally format-validated, not registry-confirmed,
unless you configure a paid provider.** The MCA portal only lets you look up
a company once you already have its CIN, and is CAPTCHA-gated for
interactive search — there's no free, official way to search by company
name. So by default this app:

- Searches public web results for the company name + "CIN".
- Scopes extraction to individual search-result blocks, and only considers a
  CIN-shaped candidate if it appears physically near an actual mention of
  the company name within that same result. This prevents attributing a
  *different* company's CIN (e.g. one listed elsewhere on a directory page)
  to the company being searched for.
- Validates each candidate's structure: correct overall shape, a real ROC
  state code, a recognized company-type code, and a plausible incorporation
  year.

**Important: unlike GSTIN, a CIN has no checksum digit.** There is no
mathematical way to confirm a CIN string wasn't simply made up — structural
validation can only confirm a candidate is *well-formed*, not that it is
genuinely registered to that company. Every CIN result is labeled with this
caveat, along with how many independent search results corroborated it.

For production-grade CIN verification, set `CIN_API_URL` (and `CIN_API_KEY`
if needed) in `.env` to point at a real MCA/CIN data provider (e.g. an
official MCA21 integration, Tofler, Zauba Corp, Probe42, etc.). When
configured, the app calls that API directly and skips the web-search
fallback.

**Confidence levels are self-reported by the LLM.** There is no independent
system-computed check behind "High/Medium/Low" — it reflects the model's own
assessment of how well-supported its answer is, per the system prompt's
instructions. CIN lookup results are never combined into or used to adjust
this confidence value.

## Deployment

This app is designed to run locally today, but the architecture doesn't need
to change to deploy it:

- `discovery.py` is stateless and has no dependency on how it's invoked —
  it doesn't know or care whether it's called from a CLI loop, a Streamlit
  callback, or (in the future) an HTTP handler.
- Configuration is read via `os.getenv(...)` in `llm_provider.py` and
  `cin_lookup.py`. Locally, `load_dotenv()` populates those from your `.env`
  file; on a hosting platform (Streamlit Community Cloud's "Secrets", Render,
  Railway, Azure, AWS, etc.) the same variables are supplied as real
  environment variables instead — no code changes needed either way.
- `ui.py` is a single Streamlit entrypoint with a `requirements.txt`
  alongside it, which is exactly what Streamlit Community Cloud (and most
  other platforms' auto-detection) expects.
- Streamlit's default layout already reflows responsively to narrow/mobile
  screens without extra work.



## Project structure

```
app.py            # Thin CLI wrapper -- calls discovery.py and prints results
discovery.py      # All business logic: LLM calls, validation, CIN lookups
ui.py             # Streamlit UI -- all display logic, no business logic
web_app.py        # Flask front end from an earlier iteration (not wired to
                  #   discovery.py; not part of the current architecture)
templates/
  index.html      # Used only by web_app.py
static/
  style.css       # Used only by web_app.py
  app.js          # Used only by web_app.py
llm_provider.py   # Provider abstraction (Gemini default, OpenAI alternative)
cin_lookup.py     # Deterministic, LLM-independent CIN lookup
prompts.py        # System prompt and prompt-building helpers
requirements.txt  # Python dependencies
.env.example      # Configuration template
```
