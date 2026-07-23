# Phase 1: LangChain Migration — Design

## Context

This is the first of four planned phases evolving the Company Discovery Agent
from a single-turn discovery/verification pipeline into a conversational
assistant:

1. **LangChain migration** (this spec)
2. Conversation memory (multi-turn state, follow-up resolution)
3. Company question answering (evidence-grounded Q&A beyond discovery)
4. Debug mode (a transparent execution trace across all stages)

Phases 2-4 are intentionally not specced yet — each gets its own
brainstorming cycle once the prior phase has landed. This document only
covers Phase 1, but two Phase-1 decisions are made with the later phases in
mind:

- `discovery.py`'s existing `history: Optional[List[ConversationTurn]]`
  parameter (currently unused) is left in place as the seam Phase 2 will
  fill in.
- New Pydantic models introduced here live strictly at the LLM
  input/output boundary. They do not replace the existing `CompanyResult` /
  `RejectedCompany` / `EvidenceItem` / `DiscoveryResult` dataclasses, so
  Phase 2-4 work (state, QA, debug trace) can build on those dataclasses
  without another rewrite of discovery.py's internals.

### Current state (verified against the repo, not assumed)

The pipeline today is:

```
User Query → discovery.py → identifier_lookup.py → (verify) → retriever.py → evidence_extractor.py
```

Identifier lookup runs *before* evidence retrieval, and gates it: a company
is only searched for general evidence once GST or CIN verification confirms
it's a real registered entity. This ordering (see `discovery.py`'s module
docstring) prevents a hallucinated company name from picking up evidence
that actually belongs to a different, similarly-named real company.

Notably:

- `qa.py` and `conversation.py`, referenced in the project history supplied
  at the start of this work, do **not exist** in this repository. Phases 2
  and 3 will create them from scratch, not extend existing files.
- `ui.py` already has a working "Show Debug Information" toggle that
  exposes per-constraint PASS/FAIL/UNKNOWN evaluation, the LLM's explicit
  ACCEPT/REJECT decision, and a "Rejected Candidates" panel. Phase 4 extends
  this existing pattern to retrieval/identifier-lookup/QA — it does not
  start from zero.
- `llm_provider.py` hand-rolls a provider abstraction: raw OpenAI SDK calls
  for OpenAI/GLM/ZAI, and the native `google-genai` SDK for Gemini (needed
  for Search grounding, unavailable on Gemini's OpenAI-compatible
  endpoint). Prompts are plain f-strings (`prompts.py`); JSON parsing is
  manual regex/fence-stripping (`json_utils.py`).
- `requirements.txt` lists `google-generativeai` (the older, deprecated
  SDK), but the code actually imports `google-genai` (the newer unified
  SDK) — a pre-existing mismatch, fixed as part of this migration since
  we're touching `llm_provider.py`'s dependencies anyway.

## Decisions made during brainstorming

| Question | Decision |
|---|---|
| ZAI's custom `web_search` tool block vs. Gemini Search grounding | Drop ZAI web search (not relied on in practice); keep Gemini grounding, using `ChatGoogleGenerativeAI`'s native grounding tool support. |
| Shape of `constraint_evaluation` (dynamic-keyed dict) under structured output | Model it as `List[LLMConstraintEvaluation]` with a `name` field per item — a fully static schema — then rebuild the `Dict[str, ConstraintEvaluation]` shape internally for existing consumers. |
| Scope of Pydantic adoption | Pydantic only at the LLM I/O boundary. Existing dataclasses (`CompanyResult`, `RejectedCompany`, `EvidenceItem`, `DiscoveryResult`) are untouched; discovery.py/evidence_extractor.py convert from the Pydantic response into them, same as they convert from parsed JSON today. |
| Shape of `get_provider()`'s return value | Returns the raw LangChain `BaseChatModel` (`ChatOpenAI` or `ChatGoogleGenerativeAI`), pre-configured from env vars. Callers build their own `ChatPromptTemplate` and call `.with_structured_output(Schema)` themselves, rather than a `generate()`-style wrapper hiding that. |
| Testing style | Keep the existing script + `check(label, condition)` harness convention (`test_gst.py`, `test_entity_matching.py`), not pytest. |
| Gemini Search grounding | **Preserved, but only for `discovery.py`.** Discovery needs web-scale recall beyond the model's training data to find companies at all; the retrieval/evidence/identifier stages operate only on documents `retriever.py` already fetched and never need grounding. Verified against the installed `langchain-google-genai` (4.3.1) source: Gemini's `google_search` tool and `with_structured_output`'s schema-constrained mode (`method="json_schema"`, the default) cannot both be active in one API call — Gemini rejects `response_schema` + `tools` together. So `discovery.py` is the one caller that must branch: when grounding is actually active for a call, it invokes the model directly (grounding tool bound, no structured-output constraint) and falls back to the pre-migration approach of prompt-instructed JSON + manual parsing, still validated through the same Pydantic schema afterward. `evidence_extractor.py` never requests grounding, so it always takes the plain `with_structured_output` path. |

## Architecture

### `llm_provider.py`

Becomes a factory returning a configured `BaseChatModel`, plus one small
helper for the grounding branch:

```python
def get_provider(*, allow_grounding: bool = False) -> BaseChatModel: ...
def is_grounded() -> bool:
    """True iff get_provider(allow_grounding=True) will return a model
    with Google Search grounding bound (i.e. LLM_PROVIDER=GEMINI and
    LLM_GROUNDING=true). Lets a caller (only discovery.py, in practice)
    decide up front whether it must take the manual-parse fallback path
    instead of with_structured_output, without duplicating the
    provider/env-var logic itself."""
```

- `OPENAI` / `GLM` / `ZAI` → `ChatOpenAI(model=..., api_key=..., base_url=..., temperature=0.3)`, base_url resolution unchanged from today (`None` for OpenAI, NVIDIA/ZAI defaults otherwise, override via `LLM_BASE_URL`). `allow_grounding` has no effect here — grounding is Gemini-only.
- `GEMINI` → `ChatGoogleGenerativeAI(model=..., api_key=..., temperature=0.3)`. Only when **both** `allow_grounding=True` (the caller opted in) **and** `LLM_GROUNDING=true` (the deployment opted in) is `google_search` bound via `.bind_tools([{"google_search": {}}])` (the documented `langchain-google-genai` API for this, confirmed against its source).
- Same required env vars as today (`LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`); `LLMProviderError` is still raised eagerly for missing/invalid config.
- The ZAI `web_search` raw tool block and its `LLM_WEB_SEARCH` env var are removed (per the earlier decision — unused in practice, and unlike Gemini grounding, has no caller that needs it preserved).
- `evidence_extractor.py` always calls plain `get_provider()` (default `allow_grounding=False`) — it never requests grounding, so it can always safely use `with_structured_output`.

### `llm_schemas.py` (new)

Pydantic models for the LLM's structured output only:

```python
class LLMConstraintEvaluation(BaseModel):
    name: str
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    reason: str

class LLMCompanyEntry(BaseModel):
    company_name: str
    constraint_evaluation: List[LLMConstraintEvaluation]
    decision: Literal["ACCEPT", "REJECT"]
    reason: str
    confidence: Literal["High", "Medium", "Low"]

class LLMDiscoveryResponse(BaseModel):
    companies: List[LLMCompanyEntry]

class LLMEvidenceItem(BaseModel):
    point: str
    source_title: str
    source_url: str

class LLMEvidenceResponse(BaseModel):
    items: List[LLMEvidenceItem]
```

Responses are wrapped in a top-level object (`companies` / `items`) rather
than a bare array, since structured-output schemas need an object root.
This is an internal contract change, invisible to everything outside
`discovery.py` / `evidence_extractor.py`.

### `prompts.py`

`SYSTEM_PROMPT` and `EVIDENCE_SYSTEM_PROMPT` are **unchanged**, including
Rule 9 / the evidence module's "OUTPUT FORMAT" sections. Originally this
spec planned to trim those sections as redundant once
`with_structured_output` enforces the shape — but discovery.py's grounded
fallback path (see above) still needs the model to follow written JSON
formatting instructions, since no schema is enforced by the API in that
branch. Leaving the instructions in place is harmless for the
`with_structured_output` path (redundant, not conflicting) and load-bearing
for the grounded path, so the simplest correct choice is to not touch this
text at all.

Both prompts are wrapped in a `ChatPromptTemplate.from_messages([("system",
...), ("human", "{user_prompt}")])`. `build_user_prompt()` /
`build_evidence_prompt()` keep building the human-turn text exactly as
today; the template just wraps it for LangChain's message format.

### `discovery.py`

Replaces `provider.generate(...)` + `extract_json_array(...)` with a branch
on `is_grounded()`:

```python
llm = get_provider(allow_grounding=True)
prompt_messages = DISCOVERY_PROMPT.format_messages(
    user_prompt=_build_prompt(query, history)
)

if is_grounded():
    # Grounding tool is bound; Gemini rejects combining it with
    # with_structured_output's schema constraint in the same call. Fall
    # back to prompt-instructed JSON (Rule 9 in prompts.py already asks
    # for this) + manual parsing, still validated through the same
    # Pydantic schema as the structured-output path below.
    raw_message = llm.invoke(prompt_messages)
    parsed_entries = extract_json_array(raw_message.text)
    response = LLMDiscoveryResponse(
        companies=[LLMCompanyEntry.model_validate(e) for e in parsed_entries]
    )
else:
    response = llm.with_structured_output(LLMDiscoveryResponse).invoke(prompt_messages)
```

Both branches funnel into the same `response.companies: List[LLMCompanyEntry]`
shape, so everything after this point in `discover()` is identical
regardless of which branch ran. `_validate_company_entry` shrinks to an
adapter: Pydantic already guarantees types and enum membership, so the
"safe default on malformed value" logic mostly goes away (a validation
failure is now a hard error, handled at the call site — see Error
Handling). The constraint dict is rebuilt by iterating
`List[LLMConstraintEvaluation]` and keying on `.name`.

Everything downstream of parsing — identifier lookup, verification gating,
evidence retrieval sequencing, `CompanyResult`/`RejectedCompany` assembly —
is unchanged.

### `evidence_extractor.py`

Same pattern: `get_provider().with_structured_output(LLMEvidenceResponse)`.
The existing URL-allowlist backstop (`_validate_evidence_entry` dropping
any entry citing a URL not in the supplied documents) is unchanged — that's
a business-logic safety net against hallucinated citations, not a parsing
concern, and stays regardless of how the response was parsed.

### `json_utils.py`

**Kept** (revised from the original plan to delete it) — `discovery.py`'s
grounded fallback branch still needs `extract_json_array()` to pull the
JSON array out of the raw model text. `evidence_extractor.py` no longer
uses it, since it never takes a grounded/manual-parse path.

### `requirements.txt`

Add `langchain`, `langchain-openai`, `langchain-google-genai`, `pydantic`
(the `llm_schemas.py` models depend on it directly, so it's declared
explicitly rather than relied on as a transitive dependency). Replace
`google-generativeai` with `google-genai` (fixing the pre-existing
mismatch between the declared and actually-imported package). Verified
compatible versions during design research: `langchain 1.3.14`,
`langchain-openai 1.4.0`, `langchain-google-genai 4.3.1`, `google-genai
2.13.0`, `pydantic 2.13.4`.

## Error handling

`invoke()` failures (provider/auth errors, or the model producing output
that fails schema validation) are caught at the same call site that
`provider.generate()` errors are caught today, and re-raised as
`DiscoveryError` (discovery.py) or swallowed to `[]` (evidence_extractor.py,
matching its existing "evidence is enrichment, not required" behavior). In
discovery.py's grounded branch, both `JsonArrayParseError` (from
`extract_json_array`) and `pydantic.ValidationError` (from
`LLMCompanyEntry.model_validate`) are caught alongside the
`with_structured_output` branch's failure modes, so either branch produces
the same `DiscoveryError` outcome from the caller's perspective.

## Testing

New `test_discovery_structured_output.py`, in the existing
script-plus-`check()` style (no pytest introduced). Stubs `get_provider()`
(and `is_grounded()`) to exercise both branches:

- **Non-grounded branch:** a fake chat model whose `.with_structured_output(...)` /
  `.invoke(...)` chain returns a canned `LLMDiscoveryResponse`. Verifies a
  valid response converts correctly to `CompanyResult`, with the constraint
  dict rebuilt from `List[LLMConstraintEvaluation]`; a `decision: "REJECT"`
  entry lands in `DiscoveryResult.rejected` with `rejection_type="llm"`; and
  a simulated failure surfaces as `DiscoveryError`.
- **Grounded branch:** a fake chat model whose plain `.invoke(...)` returns
  an `AIMessage`-like object with a raw JSON-array `.text`, confirming the
  fallback parses and validates it into the same shape, and that malformed
  JSON or a schema-violating entry both surface as `DiscoveryError`.

A similar small stub-based test covers `evidence_extractor.py`'s
URL-allowlist backstop still functioning against the new response shape.

`test_gst.py` and `test_entity_matching.py` are untouched — neither
exercises the LLM call path.

## Manual verification

`streamlit run ui.py` should behave identically to today once Phase 1
lands: `ui.py` only ever consumes `CompanyResult` / `RejectedCompany` /
`DiscoveryError`, none of which change shape in this phase.

## Out of scope for Phase 1

- Conversation memory, follow-up resolution, intent classification (Phase 2).
- Evidence-grounded free-form Q&A beyond per-company evidence bullets (Phase 3).
- A cross-stage debug/execution trace beyond what `ui.py` already renders
  for discovery (Phase 4).
- Cleanup of the `*.py copy` backup files sitting alongside the real
  modules — unrelated to this migration, flagged for later.
