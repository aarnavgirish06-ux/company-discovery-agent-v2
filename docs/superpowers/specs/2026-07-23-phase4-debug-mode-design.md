# Phase 4: Debug Mode — Design

## Context

This is the fourth and final planned phase evolving the Company Discovery
Agent into a conversational assistant:

1. LangChain migration — done
2. Conversation memory — done
3. Company question answering — done
4. **Debug mode** (this spec)

Phases 2 and 3 already left real diagnostic data lying around
uncollected or undisplayed:

- `identifier_lookup.py` already computes rich fallback-chain reasoning
  (which site was checked, what candidate was found, why a page didn't
  count) -- currently only as `print()` statements to stdout.
- `LLMIntentClassification.reasoning`/`.confidence` (Phase 2) is computed
  on every message and immediately discarded.
- `QAAnswer.sources`/`.used_new_retrieval` (Phase 3) is computed on every
  question and immediately discarded -- flagged as "currently dead data"
  in Phase 3's final review.
- `retriever.py` and `evidence_extractor.py` genuinely don't track *why*
  a page was skipped or a citation rejected, as data, at all.

Company discovery's debug view (constraint evaluation, ACCEPT/REJECT
decision, rejected-candidates panel) already exists in `ui.py` from before
this whole project started and needs no new work.

## Decision made during brainstorming

| Question | Decision |
|---|---|
| How to get real (not approximated) trace data out of `retriever.py`, `identifier_lookup.py`, `evidence_extractor.py` | Change their return types to `(result, trace)` tuples, built from the same internal decisions those modules already make. Tracing is unconditional (negligible cost -- a few dataclass appends), not optional/opt-in, since there's no reason to sometimes skip work this cheap. This is a **breaking signature change** requiring every caller (`discovery.py`, `qa.py`, and existing tests that call these functions directly) to unpack the new tuple -- confirmed acceptable in exchange for genuinely complete data rather than a coarse approximation. |

## Architecture

Every pipeline module gains a frozen trace dataclass and starts returning
`(result, trace)`. `discovery.py` collects these per company onto
`CompanyResult`/`RejectedCompany`. `conversation.py` captures the intent
classification and QA data it already computes, storing it on `ChatTurn`
instead of discarding it. `ui.py` renders all six debug sections --
extending the existing per-company debug expander for
discovery/retrieval/identifier/evidence, and adding a new turn-level block
for intent classification and QA sourcing.

**This phase is purely additive to what's already visible.** The existing
`_render_debug_html()` sections (constraints identified, constraint
evaluation rows, decision badge, confidence explanation, assumptions) and
the existing `_render_rejected_card_html()` panel are untouched in
content and appearance -- new subsections are appended after them. Every
new data field (on `CompanyResult`, `RejectedCompany`, `ChatTurn`) is
additive with a default. Everything new only renders when the existing
"Show Debug Information" toggle is already on; the toggle-off experience
is completely unchanged.

**Compliance with "never reveal the model's internal chain of thought":**
everything surfaced is either (a) a structured field the model was
explicitly asked to produce as transparent, auditable reasoning --
`constraint_evaluation`, `LLMIntentClassification.reasoning`,
`LLMQAAnswer.missing_information` -- or (b) a deterministic, code-side
fact about what Python actually did (which query ran, which URL
downloaded, which citation got dropped and why). None of it is raw model
deliberation.

## Components

### `retriever.py`

```python
@dataclass(frozen=True)
class RetrievalAttempt:
    """One page candidate considered during a retrieval call."""
    url: str
    included: bool  # True if downloaded and returned as a Document
    reason: str  # e.g. "downloaded successfully", "download failed", "exceeded max_documents cap"

@dataclass(frozen=True)
class RetrievalTrace:
    """Structured, engineered record of what one retrieval call actually did."""
    queries: List[str]
    attempts: List[RetrievalAttempt]
```

`_retrieve()` (the shared eager-retrieval primitive) builds this trace as
it works. `retrieve_for_evidence(company_name) -> Tuple[List[Document], RetrievalTrace]`
and `retrieve_for_question(company_name, question) -> Tuple[List[Document], RetrievalTrace]`
both get it for free, since both are built on `_retrieve()`.
`iter_gst_documents()` (the lazy, early-stopping registry path used by
identifier lookup) is untouched -- its trace lives in
`identifier_lookup.py` instead, since that's the module with the
early-stopping/corroboration logic worth explaining.

### `identifier_lookup.py`

```python
@dataclass(frozen=True)
class SiteCheck:
    """One page checked during the identifier fallback chain."""
    site: str
    url: str
    title: str
    company_detected: bool
    candidates_found: Dict[str, str]  # e.g. {"GST": "27ABCCE1234F1Z2"}

@dataclass(frozen=True)
class IdentifierTrace:
    """Structured, engineered record of the identifier fallback chain's execution."""
    site_checks: List[SiteCheck]
    corroboration_counts: Dict[str, int]  # e.g. {"GST": 1, "CIN": 2}
    validation_notes: Dict[str, str]  # e.g. {"GST": "Luhn mod-36 checksum valid", "CIN": "structural format valid (no checksum exists for CIN)"}
```

`get_company_identifiers(company_name) -> Tuple[Dict[str, List[IdentifierRecord] | str], IdentifierTrace]`.
Built from the same information the existing `print()` statements in
`_lookup_via_retrieval_chain` already compute -- those stay untouched
(they still serve `test_gst.py`'s interactive console debugging); this is
an additional, structured capture of the same facts, not a replacement.

### `evidence_extractor.py`

```python
@dataclass(frozen=True)
class EvidenceRejection:
    """One LLM-proposed evidence entry that was dropped, and why."""
    point: str
    source_url: str
    reason: str  # e.g. "cited a URL not in the supplied documents"

@dataclass(frozen=True)
class EvidenceTrace:
    """Structured record of evidence selection/rejection during extraction."""
    selected: List[str]  # points kept -- mirrors the returned EvidenceItem list, for a self-contained trace view
    rejected: List[EvidenceRejection]
```

`extract(company_name, user_query, discovery_reason, documents) -> Tuple[List[EvidenceItem], EvidenceTrace]`.
Captures exactly which entries `_validate_evidence_entry`'s URL-allowlist
backstop dropped, and why -- previously a silent `return None`.

### `discovery.py`

`CompanyResult` gains three new fields, all populated for every accepted
company:
```python
retrieval_trace: Optional[RetrievalTrace] = None
identifier_trace: Optional[IdentifierTrace] = None
evidence_trace: Optional[EvidenceTrace] = None
```

`RejectedCompany` gains one new field:
```python
identifier_trace: Optional[IdentifierTrace] = None
```
Populated only when `rejection_type == "verification"` -- LLM rejections
never reach identifier lookup at all (existing, unchanged behavior), so
`identifier_trace` stays `None` there, mirroring the existing `gst=None`/
`cin=None` pattern already on `RejectedCompany` for that case.

`discover()`'s three call sites (`retriever.retrieve_for_evidence(...)`,
`get_company_identifiers(...)`, `extract_evidence(...)`) unpack the new
tuples instead of a single return value. No other change to `discover()`'s
pipeline ordering, verification gating, or control flow.

### `conversation.py`

`ChatTurn` gains four new fields:
```python
intent_reasoning: str = ""
intent_confidence: str = ""
qa_sources: List[str] = field(default_factory=list)
qa_used_new_retrieval: bool = False
```
`intent_reasoning`/`intent_confidence` are populated from
`LLMIntentClassification.reasoning`/`.confidence` on every successful
classification (empty string on the rare classification-failure/empty-
message paths, where there's nothing to report). `qa_sources`/
`qa_used_new_retrieval` are populated from `QAAnswer` only on
`COMPANY_QUESTION` turns; they stay at their defaults otherwise.

### `ui.py`

- `_render_debug_html(company)` gets three new subsections appended after
  the existing ones: **Retrieval** (queries issued, pages included/
  discarded with reasons), **Identifier Lookup** (site-by-site checks,
  corroboration counts, validation notes), **Evidence Extraction**
  (selected vs. rejected entries with reasons). All read straight off
  `company.retrieval_trace`/`.identifier_trace`/`.evidence_trace`.
- `_render_rejected_card_html(rejected)` gets one new subsection
  (**Identifier Lookup**) shown only when `rejected.identifier_trace` is
  not `None` (i.e. only for verification-type rejections).
- A new per-turn debug block (rendered in `_render_turn`, only when
  `show_debug` is on) shows: detected intent, confidence, and reasoning
  for every turn; and, for `COMPANY_QUESTION` turns specifically, whether
  new retrieval was used and which source URLs were consulted.

## Testing

Same script + `check()` convention, no pytest. Every existing test file
that calls one of the three changed functions gets its call sites updated
to unpack the new tuple, plus new assertions verifying trace content is
correct: `test_retriever.py` (query list, attempt reasons),
`test_entity_matching.py` (which calls `get_company_identifiers()`
directly -- site checks, corroboration counts, validation notes),
`test_evidence_extractor_structured_output.py` (selected vs. rejected
entries with reasons), `test_discovery_structured_output.py` (traces
correctly attached to `CompanyResult`/`RejectedCompany`),
`test_conversation.py` (intent reasoning/confidence and QA sources/
retrieval-usage correctly attached to `ChatTurn`). `ui.py` gets no
automated test, same as every prior phase -- `py_compile` plus a manual
smoke test with the debug toggle on.

## Out of scope for Phase 4

- Any change to what the assistant actually decides or says -- this phase
  only adds visibility into decisions already being made, never alters
  them.
- A dedicated automated test file for `identifier_lookup.py` beyond what
  `test_entity_matching.py` already exercises -- out of scope to build one
  from scratch here.
- Persisting trace data anywhere beyond `st.session_state` for the current
  browser session -- same lifetime as everything else in `ConversationState`
  today.
