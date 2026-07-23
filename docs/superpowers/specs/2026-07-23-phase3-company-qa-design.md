# Phase 3: Company Question Answering — Design

## Context

This is the third of four planned phases evolving the Company Discovery
Agent into a conversational assistant:

1. LangChain migration — done
2. Conversation memory — done (see
   `docs/superpowers/specs/2026-07-23-phase2-conversation-memory-design.md`)
3. **Company question answering** (this spec)
4. Debug mode

Phase 2's follow-ups (`FOLLOW_UP_COMPANY`, `COMPARISON`, `RECALL`) only ever
re-present facts already sitting in a `CompanyResult` from a prior search —
zero new retrieval. Phase 3 is for questions where the answer genuinely
isn't already known ("who are its directors," "who are its competitors,"
"when was it founded") and may require fetching more evidence.

### Why this phase touches `discovery.py`

Phases 1 and 2 both deliberately left `discovery.py` untouched. Phase 3
can't: "reuse previous retrieval results whenever possible" requires the
raw documents `discovery.py` already retrieves during evidence extraction
to survive somewhere, and right now they don't — `evidence_extractor.py`
summarizes them into bullet points and the underlying pages are discarded
once that call returns. This was an explicit, considered decision (see
below), not an incidental scope creep.

## Decisions made during brainstorming

| Question | Decision |
|---|---|
| Where to persist retrieved documents for reuse | Add `CompanyResult.documents: List[Document]`, populated with the same documents `discover()` already fetches (currently discarded after evidence extraction). Additive, backward-compatible — no existing `CompanyResult` consumer needs to change. Storage footprint is modest: `retriever.py` already caps evidence retrieval at 10 documents/company, each typically 5-50KB of cleaned text, held only in `st.session_state` for the browser session (no disk/database persistence). |
| Sufficiency check structure | One combined LLM call that tries to answer and explicitly flags when it can't (`LLMQAAnswer.answered`), rather than a separate check-then-answer pair. Cheaper in the common case (existing evidence already answers most questions), simpler to reason about. |
| Retrieval when insufficient | A new, question-aware `retriever.retrieve_for_question(company_name, question)`, combining the company name with the question's own terms, rather than reusing the generic `retrieve_for_evidence()` queries verbatim — a "who are the directors" question needs a different search than the generic "about us" queries discovery already ran. |

## Architecture

`qa.py` (new) sits alongside `conversation.py`'s existing handlers.
`discovery.py` gets one additive field; `retriever.py` gets one new
function; `conversation.py` gets a new intent and handler. `ui.py` needs
**no changes** — the existing "reply text + company cards" turn rendering
from Phase 2 already covers a QA answer with no new UI work.

## Components

### `discovery.py`

`CompanyResult` gains:
```python
documents: List[Document] = field(default_factory=list)
```
Populated in `discover()`'s existing per-company loop with the same
`evidence_documents` list already fetched via
`retriever.retrieve_for_evidence(entry["company_name"])` (currently passed
only to `evidence_extractor.extract()` and then discarded). One additional
line in the existing `CompanyResult(...)` construction; nothing else in
the verification-gates-evidence pipeline ordering changes.

### `retriever.py`

New function:
```python
def retrieve_for_question(company_name: str, question: str) -> List[Document]:
```
Combines the company name with the question's own terms (e.g.
`'"Acme Corp" board of directors'`) as one query, plus a generic
`'"Acme Corp"'` fallback — built on the existing `_retrieve()` primitive,
the same pattern `retrieve_for_evidence()` already uses. No registry-domain
bias, same as evidence retrieval.

### `llm_schemas.py`

- `LLMIntentClassification.intent` gains `"COMPANY_QUESTION"` as a new
  Literal value, alongside the existing five.
- New model:
```python
class LLMQAAnswer(BaseModel):
    answered: bool  # True if `answer` is grounded in the supplied evidence
    answer: str
    missing_information: str  # what's missing, when answered=False (diagnostic; not used to rewrite the retry query)
    confidence: Literal["High", "Medium", "Low"]
```

### `prompts.py`

- `QA_SYSTEM_PROMPT` / `QA_PROMPT` — same anti-fabrication discipline as
  `EVIDENCE_SYSTEM_PROMPT`: answer strictly from supplied evidence
  (`CompanyResult.evidence` bullets and `CompanyResult.documents`' raw
  text, truncated the same way `build_evidence_prompt` already truncates),
  and say plainly when the question can't be answered from what's given.
  `build_qa_prompt(evidence, documents, question)` — structural params left
  untyped, same leaf-module convention as `build_evidence_prompt`'s
  `documents` and `build_intent_prompt`'s `discovery_history`.
- `INTENT_SYSTEM_PROMPT` gets a short addition distinguishing
  `COMPANY_QUESTION` (needs research — competitors, directors, founding
  date, "summarize everything you know") from `FOLLOW_UP_COMPANY`
  (already-known structured fields — GST, CIN, confidence, reason).

### `qa.py` (new)

```python
@dataclass(frozen=True)
class QAAnswer:
    answer: str
    confidence: str
    used_new_retrieval: bool
    sources: List[str]  # document URLs the answer could be grounded in
    documents: List[Document]  # the full pool actually used (existing + any newly retrieved)


def answer_question(company: CompanyResult, question: str) -> QAAnswer: ...
```

First attempt uses `company.evidence` + `company.documents`. If
`LLMQAAnswer.answered` is `False`, retrieves once via
`retriever.retrieve_for_question(company.company_name, question)`, merges
with the existing documents (deduplicated by URL), and tries exactly once
more with the combined pool — whatever that second attempt returns (even
"still couldn't find it") is final. No further looping, ever: bounded to
at most 2 LLM calls and 1 extra retrieval per question.

Never raises. A provider/schema failure at any point degrades to a plain
`QAAnswer` with a "couldn't find an answer" message and
`documents=company.documents` (unchanged) — matching
`evidence_extractor.py`'s "this is enrichment, not the core request"
posture, not `discovery.py`'s hard-error posture. `qa.py` never touches
`ConversationState` — it's a pure function of a `CompanyResult` and a
question; persisting the updated document pool back onto state is the
caller's job (see below), keeping the same clean layering `discovery.py`
already has with respect to `conversation.py`.

### `conversation.py`

New `_handle_company_question(state, classification, user_message)`:
resolves the referenced company via the existing `_find_company` (same
graceful "I don't have that company" degradation `_handle_follow_up_company`
already has), calls `qa.answer_question(company, user_message)`, and then
mutates `company.documents = qa_answer.documents` **in place** on the same
`CompanyResult` object already sitting in `state.discovery_history` — this
is what makes reuse work *across* questions, not just within one: a second
question about the same company benefits from whatever the first question
retrieved, since it's the same object reference in memory, not a copy.

`handle_message`'s routing gains one more branch:
`elif classification.intent == "COMPANY_QUESTION": reply = _handle_company_question(...)`.

### `ui.py`

**No changes.** `AssistantReply(text=qa_answer.answer, companies=[company])`
renders through the exact same turn-rendering code Phase 2 already built —
reply text, then the company card. Showing `sources` as clickable links is
a possible future polish, not attempted here (YAGNI — the existing
evidence-bullet links already give a citation UX; duplicating that
specifically for QA answers isn't required by anything in this phase's
scope).

## Data flow

```
"Who are its directors?"
  -> classify_intent -> COMPANY_QUESTION, referenced_company_names=["Acme Corp"]
  -> _find_company(state, "Acme Corp")
  -> qa.answer_question(company, "Who are its directors?")
       -> LLMQAAnswer from company.evidence + company.documents
       -> if answered: done
       -> if not: retriever.retrieve_for_question("Acme Corp", "Who are its directors?")
                  -> merge documents (dedup by URL), retry once, return final QAAnswer regardless
  -> company.documents updated in place (persists for the next question about Acme Corp)
  -> AssistantReply(text=answer, companies=[company])
```

## Error handling

`qa.answer_question` never raises. Unresolvable company references
degrade the same way `_handle_follow_up_company` already does (a plain "I
don't have that company" reply, no crash).

## Testing

Same script + `check()` convention, no pytest. New `test_qa.py`: answered
on first try (no retrieval), insufficient then retrieved-and-answered,
insufficient then still-insufficient after retrieval (confirms the "no
further looping" bound), provider failure degrading gracefully. New
`test_retriever.py` (first test file for this module) verifying
`retrieve_for_question`'s query construction via a monkeypatched search —
no real network access. `test_discovery_structured_output.py`,
`test_prompts_chat_templates.py`, `test_llm_schemas.py`, `test_conversation.py`
all get extended (new checks appended), not replaced.

## Out of scope for Phase 3

- Showing `sources`/citations in `ui.py` beyond what already renders via
  the reply text — a possible future polish, not required now.
- A debug trace of the sufficiency-check decision, retrieval queries, or
  which documents were used — Phase 4's job.
- Any change to `identifier_lookup.py`/`entity_matching.py` — GST/CIN
  verification is untouched by this phase.
- Capping/truncating `CompanyResult.documents`' stored text size — not
  needed at the current per-company retrieval cap (see the brainstorming
  decision above), but noted as a lever if a future phase needs it.
