# Phase 3: Company Question Answering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the assistant answer arbitrary questions about a company ("who are its directors," "who are its competitors") by reusing already-gathered evidence/documents and retrieving more only when needed, without ever hallucinating.

**Architecture:** New `qa.py` answers a question about one `CompanyResult` in at most two LLM calls: try from existing evidence/documents; if insufficient, retrieve once via a new question-aware `retriever.retrieve_for_question()` and try again with the combined pool. `discovery.py` gains one additive field (`CompanyResult.documents`) so retrieved pages survive past evidence extraction instead of being discarded. `conversation.py` gains a new intent (`COMPANY_QUESTION`) and handler that persists any newly-retrieved documents back onto the stored `CompanyResult` so a later question about the same company reuses them too. `ui.py` needs no changes.

**Tech Stack:** Same as Phases 1-2 (`langchain`, `langchain-openai`, `langchain-google-genai`, `pydantic`) — no new dependencies.

## Global Constraints

- `qa.py` never touches `ConversationState` — it's a pure function of a `CompanyResult` and a question string. Persisting the returned document pool back onto state is the caller's (`conversation.py`'s) job.
- `qa.py` never raises. Any failure (provider error, schema validation failure) degrades to a plain "couldn't find an answer" `QAAnswer` — matching `evidence_extractor.py`'s posture (enrichment, not the core request), not `discovery.py`'s (hard error).
- Exactly one retry, ever: first attempt from existing evidence/documents, and — only if insufficient — one retrieval + one more attempt. Whatever the second attempt returns (even "still couldn't find it") is final. No loop.
- References ("it," a company name) resolve to exact company names via the intent classifier, same as Phase 2 — `conversation.py` does only exact case-insensitive lookup, never fuzzy matching or ordinal math.
- `CompanyResult.documents` is additive — no existing consumer of `CompanyResult` needs to change to keep working.
- `prompts.py`'s new `build_qa_prompt` leaves its structural parameters (`evidence`, `documents`) untyped, same leaf-module convention as `build_evidence_prompt`/`build_intent_prompt`/`build_response_synthesis_prompt`.
- Testing stays in the existing script + `check(label, condition)` harness convention — no pytest.
- Full design rationale lives in `docs/superpowers/specs/2026-07-23-phase3-company-qa-design.md`.

---

### Task 1: `llm_schemas.py` — add `LLMQAAnswer` and the `COMPANY_QUESTION` intent

**Files:**
- Modify: `llm_schemas.py`
- Test: `test_llm_schemas.py` (extend)

**Interfaces:**
- Produces: `LLMQAAnswer(answered: bool, answer: str, missing_information: str, confidence: Literal["High","Medium","Low"])`. `LLMIntentClassification.intent`'s Literal gains `"COMPANY_QUESTION"` as a sixth value.

- [ ] **Step 1: Write the failing test**

In `test_llm_schemas.py`, change the import line:
```python
from llm_schemas import (
    LLMCompanyEntry,
    LLMConstraintEvaluation,
    LLMDiscoveryResponse,
    LLMEvidenceItem,
    LLMEvidenceResponse,
    LLMIntentClassification,
)
```
to:
```python
from llm_schemas import (
    LLMCompanyEntry,
    LLMConstraintEvaluation,
    LLMDiscoveryResponse,
    LLMEvidenceItem,
    LLMEvidenceResponse,
    LLMIntentClassification,
    LLMQAAnswer,
)
```

Then insert this new section immediately before the file's final block (the one starting with `print()` / `print("=" * 70)` / the `_FAILURES` check):

```python
# ---------------------------------------------------------------------
# COMPANY_QUESTION intent + LLMQAAnswer
# ---------------------------------------------------------------------

company_question_classification = LLMIntentClassification.model_validate(
    {
        "intent": "COMPANY_QUESTION",
        "referenced_company_names": ["Acme Forgings Private Limited"],
        "reasoning": "The user asked who the directors are -- research, not an already-known field.",
        "confidence": "High",
    }
)
check(
    "LLMIntentClassification accepts the new COMPANY_QUESTION intent",
    company_question_classification.intent == "COMPANY_QUESTION",
)

answered_case = LLMQAAnswer.model_validate(
    {
        "answered": True,
        "answer": "The company was founded in 2005.",
        "confidence": "High",
    }
)
check("LLMQAAnswer validates an answered=True entry", answered_case.answered is True)
check("LLMQAAnswer defaults missing_information to an empty string", answered_case.missing_information == "")

not_answered_case = LLMQAAnswer.model_validate(
    {
        "answered": False,
        "answer": "",
        "missing_information": "no director information was found in the supplied pages",
        "confidence": "Low",
    }
)
check("LLMQAAnswer validates an answered=False entry", not_answered_case.answered is False)
check(
    "LLMQAAnswer carries the missing_information explanation",
    not_answered_case.missing_information == "no director information was found in the supplied pages",
)

try:
    LLMQAAnswer.model_validate({"answered": True, "answer": "x", "confidence": "Extreme"})
    check("LLMQAAnswer rejects an invalid confidence value", False, "no exception raised")
except ValidationError:
    check("LLMQAAnswer rejects an invalid confidence value", True)

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_llm_schemas.py`
Expected: `ImportError: cannot import name 'LLMQAAnswer' from 'llm_schemas'`

- [ ] **Step 3: Update `llm_schemas.py`**

Change:
```python
    intent: Literal["NEW_DISCOVERY", "FOLLOW_UP_COMPANY", "COMPARISON", "RECALL", "UNRECOGNIZED"]
```
to:
```python
    intent: Literal["NEW_DISCOVERY", "FOLLOW_UP_COMPANY", "COMPANY_QUESTION", "COMPARISON", "RECALL", "UNRECOGNIZED"]
```

Append this class at the end of the file (after `LLMIntentClassification`):

```python


class LLMQAAnswer(BaseModel):
    """
    The outcome of one attempt to answer a question about a company from
    supplied evidence. answered=False signals that qa.py should retrieve
    more evidence and retry once -- see qa.py's module docstring.
    """

    answered: bool = Field(
        description="True if `answer` is fully grounded in the supplied evidence/documents."
    )
    answer: str = Field(
        description="The answer text if answered=True; a brief note if answered=False."
    )
    missing_information: str = Field(
        default="",
        description="What's missing to answer the question, when answered=False. Diagnostic only.",
    )
    confidence: Literal["High", "Medium", "Low"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_llm_schemas.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add llm_schemas.py test_llm_schemas.py
git commit -m "Add LLMQAAnswer schema and COMPANY_QUESTION intent for Phase 3"
```

---

### Task 2: `retriever.py` — add `retrieve_for_question`

**Files:**
- Modify: `retriever.py`
- Test: `test_retriever.py` (new)

**Interfaces:**
- Produces: `retrieve_for_question(company_name: str, question: str) -> List[Document]`.

- [ ] **Step 1: Write the failing test**

Create `test_retriever.py`:

```python
"""
test_retriever.py

Unit tests for retriever.py's retrieve_for_question(), verifying its
query construction without making real network calls --
retriever._search_company_pages is monkeypatched to capture the queries
it's called with and return no results, so _retrieve() never attempts a
real download.

Run with: python3 test_retriever.py
"""

from __future__ import annotations

import retriever
from retriever import retrieve_for_question

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


_captured_queries: list[str] = []


def _fake_search_company_pages(query, max_results=10):
    _captured_queries.append(query)
    return []


retriever._search_company_pages = _fake_search_company_pages

_captured_queries.clear()
result = retrieve_for_question("Acme Forgings Private Limited", "Who are the directors?")

check("retrieve_for_question returns an empty list when search yields no URLs", result == [])
check(
    "retrieve_for_question issues a query combining the company name and the question",
    any(
        "Acme Forgings Private Limited" in q and "Who are the directors?" in q
        for q in _captured_queries
    ),
    _captured_queries,
)
check(
    "retrieve_for_question also issues a generic company-name-only query",
    any(q == '"Acme Forgings Private Limited"' for q in _captured_queries),
    _captured_queries,
)
check("retrieve_for_question issues exactly 2 queries", len(_captured_queries) == 2, _captured_queries)


print()
print("=" * 70)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S):")
    for f in _FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_retriever.py`
Expected: `ImportError: cannot import name 'retrieve_for_question' from 'retriever'`

- [ ] **Step 3: Add `retrieve_for_question` to `retriever.py`**

Append at the very end of the file (after `iter_gst_documents`):

```python


def retrieve_for_question(company_name: str, question: str) -> List[Document]:
    """
    Retrieves pages likely to answer a specific question about
    `company_name` -- unlike retrieve_for_evidence()'s generic "about us
    products" queries, this combines the company name with the question's
    own terms, since a question like "who are the directors" needs a more
    targeted search than a generic company-name query would surface.
    Intended consumer: qa.py, when existing evidence/documents aren't
    enough to answer a question.
    """
    queries = [
        f'"{company_name}" {question}',
        f'"{company_name}"',
    ]
    return _retrieve(queries, _EVIDENCE_PREFERRED_DOMAINS, _MAX_DOCUMENTS_PER_COMPANY)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_retriever.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add retriever.py test_retriever.py
git commit -m "Add retriever.retrieve_for_question for Phase 3 QA"
```

---

### Task 3: `discovery.py` — add `CompanyResult.documents`

**Files:**
- Modify: `discovery.py`
- Test: `test_discovery_structured_output.py` (extend)

**Interfaces:**
- Produces: `CompanyResult.documents: List[retriever.Document]` (new field, populated by `discover()`, defaults to `[]`). `discover()`'s signature and every other field are unchanged.

- [ ] **Step 1: Write the failing test**

In `test_discovery_structured_output.py`, change the import block:
```python
from types import SimpleNamespace

import discovery
from identifier_lookup import IdentifierRecord
from llm_schemas import LLMCompanyEntry, LLMConstraintEvaluation, LLMDiscoveryResponse

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


def _stub_downstream_pipeline() -> None:
    """
    Stubs everything discover() calls after parsing the LLM response, so
    these tests only exercise the LLM-calling/parsing change itself.
    """
    discovery.get_company_identifiers = lambda name: {
        "GST": [
            IdentifierRecord(
                identifier_type="GST",
                value="27AAAAA0000A1Z5",
                corroboration_key="AAAAA0000A",
                corroborations=2,
                source_note="stub",
            )
        ],
        "CIN": "CIN not verified",
    }
    discovery.retriever.retrieve_for_evidence = lambda name: []
    discovery.extract_evidence = lambda *args, **kwargs: []
```
to:
```python
from types import SimpleNamespace

import discovery
from identifier_lookup import IdentifierRecord
from llm_schemas import LLMCompanyEntry, LLMConstraintEvaluation, LLMDiscoveryResponse
from retriever import Document

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


_FIXTURE_DOCUMENT = Document(url="https://example.com/about", title="About Acme", cleaned_text="Acme makes forgings.")


def _stub_downstream_pipeline() -> None:
    """
    Stubs everything discover() calls after parsing the LLM response, so
    these tests only exercise the LLM-calling/parsing change itself.
    """
    discovery.get_company_identifiers = lambda name: {
        "GST": [
            IdentifierRecord(
                identifier_type="GST",
                value="27AAAAA0000A1Z5",
                corroboration_key="AAAAA0000A",
                corroborations=2,
                source_note="stub",
            )
        ],
        "CIN": "CIN not verified",
    }
    discovery.retriever.retrieve_for_evidence = lambda name: [_FIXTURE_DOCUMENT]
    discovery.extract_evidence = lambda *args, **kwargs: []
```

Then, in the same file, change:
```python
check("Non-grounded: one LLM rejection recorded", len(result.rejected) == 1, len(result.rejected))
check(
    "Non-grounded: rejected company is tagged rejection_type='llm'",
    bool(result.rejected) and result.rejected[0].rejection_type == "llm",
)
```
to:
```python
check("Non-grounded: one LLM rejection recorded", len(result.rejected) == 1, len(result.rejected))
check(
    "Non-grounded: rejected company is tagged rejection_type='llm'",
    bool(result.rejected) and result.rejected[0].rejection_type == "llm",
)
check(
    "Non-grounded: accepted company's documents are populated from retrieve_for_evidence, not discarded",
    bool(result.accepted) and result.accepted[0].documents == [_FIXTURE_DOCUMENT],
)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_discovery_structured_output.py`
Expected: `FAIL` on the new "documents are populated" check (or a `TypeError: __init__() got an unexpected keyword argument 'documents'` if `discovery.py` doesn't yet accept it — either is an expected failure at this point).

- [ ] **Step 3: Update `discovery.py`**

Change:
```python
@dataclass
class CompanyResult:
    """A single company result, ready for display by any frontend."""
    company_name: str
    reason: str
    confidence: str  # exactly what the LLM returned: "High" | "Medium" | "Low"
    gst: Optional[str]  # representative GSTIN if found, else None ("GST not found")
    cin: Optional[str]  # representative CIN if found, else None ("CIN not found")
    pan: Optional[str]  # derived from a found GSTIN's embedded PAN; None if no GST found
    evidence: List[EvidenceItem] = field(default_factory=list)
    # -- Debug-mode fields (see module docstring). Never affect filtering/ranking. --
    constraint_evaluation: Dict[str, ConstraintEvaluation] = field(default_factory=dict)
    decision: str = "ACCEPT"  # "ACCEPT" | "REJECT" -- explicit LLM decision for this entry
```
to:
```python
@dataclass
class CompanyResult:
    """A single company result, ready for display by any frontend."""
    company_name: str
    reason: str
    confidence: str  # exactly what the LLM returned: "High" | "Medium" | "Low"
    gst: Optional[str]  # representative GSTIN if found, else None ("GST not found")
    cin: Optional[str]  # representative CIN if found, else None ("CIN not found")
    pan: Optional[str]  # derived from a found GSTIN's embedded PAN; None if no GST found
    evidence: List[EvidenceItem] = field(default_factory=list)
    documents: List[retriever.Document] = field(default_factory=list)  # raw pages retrieved for evidence, kept (not discarded) so qa.py can reuse them without re-fetching
    # -- Debug-mode fields (see module docstring). Never affect filtering/ranking. --
    constraint_evaluation: Dict[str, ConstraintEvaluation] = field(default_factory=dict)
    decision: str = "ACCEPT"  # "ACCEPT" | "REJECT" -- explicit LLM decision for this entry
```

Then change:
```python
        # STEP 3b (verified): proceed exactly as before -- retrieve, then
        # extract, evidence for this company.
        evidence_documents = retriever.retrieve_for_evidence(entry["company_name"])
        evidence = extract_evidence(entry["company_name"], query, entry["reason"], evidence_documents)

        accepted.append(
            CompanyResult(
                company_name=entry["company_name"],
                reason=entry["reason"],
                confidence=entry["confidence"],
                gst=gst_display,
                cin=cin_display,
                pan=pan_display,
                evidence=evidence,
                constraint_evaluation=entry["constraint_evaluation"],
                decision=entry["decision"],
            )
        )
```
to:
```python
        # STEP 3b (verified): proceed exactly as before -- retrieve, then
        # extract, evidence for this company.
        evidence_documents = retriever.retrieve_for_evidence(entry["company_name"])
        evidence = extract_evidence(entry["company_name"], query, entry["reason"], evidence_documents)

        accepted.append(
            CompanyResult(
                company_name=entry["company_name"],
                reason=entry["reason"],
                confidence=entry["confidence"],
                gst=gst_display,
                cin=cin_display,
                pan=pan_display,
                evidence=evidence,
                documents=evidence_documents,
                constraint_evaluation=entry["constraint_evaluation"],
                decision=entry["decision"],
            )
        )
```

(`retriever.Document` is already resolvable here without a new import: `discovery.py` already has `from __future__ import annotations` and `import retriever` at the top, so the type annotation is fine as written.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_discovery_structured_output.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add discovery.py test_discovery_structured_output.py
git commit -m "Add CompanyResult.documents so evidence pages survive for QA reuse"
```

---

### Task 4: `prompts.py` — add QA prompt, update intent prompt for `COMPANY_QUESTION`

**Files:**
- Modify: `prompts.py`
- Test: `test_prompts_chat_templates.py` (extend)

**Interfaces:**
- Produces: `QA_SYSTEM_PROMPT: str`, `build_qa_prompt(evidence, documents, question: str) -> str`, `QA_PROMPT: ChatPromptTemplate`. `INTENT_SYSTEM_PROMPT` text is updated (still the same exported name).

- [ ] **Step 1: Write the failing test**

In `test_prompts_chat_templates.py`, change the import line:
```python
from prompts import (
    DISCOVERY_PROMPT,
    EVIDENCE_PROMPT,
    EVIDENCE_SYSTEM_PROMPT,
    INTENT_PROMPT,
    INTENT_SYSTEM_PROMPT,
    RESPONSE_SYNTHESIS_PROMPT,
    RESPONSE_SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
```
to:
```python
from prompts import (
    DISCOVERY_PROMPT,
    EVIDENCE_PROMPT,
    EVIDENCE_SYSTEM_PROMPT,
    INTENT_PROMPT,
    INTENT_SYSTEM_PROMPT,
    QA_PROMPT,
    QA_SYSTEM_PROMPT,
    RESPONSE_SYNTHESIS_PROMPT,
    RESPONSE_SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
```

Then insert this new section immediately before the file's final block (the one starting with `print()` / `print("=" * 70)` / the `_FAILURES` check):

```python
check("INTENT_SYSTEM_PROMPT mentions the new COMPANY_QUESTION intent", "COMPANY_QUESTION" in INTENT_SYSTEM_PROMPT)

qa_messages = QA_PROMPT.format_messages(user_prompt='Question: "test"\n\n(no evidence)')

check("QA_PROMPT produces exactly 2 messages", len(qa_messages) == 2)
check("QA_PROMPT's first message is a SystemMessage", isinstance(qa_messages[0], SystemMessage))
check(
    "QA_PROMPT's system message content is QA_SYSTEM_PROMPT verbatim",
    qa_messages[0].content == QA_SYSTEM_PROMPT,
)
check("QA_PROMPT's second message is a HumanMessage", isinstance(qa_messages[1], HumanMessage))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_prompts_chat_templates.py`
Expected: `ImportError: cannot import name 'QA_PROMPT' from 'prompts'`

- [ ] **Step 3: Update `prompts.py`**

Replace the entire `INTENT_SYSTEM_PROMPT` block:
```python
INTENT_SYSTEM_PROMPT = """You are the Intent Classifier for a conversational Company Discovery Agent.
You will be given a short history of the conversation so far -- prior user
messages, which ones triggered a new company search, the company names that
search returned (in the order they were presented), and which company (if
any) is currently "in focus" (the company most recently discussed) -- along
with the user's newest message.

Your job is to classify the newest message into exactly one of these intents:

- NEW_DISCOVERY: the user is asking a new company-discovery question (a
  fresh search), not referring back to anything already discussed.
- FOLLOW_UP_COMPANY: the user is asking about ONE specific company already
  mentioned in the conversation (e.g. "tell me more about the second
  company", "what was its GST number", using a pronoun, an ordinal
  reference, or a company name).
- COMPARISON: the user wants TWO previously-mentioned companies compared
  against each other.
- RECALL: the user wants to be reminded what was already found or said
  earlier (e.g. "what was the first recommendation", "what did you find
  before"), without asking anything new about a specific company.
- UNRECOGNIZED: the message doesn't clearly fit any of the above, or refers
  to a company that was never actually mentioned in the supplied history.

Rules:

1. RESOLVE REFERENCES TO EXACT NAMES. When the intent is FOLLOW_UP_COMPANY
   or COMPARISON, "referenced_company_names" MUST contain the exact company
   name(s) as they appear in the supplied conversation history -- never a
   paraphrase, abbreviation, or a name not present in that history. If you
   cannot confidently resolve a reference to one of the exact names
   supplied, classify as UNRECOGNIZED instead of guessing.

2. PRONOUNS AND ORDINALS RESOLVE AGAINST THE SUPPLIED CONTEXT ONLY. "it" or
   "that company" refers to whichever company is marked as currently in
   focus. "the second company" or "the first recommendation" refers to that
   position in the most recent search's result list. Never invent a
   company that isn't in the supplied history.

3. RECALL VS FOLLOW_UP_COMPANY. If the user wants to be reminded of
   results already given (with no new question about any single company),
   classify as RECALL and leave referenced_company_names empty. If they
   want more detail about, or a specific fact about, ONE company, that is
   FOLLOW_UP_COMPANY, even if phrased as a question.

4. NEVER FABRICATE. Do not invent company names, turns, or facts that
   aren't present in the supplied conversation context.

5. EXPLAIN YOUR REASONING. Give a short, concrete explanation of why you
   chose this intent and (if applicable) how you resolved any references.

6. OUTPUT FORMAT. Respond with ONLY a JSON object (no markdown fences, no
   prose before or after) with this exact shape:

{
  "intent": "NEW_DISCOVERY" | "FOLLOW_UP_COMPANY" | "COMPARISON" | "RECALL" | "UNRECOGNIZED",
  "referenced_company_names": ["string", ...],
  "recall_ordinal": integer or null,
  "reasoning": "string",
  "confidence": "High" | "Medium" | "Low"
}

Do not include any text outside the JSON object. Do not wrap the JSON in
markdown code fences.
"""
```
with:
```python
INTENT_SYSTEM_PROMPT = """You are the Intent Classifier for a conversational Company Discovery Agent.
You will be given a short history of the conversation so far -- prior user
messages, which ones triggered a new company search, the company names that
search returned (in the order they were presented), and which company (if
any) is currently "in focus" (the company most recently discussed) -- along
with the user's newest message.

Your job is to classify the newest message into exactly one of these intents:

- NEW_DISCOVERY: the user is asking a new company-discovery question (a
  fresh search), not referring back to anything already discussed.
- FOLLOW_UP_COMPANY: the user is asking about an attribute of ONE specific
  company already mentioned in the conversation that is already part of
  its recommendation (e.g. "what was its GST number", "why was it
  recommended", "what was its confidence level"), using a pronoun, an
  ordinal reference, or a company name.
- COMPANY_QUESTION: the user is asking something about ONE specific
  already-mentioned company that requires research beyond what's already
  been recommended/verified -- e.g. "who are its competitors", "when was
  it founded", "who are the directors", "what industries does it serve",
  "summarize everything you know about this company".
- COMPARISON: the user wants TWO previously-mentioned companies compared
  against each other.
- RECALL: the user wants to be reminded what was already found or said
  earlier (e.g. "what was the first recommendation", "what did you find
  before"), without asking anything new about a specific company.
- UNRECOGNIZED: the message doesn't clearly fit any of the above, or refers
  to a company that was never actually mentioned in the supplied history.

Rules:

1. RESOLVE REFERENCES TO EXACT NAMES. When the intent is FOLLOW_UP_COMPANY,
   COMPANY_QUESTION, or COMPARISON, "referenced_company_names" MUST contain
   the exact company name(s) as they appear in the supplied conversation
   history -- never a paraphrase, abbreviation, or a name not present in
   that history. If you cannot confidently resolve a reference to one of
   the exact names supplied, classify as UNRECOGNIZED instead of guessing.

2. PRONOUNS AND ORDINALS RESOLVE AGAINST THE SUPPLIED CONTEXT ONLY. "it" or
   "that company" refers to whichever company is marked as currently in
   focus. "the second company" or "the first recommendation" refers to that
   position in the most recent search's result list. Never invent a
   company that isn't in the supplied history.

3. RECALL VS FOLLOW_UP_COMPANY VS COMPANY_QUESTION. If the user wants to be
   reminded of results already given, with no new question about any
   single company, classify as RECALL and leave referenced_company_names
   empty. If they ask about ONE company, decide between the other two by
   what the answer requires: if it's something already tracked as part of
   that company's recommendation (GST/CIN, confidence, why it was
   recommended), that's FOLLOW_UP_COMPANY; if it requires information not
   already established (competitors, founding date, directors, industries
   served, a general summary), that's COMPANY_QUESTION.

4. NEVER FABRICATE. Do not invent company names, turns, or facts that
   aren't present in the supplied conversation context.

5. EXPLAIN YOUR REASONING. Give a short, concrete explanation of why you
   chose this intent and (if applicable) how you resolved any references.

6. OUTPUT FORMAT. Respond with ONLY a JSON object (no markdown fences, no
   prose before or after) with this exact shape:

{
  "intent": "NEW_DISCOVERY" | "FOLLOW_UP_COMPANY" | "COMPANY_QUESTION" | "COMPARISON" | "RECALL" | "UNRECOGNIZED",
  "referenced_company_names": ["string", ...],
  "recall_ordinal": integer or null,
  "reasoning": "string",
  "confidence": "High" | "Medium" | "Low"
}

Do not include any text outside the JSON object. Do not wrap the JSON in
markdown code fences.
"""
```

Then append this at the very end of the file (after the existing `RESPONSE_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(...)` block):

```python


QA_SYSTEM_PROMPT = """You are the Question Answering Analyst for a conversational Company
Discovery Agent. You will be given a specific question about one company,
along with whatever evidence and source-page text is already on file for
that company. Your job is to answer strictly from what you're given, and
to say plainly when you can't.

Follow these rules strictly:

1. USE ONLY THE SUPPLIED EVIDENCE. Do not use outside knowledge, prior
   training data, or assumptions about the company. If the supplied
   evidence and documents don't say it, you cannot report it.

2. NEVER FABRICATE. Do not invent facts, names, dates, or figures. If the
   supplied material doesn't answer the question, set "answered" to false
   rather than guessing or partially answering with invented details.

3. BE HONEST ABOUT GAPS. When you cannot answer, briefly describe what
   kind of information would be needed (e.g. "no information about the
   board of directors was found in the supplied pages") in
   "missing_information" -- this helps decide whether searching for more
   evidence is worthwhile.

4. BE CONCISE AND CONCRETE. When you can answer, write a clear, direct
   answer -- a few sentences is usually enough. Do not pad with
   disclaimers once you've decided the evidence supports an answer.

5. OUTPUT FORMAT. Respond with ONLY a JSON object (no markdown fences, no
   prose before or after) with this exact shape:

{
  "answered": true | false,
  "answer": "string",
  "missing_information": "string",
  "confidence": "High" | "Medium" | "Low"
}

Do not include any text outside the JSON object. Do not wrap the JSON in
markdown code fences.
"""


def build_qa_prompt(evidence, documents, question: str) -> str:
    """
    Builds the user-turn prompt for question answering.

    `evidence` is a list of discovery.EvidenceItem-like objects (only
    `.point` is read) and `documents` is a list of retriever.Document-like
    objects (only `.title`/`.url`/`.cleaned_text` are read) -- both left
    untyped for the same leaf-module reason build_evidence_prompt()'s
    `documents` parameter is: prompts.py never imports discovery.py or
    retriever.py.
    """
    if not evidence and not documents:
        return (
            f'Question: "{question}"\n\n'
            "No evidence or documents are on file for this company yet. "
            'Respond with {"answered": false, "answer": "", '
            '"missing_information": "no evidence on file yet", "confidence": "Low"}.'
        )

    sections = []

    if evidence:
        evidence_lines = "\n".join(f"- {item.point}" for item in evidence)
        sections.append(f"Known evidence bullets:\n{evidence_lines}")

    for i, document in enumerate(documents, start=1):
        # Defensive truncation, same as build_evidence_prompt() -- a
        # fact-finding read doesn't need a page's full text.
        excerpt = document.cleaned_text[:4000]
        sections.append(
            f"--- Document {i} ---\nURL: {document.url}\nPage title: {document.title}\nContent:\n{excerpt}"
        )

    material_section = "\n\n".join(sections)

    return (
        f'Question: "{question}"\n\n'
        f"{material_section}\n\n"
        "Respond using only the JSON object format described in your instructions."
    )


QA_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=QA_SYSTEM_PROMPT), ("human", "{user_prompt}")]
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_prompts_chat_templates.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add prompts.py test_prompts_chat_templates.py
git commit -m "Add QA prompt and COMPANY_QUESTION intent description"
```

---

### Task 5: `qa.py` — new evidence-grounded question-answering module

**Files:**
- Create: `qa.py`
- Test: `test_qa.py`

**Interfaces:**
- Consumes: `CompanyResult` (`discovery.py`); `get_provider`, `structured_output_kwargs` (`llm_provider.py`); `LLMQAAnswer` (`llm_schemas.py`, Task 1); `QA_PROMPT`, `build_qa_prompt` (`prompts.py`, Task 4); `Document`, `retrieve_for_question` (`retriever.py`, Task 2).
- Produces: `QAAnswer(answer: str, confidence: str, used_new_retrieval: bool, sources: List[str], documents: List[Document])`, `answer_question(company: CompanyResult, question: str) -> QAAnswer`. Task 6 (`conversation.py`) imports `answer_question`.

- [ ] **Step 1: Write the failing test**

Create `test_qa.py`:

```python
"""
test_qa.py

Unit tests for qa.py's answer-then-retry-once question answering. No
network calls are made -- qa.get_provider and qa.retrieve_for_question are
monkeypatched.

Run with: python3 test_qa.py
"""

from __future__ import annotations

import qa
from discovery import CompanyResult
from llm_schemas import LLMQAAnswer
from retriever import Document

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


def _make_company(documents=None) -> CompanyResult:
    return CompanyResult(
        company_name="Acme Forgings Private Limited",
        reason="fits",
        confidence="High",
        gst="27ABCCE1234F1Z2",
        cin=None,
        pan=None,
        evidence=[],
        documents=documents or [],
    )


class _FakeStructuredRunnable:
    def __init__(self, response_or_exception):
        self._response_or_exception = response_or_exception

    def invoke(self, messages):
        if isinstance(self._response_or_exception, Exception):
            raise self._response_or_exception
        return self._response_or_exception


class _FakeChatModel:
    def __init__(self, response_or_exception):
        self._response_or_exception = response_or_exception

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredRunnable(self._response_or_exception)


# ---------------------------------------------------------------------
# Answered on first try: no retrieval attempted
# ---------------------------------------------------------------------

existing_document = Document(url="https://example.com/about", title="About", cleaned_text="Founded in 2005.")

qa.get_provider = lambda **kwargs: _FakeChatModel(
    LLMQAAnswer(answered=True, answer="Founded in 2005.", missing_information="", confidence="High")
)


def _fail_if_retrieval_called(company_name, question):
    raise AssertionError("retrieve_for_question should not be called when the first attempt answers")


qa.retrieve_for_question = _fail_if_retrieval_called

company = _make_company(documents=[existing_document])
result = qa.answer_question(company, "When was it founded?")

check("Answered on first try: correct answer text", result.answer == "Founded in 2005.")
check("Answered on first try: used_new_retrieval is False", result.used_new_retrieval is False)
check("Answered on first try: documents unchanged", result.documents == [existing_document])
check(
    "Answered on first try: sources include the existing document's URL",
    result.sources == ["https://example.com/about"],
)

# ---------------------------------------------------------------------
# Insufficient on first try, retrieved, answered on second try
# ---------------------------------------------------------------------

new_document = Document(url="https://example.com/directors", title="Directors", cleaned_text="Directors: Jane Doe, John Smith.")

_answers = iter(
    [
        LLMQAAnswer(answered=False, answer="", missing_information="no director info", confidence="Low"),
        LLMQAAnswer(answered=True, answer="Jane Doe and John Smith.", missing_information="", confidence="High"),
    ]
)
qa.get_provider = lambda **kwargs: _FakeChatModel(next(_answers))
qa.retrieve_for_question = lambda company_name, question: [new_document]

company = _make_company(documents=[])
result = qa.answer_question(company, "Who are the directors?")

check("Insufficient then answered: correct answer text", result.answer == "Jane Doe and John Smith.")
check("Insufficient then answered: used_new_retrieval is True", result.used_new_retrieval is True)
check("Insufficient then answered: documents include the newly retrieved one", new_document in result.documents)

# ---------------------------------------------------------------------
# Insufficient on both tries -- returns the honest "couldn't find" answer, no further looping
# ---------------------------------------------------------------------

_answers2 = iter(
    [
        LLMQAAnswer(answered=False, answer="", missing_information="no competitor info", confidence="Low"),
        LLMQAAnswer(answered=False, answer="", missing_information="still no competitor info", confidence="Low"),
    ]
)
qa.get_provider = lambda **kwargs: _FakeChatModel(next(_answers2))
qa.retrieve_for_question = lambda company_name, question: []

company = _make_company(documents=[])
result = qa.answer_question(company, "Who are its competitors?")

check(
    "Still insufficient after retrieval: honest 'couldn't find' answer, not a crash",
    result.answer == "I couldn't find enough information to answer that.",
)
check("Still insufficient after retrieval: used_new_retrieval is True", result.used_new_retrieval is True)

# ---------------------------------------------------------------------
# Provider failure degrades gracefully, never raises
# ---------------------------------------------------------------------

qa.get_provider = lambda **kwargs: _FakeChatModel(RuntimeError("simulated failure"))
qa.retrieve_for_question = lambda company_name, question: []

company = _make_company(documents=[])
result = qa.answer_question(company, "What does this company do?")

check(
    "Provider failure degrades to a graceful message, not an exception",
    result.answer == "Sorry, I couldn't find an answer to that.",
)
check("Provider failure: confidence is Low", result.confidence == "Low")


print()
print("=" * 70)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S):")
    for f in _FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_qa.py`
Expected: `ModuleNotFoundError: No module named 'qa'`

- [ ] **Step 3: Create `qa.py`**

```python
"""
qa.py

Evidence-grounded question answering for the Company Discovery Agent.
Given a specific company and an arbitrary question about it, answers using
only evidence already gathered (CompanyResult.evidence bullets and
CompanyResult.documents) -- and retrieves additional evidence via
retriever.retrieve_for_question() only when what's already on hand isn't
enough to answer.

This module never guesses: the underlying LLM call is asked to answer
strictly from supplied evidence and to explicitly say when it can't,
rather than filling gaps from its own training data. Exactly one retrieval
attempt is made if the first pass is insufficient -- this module never
loops indefinitely searching for an answer; whatever the second attempt
returns (even "still couldn't find it") is final.

This module never touches ConversationState -- it's a pure function of a
CompanyResult and a question. Persisting the returned document pool back
onto the CompanyResult (so a later question about the same company can
reuse it) is the caller's job (see conversation.py's
_handle_company_question).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from discovery import CompanyResult
from llm_provider import get_provider, structured_output_kwargs
from llm_schemas import LLMQAAnswer
from prompts import QA_PROMPT, build_qa_prompt
from retriever import Document, retrieve_for_question


@dataclass(frozen=True)
class QAAnswer:
    """The outcome of one attempt to answer a question about one company."""
    answer: str
    confidence: str
    used_new_retrieval: bool
    sources: List[str] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)


def _ask(evidence, documents: List[Document], question: str) -> LLMQAAnswer:
    """One structured-output LLM call attempting to answer from supplied material."""
    llm = get_provider()
    prompt_messages = QA_PROMPT.format_messages(user_prompt=build_qa_prompt(evidence, documents, question))
    return llm.with_structured_output(LLMQAAnswer, **structured_output_kwargs()).invoke(prompt_messages)


def _merge_documents(existing: List[Document], new: List[Document]) -> List[Document]:
    """Merges two document lists, deduplicated by URL, existing first."""
    merged = list(existing)
    seen_urls = {document.url for document in existing}
    for document in new:
        if document.url not in seen_urls:
            seen_urls.add(document.url)
            merged.append(document)
    return merged


def answer_question(company: CompanyResult, question: str) -> QAAnswer:
    """
    Attempts to answer `question` about `company`. First tries using
    company.evidence + company.documents; if that's insufficient,
    retrieves once via retriever.retrieve_for_question() and tries again
    with the combined pool. Never raises -- any failure degrades to a
    plain "couldn't find an answer" QAAnswer.

    Returns a QAAnswer whose `documents` field is the full pool actually
    used, so callers can persist it back onto `company.documents` for
    future questions to reuse.
    """
    documents = company.documents

    try:
        result = _ask(company.evidence, documents, question)
        if result.answered:
            return QAAnswer(
                answer=result.answer,
                confidence=result.confidence,
                used_new_retrieval=False,
                sources=[document.url for document in documents],
                documents=documents,
            )
    except Exception:
        pass  # fall through to the one retry, same as an explicit "not answered"

    try:
        new_documents = retrieve_for_question(company.company_name, question)
    except Exception:
        new_documents = []

    documents = _merge_documents(company.documents, new_documents)

    try:
        result = _ask(company.evidence, documents, question)
        answer_text = result.answer if result.answered else "I couldn't find enough information to answer that."
        confidence = result.confidence
    except Exception:
        answer_text = "Sorry, I couldn't find an answer to that."
        confidence = "Low"

    return QAAnswer(
        answer=answer_text,
        confidence=confidence,
        used_new_retrieval=True,
        sources=[document.url for document in documents],
        documents=documents,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_qa.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add qa.py test_qa.py
git commit -m "Add qa.py: evidence-grounded question answering with one bounded retry"
```

---

### Task 6: `conversation.py` — add `COMPANY_QUESTION` routing

**Files:**
- Modify: `conversation.py`
- Test: `test_conversation.py` (extend)

**Interfaces:**
- Consumes: `answer_question` (`qa.py`, Task 5).
- Produces: `_handle_company_question(state, classification, user_message) -> AssistantReply` (new); `handle_message`'s routing gains a `COMPANY_QUESTION` branch. `handle_message`'s public signature is unchanged.

- [ ] **Step 1: Write the failing test**

In `test_conversation.py`, change the import block:
```python
from __future__ import annotations

from types import SimpleNamespace

import conversation
from conversation import ConversationState, handle_message
from discovery import CompanyResult, DiscoveryResult
from llm_schemas import LLMIntentClassification
```
to:
```python
from __future__ import annotations

from types import SimpleNamespace

import conversation
from conversation import ConversationState, handle_message
from discovery import CompanyResult, DiscoveryResult
from llm_schemas import LLMIntentClassification
from qa import QAAnswer
from retriever import Document
```

Then insert this new section immediately before the file's final block (the one starting with `print()` / `print("=" * 70)` / the `_FAILURES` check):

```python
# ---------------------------------------------------------------------
# COMPANY_QUESTION: answers from qa.answer_question, persists new documents
# ---------------------------------------------------------------------

new_document = Document(url="https://example.com/directors", title="Directors", cleaned_text="Directors: Jane Doe.")

conversation.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=_classification("COMPANY_QUESTION", ["Beta Industries Limited"])
)
conversation.answer_question = lambda company, question: QAAnswer(
    answer="The directors are Jane Doe.",
    confidence="High",
    used_new_retrieval=True,
    sources=["https://example.com/directors"],
    documents=[new_document],
)

state, reply = handle_message(state, "Who are the directors of Beta Industries?")

check("COMPANY_QUESTION: reply uses the QA answer text", reply.text == "The directors are Jane Doe.")
check(
    "COMPANY_QUESTION: reply carries the resolved company",
    len(reply.companies) == 1 and reply.companies[0].company_name == "Beta Industries Limited",
)
check(
    "COMPANY_QUESTION: the resolved company's documents are updated in place",
    reply.companies[0].documents == [new_document],
)

_beta_result = next(r for r in state.discovery_history if any(c.company_name == "Beta Industries Limited" for c in r.accepted))
_beta_company = next(c for c in _beta_result.accepted if c.company_name == "Beta Industries Limited")
check(
    "COMPANY_QUESTION: the SAME CompanyResult object in discovery_history reflects the update",
    _beta_company.documents == [new_document],
)

# ---------------------------------------------------------------------
# COMPANY_QUESTION: unresolvable name degrades gracefully
# ---------------------------------------------------------------------

conversation.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=_classification("COMPANY_QUESTION", ["Nonexistent Company Ltd"])
)

state, reply = handle_message(state, "Who are the directors of Nonexistent Company Ltd?")

check(
    "COMPANY_QUESTION: unresolvable name yields a graceful message, not a crash",
    "don't have" in reply.text.lower(),
)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_conversation.py`
Expected: `ModuleNotFoundError: No module named 'qa'` (if Task 5 weren't already done) or an `AttributeError`/assertion failure on `conversation.answer_question` not existing / `COMPANY_QUESTION` not being routed, once Task 5 is in place. Either way, a clear failure, not a pass.

- [ ] **Step 3: Update `conversation.py`**

Change the import block:
```python
from discovery import CompanyResult, DiscoveryResult, discover
from llm_provider import get_provider
from llm_schemas import LLMIntentClassification
from prompts import (
    INTENT_PROMPT,
    RESPONSE_SYNTHESIS_PROMPT,
    build_intent_prompt,
    build_response_synthesis_prompt,
)
```
to:
```python
from discovery import CompanyResult, DiscoveryResult, discover
from llm_provider import get_provider
from llm_schemas import LLMIntentClassification
from prompts import (
    INTENT_PROMPT,
    RESPONSE_SYNTHESIS_PROMPT,
    build_intent_prompt,
    build_response_synthesis_prompt,
)
from qa import answer_question
```

Then, change:
```python
    text = _synthesize_response("RECALL", companies, user_message)
    return AssistantReply(text=text, intent="RECALL", companies=companies)


def handle_message(state: ConversationState, user_message: str) -> Tuple[ConversationState, AssistantReply]:
```
to:
```python
    text = _synthesize_response("RECALL", companies, user_message)
    return AssistantReply(text=text, intent="RECALL", companies=companies)


def _handle_company_question(
    state: ConversationState, classification: LLMIntentClassification, user_message: str
) -> AssistantReply:
    """
    Answers an arbitrary question about exactly one already-known company,
    retrieving additional evidence via qa.answer_question() when needed.
    Unlike the other handlers, this one mutates the resolved company's
    `documents` field in place -- qa.answer_question() may have retrieved
    new documents, and persisting them back onto the same CompanyResult
    object already sitting in state.discovery_history means a later
    question about the same company can reuse them too, not just this one.
    """
    if not classification.referenced_company_names:
        return AssistantReply(text="I'm not sure which company you mean -- could you name it?", intent="COMPANY_QUESTION")

    name = classification.referenced_company_names[0]
    company = _find_company(state, name)
    if company is None:
        return AssistantReply(
            text=f'I don\'t have a company called "{name}" in this conversation yet.',
            intent="COMPANY_QUESTION",
        )

    state.current_company = company.company_name
    qa_answer = answer_question(company, user_message)
    company.documents = qa_answer.documents

    return AssistantReply(text=qa_answer.answer, intent="COMPANY_QUESTION", companies=[company])


def handle_message(state: ConversationState, user_message: str) -> Tuple[ConversationState, AssistantReply]:
```

Finally, change:
```python
    if classification.intent == "NEW_DISCOVERY":
        reply = _handle_new_discovery(state, user_message)
    elif classification.intent == "FOLLOW_UP_COMPANY":
        reply = _handle_follow_up_company(state, classification, user_message)
    elif classification.intent == "COMPARISON":
        reply = _handle_comparison(state, classification, user_message)
    elif classification.intent == "RECALL":
        reply = _handle_recall(state, classification, user_message)
    else:
        reply = AssistantReply(text="I'm not sure what you're asking -- could you rephrase?", intent="UNRECOGNIZED")
```
to:
```python
    if classification.intent == "NEW_DISCOVERY":
        reply = _handle_new_discovery(state, user_message)
    elif classification.intent == "FOLLOW_UP_COMPANY":
        reply = _handle_follow_up_company(state, classification, user_message)
    elif classification.intent == "COMPANY_QUESTION":
        reply = _handle_company_question(state, classification, user_message)
    elif classification.intent == "COMPARISON":
        reply = _handle_comparison(state, classification, user_message)
    elif classification.intent == "RECALL":
        reply = _handle_recall(state, classification, user_message)
    else:
        reply = AssistantReply(text="I'm not sure what you're asking -- could you rephrase?", intent="UNRECOGNIZED")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_conversation.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add conversation.py test_conversation.py
git commit -m "Route COMPANY_QUESTION to qa.py, persisting reused documents"
```

---

### Task 7: Full regression pass

**Files:**
- None created or modified — this task only runs verification.

- [ ] **Step 1: Run every automated test script**

Run:
```bash
python3 test_llm_schemas.py && \
python3 test_llm_provider.py && \
python3 test_prompts_chat_templates.py && \
python3 test_discovery_structured_output.py && \
python3 test_evidence_extractor_structured_output.py && \
python3 test_retriever.py && \
python3 test_qa.py && \
python3 test_conversation.py && \
python3 test_entity_matching.py
```
Expected: every script prints `ALL TESTS PASSED`; the whole chain exits 0.

- [ ] **Step 2: Confirm `ui.py` still compiles and needs no changes**

Run: `python3 -m py_compile ui.py`
Expected: no output, exit code 0. (Per the design, Phase 3 requires zero `ui.py` changes — this just confirms nothing else in the phase accidentally broke it.)

- [ ] **Step 3: Manual smoke test with real credentials**

This step needs real API keys and can't be scripted here. With a working `.env`:

```bash
streamlit run ui.py
```

Verify by hand:
- Run a discovery search, then ask a genuinely new question about one of the results (e.g. "who are its competitors" or "when was it founded") and confirm you get a grounded answer (or an honest "couldn't find" message), not a hallucinated one.
- Ask a second, different question about the *same* company shortly after, and confirm it doesn't feel like it's re-searching from scratch every time (documents from the first question should already be on hand for follow-ups, per the reuse design — you can't observe this directly, but response latency for a second related question should not obviously repeat a full first-question research cycle if the first pass already surfaced enough).
- Confirm a `FOLLOW_UP_COMPANY`-style question (e.g. "what was its GST number") still routes correctly and doesn't get misclassified as `COMPANY_QUESTION` or vice versa.
- Confirm the debug toggle and existing company-card rendering are unaffected.

- [ ] **Step 4: Final commit (only if Step 3 surfaced any fixes)**

If the manual smoke test in Step 3 required any code changes, commit them now with a message describing what was fixed. If everything worked as-is, there is nothing to commit for this task.
