# Phase 2: Conversation Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-turn conversation memory to the Company Discovery Agent: a new `conversation.py` orchestrates intent classification, reference resolution, and response synthesis on top of the unchanged `discovery.py`, and `ui.py` becomes a chat interface.

**Architecture:** `conversation.py` sits above `discovery.py` (calling `discover()` unchanged) and adds two new LLM calls: a structured-output intent classifier that resolves references ("it," "the second company") to exact company names, and a plain-text response synthesizer that phrases already-known `CompanyResult` facts (never sources new ones). `ui.py` renders a scrolling chat history via Streamlit's native `st.chat_message`/`st.chat_input`, replacing the single-shot search form.

**Tech Stack:** Same as Phase 1 (`langchain`, `langchain-openai`, `langchain-google-genai`, `pydantic`) — no new dependencies. `streamlit`'s native chat widgets (`st.chat_message`, `st.chat_input`), already available at the pinned `streamlit>=1.35.0`.

## Global Constraints

- `discovery.py` is **not modified** by this phase — `conversation.py` calls `discover()` exactly as it exists today.
- `prompts.py`'s new builder functions (`build_intent_prompt`, `build_response_synthesis_prompt`) take their structural parameters (`discovery_history`, `companies`) **without type annotations**, the same way the existing `build_evidence_prompt`'s `documents` parameter is left untyped — this keeps `prompts.py` a leaf module that never imports `discovery.py` or `conversation.py`, avoiding a circular import (`discovery.py` already imports from `prompts.py`).
- References ("it," "the second company," "the previous company") resolve to **exact company names**, not ordinals or turn indices. The LLM resolves pronouns/ordinals against conversation context supplied in the prompt; `conversation.py` does a plain case-insensitive exact-name lookup afterward — no fuzzy matching.
- Response synthesis is a **plain text completion**, not `with_structured_output` — there is no JSON shape to enforce for free text.
- A failed intent classification degrades to `intent="UNRECOGNIZED"` with a clarification reply. It must **never** silently default to `NEW_DISCOVERY` (that risks running an expensive, unwanted search on a misclassified message).
- `ui.py` uses Streamlit's native `st.chat_message`/`st.chat_input`, not hand-rolled HTML chat bubbles.
- The old sidebar "click a past query to rerun it" feature is removed, replaced by a "New conversation" button that resets `st.session_state.conversation`.
- Testing stays in the existing script + `check(label, condition)` harness convention — no pytest.
- Full design rationale lives in `docs/superpowers/specs/2026-07-23-phase2-conversation-memory-design.md`.

---

### Task 1: `llm_schemas.py` — add `LLMIntentClassification`

**Files:**
- Modify: `llm_schemas.py`
- Test: `test_llm_schemas.py` (extend the existing file from Phase 1)

**Interfaces:**
- Produces: `LLMIntentClassification(intent: Literal["NEW_DISCOVERY","FOLLOW_UP_COMPANY","COMPARISON","RECALL","UNRECOGNIZED"], referenced_company_names: List[str], recall_ordinal: Optional[int], reasoning: str, confidence: Literal["High","Medium","Low"])`.

- [ ] **Step 1: Write the failing test**

In `test_llm_schemas.py`, change the import line:
```python
from llm_schemas import (
    LLMCompanyEntry,
    LLMConstraintEvaluation,
    LLMDiscoveryResponse,
    LLMEvidenceItem,
    LLMEvidenceResponse,
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
)
```

Then insert this new section immediately before the file's final block (the one starting with `print()` / `print("=" * 70)` / the `_FAILURES` check):

```python
# ---------------------------------------------------------------------
# LLMIntentClassification
# ---------------------------------------------------------------------

follow_up_classification = LLMIntentClassification.model_validate(
    {
        "intent": "FOLLOW_UP_COMPANY",
        "referenced_company_names": ["Acme Forgings Private Limited"],
        "recall_ordinal": None,
        "reasoning": "The user asked about 'it', which resolves to the company in focus.",
        "confidence": "High",
    }
)
check(
    "LLMIntentClassification validates a FOLLOW_UP_COMPANY entry",
    follow_up_classification.intent == "FOLLOW_UP_COMPANY"
    and follow_up_classification.referenced_company_names == ["Acme Forgings Private Limited"],
)

check(
    "LLMIntentClassification defaults referenced_company_names to an empty list",
    LLMIntentClassification.model_validate(
        {
            "intent": "NEW_DISCOVERY",
            "reasoning": "A fresh, unrelated request.",
            "confidence": "High",
        }
    ).referenced_company_names
    == [],
)

check(
    "LLMIntentClassification defaults recall_ordinal to None",
    LLMIntentClassification.model_validate(
        {
            "intent": "RECALL",
            "reasoning": "Recall everything.",
            "confidence": "Medium",
        }
    ).recall_ordinal
    is None,
)

recall_with_ordinal = LLMIntentClassification.model_validate(
    {
        "intent": "RECALL",
        "recall_ordinal": 1,
        "reasoning": "The user asked for the first recommendation.",
        "confidence": "High",
    }
)
check(
    "LLMIntentClassification accepts an explicit recall_ordinal",
    recall_with_ordinal.recall_ordinal == 1,
)

try:
    LLMIntentClassification.model_validate(
        {
            "intent": "SOMETHING_ELSE",
            "reasoning": "x",
            "confidence": "High",
        }
    )
    check("LLMIntentClassification rejects an invalid intent value", False, "no exception raised")
except ValidationError:
    check("LLMIntentClassification rejects an invalid intent value", True)

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_llm_schemas.py`
Expected: `ImportError: cannot import name 'LLMIntentClassification' from 'llm_schemas'`

- [ ] **Step 3: Add `LLMIntentClassification` to `llm_schemas.py`**

Change the import line:
```python
from typing import Any, List, Literal
```
to:
```python
from typing import Any, List, Literal, Optional
```

Append this class at the end of the file (after `LLMEvidenceResponse`):

```python


class LLMIntentClassification(BaseModel):
    """Classifies a conversational message and resolves any company references it makes."""

    intent: Literal["NEW_DISCOVERY", "FOLLOW_UP_COMPANY", "COMPARISON", "RECALL", "UNRECOGNIZED"]
    referenced_company_names: List[str] = Field(
        default_factory=list,
        description=(
            "Company name(s) this message refers to, resolved from the supplied "
            "conversation context -- empty for NEW_DISCOVERY/RECALL/UNRECOGNIZED, "
            "one name for FOLLOW_UP_COMPANY, two names for COMPARISON."
        ),
    )
    recall_ordinal: Optional[int] = Field(
        default=None,
        description=(
            "For RECALL only: which numbered recommendation was asked for (e.g. "
            "'the first recommendation' -> 1). None means recall the entire most "
            "recent result set."
        ),
    )
    reasoning: str = Field(description="Short explanation of why this intent/reference was chosen.")
    confidence: Literal["High", "Medium", "Low"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_llm_schemas.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add llm_schemas.py test_llm_schemas.py
git commit -m "Add LLMIntentClassification schema for Phase 2 conversation memory"
```

---

### Task 2: `prompts.py` — add intent and response-synthesis prompts

**Files:**
- Modify: `prompts.py`
- Test: `test_prompts_chat_templates.py` (extend the existing file from Phase 1)

**Interfaces:**
- Produces: `INTENT_SYSTEM_PROMPT: str`, `build_intent_prompt(discovery_history, current_company, user_message: str) -> str`, `INTENT_PROMPT: ChatPromptTemplate`; `RESPONSE_SYNTHESIS_SYSTEM_PROMPT: str`, `build_response_synthesis_prompt(intent: str, companies, user_message: str) -> str`, `RESPONSE_SYNTHESIS_PROMPT: ChatPromptTemplate`. Both `ChatPromptTemplate`s take a `user_prompt` variable via `.format_messages(user_prompt=...)`, same convention as `DISCOVERY_PROMPT`/`EVIDENCE_PROMPT`.

- [ ] **Step 1: Write the failing test**

In `test_prompts_chat_templates.py`, change the import line:
```python
from prompts import DISCOVERY_PROMPT, EVIDENCE_PROMPT, EVIDENCE_SYSTEM_PROMPT, SYSTEM_PROMPT
```
to:
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

Then insert this new section immediately before the file's final block (the one starting with `print()` / `print("=" * 70)` / the `_FAILURES` check):

```python
intent_messages = INTENT_PROMPT.format_messages(
    user_prompt='Conversation so far:\nNo companies have been discussed yet in this conversation.\n\nNo company is currently in focus.\n\nNewest user message: "test"'
)

check("INTENT_PROMPT produces exactly 2 messages", len(intent_messages) == 2)
check("INTENT_PROMPT's first message is a SystemMessage", isinstance(intent_messages[0], SystemMessage))
check(
    "INTENT_PROMPT's system message content is INTENT_SYSTEM_PROMPT verbatim",
    intent_messages[0].content == INTENT_SYSTEM_PROMPT,
)
check("INTENT_PROMPT's second message is a HumanMessage", isinstance(intent_messages[1], HumanMessage))

response_synthesis_messages = RESPONSE_SYNTHESIS_PROMPT.format_messages(
    user_prompt='Known facts:\n(none)\n\nUser message: "test"'
)

check("RESPONSE_SYNTHESIS_PROMPT produces exactly 2 messages", len(response_synthesis_messages) == 2)
check(
    "RESPONSE_SYNTHESIS_PROMPT's system message content is RESPONSE_SYNTHESIS_SYSTEM_PROMPT verbatim",
    response_synthesis_messages[0].content == RESPONSE_SYNTHESIS_SYSTEM_PROMPT,
)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_prompts_chat_templates.py`
Expected: `ImportError: cannot import name 'INTENT_PROMPT' from 'prompts'`

- [ ] **Step 3: Add the new prompts to `prompts.py`**

Append at the very end of the file (after the existing `EVIDENCE_PROMPT = ChatPromptTemplate.from_messages(...)` block):

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


def build_intent_prompt(discovery_history, current_company, user_message: str) -> str:
    """
    Builds the user-turn prompt for intent classification.

    `discovery_history` is a list of discovery.DiscoveryResult objects (one
    per past NEW_DISCOVERY turn, in order) -- left untyped here, the same
    way build_evidence_prompt()'s `documents` parameter is left untyped, so
    prompts.py never needs to import discovery.py or conversation.py and
    stays a leaf module. Only `.accepted` (a list of objects with
    `.company_name`) is read from each entry. `current_company` is the name
    of whichever company is currently in focus, or None.
    """
    if not discovery_history:
        history_section = "No companies have been discussed yet in this conversation."
    else:
        blocks = []
        for i, result in enumerate(discovery_history, start=1):
            names = [company.company_name for company in result.accepted]
            blocks.append(f"Search {i}: found {', '.join(names) if names else '(no companies)'}")
        history_section = "\n".join(blocks)

    focus_section = (
        f'Company currently in focus (what "it"/"that company" refers to): {current_company}'
        if current_company
        else "No company is currently in focus."
    )

    return (
        f"Conversation so far:\n{history_section}\n\n"
        f"{focus_section}\n\n"
        f'Newest user message: "{user_message}"\n\n'
        "Classify this message using only the JSON object format described in your instructions."
    )


INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=INTENT_SYSTEM_PROMPT), ("human", "{user_prompt}")]
)


RESPONSE_SYNTHESIS_SYSTEM_PROMPT = """You are the Response Composer for a conversational Company Discovery Agent.
You will be given the user's message, the classified intent, and a set of
already-verified facts about one or more companies (never anything you
need to look up or guess). Your only job is to phrase those facts as a
natural, concise conversational reply.

Follow these rules strictly:

1. USE ONLY THE SUPPLIED FACTS. Do not add any company name, number,
   location, or claim that isn't explicitly present in the facts you are
   given. If a fact (e.g. GST) is stated as not found/verified, say so
   plainly rather than omitting it or implying it exists.

2. NEVER FABRICATE OR GUESS. You are not being asked to research or infer
   anything -- only to phrase what you're given.

3. BE CONCISE AND CONVERSATIONAL. Write like a knowledgeable analyst
   answering a direct question, not a report. A few sentences is usually
   enough; use short bullet points only if comparing multiple companies or
   listing several facts makes that clearer.

4. ANSWER WHAT WAS ASKED. If the user asked specifically about one
   attribute (e.g. "what was its GST number"), lead with that attribute
   rather than restating everything you were given.

5. OUTPUT FORMAT. Respond with plain conversational text only -- no JSON,
   no markdown code fences, no headers.
"""


def build_response_synthesis_prompt(intent: str, companies, user_message: str) -> str:
    """
    Builds the user-turn prompt for response synthesis.

    `companies` is a list of discovery.CompanyResult objects (one for
    FOLLOW_UP_COMPANY/RECALL-with-ordinal, two for COMPARISON, or the full
    accepted list for a RECALL-everything) -- left untyped for the same
    leaf-module reason build_intent_prompt()'s `discovery_history` is. Only
    company_name/confidence/gst/cin/reason/evidence are read.
    """
    if not companies:
        return (
            f'User message: "{user_message}"\n\n'
            "No matching company facts were found. Respond with a brief, "
            "honest message explaining that you don't have that company in "
            "this conversation yet."
        )

    company_blocks = []
    for company in companies:
        evidence_lines = (
            "\n".join(f"  - {item.point}" for item in company.evidence)
            or "  (no additional evidence on file)"
        )
        company_blocks.append(
            f"Company: {company.company_name}\n"
            f"Confidence: {company.confidence}\n"
            f"GST: {company.gst or 'Not found'}\n"
            f"CIN: {company.cin or 'Not found'}\n"
            f"Why it was recommended: {company.reason}\n"
            f"Additional evidence:\n{evidence_lines}"
        )

    companies_section = "\n\n".join(company_blocks)

    return (
        f"Classified intent: {intent}\n\n"
        f'User message: "{user_message}"\n\n'
        f"Known facts:\n{companies_section}\n\n"
        "Respond with a concise, conversational reply using only the facts above."
    )


RESPONSE_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=RESPONSE_SYNTHESIS_SYSTEM_PROMPT), ("human", "{user_prompt}")]
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_prompts_chat_templates.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Run the full Phase 1 regression to confirm nothing broke**

Run: `python3 -c "import prompts"` — expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add prompts.py test_prompts_chat_templates.py
git commit -m "Add intent classification and response synthesis prompts"
```

---

### Task 3: `conversation.py` — new conversation orchestration module

**Files:**
- Create: `conversation.py`
- Test: `test_conversation.py`

**Interfaces:**
- Consumes: `get_provider() -> BaseChatModel` (`llm_provider.py`); `LLMIntentClassification` (`llm_schemas.py`, Task 1); `INTENT_PROMPT`, `RESPONSE_SYNTHESIS_PROMPT`, `build_intent_prompt`, `build_response_synthesis_prompt` (`prompts.py`, Task 2); `CompanyResult`, `DiscoveryResult`, `discover` (`discovery.py`, unchanged).
- Produces: `ChatTurn(user_message: str, intent: str, assistant_response: str, companies: List[CompanyResult], discovery_result: Optional[DiscoveryResult])`; `ConversationState(turns: List[ChatTurn], discovery_history: List[DiscoveryResult], current_company: Optional[str])`; `AssistantReply(text: str, intent: str, companies: List[CompanyResult], discovery_result: Optional[DiscoveryResult])`; `handle_message(state: ConversationState, user_message: str) -> Tuple[ConversationState, AssistantReply]`. Task 4 (`ui.py`) imports `ChatTurn`, `ConversationState`, `handle_message`.

- [ ] **Step 1: Write the failing test**

Create `test_conversation.py`:

```python
"""
test_conversation.py

Unit tests for conversation.py's intent classification, reference
resolution, and routing. No network calls are made -- conversation.get_provider
and conversation.discover are monkeypatched.

Run with: python3 test_conversation.py
"""

from __future__ import annotations

from types import SimpleNamespace

import conversation
from conversation import ConversationState, handle_message
from discovery import CompanyResult, DiscoveryResult
from llm_schemas import LLMIntentClassification

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


def _make_company(name: str, gst: str | None = "27ABCCE1234F1Z2") -> CompanyResult:
    return CompanyResult(
        company_name=name,
        reason=f"{name} fits the criteria.",
        confidence="High",
        gst=gst,
        cin=None,
        pan=None,
        evidence=[],
    )


class _FakeStructuredRunnable:
    def __init__(self, response_or_exception):
        self._response_or_exception = response_or_exception

    def invoke(self, messages):
        if isinstance(self._response_or_exception, Exception):
            raise self._response_or_exception
        return self._response_or_exception


class _FakeChatModel:
    """
    Supports both with_structured_output().invoke() (for classify_intent)
    and a plain .invoke() (for _synthesize_response).
    """

    def __init__(self, structured_response=None, plain_text=None):
        self._structured_response = structured_response
        self._plain_text = plain_text

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredRunnable(self._structured_response)

    def invoke(self, messages):
        if isinstance(self._plain_text, Exception):
            raise self._plain_text
        return SimpleNamespace(text=self._plain_text)


def _classification(intent: str, referenced_company_names=None, recall_ordinal=None) -> LLMIntentClassification:
    return LLMIntentClassification(
        intent=intent,
        referenced_company_names=referenced_company_names or [],
        recall_ordinal=recall_ordinal,
        reasoning="stub",
        confidence="High",
    )


# ---------------------------------------------------------------------
# NEW_DISCOVERY
# ---------------------------------------------------------------------

state = ConversationState()
conversation.get_provider = lambda **kwargs: _FakeChatModel(structured_response=_classification("NEW_DISCOVERY"))
conversation.discover = lambda query: DiscoveryResult(
    accepted=[_make_company("Acme Forgings Private Limited")], rejected=[]
)

state, reply = handle_message(state, "Find manufacturing companies in Thane")

check("NEW_DISCOVERY: reply mentions a match", "found" in reply.text.lower())
check("NEW_DISCOVERY: reply carries the accepted company", len(reply.companies) == 1)
check("NEW_DISCOVERY: discovery_history grew by one", len(state.discovery_history) == 1)
check("NEW_DISCOVERY: current_company reset to None", state.current_company is None)
check("NEW_DISCOVERY: turn recorded", len(state.turns) == 1 and state.turns[0].intent == "NEW_DISCOVERY")

# ---------------------------------------------------------------------
# FOLLOW_UP_COMPANY via explicit name
# ---------------------------------------------------------------------

conversation.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=_classification("FOLLOW_UP_COMPANY", ["Acme Forgings Private Limited"]),
    plain_text="Acme Forgings Private Limited's GST number is 27ABCCE1234F1Z2.",
)

state, reply = handle_message(state, "What was its GST number?")

check(
    "FOLLOW_UP_COMPANY: reply uses the synthesized text",
    reply.text == "Acme Forgings Private Limited's GST number is 27ABCCE1234F1Z2.",
)
check(
    "FOLLOW_UP_COMPANY: reply carries the resolved company",
    len(reply.companies) == 1 and reply.companies[0].company_name == "Acme Forgings Private Limited",
)
check("FOLLOW_UP_COMPANY: current_company updated", state.current_company == "Acme Forgings Private Limited")

# ---------------------------------------------------------------------
# FOLLOW_UP_COMPANY: unresolvable name degrades gracefully
# ---------------------------------------------------------------------

conversation.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=_classification("FOLLOW_UP_COMPANY", ["Nonexistent Company Ltd"])
)

state, reply = handle_message(state, "What about Nonexistent Company Ltd?")

check(
    "FOLLOW_UP_COMPANY: unresolvable name yields a graceful message, not a crash",
    "don't have" in reply.text.lower(),
)
check("FOLLOW_UP_COMPANY: unresolvable name carries no companies", reply.companies == [])

# ---------------------------------------------------------------------
# Second NEW_DISCOVERY, to set up COMPARISON/RECALL against two searches
# ---------------------------------------------------------------------

conversation.get_provider = lambda **kwargs: _FakeChatModel(structured_response=_classification("NEW_DISCOVERY"))
conversation.discover = lambda query: DiscoveryResult(
    accepted=[_make_company("Beta Industries Limited", gst=None)], rejected=[]
)

state, reply = handle_message(state, "Find pharma companies in Powai")

check("Second NEW_DISCOVERY: discovery_history now has two entries", len(state.discovery_history) == 2)

# ---------------------------------------------------------------------
# COMPARISON across both searches
# ---------------------------------------------------------------------

conversation.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=_classification("COMPARISON", ["Acme Forgings Private Limited", "Beta Industries Limited"]),
    plain_text="Acme has a verified GST; Beta's GST was not found.",
)

state, reply = handle_message(state, "Compare Acme Forgings with Beta Industries")

check("COMPARISON: reply carries both companies", len(reply.companies) == 2)
check("COMPARISON: current_company set to the last resolved company", state.current_company == "Beta Industries Limited")

# ---------------------------------------------------------------------
# RECALL with an ordinal
# ---------------------------------------------------------------------

conversation.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=_classification("RECALL", recall_ordinal=1),
    plain_text="The first recommendation from the most recent search was Beta Industries Limited.",
)

state, reply = handle_message(state, "What was the first recommendation?")

check("RECALL with ordinal: reply carries exactly one company", len(reply.companies) == 1)
check(
    "RECALL with ordinal: it's the first company of the MOST RECENT search",
    reply.companies[0].company_name == "Beta Industries Limited",
)

# ---------------------------------------------------------------------
# RECALL without an ordinal (recall everything from the most recent search)
# ---------------------------------------------------------------------

conversation.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=_classification("RECALL"),
    plain_text="The most recent search found Beta Industries Limited.",
)

state, reply = handle_message(state, "What did you find just now?")

check("RECALL without ordinal: reply carries the whole most recent result set", len(reply.companies) == 1)

# ---------------------------------------------------------------------
# Classification failure degrades to UNRECOGNIZED, no crash
# ---------------------------------------------------------------------


def _raise_provider_error(**kwargs):
    raise RuntimeError("simulated provider failure")


conversation.get_provider = _raise_provider_error

state, reply = handle_message(state, "asdkfjhaslkdfjh")

check("Classification failure yields UNRECOGNIZED, not a crash", reply.intent == "UNRECOGNIZED")
check("Classification failure's turn is still recorded", state.turns[-1].intent == "UNRECOGNIZED")

# ---------------------------------------------------------------------
# Empty message short-circuits before any LLM call
# ---------------------------------------------------------------------

state2 = ConversationState()
_, reply = handle_message(state2, "   ")
check("Empty message yields the empty-message reply, not a classification failure", reply.text == "Please enter a message.")
check("Empty message reply is UNRECOGNIZED", reply.intent == "UNRECOGNIZED")


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

Run: `python3 test_conversation.py`
Expected: `ModuleNotFoundError: No module named 'conversation'`

- [ ] **Step 3: Create `conversation.py`**

```python
"""
conversation.py

Conversation orchestration for the Company Discovery Agent. Sits above
discovery.py (calling discover() completely unchanged) and adds multi-turn
memory: classifying what a new message means given the conversation so
far, resolving references to already-discovered companies, and routing to
the right handler.

This module never retrieves new evidence or performs a new identifier
lookup for follow-ups -- FOLLOW_UP_COMPANY/COMPARISON/RECALL all read
already-known CompanyResult facts out of state.discovery_history. Questions
that genuinely require new evidence (e.g. "who are its competitors") are
Phase 3's job (not yet implemented).

INTENT TAXONOMY (see llm_schemas.LLMIntentClassification):
- NEW_DISCOVERY: run discovery.discover() for a fresh search.
- FOLLOW_UP_COMPANY: answer about exactly one already-known company.
- COMPARISON: answer about exactly two already-known companies.
- RECALL: replay already-known results (optionally one by ordinal).
- UNRECOGNIZED: classification failed or the message doesn't fit above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from discovery import CompanyResult, DiscoveryResult, discover
from llm_provider import get_provider
from llm_schemas import LLMIntentClassification
from prompts import (
    INTENT_PROMPT,
    RESPONSE_SYNTHESIS_PROMPT,
    build_intent_prompt,
    build_response_synthesis_prompt,
)


@dataclass
class ChatTurn:
    """One past turn: what the user asked, what was classified, and what came back."""
    user_message: str
    intent: str
    assistant_response: str
    companies: List[CompanyResult] = field(default_factory=list)
    discovery_result: Optional[DiscoveryResult] = None


@dataclass
class ConversationState:
    """
    All memory needed to resolve follow-up references. `discovery_history`
    holds one DiscoveryResult per NEW_DISCOVERY turn, in the order they
    happened. `current_company` is the name of whichever company was most
    recently the subject of a FOLLOW_UP_COMPANY/COMPARISON/RECALL turn --
    what "it" resolves to next.
    """
    turns: List[ChatTurn] = field(default_factory=list)
    discovery_history: List[DiscoveryResult] = field(default_factory=list)
    current_company: Optional[str] = None


@dataclass
class AssistantReply:
    """What ui.py renders for one turn: the reply text, plus any companies to show as cards."""
    text: str
    intent: str
    companies: List[CompanyResult] = field(default_factory=list)
    discovery_result: Optional[DiscoveryResult] = None


def _find_company(state: ConversationState, name: str) -> Optional[CompanyResult]:
    """
    Searches state.discovery_history most-recent-first for a company whose
    name matches `name` case-insensitively. Returns None if not found --
    callers degrade gracefully rather than crash.
    """
    normalized = name.strip().lower()
    for result in reversed(state.discovery_history):
        for company in result.accepted:
            if company.company_name.strip().lower() == normalized:
                return company
    return None


def classify_intent(state: ConversationState, user_message: str) -> LLMIntentClassification:
    """
    Runs the intent-classification LLM call. Raises on provider/schema
    failure -- handle_message catches and degrades to UNRECOGNIZED, since
    guessing an intent wrong (e.g. defaulting to NEW_DISCOVERY) risks
    running an expensive, unwanted search.
    """
    llm = get_provider()
    prompt_messages = INTENT_PROMPT.format_messages(
        user_prompt=build_intent_prompt(state.discovery_history, state.current_company, user_message)
    )
    return llm.with_structured_output(LLMIntentClassification).invoke(prompt_messages)


def _synthesize_response(intent: str, companies: List[CompanyResult], user_message: str) -> str:
    """
    Runs the response-synthesis LLM call to phrase already-known facts.
    Falls back to a plain, deterministic template built from the same
    facts if the LLM call fails -- the turn's information is never lost,
    only its phrasing degrades from conversational to mechanical.
    """
    try:
        llm = get_provider()
        prompt_messages = RESPONSE_SYNTHESIS_PROMPT.format_messages(
            user_prompt=build_response_synthesis_prompt(intent, companies, user_message)
        )
        message = llm.invoke(prompt_messages)
        text = message.text.strip()
        if text:
            return text
    except Exception:
        pass

    if not companies:
        return "I don't have a company matching that in this conversation yet."

    lines = [
        f"{company.company_name}: confidence {company.confidence}, "
        f"GST {company.gst or 'Not found'}, CIN {company.cin or 'Not found'}."
        for company in companies
    ]
    return "\n".join(lines)


def _handle_new_discovery(state: ConversationState, user_message: str) -> AssistantReply:
    """Runs a fresh discovery search via discovery.discover(), completely unchanged."""
    result = discover(user_message)
    state.discovery_history.append(result)
    state.current_company = None

    if result.accepted:
        text = f"I found {len(result.accepted)} matching compan{'y' if len(result.accepted) == 1 else 'ies'}."
    else:
        text = "I couldn't find any matching companies for that request."

    return AssistantReply(text=text, intent="NEW_DISCOVERY", companies=result.accepted, discovery_result=result)


def _handle_follow_up_company(
    state: ConversationState, classification: LLMIntentClassification, user_message: str
) -> AssistantReply:
    """Answers about exactly one already-known company."""
    if not classification.referenced_company_names:
        return AssistantReply(text="I'm not sure which company you mean -- could you name it?", intent="FOLLOW_UP_COMPANY")

    name = classification.referenced_company_names[0]
    company = _find_company(state, name)
    if company is None:
        return AssistantReply(
            text=f'I don\'t have a company called "{name}" in this conversation yet.',
            intent="FOLLOW_UP_COMPANY",
        )

    state.current_company = company.company_name
    text = _synthesize_response("FOLLOW_UP_COMPANY", [company], user_message)
    return AssistantReply(text=text, intent="FOLLOW_UP_COMPANY", companies=[company])


def _handle_comparison(
    state: ConversationState, classification: LLMIntentClassification, user_message: str
) -> AssistantReply:
    """Answers by comparing exactly two already-known companies."""
    if len(classification.referenced_company_names) < 2:
        return AssistantReply(text="I need two companies to compare -- could you name both?", intent="COMPARISON")

    companies: List[CompanyResult] = []
    missing: List[str] = []
    for name in classification.referenced_company_names[:2]:
        company = _find_company(state, name)
        if company is None:
            missing.append(name)
        else:
            companies.append(company)

    if missing:
        return AssistantReply(text=f"I don't have {' or '.join(missing)} in this conversation yet.", intent="COMPARISON")

    state.current_company = companies[-1].company_name
    text = _synthesize_response("COMPARISON", companies, user_message)
    return AssistantReply(text=text, intent="COMPARISON", companies=companies)


def _handle_recall(
    state: ConversationState, classification: LLMIntentClassification, user_message: str
) -> AssistantReply:
    """Replays already-known results: the whole most recent result set, or one by ordinal."""
    if not state.discovery_history:
        return AssistantReply(text="We haven't discussed any companies yet in this conversation.", intent="RECALL")

    most_recent = state.discovery_history[-1]

    if classification.recall_ordinal is not None:
        index = classification.recall_ordinal - 1
        if index < 0 or index >= len(most_recent.accepted):
            return AssistantReply(
                text=f"There's no recommendation #{classification.recall_ordinal} in the most recent search.",
                intent="RECALL",
            )
        companies = [most_recent.accepted[index]]
    else:
        companies = most_recent.accepted

    if companies:
        state.current_company = companies[-1].company_name

    text = _synthesize_response("RECALL", companies, user_message)
    return AssistantReply(text=text, intent="RECALL", companies=companies)


def handle_message(state: ConversationState, user_message: str) -> Tuple[ConversationState, AssistantReply]:
    """
    The main entry point: classifies user_message, routes to the
    appropriate handler, appends the resulting ChatTurn to state, and
    returns the (mutated, same-object) state alongside the AssistantReply.

    DiscoveryError, raised by discover() inside _handle_new_discovery, is
    NOT caught here -- it propagates to the caller (ui.py), exactly as it
    did before Phase 2.
    """
    if not user_message or not user_message.strip():
        return state, AssistantReply(text="Please enter a message.", intent="UNRECOGNIZED")

    try:
        classification = classify_intent(state, user_message)
    except Exception:
        reply = AssistantReply(text="Sorry, I couldn't understand that -- could you rephrase?", intent="UNRECOGNIZED")
        state.turns.append(ChatTurn(user_message=user_message, intent=reply.intent, assistant_response=reply.text))
        return state, reply

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

    state.turns.append(
        ChatTurn(
            user_message=user_message,
            intent=reply.intent,
            assistant_response=reply.text,
            companies=reply.companies,
            discovery_result=reply.discovery_result,
        )
    )
    return state, reply
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_conversation.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add conversation.py test_conversation.py
git commit -m "Add conversation.py: intent classification, reference resolution, routing"
```

---

### Task 4: `ui.py` — rewrite as a chat interface

**Files:**
- Modify: `ui.py` (full rewrite)

**Interfaces:**
- Consumes: `ChatTurn`, `ConversationState`, `handle_message` (`conversation.py`, Task 3); `CompanyResult`, `DiscoveryError`, `RejectedCompany` (`discovery.py`, unchanged).
- No other module imports from `ui.py` — this is the final task, a leaf.

- [ ] **Step 1: Replace the full contents of `ui.py`**

```python
"""
ui.py

Streamlit chat interface for the Company Discovery Agent.

All business logic (discovery, retrieval, GST/CIN extraction, evidence
extraction, and conversation orchestration) lives in discovery.py,
retriever.py, identifier_lookup.py, evidence_extractor.py, and
conversation.py. This file only handles layout, session state, and
rendering.

Conversation model (Phase 2):
  A single ConversationState (conversation.ConversationState) lives in
  st.session_state, replacing the single-shot current_query/current_results
  fields from before Phase 2. Every submitted message goes through
  conversation.handle_message(), which classifies intent, resolves any
  company references, and returns an AssistantReply. The chat history
  (state.turns) is rendered top to bottom on every rerun via Streamlit's
  native st.chat_message; each turn's companies (if any) are rendered with
  the same card HTML used since before Phase 2 -- debug mode and the
  rejected-candidates panel work exactly as they did previously, just
  scoped per turn instead of per single search.

Debug mode:
  A sidebar checkbox ("Show Debug Information") controls whether each
  company card is expanded with the LLM's per-constraint evaluation,
  explicit decision, and other diagnostic information -- unchanged from
  before Phase 2. When on, a turn that produced a DiscoveryResult also
  gets a "Rejected Candidates" expander below its cards.

Run with:
    streamlit run ui.py
"""

from __future__ import annotations

import csv
import html
import io
import json

import streamlit as st
from dotenv import load_dotenv

from conversation import ChatTurn, ConversationState, handle_message
from discovery import CompanyResult, DiscoveryError, RejectedCompany

load_dotenv()

st.set_page_config(
    page_title="Company Discovery Agent",
    page_icon="📇",
    layout="centered",
)

# -----------------------------------------------------------------------
# Styling -- unchanged from before Phase 2.
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .company-card {
        border: 1px solid rgba(150, 150, 150, 0.25);
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }
    .company-card.high { border-left: 4px solid #4c8577; }
    .company-card.medium { border-left: 4px solid #c9a227; }
    .company-card.low { border-left: 4px solid #9c4a36; }
    .company-name { font-size: 1.15rem; font-weight: 600; margin-bottom: 0.2rem; }
    .confidence-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .confidence-badge.high { background: rgba(76, 133, 119, 0.18); color: #4c8577; }
    .confidence-badge.medium { background: rgba(201, 162, 39, 0.18); color: #c9a227; }
    .confidence-badge.low { background: rgba(156, 74, 54, 0.18); color: #9c4a36; }
    .gst-line { font-family: "IBM Plex Mono", monospace; font-size: 0.85rem; color: #888; margin-bottom: 0.5rem; }
    .gst-line.not-found { font-style: italic; }
    .evidence-list { margin-top: 0.4rem; }
    .evidence-item { margin-bottom: 0.55rem; }
    .evidence-point { font-size: 0.95rem; }
    .evidence-source { font-size: 0.8rem; color: #888; margin-left: 1.1rem; margin-top: 0.05rem; }
    .evidence-source a { color: #4c8577; text-decoration: none; }
    .evidence-source a:hover { text-decoration: underline; }
    .evidence-empty { font-size: 0.85rem; color: #888; font-style: italic; }

    /* -- Debug mode -- */
    .debug-block {
        margin-top: 0.75rem;
        padding-top: 0.65rem;
        border-top: 1px dashed rgba(150, 150, 150, 0.35);
    }
    .debug-heading {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.35rem;
    }
    .debug-subtle { font-size: 0.88rem; color: #aaa; }
    .debug-empty { font-size: 0.85rem; color: #888; font-style: italic; }
    .debug-reason { font-size: 0.92rem; }
    .constraint-row {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }
    .constraint-name { min-width: 8.5rem; font-weight: 600; }
    .constraint-status {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 1px 7px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .constraint-status.status-pass { background: rgba(76, 133, 119, 0.18); color: #4c8577; }
    .constraint-status.status-fail { background: rgba(156, 74, 54, 0.18); color: #9c4a36; }
    .constraint-status.status-unknown { background: rgba(201, 162, 39, 0.18); color: #c9a227; }
    .constraint-reason { color: #999; flex: 1; }
    .decision-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 2px 10px;
        border-radius: 4px;
    }
    .decision-badge.accept { background: rgba(76, 133, 119, 0.18); color: #4c8577; }
    .decision-badge.reject { background: rgba(156, 74, 54, 0.18); color: #9c4a36; }
    .unknown-item { font-size: 0.88rem; color: #999; margin-bottom: 0.3rem; }
    .rejection-type-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 2px 9px;
        border-radius: 4px;
        margin-left: 0.5rem;
    }
    .rejection-type-badge.llm { background: rgba(156, 74, 54, 0.18); color: #9c4a36; }
    .rejection-type-badge.verification { background: rgba(80, 110, 160, 0.18); color: #4c6ea0; }
    .rejected-confidence { font-size: 0.85rem; color: #888; margin-top: 0.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation: ConversationState = ConversationState()
if "error_message" not in st.session_state:
    st.session_state.error_message: str | None = None


def run_message(user_message: str) -> None:
    """
    Sends one message through conversation.handle_message() and updates
    session state. DiscoveryError -- raised only for the NEW_DISCOVERY
    path, inside discovery.discover() -- is caught here exactly as
    run_search() caught it before Phase 2.
    """
    user_message = user_message.strip()
    if not user_message:
        return

    st.session_state.error_message = None
    with st.spinner("Thinking..."):
        try:
            st.session_state.conversation, _ = handle_message(st.session_state.conversation, user_message)
        except DiscoveryError as exc:
            st.session_state.error_message = str(exc)


def _evidence_to_flat_string(company: CompanyResult) -> str:
    """Flattens a company's evidence bullets into one delimited string for CSV export."""
    return " | ".join(f"{item.point} ({item.source_title}: {item.source_url})" for item in company.evidence)


def results_to_csv(results: list[CompanyResult]) -> str:
    """Serializes a list of CompanyResult -- unchanged from before Phase 2."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Company Name", "GST", "Confidence", "Reason", "Evidence"])
    for company in results:
        writer.writerow(
            [company.company_name, company.gst or "Not found", company.confidence, company.reason, _evidence_to_flat_string(company)]
        )
    return buffer.getvalue()


def results_to_json(results: list[CompanyResult]) -> str:
    """Serializes a list of CompanyResult -- unchanged from before Phase 2."""
    return json.dumps(
        [
            {
                "company_name": c.company_name,
                "gst": c.gst,
                "confidence": c.confidence,
                "reason": c.reason,
                "evidence": [
                    {"point": item.point, "source_title": item.source_title, "source_url": item.source_url}
                    for item in c.evidence
                ],
            }
            for c in results
        ],
        indent=2,
    )


def _render_debug_html(company: CompanyResult) -> str:
    """Builds the extra debug HTML for one company card -- unchanged from before Phase 2."""
    constraint_items = list(company.constraint_evaluation.items())

    identified_names = (
        ", ".join(name.replace("_", " ").title() for name, _ in constraint_items)
        if constraint_items
        else "No constraints were reported for this company."
    )

    if constraint_items:
        constraint_rows = "".join(
            f'<div class="constraint-row">'
            f'<div class="constraint-name">{html.escape(name.replace("_", " ").title())}</div>'
            f'<div class="constraint-status status-{ceval.status.lower()}">{html.escape(ceval.status)}</div>'
            f'<div class="constraint-reason">{html.escape(ceval.reason)}</div>'
            f'</div>'
            for name, ceval in constraint_items
        )
    else:
        constraint_rows = '<div class="debug-empty">No constraint evaluation was returned for this company.</div>'

    unknown_items = [(name, ceval) for name, ceval in constraint_items if ceval.status == "UNKNOWN"]
    if unknown_items:
        unknown_rows = "".join(
            f'<div class="unknown-item">• {html.escape(name.replace("_", " ").title())} -- '
            f'{html.escape(ceval.reason)}</div>'
            for name, ceval in unknown_items
        )
    else:
        unknown_rows = '<div class="debug-empty">No unresolved constraints -- every evaluated constraint was PASS or FAIL.</div>'

    decision_class = "accept" if company.decision == "ACCEPT" else "reject"

    return (
        '<div class="debug-block">'
        '<div class="debug-heading">Constraints identified from query</div>'
        f'<div class="debug-subtle">{html.escape(identified_names)}</div>'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Constraint evaluation</div>'
        f'{constraint_rows}'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Decision</div>'
        f'<div class="decision-badge {decision_class}">{html.escape(company.decision)}</div>'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Confidence explanation / final reason</div>'
        f'<div class="debug-reason">{html.escape(company.reason)}</div>'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Assumptions &amp; unverified constraints</div>'
        f'{unknown_rows}'
        '</div>'
    )


def _render_rejected_card_html(rejected: RejectedCompany) -> str:
    """Builds one card for the "Rejected Candidates" debug section -- unchanged from before Phase 2."""
    constraint_items = list(rejected.constraint_evaluation.items())

    if constraint_items:
        constraint_rows = "".join(
            f'<div class="constraint-row">'
            f'<div class="constraint-name">{html.escape(name.replace("_", " ").title())}</div>'
            f'<div class="constraint-status status-{ceval.status.lower()}">{html.escape(ceval.status)}</div>'
            f'<div class="constraint-reason">{html.escape(ceval.reason)}</div>'
            f'</div>'
            for name, ceval in constraint_items
        )
    else:
        constraint_rows = '<div class="debug-empty">No constraint evaluation was returned for this candidate.</div>'

    is_verification = rejected.rejection_type == "verification"
    rejection_type_label = "Verification Rejection" if is_verification else "LLM Rejection"
    rejection_type_class = "verification" if is_verification else "llm"

    identifiers_html = (
        f'<div class="gst-line not-found">GST: Not found</div>'
        f'<div class="gst-line not-found">CIN: Not found</div>'
        if is_verification
        else ""
    )

    return (
        '<div class="company-card low" style="opacity:0.9;">'
        f'<div class="decision-badge reject">{html.escape(rejected.decision)}</div>'
        f'<span class="rejection-type-badge {rejection_type_class}">{html.escape(rejection_type_label)}</span>'
        f'<div class="company-name">{rejected.company_name}</div>'
        f'<div class="rejected-confidence">Original LLM confidence: {html.escape(rejected.confidence)}</div>'
        f'{identifiers_html}'
        '<div class="debug-block" style="margin-top:0.3rem;">'
        '<div class="debug-heading">Constraint evaluation</div>'
        f'{constraint_rows}'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Why it was rejected</div>'
        f'<div class="debug-reason">{html.escape(rejected.reason)}</div>'
        '</div>'
        '</div>'
    )


def _company_card_html(company: CompanyResult, show_debug: bool) -> str:
    """Builds one company card's HTML -- unchanged rendering logic from before Phase 2."""
    confidence_class = company.confidence.lower() if company.confidence.lower() in {"high", "medium", "low"} else "low"
    gst_display = company.gst if company.gst else "Not found"
    gst_class = "" if company.gst else "not-found"
    cin_display = company.cin if company.cin else "Not found"

    if company.evidence:
        evidence_html = "".join(
            f'<div class="evidence-item">'
            f'<div class="evidence-point">• {html.escape(item.point)}</div>'
            f'<div class="evidence-source">🔗 '
            f'<a href="{html.escape(item.source_url, quote=True)}" target="_blank" rel="noopener noreferrer">'
            f'{html.escape(item.source_title)}</a></div>'
            f'</div>'
            for item in company.evidence
        )
    else:
        evidence_html = '<div class="evidence-empty">No sourced evidence found for this company.</div>'

    debug_html = _render_debug_html(company) if show_debug else ""
    evidence_heading_html = '<div class="debug-heading" style="margin-top:0.75rem;">Evidence</div>' if show_debug else ""

    return (
        f'<div class="company-card {confidence_class}">'
        f'<div class="confidence-badge {confidence_class}">{company.confidence} confidence</div>'
        f'<div class="company-name">{company.company_name}</div>'
        f'<div class="gst-line {gst_class}">GST: {gst_display}</div>'
        f'<div class="gst-line">CIN: {cin_display}</div>'
        f'{debug_html}'
        f'{evidence_heading_html}'
        f'<div class="evidence-list">{evidence_html}</div>'
        f'</div>'
    )


def _render_turn(turn: ChatTurn, show_debug: bool) -> None:
    """Renders one past turn: the user's message, the assistant's reply, and any company cards."""
    with st.chat_message("user"):
        st.write(turn.user_message)

    with st.chat_message("assistant"):
        st.write(turn.assistant_response)

        for company in turn.companies:
            st.markdown(_company_card_html(company, show_debug), unsafe_allow_html=True)

        if show_debug and turn.discovery_result is not None:
            with st.expander(f"Rejected Candidates ({len(turn.discovery_result.rejected)})", expanded=False):
                if turn.discovery_result.rejected:
                    st.caption(
                        "Candidates that did not make it into the results above, for two "
                        "possible reasons: the model itself ruled them out because an explicit "
                        "constraint failed (\"LLM Rejection\"), or the model recommended them but "
                        "deterministic GST/CIN verification couldn't confirm the legal entity "
                        "exists (\"Verification Rejection\")."
                    )
                    for rejected in turn.discovery_result.rejected:
                        st.markdown(_render_rejected_card_html(rejected), unsafe_allow_html=True)
                else:
                    st.caption("No rejected candidates were reported for this search.")


# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("📇 Company Discovery Agent")
st.caption(
    "Ask a plain-language question, or ask a follow-up about a company already "
    "discussed. Every match comes with sourced evidence from the web and a "
    "deterministic GST check -- never guessed, never fabricated."
)

# -----------------------------------------------------------------------
# Sidebar: debug toggle + new-conversation reset
# -----------------------------------------------------------------------
with st.sidebar:
    show_debug = st.checkbox(
        "Show Debug Information",
        value=False,
        key="show_debug_toggle",
        help=(
            "Expands every company card with the LLM's full per-constraint "
            "evaluation, its explicit decision, and any assumptions made -- "
            "useful for diagnosing retrieval and prompt quality. The normal "
            "view is unaffected when this is off."
        ),
    )

    st.divider()

    if st.button("New conversation", use_container_width=True):
        st.session_state.conversation = ConversationState()
        st.session_state.error_message = None
        st.rerun()

# -----------------------------------------------------------------------
# Chat history
# -----------------------------------------------------------------------
conversation_state: ConversationState = st.session_state.conversation

if st.session_state.error_message:
    st.error(st.session_state.error_message)

for turn in conversation_state.turns:
    _render_turn(turn, show_debug)

# -----------------------------------------------------------------------
# Message input -- st.chat_input pins itself to the bottom of the
# viewport regardless of where it's called in the script.
# -----------------------------------------------------------------------
prompt = st.chat_input("Ask a question or a follow-up...")
if prompt:
    run_message(prompt)
    st.rerun()

# -----------------------------------------------------------------------
# Export the most recent turn that produced companies
# -----------------------------------------------------------------------
_most_recent_companies: list[CompanyResult] = []
for turn in reversed(conversation_state.turns):
    if turn.companies:
        _most_recent_companies = turn.companies
        break

if _most_recent_companies:
    st.divider()
    st.subheader("Export most recent result set")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            data=results_to_csv(_most_recent_companies),
            file_name="company_discovery_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download JSON",
            data=results_to_json(_most_recent_companies),
            file_name="company_discovery_results.json",
            mime="application/json",
            use_container_width=True,
        )
```

- [ ] **Step 2: Syntax/import sanity check**

Run: `python3 -m py_compile ui.py`
Expected: no output, exit code 0 (this only checks for syntax errors — it does not execute Streamlit's top-level calls, which require a running Streamlit script context).

- [ ] **Step 3: Commit**

```bash
git add ui.py
git commit -m "Rewrite ui.py as a chat interface (Phase 2 conversation memory)"
```

- [ ] **Step 4: Manual smoke test with real credentials**

This step needs real API keys and can't be scripted here. With a working `.env` (or exported env vars):

```bash
streamlit run ui.py
```

Verify by hand:
- A fresh conversation: type a discovery query (e.g. "Find manufacturing companies in Mumbai with turnover between 20 and 100 crore") and confirm company cards render inline in the chat, exactly as they did before Phase 2.
- Ask a follow-up referencing "it" or "the second company" and confirm the reply correctly identifies and describes that company (GST/CIN/confidence), without re-running a new search.
- Ask for a comparison between two previously-mentioned companies and confirm both appear in the reply.
- Ask "what was the first recommendation" and confirm it recalls the most recent search's first company.
- Toggle "Show Debug Information" on and confirm the constraint-evaluation debug block and "Rejected Candidates" panel still work on `NEW_DISCOVERY` turns.
- Click "New conversation" and confirm the chat history clears and a fresh `ConversationState` is in effect.
- Trigger a deliberately malformed/empty message and confirm the app doesn't crash.

- [ ] **Step 5: Final commit (only if Step 4 surfaced any fixes)**

If the manual smoke test in Step 4 required any code changes, commit them now with a message describing what was fixed. If everything worked as-is, there is nothing to commit for this task.
