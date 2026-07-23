# Phase 2: Conversation Memory — Design

## Context

This is the second of four planned phases evolving the Company Discovery
Agent from a single-turn discovery/verification pipeline into a
conversational assistant:

1. LangChain migration — done (see
   `docs/superpowers/specs/2026-07-23-phase1-langchain-migration-design.md`)
2. **Conversation memory** (this spec)
3. Company question answering (evidence-grounded Q&A beyond discovery)
4. Debug mode (a transparent execution trace across all stages)

The original plan asked for Phase 4 before Phase 2. Scoping that revealed a
hard dependency: two of Phase 4's six debug-panel sections (Intent
Classification, Final Answer Generation) only have data once Phase 2/3
exist. The order reverts to the original 2 → 3 → 4.

### Why Phase 2 needs no changes to `discovery.py`

`discovery.py` already has an unused `history: Optional[List[ConversationTurn]]`
parameter, reserved from before Phase 1. This spec deliberately does **not**
wire that parameter up. Phase 2's follow-up examples ("tell me more about
the second company," "what was its GST number," "compare it with the
previous company," "what was the first recommendation") are all about
**referencing and re-presenting already-known results**, not about feeding
prior queries back into a fresh discovery search. `discovery.py`'s
`history` parameter would matter for a different feature (e.g. "narrow
that down to Maharashtra," which refines a NEW search using context) — out
of scope here. Because of this, `conversation.py` sits entirely above
`discovery.py` and calls `discover()` completely unchanged.

## Decisions made during brainstorming

| Question | Decision |
|---|---|
| Debug-panel sections with no data source yet (Intent Classification, Final Answer Generation) | Resolved by reordering: implement Phase 2 and Phase 3 first, then Phase 4 (all six sections have real data by then). |
| Intent classification: separate call vs. folded into one combined call | Separate, small structured-output call. Keeps intent classification independently auditable/testable now, and is exactly what Phase 4's "Intent Classification" debug section will read from later. |
| How to produce reply text for follow-ups (facts are already known, no new retrieval) | Light grounded LLM phrasing: the LLM turns already-resolved `CompanyResult` fields into a natural sentence, but never sources new facts — it is handed only the fixed, already-verified values. Zero hallucination risk since inputs are fixed. |
| UI shape | Chat-style interface: scrolling message history with company cards rendered inline, replacing the single-shot search form. |

## Architecture

`conversation.py` (new) sits above `discovery.py`, orchestrating it without
modifying it. Two new LLM calls:

1. **Intent classification** — structured output. Resolves what the user
   means and, for follow-ups, which company they're referring to.
2. **Response synthesis** — plain text completion (no JSON schema; there is
   nothing to structure in free text). Phrases already-known facts. Never
   asked to source new facts, so it cannot hallucinate a fact not already
   in the `CompanyResult`/comparison data it's handed.

`ui.py` becomes a chat interface driven by a `ConversationState` object
held in `st.session_state`, replacing the current scalar
`current_query`/`current_results`/`current_rejected` session fields.

## Components

### `llm_schemas.py` (add to existing file)

```python
class LLMIntentClassification(BaseModel):
    intent: Literal["NEW_DISCOVERY", "FOLLOW_UP_COMPANY", "COMPARISON", "RECALL", "UNRECOGNIZED"]
    referenced_company_names: List[str]  # resolved company name(s), not ordinals/turn-indices
    recall_ordinal: Optional[int]  # for RECALL, e.g. "the first recommendation" -> 1; None means "recall everything"
    reasoning: str  # short explanation of the classification -- debug-friendly, feeds Phase 4 later
    confidence: Literal["High", "Medium", "Low"]
```

References resolve directly to **company names**, not ordinal positions or
turn indices. The classification prompt supplies the LLM with prior turns'
company lists plus which company is currently "in focus," so it can
resolve "it," "the second company," "the previous company," etc. into a
concrete name — the same deterministic key `CompanyResult` already
carries. This sidesteps building separate ordinal/pronoun-resolution logic
in Python; the LLM does reference resolution, `conversation.py` does exact
name lookup afterward (no fuzzy matching needed, since the LLM is shown the
exact names to choose from).

### `prompts.py` (add to existing file)

- `INTENT_SYSTEM_PROMPT` / `INTENT_PROMPT` (`ChatPromptTemplate`, same
  `SystemMessage`-wrapping convention as `DISCOVERY_PROMPT` for brace
  safety) — describes the intent taxonomy and reference-resolution rules.
  `build_intent_prompt(state, user_message)` renders the conversation
  context (each past `NEW_DISCOVERY`/`RECALL` turn's company names, in
  order, plus which company is currently focused) and the new message.
- `RESPONSE_SYNTHESIS_SYSTEM_PROMPT` / `RESPONSE_SYNTHESIS_PROMPT` —
  instructs the LLM to phrase only the facts it's given, with the same
  anti-fabrication rules already used in `EVIDENCE_SYSTEM_PROMPT`. No
  `with_structured_output` here since the desired output is free text, not
  structured data. `build_response_synthesis_prompt(intent, companies, user_message)`
  renders the intent, the resolved compan(ies)' known fields (name,
  confidence, GST, CIN, evidence points, reason), and the user's original
  message.

### `conversation.py` (new)

```python
@dataclass
class ChatTurn:
    """One past turn: what the user asked, what was classified, and what came back."""
    user_message: str
    intent: str
    assistant_response: str
    companies: List[CompanyResult] = field(default_factory=list)

@dataclass
class ConversationState:
    turns: List[ChatTurn] = field(default_factory=list)
    discovery_history: List[DiscoveryResult] = field(default_factory=list)  # one per NEW_DISCOVERY turn, in order
    current_company: Optional[str] = None  # name of the most recently focused company

@dataclass
class AssistantReply:
    text: str
    intent: str
    companies: List[CompanyResult] = field(default_factory=list)  # for ui.py to render as cards
    discovery_result: Optional[DiscoveryResult] = None  # only for NEW_DISCOVERY, so ui.py's rejected-candidates panel still works

def handle_message(state: ConversationState, user_message: str) -> Tuple[ConversationState, AssistantReply]: ...
```

- `_find_company(state, name) -> Optional[CompanyResult]` — searches
  `state.discovery_history` most-recent-first for an exact
  (case-insensitive) name match.
- `classify_intent(state, user_message) -> LLMIntentClassification` — the
  structured-output call.
- One handler per intent:
  - `_handle_new_discovery(state, user_message)` — calls
    `discovery.discover(user_message)` unchanged, appends the result to
    `discovery_history`, sets `current_company` to `None` (a fresh search
    has no single focus yet).
  - `_handle_follow_up_company(state, classification)` — looks up the one
    referenced company via `_find_company`, calls the response-synthesis
    LLM with its known fields, sets `current_company` to that name.
  - `_handle_comparison(state, classification)` — looks up the two
    referenced companies, calls response synthesis with both companies'
    known fields side by side.
  - `_handle_recall(state, classification)` — replays either the whole
    most recent `DiscoveryResult.accepted` list or, if `recall_ordinal` is
    set, that one company (1-indexed into the most recent
    `DiscoveryResult.accepted`), via response synthesis.
- `handle_message` classifies, routes, builds the new `ChatTurn`, appends
  it to `state.turns`, and returns `(state, AssistantReply)`.

### `ui.py` (rewritten)

- Session state holds one `ConversationState` (`st.session_state.conversation`)
  instead of the current scalar `current_query`/`current_results`/`current_rejected`
  fields.
- Renders `state.turns` as a scrolling chat history: each turn's
  `user_message`, then `assistant_response`, then (if `turn.companies` is
  non-empty) the **existing** company-card rendering functions reused
  as-is (confidence badge, GST/CIN lines, evidence list, and — when the
  debug toggle is on — the existing constraint-evaluation/decision debug
  block). No new card-rendering code; only the surrounding chat-history
  loop is new.
- A single message input is always available (not just for the first
  query), submitting through `handle_message`.
- The existing "Show Debug Information" toggle and "Rejected Candidates"
  panel keep working exactly as today, scoped to whichever turn(s)
  produced a `DiscoveryResult`.
- The old "click a past query to rerun it fresh" sidebar history is
  **removed** — it doesn't make sense once conversation state is real
  memory (rerunning a query "fresh" would fork the conversation
  confusingly). Replaced with a "New conversation" button that resets
  `st.session_state.conversation` to a blank `ConversationState`.
- CSV/JSON export buttons scope to the most recent turn that produced
  companies (`discovery_result` or `companies` non-empty), preserving
  today's "export current search" semantics, just re-scoped to "most
  recent search-producing turn" instead of "the one and only search."

## Data flow

```
user_message
  -> classify_intent(state, user_message)   [structured-output LLM call]
  -> route on classification.intent:
       NEW_DISCOVERY      -> discovery.discover(user_message)          [unchanged]
       FOLLOW_UP_COMPANY  -> _find_company(state, name)                [in-memory lookup, no retrieval]
       COMPARISON         -> _find_company(state, name) x2             [in-memory lookup, no retrieval]
       RECALL             -> most recent DiscoveryResult.accepted[...]  [in-memory replay, no retrieval]
       UNRECOGNIZED       -> clarification reply, no LLM call beyond classification
  -> (for FOLLOW_UP_COMPANY/COMPARISON/RECALL) response-synthesis LLM call
     phrases the already-known fields                                  [plain text, no new facts]
  -> AssistantReply { text, companies, discovery_result }
  -> new ChatTurn appended to state.turns; current_company updated
```

## Error handling

- Intent classification failure (provider error, schema validation
  failure) degrades to `intent="UNRECOGNIZED"` with a plain clarification
  reply ("Sorry, I couldn't understand that — could you rephrase?") rather
  than defaulting to `NEW_DISCOVERY` and silently running a wrong,
  possibly expensive search.
- A referenced company name that doesn't resolve (`_find_company` returns
  `None` — e.g. the LLM hallucinated a name not actually in history, or the
  user asked about something never discussed) degrades to a plain
  "I don't have a company matching that in this conversation yet" reply,
  not a crash.
- Response-synthesis failure degrades to a plain fallback message built
  from the same structured facts via an f-string template (never silently
  drops the turn), mirroring `evidence_extractor.py`'s "never let an LLM
  hiccup break the pipeline" posture from Phase 1.
- `discovery.discover()`'s existing `DiscoveryError` propagates through
  `_handle_new_discovery` exactly as it does today (ui.py already has a
  catch for it).

## Testing

Same script + `check()` convention as Phase 1 (no pytest). New
`test_conversation.py` stubs `get_provider()` for both the intent and
synthesis calls, and stubs `discovery.discover()` for the `NEW_DISCOVERY`
path (a fixture `DiscoveryResult`, not a real search). Covers: a
`NEW_DISCOVERY` turn populating `discovery_history` and setting
`current_company` correctly; a `FOLLOW_UP_COMPANY` turn resolving "it" via
`current_company` and via an explicit name; a `COMPARISON` turn resolving
two names; a `RECALL` turn with and without an ordinal; an unresolvable
company name degrading gracefully; and a classification failure degrading
to `UNRECOGNIZED`.

## Out of scope for Phase 2

- Arbitrary new questions requiring new evidence retrieval ("what does
  this company do," "who are its competitors") — Phase 3.
- A debug trace for intent classification or response synthesis beyond
  what's naturally available in `LLMIntentClassification.reasoning` —
  Phase 4 wires that into a visible panel.
- Wiring up `discovery.py`'s existing `history` parameter (context-aware
  *new* discovery searches, e.g. "narrow that down to Maharashtra") — a
  different feature than what Phase 2's examples call for; not attempted
  here.
