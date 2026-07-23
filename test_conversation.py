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
from qa import QAAnswer
from retriever import Document

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
check("NEW_DISCOVERY: turn records intent_reasoning", state.turns[0].intent_reasoning == "stub")
check("NEW_DISCOVERY: turn records intent_confidence", state.turns[0].intent_confidence == "High")
check("NEW_DISCOVERY: turn's qa_sources stays empty (not a COMPANY_QUESTION turn)", state.turns[0].qa_sources == [])
check(
    "NEW_DISCOVERY: turn's qa_used_new_retrieval stays False (not a COMPANY_QUESTION turn)",
    state.turns[0].qa_used_new_retrieval is False,
)

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
check(
    "Classification failure's turn has no intent reasoning (classification never completed)",
    state.turns[-1].intent_reasoning == "",
)

# ---------------------------------------------------------------------
# Empty message short-circuits before any LLM call
# ---------------------------------------------------------------------

state2 = ConversationState()
_, reply = handle_message(state2, "   ")
check("Empty message yields the empty-message reply, not a classification failure", reply.text == "Please enter a message.")
check("Empty message reply is UNRECOGNIZED", reply.intent == "UNRECOGNIZED")


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
check("COMPANY_QUESTION: reply carries QA sources", reply.qa_sources == ["https://example.com/directors"])
check("COMPANY_QUESTION: reply carries qa_used_new_retrieval", reply.qa_used_new_retrieval is True)
check(
    "COMPANY_QUESTION: the recorded turn carries QA sources",
    state.turns[-1].qa_sources == ["https://example.com/directors"],
)
check("COMPANY_QUESTION: the recorded turn carries qa_used_new_retrieval", state.turns[-1].qa_used_new_retrieval is True)

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


print()
print("=" * 70)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S):")
    for f in _FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
