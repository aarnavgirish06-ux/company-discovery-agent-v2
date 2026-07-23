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
qa.retrieve_for_question = lambda company_name, question: ([new_document], None)

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
qa.retrieve_for_question = lambda company_name, question: ([], None)

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
qa.retrieve_for_question = lambda company_name, question: ([], None)

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
