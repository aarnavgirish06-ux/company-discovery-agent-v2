"""
test_evidence_extractor_structured_output.py

Unit tests for evidence_extractor.py's LangChain-based structured-output
call. No network calls are made -- evidence_extractor.get_provider is
monkeypatched.

Run with: python3 test_evidence_extractor_structured_output.py
"""

from __future__ import annotations

import evidence_extractor
from llm_schemas import LLMEvidenceItem, LLMEvidenceResponse
from retriever import Document

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


class _FakeStructuredRunnable:
    def __init__(self, response_or_exception):
        self._response_or_exception = response_or_exception

    def invoke(self, messages):
        if isinstance(self._response_or_exception, Exception):
            raise self._response_or_exception
        return self._response_or_exception


class _FakeChatModel:
    def __init__(self, structured_response):
        self._structured_response = structured_response

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredRunnable(self._structured_response)


documents = [
    Document(url="https://example.com/about", title="About Acme", cleaned_text="Acme makes forgings."),
]

# ---------------------------------------------------------------------
# Valid response, one entry citing a real document URL, one hallucinated
# ---------------------------------------------------------------------

evidence_extractor.get_provider = lambda **kwargs: _FakeChatModel(
    LLMEvidenceResponse(
        items=[
            LLMEvidenceItem(
                point="Manufactures precision forgings",
                source_title="Official Website",
                source_url="https://example.com/about",
            ),
            LLMEvidenceItem(
                point="Hallucinated fact",
                source_title="Nowhere",
                source_url="https://not-supplied.example.com/",
            ),
        ]
    )
)

result, trace = evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", documents)

check("Only the entry citing a supplied URL survives", len(result) == 1, len(result))
check(
    "The surviving entry's fields match the supplied document's URL",
    bool(result) and result[0].source_url == "https://example.com/about",
)
check("The trace records the selected point", trace.selected == ["Manufactures precision forgings"], trace.selected)
check(
    "The trace records the hallucinated entry as rejected, with a reason",
    len(trace.rejected) == 1
    and trace.rejected[0].point == "Hallucinated fact"
    and "not in the supplied documents" in trace.rejected[0].reason,
    trace.rejected,
)

# ---------------------------------------------------------------------
# No documents -> empty list, no LLM call attempted
# ---------------------------------------------------------------------


def _fail_if_called(**kwargs):
    raise AssertionError("get_provider() should not be called when there are no documents")


evidence_extractor.get_provider = _fail_if_called
no_docs_result, no_docs_trace = evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", [])
check("extract() with no documents returns [] without calling the LLM", no_docs_result == [])
check("extract() with no documents returns an empty trace", no_docs_trace.selected == [] and no_docs_trace.rejected == [])

# ---------------------------------------------------------------------
# Provider/invoke failure degrades to an empty list, not an exception
# ---------------------------------------------------------------------

evidence_extractor.get_provider = lambda **kwargs: _FakeChatModel(RuntimeError("simulated failure"))
failure_result, failure_trace = evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", documents)
check("A provider/invoke failure returns [] rather than raising", failure_result == [])
check("A provider/invoke failure returns an empty trace", failure_trace.selected == [] and failure_trace.rejected == [])


print()
print("=" * 70)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S):")
    for f in _FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
