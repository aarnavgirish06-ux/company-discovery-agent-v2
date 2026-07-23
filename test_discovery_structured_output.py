"""
test_discovery_structured_output.py

Unit tests for discovery.py's LangChain-based LLM call, covering both
branches: the plain with_structured_output path, and the grounded
fallback path used when Gemini Search grounding is active (which cannot
be combined with with_structured_output -- see llm_provider.py). No
network calls are made -- discovery.get_provider, discovery.is_grounded,
discovery.get_company_identifiers, discovery.retriever.retrieve_for_evidence,
and discovery.extract_evidence are all monkeypatched.

Run with: python3 test_discovery_structured_output.py
"""

from __future__ import annotations

from types import SimpleNamespace

import discovery
import retriever
from evidence_extractor import EvidenceTrace
from identifier_lookup import IdentifierRecord, IdentifierTrace
from llm_schemas import LLMCompanyEntry, LLMConstraintEvaluation, LLMDiscoveryResponse
from retriever import Document

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


_FIXTURE_DOCUMENT = Document(url="https://example.com/about", title="About Acme", cleaned_text="Acme makes forgings.")


_FIXTURE_IDENTIFIER_TRACE = IdentifierTrace(site_checks=[], corroboration_counts={"GST": 2}, validation_notes={"GST": "stub"})
_FIXTURE_RETRIEVAL_TRACE = retriever.RetrievalTrace(queries=['"Acme Forgings Private Limited"'], attempts=[])
_FIXTURE_EVIDENCE_TRACE = EvidenceTrace(selected=["Manufactures forgings"], rejected=[])


def _stub_downstream_pipeline() -> None:
    """
    Stubs everything discover() calls after parsing the LLM response, so
    these tests only exercise the LLM-calling/parsing change itself.
    """
    discovery.get_company_identifiers = lambda name: (
        {
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
        },
        _FIXTURE_IDENTIFIER_TRACE,
    )
    discovery.retriever.retrieve_for_evidence = lambda name: ([_FIXTURE_DOCUMENT], _FIXTURE_RETRIEVAL_TRACE)
    discovery.extract_evidence = lambda *args, **kwargs: ([], _FIXTURE_EVIDENCE_TRACE)


class _FakeStructuredRunnable:
    def __init__(self, response_or_exception):
        self._response_or_exception = response_or_exception

    def invoke(self, messages):
        if isinstance(self._response_or_exception, Exception):
            raise self._response_or_exception
        return self._response_or_exception


class _FakeChatModel:
    def __init__(self, structured_response=None, raw_text=None):
        self._structured_response = structured_response
        self._raw_text = raw_text

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredRunnable(self._structured_response)

    def invoke(self, messages):
        return SimpleNamespace(text=self._raw_text)


# ---------------------------------------------------------------------
# Non-grounded branch (with_structured_output)
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: False
discovery.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=LLMDiscoveryResponse(
        companies=[
            LLMCompanyEntry(
                company_name="Acme Forgings Private Limited",
                constraint_evaluation=[
                    LLMConstraintEvaluation(name="location", status="PASS", reason="HQ in Thane.")
                ],
                decision="ACCEPT",
                reason="Matches all constraints.",
                confidence="High",
            ),
            LLMCompanyEntry(
                company_name="Bad Corp Limited",
                constraint_evaluation=[
                    LLMConstraintEvaluation(name="location", status="FAIL", reason="HQ in Pune, not Thane.")
                ],
                decision="REJECT",
                reason="Location fails.",
                confidence="Medium",
            ),
        ]
    )
)

result = discovery.discover("Find manufacturing companies in Thane")

check("Non-grounded: one company accepted", len(result.accepted) == 1, len(result.accepted))
check(
    "Non-grounded: accepted company name matches",
    bool(result.accepted) and result.accepted[0].company_name == "Acme Forgings Private Limited",
)
check(
    "Non-grounded: accepted company's constraint_evaluation dict is rebuilt correctly",
    bool(result.accepted)
    and result.accepted[0].constraint_evaluation.get("location") is not None
    and result.accepted[0].constraint_evaluation["location"].status == "PASS",
)
check("Non-grounded: one LLM rejection recorded", len(result.rejected) == 1, len(result.rejected))
check(
    "Non-grounded: rejected company is tagged rejection_type='llm'",
    bool(result.rejected) and result.rejected[0].rejection_type == "llm",
)
check(
    "Non-grounded: accepted company's documents are populated from retrieve_for_evidence, not discarded",
    bool(result.accepted) and result.accepted[0].documents == [_FIXTURE_DOCUMENT],
)
check(
    "Non-grounded: accepted company's retrieval_trace is attached",
    bool(result.accepted) and result.accepted[0].retrieval_trace == _FIXTURE_RETRIEVAL_TRACE,
)
check(
    "Non-grounded: accepted company's identifier_trace is attached",
    bool(result.accepted) and result.accepted[0].identifier_trace == _FIXTURE_IDENTIFIER_TRACE,
)
check(
    "Non-grounded: accepted company's evidence_trace is attached",
    bool(result.accepted) and result.accepted[0].evidence_trace == _FIXTURE_EVIDENCE_TRACE,
)

# ---------------------------------------------------------------------
# Non-grounded branch: invoke() failure surfaces as DiscoveryError
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: False
discovery.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=RuntimeError("simulated provider failure")
)

try:
    discovery.discover("Find manufacturing companies in Thane")
    check("Non-grounded: provider failure raises DiscoveryError", False, "no exception raised")
except discovery.DiscoveryError:
    check("Non-grounded: provider failure raises DiscoveryError", True)

# ---------------------------------------------------------------------
# Grounded fallback branch (manual JSON parse + Pydantic validation)
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: True
discovery.get_provider = lambda **kwargs: _FakeChatModel(
    raw_text="""
    [
      {
        "company_name": "Acme Forgings Private Limited",
        "constraint_evaluation": {
          "location": {"status": "PASS", "reason": "HQ in Thane."}
        },
        "decision": "ACCEPT",
        "reason": "Matches all constraints.",
        "confidence": "High"
      }
    ]
    """
)

grounded_result = discovery.discover("Find manufacturing companies in Thane")

check(
    "Grounded: dict-shaped constraint_evaluation from the raw prompt path is accepted",
    len(grounded_result.accepted) == 1,
    len(grounded_result.accepted),
)
check(
    "Grounded: accepted company's constraint_evaluation dict is rebuilt correctly",
    bool(grounded_result.accepted)
    and grounded_result.accepted[0].constraint_evaluation.get("location") is not None
    and grounded_result.accepted[0].constraint_evaluation["location"].status == "PASS",
)

# ---------------------------------------------------------------------
# Grounded fallback branch: malformed JSON surfaces as DiscoveryError
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: True
discovery.get_provider = lambda **kwargs: _FakeChatModel(raw_text="not valid json at all")

try:
    discovery.discover("Find manufacturing companies in Thane")
    check("Grounded: malformed JSON raises DiscoveryError", False, "no exception raised")
except discovery.DiscoveryError:
    check("Grounded: malformed JSON raises DiscoveryError", True)

# ---------------------------------------------------------------------
# Grounded fallback branch: schema-violating entry surfaces as DiscoveryError
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: True
discovery.get_provider = lambda **kwargs: _FakeChatModel(
    raw_text='[{"company_name": "Acme Ltd", "decision": "MAYBE", "reason": "x", "confidence": "High"}]'
)

try:
    discovery.discover("Find manufacturing companies in Thane")
    check("Grounded: schema-violating entry raises DiscoveryError", False, "no exception raised")
except discovery.DiscoveryError:
    check("Grounded: schema-violating entry raises DiscoveryError", True)


# ---------------------------------------------------------------------
# Verification rejection: identifier_trace is still attached (identifier
# lookup ran and produced this trace even though verification failed)
# ---------------------------------------------------------------------

discovery.get_company_identifiers = lambda name: (
    {"GST": "GST not verified", "CIN": "CIN not verified"},
    _FIXTURE_IDENTIFIER_TRACE,
)
discovery.retriever.retrieve_for_evidence = lambda name: ([_FIXTURE_DOCUMENT], _FIXTURE_RETRIEVAL_TRACE)
discovery.extract_evidence = lambda *args, **kwargs: ([], _FIXTURE_EVIDENCE_TRACE)
discovery.is_grounded = lambda: False
discovery.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=LLMDiscoveryResponse(
        companies=[
            LLMCompanyEntry(
                company_name="Unverifiable Corp Limited",
                constraint_evaluation=[],
                decision="ACCEPT",
                reason="Looked promising.",
                confidence="Medium",
            ),
        ]
    )
)

verification_rejected_result = discovery.discover("Find manufacturing companies in Thane")

check(
    "Verification rejection: exactly one company rejected",
    len(verification_rejected_result.rejected) == 1,
    len(verification_rejected_result.rejected),
)
check(
    "Verification rejection: identifier_trace is attached even though verification failed",
    bool(verification_rejected_result.rejected)
    and verification_rejected_result.rejected[0].identifier_trace == _FIXTURE_IDENTIFIER_TRACE,
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
