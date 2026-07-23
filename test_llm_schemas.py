"""
test_llm_schemas.py

Unit tests for llm_schemas.py's Pydantic models, especially the
constraint_evaluation dict/list normalization that lets discovery.py's
grounded and non-grounded paths share one schema.

Run with: python3 test_llm_schemas.py
"""

from __future__ import annotations

from pydantic import ValidationError

from llm_schemas import (
    LLMCompanyEntry,
    LLMConstraintEvaluation,
    LLMDiscoveryResponse,
    LLMEvidenceItem,
    LLMEvidenceResponse,
    LLMIntentClassification,
    LLMQAAnswer,
)

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


# ---------------------------------------------------------------------
# LLMConstraintEvaluation / LLMCompanyEntry
# ---------------------------------------------------------------------

list_shape_entry = LLMCompanyEntry.model_validate(
    {
        "company_name": "Acme Forgings Private Limited",
        "constraint_evaluation": [
            {"name": "location", "status": "PASS", "reason": "HQ in Thane."}
        ],
        "decision": "ACCEPT",
        "reason": "Matches all constraints.",
        "confidence": "High",
    }
)
check(
    "List-shape constraint_evaluation (with_structured_output path) validates",
    list_shape_entry.constraint_evaluation == [
        LLMConstraintEvaluation(name="location", status="PASS", reason="HQ in Thane.")
    ],
)

dict_shape_entry = LLMCompanyEntry.model_validate(
    {
        "company_name": "Acme Forgings Private Limited",
        "constraint_evaluation": {
            "location": {"status": "PASS", "reason": "HQ in Thane."}
        },
        "decision": "ACCEPT",
        "reason": "Matches all constraints.",
        "confidence": "High",
    }
)
check(
    "Dict-shape constraint_evaluation (grounded-fallback prompt shape) normalizes to a list",
    dict_shape_entry.constraint_evaluation == [
        LLMConstraintEvaluation(name="location", status="PASS", reason="HQ in Thane.")
    ],
)

check(
    "Empty constraint_evaluation defaults to an empty list",
    LLMCompanyEntry.model_validate(
        {
            "company_name": "Acme Forgings Private Limited",
            "decision": "ACCEPT",
            "reason": "No explicit constraints in this query.",
            "confidence": "Medium",
        }
    ).constraint_evaluation
    == [],
)

try:
    LLMCompanyEntry.model_validate(
        {
            "company_name": "Acme Forgings Private Limited",
            "decision": "MAYBE",
            "reason": "x",
            "confidence": "High",
        }
    )
    check("Invalid decision value is rejected by validation", False, "no exception raised")
except ValidationError:
    check("Invalid decision value is rejected by validation", True)

# ---------------------------------------------------------------------
# LLMDiscoveryResponse
# ---------------------------------------------------------------------

discovery_response = LLMDiscoveryResponse.model_validate(
    {
        "companies": [
            {
                "company_name": "Acme Forgings Private Limited",
                "constraint_evaluation": [],
                "decision": "ACCEPT",
                "reason": "Fits.",
                "confidence": "High",
            }
        ]
    }
)
check(
    "LLMDiscoveryResponse wraps companies under a 'companies' key",
    len(discovery_response.companies) == 1
    and discovery_response.companies[0].company_name == "Acme Forgings Private Limited",
)

check(
    "LLMDiscoveryResponse defaults to an empty companies list",
    LLMDiscoveryResponse.model_validate({}).companies == [],
)

# ---------------------------------------------------------------------
# LLMEvidenceItem / LLMEvidenceResponse
# ---------------------------------------------------------------------

evidence_response = LLMEvidenceResponse.model_validate(
    {
        "items": [
            {
                "point": "Manufactures precision automotive components",
                "source_title": "Official Website",
                "source_url": "https://example.com/about",
            }
        ]
    }
)
check(
    "LLMEvidenceResponse wraps items under an 'items' key",
    len(evidence_response.items) == 1
    and evidence_response.items[0].source_url == "https://example.com/about",
)

try:
    LLMEvidenceItem.model_validate({"point": "x", "source_title": "y"})
    check("LLMEvidenceItem requires source_url", False, "no exception raised")
except ValidationError:
    check("LLMEvidenceItem requires source_url", True)

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


print()
print("=" * 70)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S):")
    for f in _FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
