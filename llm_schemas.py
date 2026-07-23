"""
llm_schemas.py

Pydantic models describing the LLM's structured output shape for the
Company Discovery Agent. These models exist only at the LLM I/O boundary:
callers (discovery.py, evidence_extractor.py) convert instances of these
into the project's existing plain dataclasses
(CompanyResult/RejectedCompany/EvidenceItem/DiscoveryResult) before
returning to their own callers. Nothing outside discovery.py and
evidence_extractor.py should import from this module.

Responses are wrapped in a top-level object (`companies` / `items`)
rather than a bare array, since structured-output/function-calling JSON
schemas require an object root.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class LLMConstraintEvaluation(BaseModel):
    """One named constraint's PASS/FAIL/UNKNOWN judgement, as returned by the LLM."""

    name: str = Field(description="Short snake_case constraint name, e.g. 'location', 'turnover'.")
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    reason: str = Field(description="One or two sentence explanation for this status.")


class LLMCompanyEntry(BaseModel):
    """One company entry (accepted or rejected) as returned by the discovery LLM call."""

    company_name: str = Field(description="The full legal or commonly used company name.")
    constraint_evaluation: List[LLMConstraintEvaluation] = Field(
        default_factory=list,
        description=(
            "Per-constraint PASS/FAIL/UNKNOWN evaluation, one entry per "
            "explicit constraint relevant to the query."
        ),
    )
    decision: Literal["ACCEPT", "REJECT"]
    reason: str = Field(description="Executive summary of why this company was accepted or rejected.")
    confidence: Literal["High", "Medium", "Low"]

    @field_validator("constraint_evaluation", mode="before")
    @classmethod
    def _normalize_constraint_evaluation(cls, value: Any) -> Any:
        """
        Accepts either the schema-enforced shape (a list of
        {name, status, reason} objects -- what with_structured_output
        forces the model into, regardless of prompt wording) or a dict
        keyed by constraint name (the shape prompts.py's Rule 9 describes
        in prose, which is what discovery.py's grounded fallback path
        actually receives, since no JSON-schema constraint is enforced
        there). Normalizes a dict into the list shape; a list (or a list
        of already-built LLMConstraintEvaluation instances) passes through
        untouched.
        """
        if isinstance(value, dict):
            return [
                {"name": name, **item}
                for name, item in value.items()
                if isinstance(item, dict)
            ]
        return value


class LLMDiscoveryResponse(BaseModel):
    """Top-level structured response for a discovery request."""

    companies: List[LLMCompanyEntry] = Field(default_factory=list)


class LLMEvidenceItem(BaseModel):
    """One sourced fact extracted from already-downloaded documents."""

    point: str = Field(description="One short, concrete fact (roughly 3-12 words).")
    source_title: str = Field(description="Short human-readable label for the source.")
    source_url: str = Field(description="Copied exactly from one of the supplied document URLs.")


class LLMEvidenceResponse(BaseModel):
    """Top-level structured response for an evidence-extraction request."""

    items: List[LLMEvidenceItem] = Field(default_factory=list)


class LLMIntentClassification(BaseModel):
    """Classifies a conversational message and resolves any company references it makes."""

    intent: Literal["NEW_DISCOVERY", "FOLLOW_UP_COMPANY", "COMPANY_QUESTION", "COMPARISON", "RECALL", "UNRECOGNIZED"]
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
