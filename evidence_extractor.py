"""
evidence_extractor.py

LLM-based evidence extraction for the Company Discovery Agent.

Given a company name and a list of already-downloaded `retriever.Document`
objects, asks the LLM to summarize what those documents actually say about
the company as a list of short, sourced bullet points.

This module NEVER searches the web or downloads anything itself -- it only
reads the `cleaned_text` of documents the retriever already fetched (see
retriever.py). It is also purely a summarizer, never a determiner of
identifiers: the LLM is never asked to find or guess a GST number here --
that stays identifier_lookup.py's deterministic job.

Unlike discovery.py, this module never requests Gemini Search grounding
(get_provider() is always called with no arguments) -- it only ever
summarizes documents retriever.py already fetched, so it always takes the
plain with_structured_output path, with no grounded-fallback branch.

As a deterministic backstop against a hallucinated fact or source URL
slipping through despite the prompt's instructions not to, `extract()` also
drops any evidence entry whose `source_url` isn't one of the URLs actually
supplied to the LLM -- so even if the model hallucinates a citation, it
can't reach the final result.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Tuple

from llm_provider import get_provider, structured_output_kwargs
from llm_schemas import LLMEvidenceItem, LLMEvidenceResponse
from prompts import EVIDENCE_PROMPT, build_evidence_prompt
from retriever import Document


@dataclass(frozen=True)
class EvidenceItem:
    """One sourced, LLM-summarized fact about a company."""
    point: str
    source_title: str
    source_url: str


@dataclass(frozen=True)
class EvidenceRejection:
    """One LLM-proposed evidence entry that was dropped, and why."""
    point: str
    source_url: str
    reason: str


@dataclass(frozen=True)
class EvidenceTrace:
    """Structured record of evidence selection/rejection during extraction."""
    selected: List[str]  # points kept -- mirrors the returned EvidenceItem list, for a self-contained trace view
    rejected: List[EvidenceRejection]


def _validate_evidence_entry(entry: LLMEvidenceItem, valid_urls: set[str]) -> Tuple[EvidenceItem | None, str | None]:
    """
    Validates one structured LLMEvidenceItem against the documents actually
    supplied to the LLM. Pydantic has already guaranteed point/source_title/
    source_url are strings; this is the deterministic backstop against a
    hallucinated citation, dropping any entry whose source_url isn't one of
    the documents actually supplied to the LLM.

    Returns (item, None) when valid, or (None, reason) when rejected, so
    the caller can build an EvidenceTrace without duplicating this logic.
    """
    point = entry.point.strip()
    source_title = entry.source_title.strip()
    source_url = entry.source_url.strip()

    if not point or not source_url:
        return None, "missing point or source_url"
    if source_url not in valid_urls:
        return None, "cited a URL not in the supplied documents"

    return (
        EvidenceItem(
            point=point,
            source_title=source_title or source_url,
            source_url=source_url,
        ),
        None,
    )


def extract(
    company_name: str, user_query: str, discovery_reason: str, documents: List[Document]
) -> Tuple[List[EvidenceItem], EvidenceTrace]:
    """
    Asks the LLM to summarize `documents` into short, sourced bullet points
    about `company_name`.

    Returns ([], an empty trace) -- rather than raising -- if there are no
    documents to summarize, the LLM call fails, or its response can't be
    parsed. Evidence is an enrichment on top of a company result, not a
    required field, so a failure here should never block the rest of a
    company's result from being returned.

    The returned EvidenceTrace records which LLM-proposed points were kept
    and which were rejected (and why), for Phase 4's debug mode.
    """
    if not documents:
        return [], EvidenceTrace(selected=[], rejected=[])

    valid_urls = {document.url for document in documents}

    prompt_messages = EVIDENCE_PROMPT.format_messages(
        user_prompt=build_evidence_prompt(company_name, user_query, discovery_reason, documents)
    )

    try:
        llm = get_provider()
        response = llm.with_structured_output(
            LLMEvidenceResponse, **structured_output_kwargs()
        ).invoke(prompt_messages)
    except Exception as exc:
        print(f"Warning: evidence extraction failed for {company_name!r}: {exc}", file=sys.stderr)
        return [], EvidenceTrace(selected=[], rejected=[])

    evidence: List[EvidenceItem] = []
    rejected: List[EvidenceRejection] = []
    for entry in response.items:
        item, reason = _validate_evidence_entry(entry, valid_urls)
        if item is not None:
            evidence.append(item)
        else:
            rejected.append(EvidenceRejection(point=entry.point, source_url=entry.source_url, reason=reason))

    trace = EvidenceTrace(selected=[item.point for item in evidence], rejected=rejected)
    return evidence, trace
