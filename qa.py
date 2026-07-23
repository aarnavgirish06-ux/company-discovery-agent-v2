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
        new_documents, _retrieval_trace = retrieve_for_question(company.company_name, question)
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
