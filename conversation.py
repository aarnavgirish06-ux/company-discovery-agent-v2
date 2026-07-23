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
from qa import answer_question


@dataclass
class ChatTurn:
    """One past turn: what the user asked, what was classified, and what came back."""
    user_message: str
    intent: str
    assistant_response: str
    companies: List[CompanyResult] = field(default_factory=list)
    discovery_result: Optional[DiscoveryResult] = None
    intent_reasoning: str = ""  # LLMIntentClassification.reasoning, for every successfully classified turn (Phase 4 debug mode)
    intent_confidence: str = ""  # LLMIntentClassification.confidence, same as above
    qa_sources: List[str] = field(default_factory=list)  # QAAnswer.sources, only for COMPANY_QUESTION turns (Phase 4 debug mode)
    qa_used_new_retrieval: bool = False  # QAAnswer.used_new_retrieval, only for COMPANY_QUESTION turns (Phase 4 debug mode)


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
    qa_sources: List[str] = field(default_factory=list)  # only for COMPANY_QUESTION turns (Phase 4 debug mode)
    qa_used_new_retrieval: bool = False  # only for COMPANY_QUESTION turns (Phase 4 debug mode)


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


def _handle_company_question(
    state: ConversationState, classification: LLMIntentClassification, user_message: str
) -> AssistantReply:
    """
    Answers an arbitrary question about exactly one already-known company,
    retrieving additional evidence via qa.answer_question() when needed.
    Unlike the other handlers, this one mutates the resolved company's
    `documents` field in place -- qa.answer_question() may have retrieved
    new documents, and persisting them back onto the same CompanyResult
    object already sitting in state.discovery_history means a later
    question about the same company can reuse them too, not just this one.
    """
    if not classification.referenced_company_names:
        return AssistantReply(text="I'm not sure which company you mean -- could you name it?", intent="COMPANY_QUESTION")

    name = classification.referenced_company_names[0]
    company = _find_company(state, name)
    if company is None:
        return AssistantReply(
            text=f'I don\'t have a company called "{name}" in this conversation yet.',
            intent="COMPANY_QUESTION",
        )

    state.current_company = company.company_name
    qa_answer = answer_question(company, user_message)
    company.documents = qa_answer.documents

    return AssistantReply(
        text=qa_answer.answer,
        intent="COMPANY_QUESTION",
        companies=[company],
        qa_sources=qa_answer.sources,
        qa_used_new_retrieval=qa_answer.used_new_retrieval,
    )


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
    elif classification.intent == "COMPANY_QUESTION":
        reply = _handle_company_question(state, classification, user_message)
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
            intent_reasoning=classification.reasoning,
            intent_confidence=classification.confidence,
            qa_sources=reply.qa_sources,
            qa_used_new_retrieval=reply.qa_used_new_retrieval,
        )
    )
    return state, reply
