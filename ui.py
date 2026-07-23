"""
ui.py

Streamlit chat interface for the Company Discovery Agent.

All business logic (discovery, retrieval, GST/CIN extraction, evidence
extraction, and conversation orchestration) lives in discovery.py,
retriever.py, identifier_lookup.py, evidence_extractor.py, and
conversation.py. This file only handles layout, session state, and
rendering.

Conversation model (Phase 2):
  A single ConversationState (conversation.ConversationState) lives in
  st.session_state, replacing the single-shot current_query/current_results
  fields from before Phase 2. Every submitted message goes through
  conversation.handle_message(), which classifies intent, resolves any
  company references, and returns an AssistantReply. The chat history
  (state.turns) is rendered top to bottom on every rerun via Streamlit's
  native st.chat_message; each turn's companies (if any) are rendered with
  the same card HTML used since before Phase 2 -- debug mode and the
  rejected-candidates panel work exactly as they did previously, just
  scoped per turn instead of per single search.

Debug mode:
  A sidebar checkbox ("Show Debug Information") controls whether each
  company card is expanded with the LLM's per-constraint evaluation,
  explicit decision, and other diagnostic information -- unchanged from
  before Phase 2. When on, a turn that produced a DiscoveryResult also
  gets a "Rejected Candidates" expander below its cards.

Run with:
    streamlit run ui.py
"""

from __future__ import annotations

import csv
import html
import io
import json

import streamlit as st
from dotenv import load_dotenv

from conversation import ChatTurn, ConversationState, handle_message
from discovery import CompanyResult, DiscoveryError, RejectedCompany

load_dotenv()

st.set_page_config(
    page_title="Company Discovery Agent",
    page_icon="📇",
    layout="centered",
)

# -----------------------------------------------------------------------
# Styling -- unchanged from before Phase 2.
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .company-card {
        border: 1px solid rgba(150, 150, 150, 0.25);
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }
    .company-card.high { border-left: 4px solid #4c8577; }
    .company-card.medium { border-left: 4px solid #c9a227; }
    .company-card.low { border-left: 4px solid #9c4a36; }
    .company-name { font-size: 1.15rem; font-weight: 600; margin-bottom: 0.2rem; }
    .confidence-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .confidence-badge.high { background: rgba(76, 133, 119, 0.18); color: #4c8577; }
    .confidence-badge.medium { background: rgba(201, 162, 39, 0.18); color: #c9a227; }
    .confidence-badge.low { background: rgba(156, 74, 54, 0.18); color: #9c4a36; }
    .gst-line { font-family: "IBM Plex Mono", monospace; font-size: 0.85rem; color: #888; margin-bottom: 0.5rem; }
    .gst-line.not-found { font-style: italic; }
    .evidence-list { margin-top: 0.4rem; }
    .evidence-item { margin-bottom: 0.55rem; }
    .evidence-point { font-size: 0.95rem; }
    .evidence-source { font-size: 0.8rem; color: #888; margin-left: 1.1rem; margin-top: 0.05rem; }
    .evidence-source a { color: #4c8577; text-decoration: none; }
    .evidence-source a:hover { text-decoration: underline; }
    .evidence-empty { font-size: 0.85rem; color: #888; font-style: italic; }

    /* -- Debug mode -- */
    .debug-block {
        margin-top: 0.75rem;
        padding-top: 0.65rem;
        border-top: 1px dashed rgba(150, 150, 150, 0.35);
    }
    .debug-heading {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.35rem;
    }
    .debug-subtle { font-size: 0.88rem; color: #aaa; }
    .debug-empty { font-size: 0.85rem; color: #888; font-style: italic; }
    .debug-reason { font-size: 0.92rem; }
    .constraint-row {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }
    .constraint-name { min-width: 8.5rem; font-weight: 600; }
    .constraint-status {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 1px 7px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .constraint-status.status-pass { background: rgba(76, 133, 119, 0.18); color: #4c8577; }
    .constraint-status.status-fail { background: rgba(156, 74, 54, 0.18); color: #9c4a36; }
    .constraint-status.status-unknown { background: rgba(201, 162, 39, 0.18); color: #c9a227; }
    .constraint-reason { color: #999; flex: 1; }
    .decision-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 2px 10px;
        border-radius: 4px;
    }
    .decision-badge.accept { background: rgba(76, 133, 119, 0.18); color: #4c8577; }
    .decision-badge.reject { background: rgba(156, 74, 54, 0.18); color: #9c4a36; }
    .unknown-item { font-size: 0.88rem; color: #999; margin-bottom: 0.3rem; }
    .rejection-type-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 2px 9px;
        border-radius: 4px;
        margin-left: 0.5rem;
    }
    .rejection-type-badge.llm { background: rgba(156, 74, 54, 0.18); color: #9c4a36; }
    .rejection-type-badge.verification { background: rgba(80, 110, 160, 0.18); color: #4c6ea0; }
    .rejected-confidence { font-size: 0.85rem; color: #888; margin-top: 0.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation: ConversationState = ConversationState()
if "error_message" not in st.session_state:
    st.session_state.error_message: str | None = None


def run_message(user_message: str) -> None:
    """
    Sends one message through conversation.handle_message() and updates
    session state. DiscoveryError -- raised only for the NEW_DISCOVERY
    path, inside discovery.discover() -- is caught here exactly as
    run_search() caught it before Phase 2.
    """
    user_message = user_message.strip()
    if not user_message:
        return

    st.session_state.error_message = None
    with st.spinner("Thinking..."):
        try:
            st.session_state.conversation, _ = handle_message(st.session_state.conversation, user_message)
        except DiscoveryError as exc:
            st.session_state.error_message = str(exc)


def _evidence_to_flat_string(company: CompanyResult) -> str:
    """Flattens a company's evidence bullets into one delimited string for CSV export."""
    return " | ".join(f"{item.point} ({item.source_title}: {item.source_url})" for item in company.evidence)


def results_to_csv(results: list[CompanyResult]) -> str:
    """Serializes a list of CompanyResult -- unchanged from before Phase 2."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Company Name", "GST", "Confidence", "Reason", "Evidence"])
    for company in results:
        writer.writerow(
            [company.company_name, company.gst or "Not found", company.confidence, company.reason, _evidence_to_flat_string(company)]
        )
    return buffer.getvalue()


def results_to_json(results: list[CompanyResult]) -> str:
    """Serializes a list of CompanyResult -- unchanged from before Phase 2."""
    return json.dumps(
        [
            {
                "company_name": c.company_name,
                "gst": c.gst,
                "confidence": c.confidence,
                "reason": c.reason,
                "evidence": [
                    {"point": item.point, "source_title": item.source_title, "source_url": item.source_url}
                    for item in c.evidence
                ],
            }
            for c in results
        ],
        indent=2,
    )


def _render_debug_html(company: CompanyResult) -> str:
    """Builds the extra debug HTML for one company card -- unchanged from before Phase 2."""
    constraint_items = list(company.constraint_evaluation.items())

    identified_names = (
        ", ".join(name.replace("_", " ").title() for name, _ in constraint_items)
        if constraint_items
        else "No constraints were reported for this company."
    )

    if constraint_items:
        constraint_rows = "".join(
            f'<div class="constraint-row">'
            f'<div class="constraint-name">{html.escape(name.replace("_", " ").title())}</div>'
            f'<div class="constraint-status status-{ceval.status.lower()}">{html.escape(ceval.status)}</div>'
            f'<div class="constraint-reason">{html.escape(ceval.reason)}</div>'
            f'</div>'
            for name, ceval in constraint_items
        )
    else:
        constraint_rows = '<div class="debug-empty">No constraint evaluation was returned for this company.</div>'

    unknown_items = [(name, ceval) for name, ceval in constraint_items if ceval.status == "UNKNOWN"]
    if unknown_items:
        unknown_rows = "".join(
            f'<div class="unknown-item">• {html.escape(name.replace("_", " ").title())} -- '
            f'{html.escape(ceval.reason)}</div>'
            for name, ceval in unknown_items
        )
    else:
        unknown_rows = '<div class="debug-empty">No unresolved constraints -- every evaluated constraint was PASS or FAIL.</div>'

    decision_class = "accept" if company.decision == "ACCEPT" else "reject"

    if company.retrieval_trace is not None:
        queries_text = ", ".join(company.retrieval_trace.queries) or "(no queries issued)"
        attempt_rows = "".join(
            f'<div class="constraint-row">'
            f'<div class="constraint-status status-{"pass" if a.included else "fail"}">'
            f'{"INCLUDED" if a.included else "DISCARDED"}</div>'
            f'<div class="constraint-reason">{html.escape(a.url)} -- {html.escape(a.reason)}</div>'
            f'</div>'
            for a in company.retrieval_trace.attempts
        ) or '<div class="debug-empty">No candidate pages were found.</div>'
        retrieval_html = (
            '<div class="debug-block">'
            '<div class="debug-heading">Retrieval</div>'
            f'<div class="debug-subtle">Queries issued: {html.escape(queries_text)}</div>'
            f'{attempt_rows}'
            '</div>'
        )
    else:
        retrieval_html = (
            '<div class="debug-block">'
            '<div class="debug-heading">Retrieval</div>'
            '<div class="debug-empty">No retrieval trace is available for this company.</div>'
            '</div>'
        )

    if company.identifier_trace is not None:
        site_rows = "".join(
            f'<div class="constraint-row">'
            f'<div class="constraint-status status-{"pass" if sc.company_detected else "unknown"}">'
            f'{"DETECTED" if sc.company_detected else "NOT DETECTED"}</div>'
            f'<div class="constraint-reason">{html.escape(sc.site)}: {html.escape(sc.title)}'
            + (f" -- found {html.escape(str(sc.candidates_found))}" if sc.candidates_found else "")
            + '</div>'
            f'</div>'
            for sc in company.identifier_trace.site_checks
        ) or '<div class="debug-empty">No registry sites were checked.</div>'
        corroboration_text = (
            ", ".join(f"{k}: {v} site(s)" for k, v in company.identifier_trace.corroboration_counts.items())
            or "(none)"
        )
        validation_text = (
            ", ".join(f"{k}: {v}" for k, v in company.identifier_trace.validation_notes.items()) or "(none)"
        )
        identifier_html = (
            '<div class="debug-block">'
            '<div class="debug-heading">Identifier Lookup</div>'
            f'{site_rows}'
            f'<div class="debug-subtle">Corroboration: {html.escape(corroboration_text)}</div>'
            f'<div class="debug-subtle">Validation: {html.escape(validation_text)}</div>'
            '</div>'
        )
    else:
        identifier_html = (
            '<div class="debug-block">'
            '<div class="debug-heading">Identifier Lookup</div>'
            '<div class="debug-empty">No identifier-lookup trace is available for this company.</div>'
            '</div>'
        )

    if company.evidence_trace is not None:
        selected_rows = "".join(
            f'<div class="unknown-item">• {html.escape(point)}</div>' for point in company.evidence_trace.selected
        ) or '<div class="debug-empty">No evidence points were selected.</div>'
        rejected_rows = "".join(
            f'<div class="unknown-item">• {html.escape(r.point)} -- {html.escape(r.reason)}</div>'
            for r in company.evidence_trace.rejected
        ) or '<div class="debug-empty">No evidence points were rejected.</div>'
        evidence_trace_html = (
            '<div class="debug-block">'
            '<div class="debug-heading">Evidence Extraction</div>'
            '<div class="debug-subtle">Selected</div>'
            f'{selected_rows}'
            '<div class="debug-subtle" style="margin-top:0.4rem;">Rejected</div>'
            f'{rejected_rows}'
            '</div>'
        )
    else:
        evidence_trace_html = (
            '<div class="debug-block">'
            '<div class="debug-heading">Evidence Extraction</div>'
            '<div class="debug-empty">No evidence-extraction trace is available for this company.</div>'
            '</div>'
        )

    return (
        '<div class="debug-block">'
        '<div class="debug-heading">Constraints identified from query</div>'
        f'<div class="debug-subtle">{html.escape(identified_names)}</div>'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Constraint evaluation</div>'
        f'{constraint_rows}'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Decision</div>'
        f'<div class="decision-badge {decision_class}">{html.escape(company.decision)}</div>'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Confidence explanation / final reason</div>'
        f'<div class="debug-reason">{html.escape(company.reason)}</div>'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Assumptions &amp; unverified constraints</div>'
        f'{unknown_rows}'
        '</div>'
        f'{retrieval_html}'
        f'{identifier_html}'
        f'{evidence_trace_html}'
    )


def _render_rejected_card_html(rejected: RejectedCompany) -> str:
    """Builds one card for the "Rejected Candidates" debug section -- unchanged from before Phase 2."""
    constraint_items = list(rejected.constraint_evaluation.items())

    if constraint_items:
        constraint_rows = "".join(
            f'<div class="constraint-row">'
            f'<div class="constraint-name">{html.escape(name.replace("_", " ").title())}</div>'
            f'<div class="constraint-status status-{ceval.status.lower()}">{html.escape(ceval.status)}</div>'
            f'<div class="constraint-reason">{html.escape(ceval.reason)}</div>'
            f'</div>'
            for name, ceval in constraint_items
        )
    else:
        constraint_rows = '<div class="debug-empty">No constraint evaluation was returned for this candidate.</div>'

    is_verification = rejected.rejection_type == "verification"
    rejection_type_label = "Verification Rejection" if is_verification else "LLM Rejection"
    rejection_type_class = "verification" if is_verification else "llm"

    identifiers_html = (
        f'<div class="gst-line not-found">GST: Not found</div>'
        f'<div class="gst-line not-found">CIN: Not found</div>'
        if is_verification
        else ""
    )

    if rejected.identifier_trace is not None:
        site_rows = "".join(
            f'<div class="constraint-row">'
            f'<div class="constraint-status status-{"pass" if sc.company_detected else "unknown"}">'
            f'{"DETECTED" if sc.company_detected else "NOT DETECTED"}</div>'
            f'<div class="constraint-reason">{html.escape(sc.site)}: {html.escape(sc.title)}'
            + (f" -- found {html.escape(str(sc.candidates_found))}" if sc.candidates_found else "")
            + '</div>'
            f'</div>'
            for sc in rejected.identifier_trace.site_checks
        ) or '<div class="debug-empty">No registry sites were checked.</div>'
        identifier_trace_html = (
            '<div class="debug-block">'
            '<div class="debug-heading">Identifier Lookup</div>'
            f'{site_rows}'
            '</div>'
        )
    else:
        identifier_trace_html = ""

    return (
        '<div class="company-card low" style="opacity:0.9;">'
        f'<div class="decision-badge reject">{html.escape(rejected.decision)}</div>'
        f'<span class="rejection-type-badge {rejection_type_class}">{html.escape(rejection_type_label)}</span>'
        f'<div class="company-name">{rejected.company_name}</div>'
        f'<div class="rejected-confidence">Original LLM confidence: {html.escape(rejected.confidence)}</div>'
        f'{identifiers_html}'
        '<div class="debug-block" style="margin-top:0.3rem;">'
        '<div class="debug-heading">Constraint evaluation</div>'
        f'{constraint_rows}'
        '</div>'
        '<div class="debug-block">'
        '<div class="debug-heading">Why it was rejected</div>'
        f'<div class="debug-reason">{html.escape(rejected.reason)}</div>'
        '</div>'
        f'{identifier_trace_html}'
        '</div>'
    )


def _company_card_html(company: CompanyResult, show_debug: bool) -> str:
    """Builds one company card's HTML -- unchanged rendering logic from before Phase 2."""
    confidence_class = company.confidence.lower() if company.confidence.lower() in {"high", "medium", "low"} else "low"
    gst_display = company.gst if company.gst else "Not found"
    gst_class = "" if company.gst else "not-found"
    cin_display = company.cin if company.cin else "Not found"

    if company.evidence:
        evidence_html = "".join(
            f'<div class="evidence-item">'
            f'<div class="evidence-point">• {html.escape(item.point)}</div>'
            f'<div class="evidence-source">🔗 '
            f'<a href="{html.escape(item.source_url, quote=True)}" target="_blank" rel="noopener noreferrer">'
            f'{html.escape(item.source_title)}</a></div>'
            f'</div>'
            for item in company.evidence
        )
    else:
        evidence_html = '<div class="evidence-empty">No sourced evidence found for this company.</div>'

    debug_html = _render_debug_html(company) if show_debug else ""
    evidence_heading_html = '<div class="debug-heading" style="margin-top:0.75rem;">Evidence</div>' if show_debug else ""

    return (
        f'<div class="company-card {confidence_class}">'
        f'<div class="confidence-badge {confidence_class}">{company.confidence} confidence</div>'
        f'<div class="company-name">{company.company_name}</div>'
        f'<div class="gst-line {gst_class}">GST: {gst_display}</div>'
        f'<div class="gst-line">CIN: {cin_display}</div>'
        f'{debug_html}'
        f'{evidence_heading_html}'
        f'<div class="evidence-list">{evidence_html}</div>'
        f'</div>'
    )


def _render_intent_debug_html(turn: ChatTurn) -> str:
    """
    Builds the turn-level debug block: detected intent, confidence, and
    reasoning for every turn, plus (only for COMPANY_QUESTION turns)
    whether new retrieval was used and which source URLs were consulted.
    """
    qa_html = ""
    if turn.intent == "COMPANY_QUESTION":
        sources_text = ", ".join(turn.qa_sources) if turn.qa_sources else "(none)"
        qa_html = (
            '<div class="debug-block">'
            '<div class="debug-heading">Final Answer Generation</div>'
            f'<div class="debug-subtle">Used new retrieval: {"Yes" if turn.qa_used_new_retrieval else "No"}</div>'
            f'<div class="debug-subtle">Sources: {html.escape(sources_text)}</div>'
            '</div>'
        )

    return (
        '<div class="debug-block">'
        '<div class="debug-heading">Intent Classification</div>'
        f'<div class="debug-subtle">Detected intent: {html.escape(turn.intent)} '
        f'({html.escape(turn.intent_confidence) or "n/a"} confidence)</div>'
        f'<div class="debug-reason">{html.escape(turn.intent_reasoning) or "(no reasoning recorded)"}</div>'
        '</div>'
        f'{qa_html}'
    )


def _render_turn(turn: ChatTurn, show_debug: bool) -> None:
    """Renders one past turn: the user's message, the assistant's reply, and any company cards."""
    with st.chat_message("user"):
        st.write(turn.user_message)

    with st.chat_message("assistant"):
        if show_debug:
            st.markdown(_render_intent_debug_html(turn), unsafe_allow_html=True)

        st.write(turn.assistant_response)

        for company in turn.companies:
            st.markdown(_company_card_html(company, show_debug), unsafe_allow_html=True)

        if show_debug and turn.discovery_result is not None:
            with st.expander(f"Rejected Candidates ({len(turn.discovery_result.rejected)})", expanded=False):
                if turn.discovery_result.rejected:
                    st.caption(
                        "Candidates that did not make it into the results above, for two "
                        "possible reasons: the model itself ruled them out because an explicit "
                        "constraint failed (\"LLM Rejection\"), or the model recommended them but "
                        "deterministic GST/CIN verification couldn't confirm the legal entity "
                        "exists (\"Verification Rejection\")."
                    )
                    for rejected in turn.discovery_result.rejected:
                        st.markdown(_render_rejected_card_html(rejected), unsafe_allow_html=True)
                else:
                    st.caption("No rejected candidates were reported for this search.")


# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("📇 Company Discovery Agent")
st.caption(
    "Ask a plain-language question, or ask a follow-up about a company already "
    "discussed. Every match comes with sourced evidence from the web and a "
    "deterministic GST check -- never guessed, never fabricated."
)

# -----------------------------------------------------------------------
# Sidebar: debug toggle + new-conversation reset
# -----------------------------------------------------------------------
with st.sidebar:
    show_debug = st.checkbox(
        "Show Debug Information",
        value=False,
        key="show_debug_toggle",
        help=(
            "Expands every company card with the LLM's full per-constraint "
            "evaluation, its explicit decision, and any assumptions made -- "
            "useful for diagnosing retrieval and prompt quality. The normal "
            "view is unaffected when this is off."
        ),
    )

    st.divider()

    if st.button("New conversation", use_container_width=True):
        st.session_state.conversation = ConversationState()
        st.session_state.error_message = None
        st.rerun()

# -----------------------------------------------------------------------
# Chat history
# -----------------------------------------------------------------------
conversation_state: ConversationState = st.session_state.conversation

if st.session_state.error_message:
    st.error(st.session_state.error_message)

for turn in conversation_state.turns:
    _render_turn(turn, show_debug)

# -----------------------------------------------------------------------
# Message input -- st.chat_input pins itself to the bottom of the
# viewport regardless of where it's called in the script.
# -----------------------------------------------------------------------
prompt = st.chat_input("Ask a question or a follow-up...")
if prompt:
    run_message(prompt)
    st.rerun()

# -----------------------------------------------------------------------
# Export the most recent turn that produced companies
# -----------------------------------------------------------------------
_most_recent_companies: list[CompanyResult] = []
for turn in reversed(conversation_state.turns):
    if turn.companies:
        _most_recent_companies = turn.companies
        break

if _most_recent_companies:
    st.divider()
    st.subheader("Export most recent result set")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            data=results_to_csv(_most_recent_companies),
            file_name="company_discovery_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download JSON",
            data=results_to_json(_most_recent_companies),
            file_name="company_discovery_results.json",
            mime="application/json",
            use_container_width=True,
        )
