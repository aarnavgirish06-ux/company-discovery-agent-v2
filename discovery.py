"""
discovery.py

Reusable business logic for the Company Discovery Agent. This module owns
all orchestration: calling the LLM for company discovery, validating its
output, retrieving webpages for each company, and fanning the resulting
documents out to both the deterministic identifier extractor and the
LLM-based evidence extractor.

Design notes:

- No UI-specific formatting lives here. Every public function returns plain
  dataclasses/lists -- callers (app.py, ui.py, or any future frontend such
  as a FastAPI service or React app) decide how to display them.
- discovery.py itself never searches the web, downloads a page, or parses
  HTML. All of that lives in retriever.py and (for GST/CIN)
  identifier_lookup.py.
- PIPELINE ORDER (identifier lookup -> entity verification -> evidence):
  for each accepted company, identifier lookup runs first and is awaited
  before anything else happens. Its result decides whether the company is
  a verified legal entity (GST exists OR CIN exists). Only if verified do
  we go on to retrieve and extract evidence; if not, evidence retrieval is
  skipped entirely and the CompanyResult is populated with GST/CIN/PAN as
  None and no evidence, with a reason explaining that verification failed.

  This ordering is a deliberate architectural choice, not an incidental
  side effect: evidence retrieval is a plain web search for the company
  name, and when the LLM has hallucinated or misremembered a company that
  doesn't actually exist, that search reliably turns up pages about
  other, similarly-named real companies. If evidence retrieval ran before
  (or independently of) identifier verification, that unrelated evidence
  would get attached to a company we already know isn't real, making a
  hallucination look well-sourced. Gating evidence retrieval on a
  successful GST/CIN lookup means we only ever spend a web search -- and
  only ever show evidence -- on a company we've already confirmed exists.

  Because of this, identifier lookup and evidence retrieval are no longer
  independent branches that can run concurrently within a company: evidence
  retrieval now depends on the identifier lookup's outcome, so they run
  sequentially. (A `ThreadPoolExecutor` is not used here anymore for that
  reason -- there is nothing left within a single company to run in
  parallel. It would still be the right tool if, in the future, identifier
  lookups for *different* companies were parallelized across the loop
  below; that's a possible follow-up but out of scope for this change.)
- Per-company work still proceeds one company at a time (the LLM typically
  returns a handful of companies, 3-10), so the simplicity of a plain loop
  outweighs the complexity of concurrency across companies for now.
- `discover()` accepts an optional `history` parameter reserved for future
  multi-turn conversations. It is currently unused (each call is treated
  independently), but accepting it now means callers can start threading
  conversation context through without a future signature change breaking
  them. See `_build_prompt(...)` for the single place that would need to
  change to actually use it.

Debug mode note:
  The LLM's JSON response now also includes "constraint_evaluation" (a
  per-constraint PASS/FAIL/UNKNOWN breakdown with reasons) and "decision"
  (the LLM's explicit ACCEPT/REJECT call for that entry). These are purely
  diagnostic/debug fields -- see prompts.py rule 8/9 and ui.py's "Show
  Debug Information" toggle. They are parsed defensively here (missing or
  malformed values fall back to safe defaults) so that older prompt
  versions or slightly malformed LLM output never break discovery; they do
  not affect which companies are returned or how they are ranked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic import ValidationError

import retriever
from evidence_extractor import EvidenceItem, EvidenceTrace, extract as extract_evidence
from identifier_lookup import IdentifierRecord, IdentifierTrace, get_company_identifiers
from json_utils import JsonArrayParseError, extract_json_array
from llm_provider import LLMProviderError, get_provider, is_grounded, structured_output_kwargs
from llm_schemas import LLMCompanyEntry, LLMConstraintEvaluation, LLMDiscoveryResponse
from prompts import DISCOVERY_PROMPT, build_user_prompt


@dataclass
class ConstraintEvaluation:
    """The LLM's PASS/FAIL/UNKNOWN judgement for a single explicit user constraint."""
    status: str  # "PASS" | "FAIL" | "UNKNOWN"
    reason: str


@dataclass
class CompanyResult:
    """A single company result, ready for display by any frontend."""
    company_name: str
    reason: str
    confidence: str  # exactly what the LLM returned: "High" | "Medium" | "Low"
    gst: Optional[str]  # representative GSTIN if found, else None ("GST not found")
    cin: Optional[str]  # representative CIN if found, else None ("CIN not found")
    pan: Optional[str]  # derived from a found GSTIN's embedded PAN; None if no GST found
    evidence: List[EvidenceItem] = field(default_factory=list)
    documents: List[retriever.Document] = field(default_factory=list)  # raw pages retrieved for evidence, kept (not discarded) so qa.py can reuse them without re-fetching
    # -- Debug-mode fields (see module docstring). Never affect filtering/ranking. --
    constraint_evaluation: Dict[str, ConstraintEvaluation] = field(default_factory=dict)
    decision: str = "ACCEPT"  # "ACCEPT" | "REJECT" -- explicit LLM decision for this entry
    retrieval_trace: Optional[retriever.RetrievalTrace] = None  # engineered record of the evidence retrieval call (Phase 4 debug mode)
    identifier_trace: Optional[IdentifierTrace] = None  # engineered record of the identifier fallback chain (Phase 4 debug mode)
    evidence_trace: Optional[EvidenceTrace] = None  # engineered record of evidence selection/rejection (Phase 4 debug mode)


@dataclass
class RejectedCompany:
    """
    A candidate that did not make it into `accepted`, for the debug-only
    "Rejected Candidates" section. Never shown in the normal (non-debug)
    UI. `rejection_type` distinguishes the two distinct ways a candidate
    can end up here, so the UI never has to infer the source from the
    reason text:

    - "llm": the LLM itself rejected the candidate under Rule 2/8's
      decision rules (at least one explicit constraint was FAIL). Never
      went through identifier lookup or evidence retrieval.
    - "verification": the LLM said ACCEPT, but deterministic GST/CIN
      verification failed (no GST or CIN record found), so the verifier
      overrode the LLM's decision to REJECT. Went through identifier
      lookup (hence `gst`/`cin` are present, and always None here by
      definition -- if either had been found, verification would have
      passed) but never reached evidence retrieval/extraction.

    Either way, evidence is never retrieved or shown for a rejected
    candidate -- there is no `evidence` field on this class at all.
    """
    company_name: str
    reason: str
    confidence: str  # the LLM's original confidence -- never overwritten, even for verification rejections
    rejection_type: str  # "llm" | "verification"
    constraint_evaluation: Dict[str, ConstraintEvaluation] = field(default_factory=dict)
    decision: str = "REJECT"
    gst: Optional[str] = None  # always None here; present so the UI can display "GST: Not found" for verification rejections
    cin: Optional[str] = None  # always None here; present so the UI can display "CIN: Not found" for verification rejections
    identifier_trace: Optional[IdentifierTrace] = None  # populated only for rejection_type="verification" -- LLM rejections never reach identifier lookup (Phase 4 debug mode)


# Defensive upper bound on how many LLM-reported rejected candidates we'll
# ever hold onto from one discover() call, independent of what the prompt
# asks the model for. Purely a safety net against a malformed/unbounded LLM
# response -- the prompt itself asks for at most 5. This only bounds
# rejection_type="llm" entries; rejection_type="verification" entries are
# naturally bounded by how many companies the LLM accepted in the first
# place, so no separate cap is needed for those.
_MAX_REJECTED_CANDIDATES = 10


@dataclass
class DiscoveryResult:
    """
    The full outcome of one discover() call.

    `accepted` is exactly what the normal (non-debug) UI has always shown --
    fully resolved CompanyResult objects with GST/CIN/PAN and evidence.
    Every entry here is a verified legal entity (GST or CIN was found);
    accepting a company and failing verification are mutually exclusive.

    `rejected` is debug-only, for the "Rejected Candidates" section, and
    contains two kinds of entries (see RejectedCompany.rejection_type):
    candidates the LLM itself rejected ("llm"), and candidates the LLM
    accepted but that failed deterministic GST/CIN verification
    ("verification"). Empty whenever debug mode isn't being used, or when
    there were no rejections of either kind.
    """
    accepted: List[CompanyResult] = field(default_factory=list)
    rejected: List[RejectedCompany] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """
    One past turn in a conversation: the query that was asked and the
    results it produced. Reserved for future multi-turn support -- not
    used by the current single-turn `discover()` flow, but the shape
    future history-aware callers would build up and pass back in.
    """
    query: str
    companies: List[CompanyResult] = field(default_factory=list)


class DiscoveryError(Exception):
    """Raised when a discovery request cannot be completed."""


def _constraint_dict_from_llm(
    entries: List[LLMConstraintEvaluation],
) -> Dict[str, ConstraintEvaluation]:
    """
    Rebuilds the internal constraint_name -> ConstraintEvaluation dict from
    the LLM's structured List[LLMConstraintEvaluation]. Pydantic has
    already validated `status` against PASS/FAIL/UNKNOWN and guaranteed
    `name`/`reason` are strings, so this is a pure reshape, not validation.
    """
    return {
        item.name.strip(): ConstraintEvaluation(status=item.status, reason=item.reason.strip())
        for item in entries
        if item.name.strip()
    }


def _entry_from_llm(entry: LLMCompanyEntry) -> dict:
    """
    Adapts one Pydantic-validated LLMCompanyEntry into the plain dict shape
    the rest of discover() expects (unchanged since before the LangChain
    migration, so everything downstream of this function is untouched).
    """
    return {
        "company_name": entry.company_name.strip(),
        "reason": entry.reason.strip(),
        "confidence": entry.confidence,
        "decision": entry.decision,
        "constraint_evaluation": _constraint_dict_from_llm(entry.constraint_evaluation),
    }


def _identifier_value_to_display(records_or_error: List[IdentifierRecord] | str) -> Optional[str]:
    """
    Translates one entry of identifier_lookup.get_company_identifiers()'s
    return value -- a List[IdentifierRecord] (currently at most one
    representative record) or a literal "<TYPE> not verified" string --
    into the simplified, UI-oriented representation CompanyResult exposes:
    either the representative identifier string, or None.
    """
    if isinstance(records_or_error, str):
        return None
    if not records_or_error:
        return None
    return ", ".join(record.value for record in records_or_error)


def _pan_to_display(gst_records_or_error: List[IdentifierRecord] | str) -> Optional[str]:
    """
    Derives a display PAN from the GST identifier result. For GST records,
    `corroboration_key` holds the embedded PAN (see identifier_lookup.py),
    so no separate extraction is needed -- PAN is only ever available when
    a GSTIN was found.
    """
    if isinstance(gst_records_or_error, str):
        return None
    if not gst_records_or_error:
        return None
    pans = {record.corroboration_key for record in gst_records_or_error}
    return ", ".join(sorted(pans))


def _build_prompt(query: str, history: Optional[List[ConversationTurn]] = None) -> str:
    """
    Builds the user-turn prompt sent to the LLM. `history` is accepted for
    future multi-turn support but currently ignored -- this is the single
    place a future implementation would splice prior turns into the prompt.
    """
    # Future: incorporate `history` (e.g. prior queries/results) here to
    # support follow-up questions like "narrow that down to Maharashtra".
    return build_user_prompt(query)


def discover(
    query: str, history: Optional[List[ConversationTurn]] = None
) -> DiscoveryResult:
    """
    Runs a full discovery request: LLM company discovery, validation, and
    for each ACCEPTED company, identifier lookup (GST + CIN) followed by
    entity verification. Verification is the final authority: a company the
    LLM accepted is only included in `.accepted` if GST or CIN was actually
    found; otherwise it is moved to `.rejected` with the LLM's ACCEPT
    overridden to REJECT. Evidence retrieval and extraction only happen for
    companies that pass verification -- see the module docstring's
    "PIPELINE ORDER" note for why this ordering matters.

    The LLM's response may also include a small number of REJECTED
    candidates (debug-only -- see prompts.py Rule 8). Those never go
    through identifier lookup or evidence retrieval, since that work is
    only meaningful for companies actually being recommended; they're
    returned as lightweight RejectedCompany objects for the UI's debug
    "Rejected Candidates" section, alongside any verification-failure
    rejections (distinguished by `rejection_type`).

    Args:
        query: The natural-language discovery query.
        history: Reserved for future multi-turn support. Currently unused.

    Returns:
        A DiscoveryResult with `.accepted` (verified companies only -- what
        the normal UI has always shown) and `.rejected` (debug-only: both
        LLM-rejected candidates and companies the LLM accepted but that
        failed deterministic GST/CIN verification).

    Raises:
        DiscoveryError: if the LLM provider fails or its response can't be
            parsed as the expected JSON array.
    """
    if not query or not query.strip():
        raise DiscoveryError("Query cannot be empty.")

    prompt_messages = DISCOVERY_PROMPT.format_messages(
        user_prompt=_build_prompt(query, history)
    )

    try:
        llm = get_provider(allow_grounding=True)

        if is_grounded():
            # Gemini's google_search grounding tool cannot be combined with
            # with_structured_output's schema constraint in the same call
            # (see llm_provider.py's module docstring). Fall back to
            # prompt-instructed JSON (prompts.py Rule 9) + manual parsing,
            # still validated through the same Pydantic schema used below.
            raw_message = llm.invoke(prompt_messages)
            parsed_entries = extract_json_array(raw_message.text)
            response = LLMDiscoveryResponse(
                companies=[LLMCompanyEntry.model_validate(e) for e in parsed_entries]
            )
        else:
            response = llm.with_structured_output(
                LLMDiscoveryResponse, **structured_output_kwargs()
            ).invoke(prompt_messages)
    except LLMProviderError as exc:
        raise DiscoveryError(f"LLM provider error: {exc}") from exc
    except JsonArrayParseError as exc:
        raise DiscoveryError(str(exc)) from exc
    except ValidationError as exc:
        raise DiscoveryError(f"LLM response failed schema validation: {exc}") from exc
    except Exception as exc:
        raise DiscoveryError(f"LLM request failed: {exc}") from exc

    validated_entries = [_entry_from_llm(entry) for entry in response.companies]

    # Split by the LLM's explicit decision. ACCEPT entries are candidates
    # for recommendation and go through identifier lookup + verification
    # below -- but verification, not the LLM, has the final say on whether
    # any of them actually end up in `accepted` (see the loop below). REJECT
    # entries are the LLM's own rejections and never reach identifier
    # lookup or evidence retrieval at all.
    accepted_entries = [e for e in validated_entries if e["decision"] == "ACCEPT"]
    llm_rejected_entries = [e for e in validated_entries if e["decision"] == "REJECT"][:_MAX_REJECTED_CANDIDATES]

    accepted: List[CompanyResult] = []
    rejected: List[RejectedCompany] = []

    # The LLM's own rejections go straight into `rejected`, tagged so the
    # UI can label them distinctly from verification failures.
    for entry in llm_rejected_entries:
        rejected.append(
            RejectedCompany(
                company_name=entry["company_name"],
                reason=entry["reason"],
                confidence=entry["confidence"],
                rejection_type="llm",
                constraint_evaluation=entry["constraint_evaluation"],
                decision=entry["decision"],
            )
        )

    for entry in accepted_entries:
        # STEP 1: Identifier lookup runs first and is fully awaited before
        # anything else. get_company_identifiers() drives its own retrieval
        # internally (a lazy, early-stopping fallback chain across registry
        # sites) and returns BOTH GST and CIN from one pass.
        identifiers, identifier_trace = get_company_identifiers(entry["company_name"])
        gst_result = identifiers.get("GST", "GST not verified")
        cin_result = identifiers.get("CIN", "CIN not verified")

        gst_display = _identifier_value_to_display(gst_result)
        cin_display = _identifier_value_to_display(cin_result)
        pan_display = _pan_to_display(gst_result)

        # STEP 2: Entity verification. verified = GST exists OR CIN exists.
        # This is the final authority on whether a company is recommended
        # -- the LLM's own ACCEPT decision is overridden below if
        # verification fails.
        verified = gst_display is not None or cin_display is not None

        if not verified:
            # STEP 3a (unverified): Deterministic verification overrides the
            # LLM's ACCEPT -> the company is rejected, not accepted with
            # empty evidence. Do NOT call retriever.retrieve_for_evidence()
            # or evidence_extractor.extract() -- a web search for a company
            # name that has no GST/CIN record (i.e. we have no independent
            # confirmation it's a real registered entity) reliably surfaces
            # pages about other, similarly-named companies instead. The
            # LLM's original confidence is preserved rather than overwritten
            # -- a high LLM confidence paired with a failed verification is
            # itself useful debugging signal (the model hallucinated
            # confidently), not something to hide by resetting it to Low.
            rejected.append(
                RejectedCompany(
                    company_name=entry["company_name"],
                    reason=(
                        "Could not be verified as a registered legal entity -- no "
                        "GST or CIN record was found -- so this candidate was "
                        "rejected by deterministic verification, overriding the "
                        "LLM's ACCEPT decision."
                    ),
                    confidence=entry["confidence"],  # preserved, not overwritten
                    rejection_type="verification",
                    constraint_evaluation=entry["constraint_evaluation"],
                    decision="REJECT",  # overridden from the LLM's original ACCEPT
                    gst=None,
                    cin=None,
                    identifier_trace=identifier_trace,
                )
            )
            continue

        # STEP 3b (verified): proceed exactly as before -- retrieve, then
        # extract, evidence for this company.
        evidence_documents, retrieval_trace = retriever.retrieve_for_evidence(entry["company_name"])
        evidence, evidence_trace = extract_evidence(entry["company_name"], query, entry["reason"], evidence_documents)

        accepted.append(
            CompanyResult(
                company_name=entry["company_name"],
                reason=entry["reason"],
                confidence=entry["confidence"],
                gst=gst_display,
                cin=cin_display,
                pan=pan_display,
                evidence=evidence,
                documents=evidence_documents,
                constraint_evaluation=entry["constraint_evaluation"],
                decision=entry["decision"],
                retrieval_trace=retrieval_trace,
                identifier_trace=identifier_trace,
                evidence_trace=evidence_trace,
            )
        )

    return DiscoveryResult(accepted=accepted, rejected=rejected)
