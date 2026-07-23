"""
identifier_lookup.py

Deterministic, LLM-independent company-identifier extraction: GSTIN (Goods
and Services Tax Identification Number) and CIN (Corporate Identification
Number).

This module performs NO web searching, downloading, or HTTP requests of
its own (except for the optional configured-API paths described below,
which are independent official data sources, not a web search) -- it
drives retrieval (via retriever.iter_gst_documents()) because doing so is
what lets it stop early.

WHY GST AND CIN LIVE IN ONE MODULE:

Testing showed that registry sites split along identifier lines: The
Company Check, ZaubaCorp, and Tofler reliably expose CIN but not GSTIN;
KnowYourGST reliably exposes GSTIN but not CIN. So GST and CIN are
retrieved from the SAME site-priority chain, over the SAME downloaded
pages, using the SAME "verify the page is about this company, then look
for an identifier-shaped string near that mention" mechanism -- they just
apply a different pattern/validator/corroboration-key to each page. Rather
than duplicate that mechanism, one generic engine below (organized around
`_IdentifierSpec`) drives both, and `get_gst_details()` / `get_cin_details()`
are thin, GST/CIN-specific views over it.

COMPANY VERIFICATION:

"Is this page about the right company" is NOT decided in this module. It
is delegated entirely to entity_matching.py, which performs deterministic
legal-entity matching (exact substantive-name matching with normalized-
but-preserved legal form, plus symmetric boundary checks) rather than
fuzzy string similarity. See entity_matching.py's module docstring for the
full rationale. This module only knows that it needs to call
`parse_entity_name()` once per company and `find_entity_mentions()` per
document -- it has no matching logic of its own.

IMPORTANT LIMITATIONS (documented, not hidden):

GST:

1. There is no free, official, public "search by company name" API for GST.
   The GST portal (services.gst.gov.in) only lets you look up a taxpayer's
   details once you already have their GSTIN or PAN, and is CAPTCHA gated
   for interactive search. So, as with any name -> identifier resolution
   problem here, there are two paths: (a) a paid/official API, or (b)
   extracting from public web pages that mention the identifier.

2. A company can legitimately hold MULTIPLE GSTINs -- one per state it is
   registered in. This is expected and normal, not an error. Because of
   this, "the one correct GSTIN" isn't a well-posed question the way "the
   one correct CIN" is. What identifies the company uniquely is the PAN
   embedded in every GSTIN it holds (characters 3-12), so GST corroboration
   is keyed by that embedded PAN, not by the GSTIN string itself -- two
   different-looking GSTINs from two different sites still count as
   agreement if their embedded PANs match.

3. GSTIN DOES have a checksum (a Luhn mod-36 check digit over its first 14
   characters), so a candidate can be confirmed not just well-formed but
   internally self-consistent -- i.e. not a typo or a fabricated-looking
   string. It still cannot confirm the GSTIN is actually registered to
   this company; only the GST portal itself can do that. As an additional
   deterministic check (independent of the checksum), a GST candidate's
   embedded PAN category is cross-checked against the requested entity's
   legal form via entity_matching.pan_category_matches() -- see that
   module for details and its deliberately-limited scope.

CIN:

1. A CIN is a single canonical identifier per company (no multi-state
   variants the way GST has), so CIN corroboration is keyed by the CIN
   value itself, not by any derived sub-key.

2. Unlike GSTIN, a CIN has NO checksum digit. Validation here is
   structural/regex only (1 letter listing status + 5 digit industry code
   + 2 letter state code + 4 digit incorporation year + 3 letter
   entity-type code + 6 digit serial), with an additional plausibility
   check on the embedded year. This is a strictly weaker verification
   guarantee than GST's checksum -- a well-formed-looking CIN has NOT been
   confirmed internally self-consistent the way a checksum-valid GSTIN
   has, only "correctly shaped." There is no unambiguous CIN <-> legal-form
   cross-check analogous to GST's PAN category (CIN's state code is MCA's
   own code, not derived from entity type), so none is applied.

RETRIEVAL STRATEGY (fallback chain, not a single batch fetch):

Registry sites are treated as ALTERNATIVES within an identifier type, not
complementary sources -- they mostly republish the same underlying
MCA/GST-portal data, so querying all of them once an identifier is already
corroborated wastes searches, downloads, and time.
`get_company_identifiers()` pulls candidate pages from
`retriever.iter_gst_documents()` (which searches one registry site at a
time, in `retriever.GST_SITE_PRIORITY` order, and does so lazily) and, for
every page it downloads, tries EVERY identifier type it's still looking
for (no hardcoded "this site gives CIN" mapping -- a pattern that doesn't
match a given page's text just yields nothing, at negligible cost). A
site's remaining pages are skipped as soon as it contributes a candidate
for ANY identifier still outstanding. The whole chain stops once every
identifier being pursued has reached its own corroboration threshold, or
the site list is exhausted -- at which point each identifier
independently falls back to whatever is best-corroborated so far (even if
that's only one site) -- a single source is weaker evidence than two
agreeing sources, but still better than nothing.

Each identifier type also has an optional configured-API override
(GST_API_URL / CIN_API_URL) that bypasses retrieval entirely for that
identifier, checked before the retrieval chain runs. If both are
configured, no retrieval happens at all for this company.

No identifier is ever guessed or generated by an LLM.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Dict, List, Tuple

import requests

from entity_matching import ParsedEntity, find_entity_mentions, pan_category_matches, parse_entity_name
from retriever import Document, iter_gst_documents

# ---------------------------------------------------------------------
# GSTIN structure (15 characters), per GSTN specification:
#   [2]  State code (numeric, Census 2011 state/UT code)
#   [10] PAN of the taxpayer (5 letters + 4 digits + 1 letter)
#   [1]  Entity number: registration count for this PAN within the state
#        (1-9, then A-Z for the 10th+ registration)
#   [1]  Always "Z" (reserved for future use)
#   [1]  Checksum character (Luhn mod-36 check digit over characters 1-14)
GST_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")

# Character set used by the GSTIN checksum algorithm: digits 0-9 map to
# their own value, then A-Z map to 10-35 (a standard Luhn mod-36 variant).
_GST_CHECKSUM_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ---------------------------------------------------------------------
# CIN structure (21 characters), per MCA specification:
#   [1]  Listing status: L (listed) or U (unlisted)
#   [5]  Industry/economic activity code
#   [2]  State code (alphabetic, MCA's own codes, not GST's numeric ones)
#   [4]  Year of incorporation
#   [3]  Ownership/entity type (e.g. PLC, PTC, GOI, NPL, ...)
#   [6]  Registration serial number
CIN_PATTERN = re.compile(r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b")

# How many distinct registry sites must independently corroborate the same
# candidate before the fallback chain treats that identifier as settled.
# Kept equal for GST and CIN even though empirically only ~1 site reliably
# yields GST vs ~3 for CIN: GST simply falls back to "best available, even
# at 1 corroboration" almost every time, which is the same graceful
# degradation used when a chain is exhausted early for any other reason.
_MIN_CORROBORATING_SITES = 2


@dataclass(frozen=True)
class IdentifierRecord:
    """
    A single format-and-validated identifier result.

    `corroboration_key` is what corroboration across sites/documents is
    grouped by -- for GST this is the embedded PAN (see module docstring
    for why); for CIN it's simply the CIN value itself.
    """
    identifier_type: str  # "GST" | "CIN"
    value: str
    corroboration_key: str
    corroborations: int
    source_note: str


@dataclass(frozen=True)
class SiteCheck:
    """One page checked during the identifier fallback chain."""
    site: str
    url: str
    title: str
    company_detected: bool
    candidates_found: Dict[str, str]  # e.g. {"GST": "27ABCCE1234F1Z2"} for identifiers found on this page


@dataclass(frozen=True)
class IdentifierTrace:
    """Structured, engineered record of the identifier fallback chain's execution."""
    site_checks: List[SiteCheck]
    corroboration_counts: Dict[str, int]  # e.g. {"GST": 1, "CIN": 2}
    validation_notes: Dict[str, str]  # e.g. {"GST": "...checksum...", "CIN": "...no checksum exists for CIN..."}


@dataclass(frozen=True)
class GSTRecord:
    """Thin, backward-compatible named view over an IdentifierRecord(identifier_type='GST')."""
    gstin: str
    pan: str
    state_code: str
    source_note: str


@dataclass(frozen=True)
class CINRecord:
    """Thin, backward-compatible named view over an IdentifierRecord(identifier_type='CIN')."""
    cin: str
    source_note: str


def _to_gst_record(record: IdentifierRecord) -> GSTRecord:
    return GSTRecord(
        gstin=record.value,
        pan=record.corroboration_key,
        state_code=record.value[0:2],
        source_note=record.source_note,
    )


def _to_cin_record(record: IdentifierRecord) -> CINRecord:
    return CINRecord(cin=record.value, source_note=record.source_note)


def _extract_pan(gstin: str) -> str:
    """
    Extracts the PAN embedded in a GSTIN. Characters 3-12 (0-indexed 2:12)
    of a GSTIN are always that taxpayer's PAN -- this is what actually
    identifies the company, since a company can hold multiple GSTINs (one
    per state) that all share the same PAN.
    """
    return gstin[2:12]


def _gst_checksum_char(gstin_prefix: str) -> str:
    """
    Computes the expected 15th (checksum) character of a GSTIN from its
    first 14 characters, using the published Luhn mod-36 algorithm: each
    character is mapped to a value (0-9 for digits, 10-35 for A-Z),
    alternately weighted by 1 and 2 from the left, each product reduced via
    (value // 36) + (value % 36), summed, and the check character is
    whichever one makes the total a multiple of 36.
    """
    modulus = len(_GST_CHECKSUM_CHARSET)
    factor = 1
    total = 0
    for char in gstin_prefix:
        value = _GST_CHECKSUM_CHARSET.index(char) * factor
        value = (value // modulus) + (value % modulus)
        total += value
        factor = 2 if factor == 1 else 1

    check_code_point = (modulus - (total % modulus)) % modulus
    return _GST_CHECKSUM_CHARSET[check_code_point]


def _gst_format_is_valid(gstin: str) -> bool:
    """
    Validates a GSTIN: correct overall structural shape (regex) AND a
    correct Luhn mod-36 checksum over its first 14 characters.

    NOTE: Passing both checks confirms the GSTIN is internally
    self-consistent (not a typo or an arbitrary fabricated-looking string).
    It does NOT confirm the number is genuinely registered, or registered
    to this particular company -- only the GST portal itself can do that.
    """
    if not GST_PATTERN.fullmatch(gstin):
        return False

    expected_checksum = _gst_checksum_char(gstin[:14])
    return gstin[14] == expected_checksum


def _cin_format_is_valid(cin: str) -> bool:
    """
    Validates a CIN: correct overall structural shape (regex) plus a
    plausibility check on the embedded incorporation year.

    Unlike GSTIN, a CIN has NO checksum digit, so this is a strictly
    weaker guarantee than `_gst_format_is_valid` -- it confirms the string
    is correctly SHAPED, not that it's internally self-consistent the way
    a checksum can. It does NOT confirm the CIN is genuinely registered,
    or registered to this particular company.
    """
    if not CIN_PATTERN.fullmatch(cin):
        return False

    year = int(cin[8:12])
    current_year = date.today().year
    return 1900 <= year <= current_year + 1


@dataclass(frozen=True)
class _IdentifierSpec:
    """Everything the generic engine needs to look for one identifier type."""
    identifier_type: str
    pattern: re.Pattern
    validator: Callable[[str], bool]
    corroboration_key_fn: Callable[[str], str]
    min_corroborating_sites: int
    api_url_env: str
    api_key_env: str
    # Optional second-pass check, applied alongside `validator`, that also
    # needs to know which entity was requested (e.g. PAN-category
    # cross-checking for GST -- see entity_matching.pan_category_matches).
    # None for identifier types with no such check (e.g. CIN).
    extra_validator: Callable[[str, ParsedEntity], bool] | None = None


_GST_SPEC = _IdentifierSpec(
    identifier_type="GST",
    pattern=GST_PATTERN,
    validator=_gst_format_is_valid,
    corroboration_key_fn=_extract_pan,
    min_corroborating_sites=_MIN_CORROBORATING_SITES,
    api_url_env="GST_API_URL",
    api_key_env="GST_API_KEY",
    extra_validator=pan_category_matches,
)

_CIN_SPEC = _IdentifierSpec(
    identifier_type="CIN",
    pattern=CIN_PATTERN,
    validator=_cin_format_is_valid,
    corroboration_key_fn=lambda value: value,
    min_corroborating_sites=_MIN_CORROBORATING_SITES,
    api_url_env="CIN_API_URL",
    api_key_env="CIN_API_KEY",
    # No unambiguous CIN <-> legal-form cross-check exists -- see module
    # docstring -- so no extra_validator.
)

_ALL_SPECS = (_GST_SPEC, _CIN_SPEC)


def _lookup_via_configured_api(company_name: str, spec: _IdentifierSpec) -> List[IdentifierRecord]:
    """
    Calls a user-configured lookup API (GST_API_URL or CIN_API_URL,
    depending on `spec`). Expected to return JSON shaped like:
        {"results": [{"value": "...", ...}]}
    Adjust the parsing below if your provider's response shape differs.

    This is an independent official data source, not a web search -- it
    needs no documents and is checked before any retrieval is attempted
    for this identifier type.
    """
    api_url = os.getenv(spec.api_url_env, "").strip()
    if not api_url:
        return []

    api_key = os.getenv(spec.api_key_env, "").strip()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        response = requests.get(
            api_url,
            params={"company_name": company_name},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []

    records: List[IdentifierRecord] = []
    for item in data.get("results", []):
        value = str(item.get("value", "")).strip().upper()
        if spec.validator(value):
            records.append(
                IdentifierRecord(
                    identifier_type=spec.identifier_type,
                    value=value,
                    corroboration_key=spec.corroboration_key_fn(value),
                    corroborations=1,
                    source_note=f"Configured {spec.api_url_env} provider",
                )
            )
    return records


def _extract_candidate(
    page_text: str,
    mention_positions: List[int],
    spec: _IdentifierSpec,
    requested: ParsedEntity,
) -> str | None:
    """
    Searches the ENTIRE document text for strings matching `spec.pattern`,
    validates each one via `spec.validator` (and, if present,
    `spec.extra_validator` against the requested entity), and returns the
    single value physically closest to any verified company mention --
    rather than returning every valid match in the document.

    A long document can legitimately contain other companies' identifiers
    (vendor lists, related filings, footnotes about a registrar, etc.).
    All of those can pass format/checksum validation too, so validation
    alone can't tell them apart from the right one. Proximity to an actual
    verified mention of the requested company is the best available
    signal for which identifier the document is really about.
    """
    best_value: str | None = None
    best_distance = float("inf")

    for match in spec.pattern.finditer(page_text):
        candidate = match.group(0)
        if not spec.validator(candidate):
            continue
        if spec.extra_validator is not None and not spec.extra_validator(candidate, requested):
            continue

        start, end = match.start(), match.end()
        distance = min(
            (
                0
                if start <= position <= end
                else min(abs(start - position), abs(end - position))
            )
            for position in mention_positions
        )

        if distance < best_distance:
            best_distance = distance
            best_value = candidate

    return best_value


def _best_corroboration_count(aggregated: Dict[str, Dict[str, Any]]) -> int:
    """Returns the highest corroboration count across all candidates seen so far for one identifier type, or 0."""
    if not aggregated:
        return 0
    return max(info["corroborations"] for info in aggregated.values())


def _finalize_best_record(
    company_name: str,
    spec: _IdentifierSpec,
    aggregated: Dict[str, Dict[str, Any]],
) -> List[IdentifierRecord]:
    """
    Picks the best-corroborated candidate accumulated for one identifier
    type and builds its representative IdentifierRecord. Dicts preserve
    insertion order, so max() with a stable key naturally picks the
    first-encountered (i.e. highest-priority-site) candidate on a
    corroboration-count tie.
    """
    if not aggregated:
        return []

    best_key, best_info = max(aggregated.items(), key=lambda item: item[1]["corroborations"])

    corroboration_note = (
        f"seen near \"{company_name}\" on {best_info['corroborations']} independent registry "
        f"site(s), identified by {best_key}"
    )
    source_note = (
        f"Found via retriever.GST_SITE_PRIORITY fallback chain ({corroboration_note}) and "
        f"validated by structure"
        + (" and Luhn mod-36 checksum" if spec.identifier_type == "GST" else " (no checksum exists for CIN)")
        + ". checksum/structural validity confirms internal consistency and/or correct shape "
        "only -- it does NOT confirm the value is genuinely registered to this company."
    )

    return [
        IdentifierRecord(
            identifier_type=spec.identifier_type,
            value=best_info["value"],
            corroboration_key=best_key,
            corroborations=best_info["corroborations"],
            source_note=source_note,
        )
    ]


def _lookup_via_retrieval_chain(
    company_name: str,
    specs: List[_IdentifierSpec],
) -> Tuple[Dict[str, List[IdentifierRecord]], IdentifierTrace]:
    """
    Drives the identifier retrieval fallback chain for every spec in
    `specs` at once: pulls candidate pages from
    `retriever.iter_gst_documents()` one site at a time, in priority
    order, and for each page tries EVERY still-unsatisfied spec against
    it (no hardcoded site-to-identifier-type mapping -- a spec whose
    pattern doesn't match a given page's text simply contributes nothing,
    at negligible cost).

    The requested company name is parsed into a `ParsedEntity` once, up
    front -- not per document -- since it doesn't depend on the page being
    examined. Every document's mentions are then verified via
    entity_matching.find_entity_mentions(), which performs exact
    substantive-name matching with normalized legal form and symmetric
    boundary checks (see entity_matching.py) rather than fuzzy similarity.

    Within each site: pages are pulled one at a time (lazily -- see
    retriever.py), and as soon as one page contributes a candidate for ANY
    still-unsatisfied identifier, the rest of that site's pages are
    skipped. A spec that has already reached its corroboration threshold
    is no longer tried against subsequent pages.

    The whole chain stops once every spec in `specs` has reached its own
    `min_corroborating_sites`, or the site list is exhausted. Any spec
    that never reaches its threshold falls back to whatever's best
    corroborated (even a single site) once the chain ends.

    Also builds an IdentifierTrace alongside the existing print()
    statements (which are unchanged and still serve test_gst.py's
    interactive console debugging) -- a SiteCheck per page examined, plus
    the final corroboration counts and validation notes per identifier
    type, for Phase 4's debug mode.
    """
    requested = parse_entity_name(company_name)
    aggregated: Dict[str, Dict[str, Dict[str, Any]]] = {spec.identifier_type: {} for spec in specs}
    satisfied: set[str] = set()
    all_types = {spec.identifier_type for spec in specs}
    site_checks: List[SiteCheck] = []

    for site, site_documents in iter_gst_documents(company_name):
        print("\n" + "=" * 80)
        print(f"Checking site: {site} (still looking for: {sorted(all_types - satisfied)})")

        for document in site_documents:
            print(f"  - {document.title} ({document.url})")
            mention_positions = find_entity_mentions(document.cleaned_text, requested)
            if not mention_positions:
                print("    \u274c company not detected on this page")
                site_checks.append(
                    SiteCheck(
                        site=site, url=document.url, title=document.title, company_detected=False, candidates_found={}
                    )
                )
                continue

            contributed = False
            candidates_found: Dict[str, str] = {}
            for spec in specs:
                if spec.identifier_type in satisfied:
                    continue
                candidate = _extract_candidate(document.cleaned_text, mention_positions, spec, requested)
                if candidate is None:
                    continue

                key = spec.corroboration_key_fn(candidate)
                entry = aggregated[spec.identifier_type].setdefault(
                    key, {"value": candidate, "corroborations": 0, "sources": set()}
                )
                entry["corroborations"] += 1
                entry["sources"].add(document.url)
                contributed = True
                candidates_found[spec.identifier_type] = candidate
                print(f"    \u2705 {spec.identifier_type} candidate: {candidate}")

            site_checks.append(
                SiteCheck(
                    site=site,
                    url=document.url,
                    title=document.title,
                    company_detected=True,
                    candidates_found=candidates_found,
                )
            )

            if contributed:
                print(f"    skipping remaining pages for {site}")
                break
            else:
                print("    no verified candidate on this page")

        for spec in specs:
            if spec.identifier_type in satisfied:
                continue
            if _best_corroboration_count(aggregated[spec.identifier_type]) >= spec.min_corroborating_sites:
                satisfied.add(spec.identifier_type)
                print(f"\u2705 {spec.identifier_type} reached its corroboration threshold")

        if satisfied == all_types:
            print("\u2705 all identifiers satisfied, stopping the fallback chain early")
            break

    finalized = {
        spec.identifier_type: _finalize_best_record(company_name, spec, aggregated[spec.identifier_type])
        for spec in specs
    }

    corroboration_counts = {
        spec.identifier_type: _best_corroboration_count(aggregated[spec.identifier_type]) for spec in specs
    }
    validation_notes = {
        spec.identifier_type: (
            finalized[spec.identifier_type][0].source_note if finalized[spec.identifier_type] else "no candidate found"
        )
        for spec in specs
    }

    return finalized, IdentifierTrace(
        site_checks=site_checks,
        corroboration_counts=corroboration_counts,
        validation_notes=validation_notes,
    )


def get_company_identifiers(company_name: str) -> Tuple[Dict[str, List[IdentifierRecord] | str], IdentifierTrace]:
    """
    The main entry point: retrieves and extracts BOTH GST and CIN for
    `company_name` in a single retrieval pass, returning
    `({"GST": [...] | "GST not verified", "CIN": [...] | "CIN not verified"}, trace)`.

    Checks each identifier's configured API override first
    (GST_API_URL / CIN_API_URL); whichever identifiers aren't resolved
    that way go through the shared retrieval fallback chain together, so
    a company needing both still costs one retrieval pass, not two.

    The returned IdentifierTrace is empty (no site_checks) whenever both
    identifiers resolved via a configured API, or the company name was
    empty -- there is genuinely nothing to report in either case, since
    the fallback chain never ran.
    """
    empty_trace = IdentifierTrace(site_checks=[], corroboration_counts={}, validation_notes={})

    if not company_name or not company_name.strip():
        return {spec.identifier_type: f"{spec.identifier_type} not verified" for spec in _ALL_SPECS}, empty_trace

    results: Dict[str, List[IdentifierRecord] | str] = {}
    remaining_specs: List[_IdentifierSpec] = []

    for spec in _ALL_SPECS:
        api_records = _lookup_via_configured_api(company_name, spec)
        if api_records:
            results[spec.identifier_type] = api_records
        else:
            remaining_specs.append(spec)

    trace = empty_trace
    if remaining_specs:
        chain_results, trace = _lookup_via_retrieval_chain(company_name, remaining_specs)
        for identifier_type, records in chain_results.items():
            results[identifier_type] = records if records else f"{identifier_type} not verified"

    return results, trace


def get_gst_details(company_name: str) -> List[GSTRecord] | str:
    """
    Backward-compatible GST-only entry point. NOTE: if you also need CIN
    for the same company, prefer calling `get_company_identifiers()`
    directly and reading both keys -- calling this alongside
    `get_cin_details()` separately runs the retrieval chain twice.
    """
    identifiers, _trace = get_company_identifiers(company_name)
    result = identifiers.get("GST", "GST not verified")
    if isinstance(result, str):
        return result
    return [_to_gst_record(record) for record in result]


def get_cin_details(company_name: str) -> List[CINRecord] | str:
    """
    Backward-compatible CIN-only entry point. NOTE: if you also need GST
    for the same company, prefer calling `get_company_identifiers()`
    directly and reading both keys -- calling this alongside
    `get_gst_details()` separately runs the retrieval chain twice.
    """
    identifiers, _trace = get_company_identifiers(company_name)
    result = identifiers.get("CIN", "CIN not verified")
    if isinstance(result, str):
        return result
    return [_to_cin_record(record) for record in result]
