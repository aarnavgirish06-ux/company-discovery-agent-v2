"""
test_entity_matching.py

Tests for the new deterministic entity-matching layer. Two levels:

1. Direct unit tests of entity_matching.py's public functions.
2. Regression tests reproducing the three reported bugs end-to-end,
   by monkeypatching identifier_lookup.iter_gst_documents with fixture
   Document objects (no real network call -- retriever.py itself is
   untouched by this change, so it doesn't need to be exercised here).

Run with: python3 test_entity_matching.py
"""

from __future__ import annotations

import identifier_lookup
from entity_matching import find_entity_mentions, pan_category_matches, parse_entity_name
from identifier_lookup import _gst_checksum_char, get_company_identifiers
from retriever import Document

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


def make_gstin(state_code: str, pan: str, entity_num: str = "1") -> str:
    """Builds a structurally valid, checksum-correct GSTIN for a given state code + PAN."""
    assert len(state_code) == 2 and state_code.isdigit()
    assert len(pan) == 10
    prefix14 = f"{state_code}{pan}{entity_num}Z"
    assert len(prefix14) == 14
    checksum = _gst_checksum_char(prefix14)
    gstin = prefix14 + checksum
    assert len(gstin) == 15
    return gstin


# =====================================================================
# 1. Direct unit tests of entity_matching.py
# =====================================================================

print("=" * 70)
print("Direct entity_matching.py unit tests")
print("=" * 70)

# --- parse_entity_name -------------------------------------------------

pe = parse_entity_name("Infosys Limited")
check("parse: 'Infosys Limited' substantive", pe.substantive_tokens == ("infosys",), pe.substantive_tokens)
check("parse: 'Infosys Limited' legal_form", pe.legal_form == ("limited",), pe.legal_form)

pe2 = parse_entity_name("Shree Samarth Packaging Pvt Ltd")
check(
    "parse: 'Shree Samarth Packaging Pvt Ltd' substantive",
    pe2.substantive_tokens == ("shree", "samarth", "packaging"),
    pe2.substantive_tokens,
)
check("parse: 'Shree Samarth Packaging Pvt Ltd' legal_form", pe2.legal_form == ("private", "limited"), pe2.legal_form)

pe3 = parse_entity_name("Shree Samarth Packaging Private Limited")
check(
    "parse: 'Pvt Ltd' and 'Private Limited' canonicalize identically",
    pe2.legal_form == pe3.legal_form and pe2.substantive_tokens == pe3.substantive_tokens,
)

pe4 = parse_entity_name("Infosys")
check("parse: bare 'Infosys' has no legal_form", pe4.legal_form is None)

# --- find_entity_mentions: Example 1 (Infosys) --------------------------

query_entity = parse_entity_name("Infosys Limited")

page_correct = "Infosys Limited is an Indian IT company. Infosys Limited was founded in 1981."
positions_correct = find_entity_mentions(page_correct, query_entity)
check("Example 1: 'Infosys Limited' page matches", len(positions_correct) > 0, positions_correct)

page_wrong = "Infosys Consulting India Limited provides consulting services to clients."
positions_wrong = find_entity_mentions(page_wrong, query_entity)
check(
    "Example 1: 'Infosys Consulting India Limited' page must NOT match 'Infosys Limited'",
    len(positions_wrong) == 0,
    positions_wrong,
)

# --- find_entity_mentions: Example 2 (Shree Samarth Packaging) ---------

query_entity2 = parse_entity_name("Shree Samarth Packaging Pvt Ltd")

page2_correct = "Shree Samarth Packaging Private Limited manufactures corrugated boxes."
positions2_correct = find_entity_mentions(page2_correct, query_entity2)
check("Example 2: exact-entity page matches (legal form normalized)", len(positions2_correct) > 0, positions2_correct)

page2_wrong = "Shree Swami Samarth Packaging Pvt Ltd is located in Pune."
positions2_wrong = find_entity_mentions(page2_wrong, query_entity2)
check(
    "Example 2: 'Shree Swami Samarth Packaging Pvt Ltd' must NOT match 'Shree Samarth Packaging Pvt Ltd'",
    len(positions2_wrong) == 0,
    positions2_wrong,
)

# --- symmetric boundary check (leading side) ----------------------------

query_entity5 = parse_entity_name("Samarth Packaging")
page5 = "Shree Samarth Packaging Private Limited is a manufacturer."
positions5 = find_entity_mentions(page5, query_entity5)
check(
    "Leading-boundary check: 'Samarth Packaging' must NOT match inside 'Shree Samarth Packaging ...'",
    len(positions5) == 0,
    positions5,
)

# --- legal form mismatch is a rejection, absence of legal form is not ---

query_llp = parse_entity_name("Acme Traders LLP")
page_llp_match = "Acme Traders LLP is a trading firm based in Mumbai."
check("Legal form match: LLP page matches LLP query", len(find_entity_mentions(page_llp_match, query_llp)) > 0)

page_llp_mismatch = "Acme Traders Private Limited is a trading company."
check(
    "Legal form mismatch: 'Acme Traders Private Limited' must NOT match 'Acme Traders LLP' query",
    len(find_entity_mentions(page_llp_mismatch, query_llp)) == 0,
)

page_llp_unstated = "Acme Traders is listed on the state registry portal."
check(
    "Legal form unstated on page: treated as unconstrained, still matches",
    len(find_entity_mentions(page_llp_unstated, query_llp)) > 0,
)

# --- pan_category_matches: Example 3 -------------------------------------

requested_company = parse_entity_name("Shree Samarth Packaging Pvt Ltd")  # -> legal_form ("private","limited") -> "C"
individual_gstin = make_gstin("27", "ABBPO1025B")  # PAN 4th char 'P' = Individual
check(
    "Example 3: individual-category GSTIN rejected for a Private Limited request",
    pan_category_matches(individual_gstin, requested_company) is False,
    individual_gstin,
)

company_gstin = make_gstin("27", "ABBCO1025B")  # PAN 4th char 'C' = Company
check(
    "Company-category GSTIN accepted for a Private Limited request",
    pan_category_matches(company_gstin, requested_company) is True,
    company_gstin,
)

requested_llp = parse_entity_name("Acme Traders LLP")
check(
    "PAN-category check skipped (not rejected) for an unmapped legal form like LLP",
    pan_category_matches(individual_gstin, requested_llp) is True,
)

requested_no_form = parse_entity_name("Infosys")
check(
    "PAN-category check skipped when query specifies no legal form at all",
    pan_category_matches(individual_gstin, requested_no_form) is True,
)


# =====================================================================
# 2. End-to-end regression tests via identifier_lookup.get_company_identifiers
#    (iter_gst_documents monkeypatched with fixture pages; no real network)
# =====================================================================

print()
print("=" * 70)
print("End-to-end regression tests (fixture pages, no live network)")
print("=" * 70)


def fixture_chain(pages_by_site):
    """Builds a fake iter_gst_documents() generator from {site: [Document, ...]}."""
    def _iter(company_name):
        for site, docs in pages_by_site.items():
            yield site, iter(docs)
    return _iter


# --- Example 1: Infosys Limited must not pick up Infosys Consulting India Limited's CIN

infosys_cin = "L85110KA1981PLC013115"  # Infosys Limited's real-shaped CIN (structurally valid, illustrative)
infosys_consulting_cin = "U74140MH2000PLC128105"  # a different, structurally valid CIN

pages_example1 = {
    "thecompanycheck.com": [
        Document(
            url="https://thecompanycheck.com/company/infosys-consulting-india-limited",
            title="Infosys Consulting India Limited",
            cleaned_text=(
                f"Infosys Consulting India Limited CIN {infosys_consulting_cin}. "
                "Infosys Consulting India Limited offers consulting services."
            ),
        ),
        Document(
            url="https://thecompanycheck.com/company/infosys-limited",
            title="Infosys Limited",
            cleaned_text=(
                f"Infosys Limited CIN {infosys_cin}. Infosys Limited is headquartered in Bengaluru."
            ),
        ),
    ],
    "zaubacorp.com": [
        Document(
            url="https://zaubacorp.com/company/infosys-limited",
            title="Infosys Limited - ZaubaCorp",
            cleaned_text=f"Infosys Limited, CIN: {infosys_cin}, is a public company.",
        ),
    ],
}

identifier_lookup.iter_gst_documents = fixture_chain(pages_example1)
result1, trace1 = get_company_identifiers("Infosys Limited")
cin_result1 = result1["CIN"]
cin_value1 = None if isinstance(cin_result1, str) else cin_result1[0].value
check(
    "Example 1 (end-to-end): correct CIN (Infosys Limited's) returned, not the consulting entity's",
    cin_value1 == infosys_cin,
    f"got {cin_value1!r}, expected {infosys_cin!r}",
)
check(
    "Example 1 (end-to-end) trace: at least one site_check recorded",
    len(trace1.site_checks) > 0,
    len(trace1.site_checks),
)
check(
    "Example 1 (end-to-end) trace: the consulting entity's page is recorded as NOT detected (correctly rejected)",
    any(
        sc.url == "https://thecompanycheck.com/company/infosys-consulting-india-limited" and not sc.company_detected
        for sc in trace1.site_checks
    ),
    trace1.site_checks,
)
check(
    "Example 1 (end-to-end) trace: the correct Infosys Limited page is recorded as detected with a CIN candidate",
    any(
        sc.url == "https://thecompanycheck.com/company/infosys-limited"
        and sc.company_detected
        and sc.candidates_found.get("CIN") == infosys_cin
        for sc in trace1.site_checks
    ),
    trace1.site_checks,
)

# --- Example 2 + 3: Shree Samarth Packaging Pvt Ltd must not match Shree Swami Samarth Packaging Pvt Ltd,
#     and the individual-PAN GSTIN found on KnowYourGST must be rejected.

correct_company_gstin = make_gstin("27", "ABCCE1234F")  # PAN[3]='C' -> Company, matches request
wrong_entity_gstin = make_gstin("27", "PQRSD5678G")      # belongs to the OTHER (Swami Samarth) entity
individual_pan_gstin = make_gstin("27", "ABBPO1025B")     # PAN 4th char 'P' -> Individual, from Example 3

pages_example2 = {
    "thecompanycheck.com": [
        Document(
            url="https://thecompanycheck.com/company/shree-swami-samarth-packaging-pvt-ltd",
            title="Shree Swami Samarth Packaging Pvt Ltd",
            cleaned_text=(
                f"Shree Swami Samarth Packaging Pvt Ltd GSTIN {wrong_entity_gstin}. "
                "Shree Swami Samarth Packaging Pvt Ltd is based in Pune."
            ),
        ),
    ],
    "zaubacorp.com": [
        Document(
            url="https://zaubacorp.com/company/shree-samarth-packaging-pvt-ltd",
            title="Shree Samarth Packaging Pvt Ltd",
            cleaned_text=(
                f"Shree Samarth Packaging Pvt Ltd GSTIN {correct_company_gstin}. "
                "Shree Samarth Packaging Pvt Ltd manufactures corrugated boxes."
            ),
        ),
    ],
    "knowyourgst.com": [
        Document(
            url="https://knowyourgst.com/company/shree-samarth-packaging",
            title="Shree Samarth Packaging - GST Search",
            cleaned_text=(
                f"Shree Samarth Packaging Pvt Ltd GSTIN {individual_pan_gstin}. "
                "Shree Samarth Packaging Pvt Ltd registered taxpayer details."
            ),
        ),
    ],
}

identifier_lookup.iter_gst_documents = fixture_chain(pages_example2)
result2, trace2 = get_company_identifiers("Shree Samarth Packaging Pvt Ltd")
gst_result2 = result2["GST"]
gst_values2 = [] if isinstance(gst_result2, str) else [r.value for r in gst_result2]

check(
    "Example 2 (end-to-end): wrong entity's GSTIN never accepted",
    wrong_entity_gstin not in gst_values2,
    gst_values2,
)
check(
    "Example 3 (end-to-end): individual-PAN GSTIN never accepted for a Pvt Ltd request",
    individual_pan_gstin not in gst_values2,
    gst_values2,
)
check(
    "Example 2/3 (end-to-end): correct company-category GSTIN is the one returned",
    gst_values2 == [correct_company_gstin],
    gst_values2,
)

check(
    "Example 2/3 (end-to-end) trace: corroboration_counts recorded for GST",
    trace2.corroboration_counts.get("GST", 0) >= 1,
    trace2.corroboration_counts,
)
check(
    "Example 2/3 (end-to-end) trace: validation_notes mentions checksum for GST",
    "checksum" in trace2.validation_notes.get("GST", "").lower(),
    trace2.validation_notes,
)


# =====================================================================

print()
print("=" * 70)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S):")
    for f in _FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
