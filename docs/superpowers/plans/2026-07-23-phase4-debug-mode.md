# Phase 4: Debug Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface real, engineered trace data for all six debug sections (intent classification, company discovery, retrieval, identifier lookup, evidence extraction, final answer) without changing any existing decision the assistant makes or removing anything currently visible in the debug view.

**Architecture:** `retriever.py`, `identifier_lookup.py`, and `evidence_extractor.py` each gain a frozen trace dataclass and change their return type to `(result, trace)`. `discovery.py` collects these per company onto `CompanyResult`/`RejectedCompany`. `conversation.py` captures intent-classification and QA data it already computes (currently discarded) onto `ChatTurn`. `ui.py` renders all six sections, extending the existing per-company debug expander and adding a new turn-level debug block.

**Tech Stack:** No new dependencies — pure Python dataclasses and existing Streamlit/HTML rendering patterns.

## Global Constraints

- Tracing is unconditional (always collected), never gated behind a flag — the existing "Show Debug Information" toggle controls *display* only, exactly like `constraint_evaluation` already works.
- `retrieve_for_evidence()`, `retrieve_for_question()`, `get_company_identifiers()`, and `extract()` all change their return type to a 2-tuple `(result, trace)`. Every caller (including existing test files that call these directly) must be updated to unpack the tuple in the same task that changes the signature, so every task leaves the full test suite green.
- Nothing currently visible in the debug view changes in content or appearance — `_render_debug_html()`'s existing sections and `_render_rejected_card_html()` stay as-is; new sections are appended after them.
- No `print()` statements in `identifier_lookup.py` are removed — the new `IdentifierTrace` is built *alongside* them, not instead of them (they still serve `test_gst.py`'s interactive console debugging).
- Every new dataclass field (on `CompanyResult`, `RejectedCompany`, `ChatTurn`) is additive with a default — no existing field is renamed, removed, or reshaped.
- Testing stays in the existing script + `check(label, condition)` harness convention — no pytest.
- Full design rationale lives in `docs/superpowers/specs/2026-07-23-phase4-debug-mode-design.md`.

---

### Task 1: `retriever.py` — add `RetrievalTrace`, change return types

**Files:**
- Modify: `retriever.py`, `discovery.py`, `qa.py`
- Test: `test_retriever.py` (extend), `test_discovery_structured_output.py` (extend), `test_qa.py` (extend)

**Interfaces:**
- Produces: `RetrievalAttempt(url: str, included: bool, reason: str)`, `RetrievalTrace(queries: List[str], attempts: List[RetrievalAttempt])`. `retrieve_for_evidence(company_name) -> Tuple[List[Document], RetrievalTrace]`, `retrieve_for_question(company_name, question) -> Tuple[List[Document], RetrievalTrace]`.
- `discovery.py`'s `discover()` and `qa.py`'s `answer_question()` unpack the new tuple but do not yet store the trace anywhere (that's Task 4/5) — the unpacked trace variable is prefixed with `_` to signal "intentionally unused for now."

- [ ] **Step 1: Write the failing test**

In `test_retriever.py`, change:
```python
_captured_queries.clear()
result = retrieve_for_question("Acme Forgings Private Limited", "Who are the directors?")

check("retrieve_for_question returns an empty list when search yields no URLs", result == [])
check(
    "retrieve_for_question issues a query combining the company name and the question",
    any(
        "Acme Forgings Private Limited" in q and "Who are the directors?" in q
        for q in _captured_queries
    ),
    _captured_queries,
)
check(
    "retrieve_for_question also issues a generic company-name-only query",
    any(q == '"Acme Forgings Private Limited"' for q in _captured_queries),
    _captured_queries,
)
check("retrieve_for_question issues exactly 2 queries", len(_captured_queries) == 2, _captured_queries)


print()
print("=" * 70)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S):")
    for f in _FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
```
to:
```python
_captured_queries.clear()
result, trace = retrieve_for_question("Acme Forgings Private Limited", "Who are the directors?")

check("retrieve_for_question returns an empty list when search yields no URLs", result == [])
check(
    "retrieve_for_question issues a query combining the company name and the question",
    any(
        "Acme Forgings Private Limited" in q and "Who are the directors?" in q
        for q in _captured_queries
    ),
    _captured_queries,
)
check(
    "retrieve_for_question also issues a generic company-name-only query",
    any(q == '"Acme Forgings Private Limited"' for q in _captured_queries),
    _captured_queries,
)
check("retrieve_for_question issues exactly 2 queries", len(_captured_queries) == 2, _captured_queries)
check("retrieve_for_question's trace records both queries issued", trace.queries == _captured_queries, trace.queries)
check("retrieve_for_question's trace has no attempts when no URLs were found", trace.attempts == [])

# ---------------------------------------------------------------------
# retrieve_for_question: trace records download outcomes (included / failed / capped)
# ---------------------------------------------------------------------


def _fake_search_with_urls(query, max_results=10):
    return [f"https://example.com/page{i}" for i in range(12)]


def _fake_download_page(url, timeout=10):
    if url == "https://example.com/page0":
        return "<html><title>Page 0</title><body>Content</body></html>"
    return None  # every other URL "fails to download"


retriever._search_company_pages = _fake_search_with_urls
retriever._download_page = _fake_download_page

result2, trace2 = retrieve_for_question("Acme Forgings Private Limited", "Who are the directors?")

check("retrieve_for_question: exactly one page downloaded successfully", len(result2) == 1, len(result2))
check(
    "retrieve_for_question's trace marks the successful page as included",
    any(
        a.url == "https://example.com/page0" and a.included and a.reason == "downloaded successfully"
        for a in trace2.attempts
    ),
)
check(
    "retrieve_for_question's trace marks a failed download as not included",
    any(
        a.url == "https://example.com/page1" and not a.included and a.reason == "download failed"
        for a in trace2.attempts
    ),
)
check(
    "retrieve_for_question's trace marks URLs beyond the cap as excluded for that reason",
    any(not a.included and a.reason == "exceeded max_documents cap" for a in trace2.attempts),
)
check("retrieve_for_question's trace has one attempt per discovered URL", len(trace2.attempts) == 12, len(trace2.attempts))


print()
print("=" * 70)
if _FAILURES:
    print(f"{len(_FAILURES)} FAILURE(S):")
    for f in _FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL TESTS PASSED")
```

Also, in `test_discovery_structured_output.py`, change the import block:
```python
from types import SimpleNamespace

import discovery
from identifier_lookup import IdentifierRecord
from llm_schemas import LLMCompanyEntry, LLMConstraintEvaluation, LLMDiscoveryResponse
from retriever import Document
```
to:
```python
from types import SimpleNamespace

import discovery
import retriever
from identifier_lookup import IdentifierRecord
from llm_schemas import LLMCompanyEntry, LLMConstraintEvaluation, LLMDiscoveryResponse
from retriever import Document
```
and change:
```python
    discovery.retriever.retrieve_for_evidence = lambda name: [_FIXTURE_DOCUMENT]
```
to:
```python
    discovery.retriever.retrieve_for_evidence = lambda name: (
        [_FIXTURE_DOCUMENT],
        retriever.RetrievalTrace(queries=[], attempts=[]),
    )
```

Also, in `test_qa.py`, change every stub of `qa.retrieve_for_question` to return a 2-tuple. Change:
```python
qa.retrieve_for_question = lambda company_name, question: [new_document]
```
to:
```python
qa.retrieve_for_question = lambda company_name, question: ([new_document], None)
```
(both occurrences of `lambda company_name, question: []` become `lambda company_name, question: ([], None)` -- there are two such occurrences, in the "insufficient on both tries" and "provider failure" sections).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 test_retriever.py`
Expected: `ImportError` or `AttributeError` referencing `RetrievalTrace`, or a tuple-unpacking `ValueError` (too many/few values to unpack) since `retrieve_for_question` still returns a bare list.

Run: `python3 test_discovery_structured_output.py` and `python3 test_qa.py`
Expected: failures, since the stubs now return tuples but the real functions haven't changed yet (or vice versa) -- either way, a clear failure, not a pass.

- [ ] **Step 3: Update `retriever.py`**

Change the typing import:
```python
from typing import Iterator, List, Sequence, Tuple
```
This line is already present unchanged (no edit needed here -- `Tuple` is already imported).

Add after the `Document` dataclass:
```python
@dataclass(frozen=True)
class Document:
    """A single downloaded, cleaned webpage, ready for downstream extractors."""
    url: str
    title: str
    cleaned_text: str
```
becomes:
```python
@dataclass(frozen=True)
class Document:
    """A single downloaded, cleaned webpage, ready for downstream extractors."""
    url: str
    title: str
    cleaned_text: str


@dataclass(frozen=True)
class RetrievalAttempt:
    """One page candidate considered during a retrieval call."""
    url: str
    included: bool  # True if downloaded and returned as a Document
    reason: str  # e.g. "downloaded successfully", "download failed", "exceeded max_documents cap"


@dataclass(frozen=True)
class RetrievalTrace:
    """Structured, engineered record of what one retrieval call actually did."""
    queries: List[str]
    attempts: List[RetrievalAttempt]
```

Replace the whole `_retrieve()` function:
```python
def _retrieve(
    queries: Sequence[str],
    preferred_domains: Sequence[str],
    max_documents: int,
) -> List[Document]:
    """
    Generic eager retrieval primitive. Runs each of `queries` as an
    independent search, merges and deduplicates their result URLs (earlier
    queries' URLs take precedence in ordering), prioritizes
    `preferred_domains`, downloads up to `max_documents` pages, and
    returns each as a cleaned `Document`. Pages that fail to download are
    skipped gracefully.

    Used by `retrieve_for_evidence()`. NOT used by the identifier-lookup
    path -- see module docstring for why.
    """
    urls = _prioritize_urls(
        _merge_and_dedupe_urls(*[_search_company_pages(query) for query in queries]),
        preferred_domains,
    )

    documents: List[Document] = []
    for url in urls[:max_documents]:
        html = _download_page(url)
        if not html:
            continue
        title, cleaned_text = _parse_page(html)
        documents.append(Document(url=url, title=title, cleaned_text=cleaned_text))

    return documents
```
with:
```python
def _retrieve(
    queries: Sequence[str],
    preferred_domains: Sequence[str],
    max_documents: int,
) -> Tuple[List[Document], RetrievalTrace]:
    """
    Generic eager retrieval primitive. Runs each of `queries` as an
    independent search, merges and deduplicates their result URLs (earlier
    queries' URLs take precedence in ordering), prioritizes
    `preferred_domains`, downloads up to `max_documents` pages, and
    returns each as a cleaned `Document`. Pages that fail to download are
    skipped gracefully.

    Also builds a RetrievalTrace recording every query issued and, for
    every URL found, whether it was included and why not if not -- either
    it exceeded `max_documents` (never attempted) or its download failed.

    Used by `retrieve_for_evidence()` and `retrieve_for_question()`. NOT
    used by the identifier-lookup path -- see module docstring for why.
    """
    urls = _prioritize_urls(
        _merge_and_dedupe_urls(*[_search_company_pages(query) for query in queries]),
        preferred_domains,
    )

    documents: List[Document] = []
    attempts: List[RetrievalAttempt] = []
    for url in urls[:max_documents]:
        html = _download_page(url)
        if not html:
            attempts.append(RetrievalAttempt(url=url, included=False, reason="download failed"))
            continue
        title, cleaned_text = _parse_page(html)
        documents.append(Document(url=url, title=title, cleaned_text=cleaned_text))
        attempts.append(RetrievalAttempt(url=url, included=True, reason="downloaded successfully"))

    for url in urls[max_documents:]:
        attempts.append(RetrievalAttempt(url=url, included=False, reason="exceeded max_documents cap"))

    return documents, RetrievalTrace(queries=list(queries), attempts=attempts)
```

Replace `retrieve_for_evidence()`:
```python
def retrieve_for_evidence(company_name: str) -> List[Document]:
    """
    Retrieves general pages about `company_name` -- official site,
    directory listings, "about us" / product pages -- useful for
    evidence extraction. No registry-domain bias. Intended consumer:
    evidence_extractor.extract().
    """
    queries = [
        f'"{company_name}"',
        f'"{company_name}" about us products',
    ]
    return _retrieve(queries, _EVIDENCE_PREFERRED_DOMAINS, _MAX_DOCUMENTS_PER_COMPANY)
```
with:
```python
def retrieve_for_evidence(company_name: str) -> Tuple[List[Document], RetrievalTrace]:
    """
    Retrieves general pages about `company_name` -- official site,
    directory listings, "about us" / product pages -- useful for
    evidence extraction. No registry-domain bias. Intended consumer:
    evidence_extractor.extract().

    Returns (documents, trace) -- the trace is an engineered record of
    which queries were issued and which candidate pages were included or
    discarded and why, for Phase 4's debug mode.
    """
    queries = [
        f'"{company_name}"',
        f'"{company_name}" about us products',
    ]
    return _retrieve(queries, _EVIDENCE_PREFERRED_DOMAINS, _MAX_DOCUMENTS_PER_COMPANY)
```

Replace `retrieve_for_question()`:
```python
def retrieve_for_question(company_name: str, question: str) -> List[Document]:
    """
    Retrieves pages likely to answer a specific question about
    `company_name` -- unlike retrieve_for_evidence()'s generic "about us
    products" queries, this combines the company name with the question's
    own terms, since a question like "who are the directors" needs a more
    targeted search than a generic company-name query would surface.
    Intended consumer: qa.py, when existing evidence/documents aren't
    enough to answer a question.
    """
    queries = [
        f'"{company_name}" {question}',
        f'"{company_name}"',
    ]
    return _retrieve(queries, _EVIDENCE_PREFERRED_DOMAINS, _MAX_DOCUMENTS_PER_COMPANY)
```
with:
```python
def retrieve_for_question(company_name: str, question: str) -> Tuple[List[Document], RetrievalTrace]:
    """
    Retrieves pages likely to answer a specific question about
    `company_name` -- unlike retrieve_for_evidence()'s generic "about us
    products" queries, this combines the company name with the question's
    own terms, since a question like "who are the directors" needs a more
    targeted search than a generic company-name query would surface.
    Intended consumer: qa.py, when existing evidence/documents aren't
    enough to answer a question.

    Returns (documents, trace) -- see retrieve_for_evidence()'s docstring
    for what the trace records.
    """
    queries = [
        f'"{company_name}" {question}',
        f'"{company_name}"',
    ]
    return _retrieve(queries, _EVIDENCE_PREFERRED_DOMAINS, _MAX_DOCUMENTS_PER_COMPANY)
```

- [ ] **Step 4: Update `discovery.py`'s call site**

Change:
```python
        # STEP 3b (verified): proceed exactly as before -- retrieve, then
        # extract, evidence for this company.
        evidence_documents = retriever.retrieve_for_evidence(entry["company_name"])
        evidence = extract_evidence(entry["company_name"], query, entry["reason"], evidence_documents)
```
to:
```python
        # STEP 3b (verified): proceed exactly as before -- retrieve, then
        # extract, evidence for this company.
        evidence_documents, _evidence_retrieval_trace = retriever.retrieve_for_evidence(entry["company_name"])
        evidence = extract_evidence(entry["company_name"], query, entry["reason"], evidence_documents)
```

- [ ] **Step 5: Update `qa.py`'s call site**

Change:
```python
    try:
        new_documents = retrieve_for_question(company.company_name, question)
    except Exception:
        new_documents = []
```
to:
```python
    try:
        new_documents, _retrieval_trace = retrieve_for_question(company.company_name, question)
    except Exception:
        new_documents = []
```

- [ ] **Step 6: Run all three tests to verify they pass**

Run: `python3 test_retriever.py && python3 test_discovery_structured_output.py && python3 test_qa.py`
Expected: all three print `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add retriever.py discovery.py qa.py test_retriever.py test_discovery_structured_output.py test_qa.py
git commit -m "Add RetrievalTrace; retriever.py retrieval functions now return (documents, trace)"
```

---

### Task 2: `identifier_lookup.py` — add `IdentifierTrace`, change return type

**Files:**
- Modify: `identifier_lookup.py`, `discovery.py`
- Test: `test_entity_matching.py` (extend), `test_discovery_structured_output.py` (extend -- its `get_company_identifiers` stub must also return a tuple, or `discover()`'s new unpacking will raise `AttributeError` once this task's `discovery.py` edit lands)

**Interfaces:**
- Produces: `SiteCheck(site: str, url: str, title: str, company_detected: bool, candidates_found: Dict[str, str])`, `IdentifierTrace(site_checks: List[SiteCheck], corroboration_counts: Dict[str, int], validation_notes: Dict[str, str])`. `get_company_identifiers(company_name) -> Tuple[Dict[str, List[IdentifierRecord] | str], IdentifierTrace]`.
- `discovery.py`'s `discover()` unpacks the new tuple but does not yet store the trace (that's Task 4) — prefixed `_identifier_trace`.

- [ ] **Step 1: Write the failing test**

In `test_entity_matching.py`, change:
```python
identifier_lookup.iter_gst_documents = fixture_chain(pages_example1)
result1 = get_company_identifiers("Infosys Limited")
cin_result1 = result1["CIN"]
cin_value1 = None if isinstance(cin_result1, str) else cin_result1[0].value
check(
    "Example 1 (end-to-end): correct CIN (Infosys Limited's) returned, not the consulting entity's",
    cin_value1 == infosys_cin,
    f"got {cin_value1!r}, expected {infosys_cin!r}",
)
```
to:
```python
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
```

Then change:
```python
identifier_lookup.iter_gst_documents = fixture_chain(pages_example2)
result2 = get_company_identifiers("Shree Samarth Packaging Pvt Ltd")
gst_result2 = result2["GST"]
gst_values2 = [] if isinstance(gst_result2, str) else [r.value for r in gst_result2]
```
to:
```python
identifier_lookup.iter_gst_documents = fixture_chain(pages_example2)
result2, trace2 = get_company_identifiers("Shree Samarth Packaging Pvt Ltd")
gst_result2 = result2["GST"]
gst_values2 = [] if isinstance(gst_result2, str) else [r.value for r in gst_result2]
```

Then, immediately before the file's final block (the one starting with `print()` / `print("=" * 70)` / the `_FAILURES` check at the very end of the file), insert:
```python
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

```

Also, in `test_discovery_structured_output.py`, change the import line:
```python
from identifier_lookup import IdentifierRecord
```
to:
```python
from identifier_lookup import IdentifierRecord, IdentifierTrace
```
and change:
```python
    discovery.get_company_identifiers = lambda name: {
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
    }
```
to:
```python
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
        IdentifierTrace(site_checks=[], corroboration_counts={"GST": 2}, validation_notes={"GST": "stub"}),
    )
```
(This stub lives inside `_stub_downstream_pipeline()` -- the same function Task 1 already touched for the `retrieve_for_evidence` stub.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 test_entity_matching.py`
Expected: `ValueError: too many values to unpack` (or similar), since `get_company_identifiers` still returns a bare dict.

Run: `python3 test_discovery_structured_output.py`
Expected: passes for now (nothing in `discovery.py` has changed yet in this task) -- this run is just a baseline; it will be re-run in Step 5 after `discovery.py`'s call site changes to confirm the updated stub still works.

- [ ] **Step 3: Update `identifier_lookup.py`**

Change the typing import:
```python
from typing import Any, Callable, Dict, List
```
to:
```python
from typing import Any, Callable, Dict, List, Tuple
```

Add after the `IdentifierRecord` dataclass:
```python
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
```
becomes:
```python
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
```

Replace the whole `_lookup_via_retrieval_chain()` function:
```python
def _lookup_via_retrieval_chain(
    company_name: str,
    specs: List[_IdentifierSpec],
) -> Dict[str, List[IdentifierRecord]]:
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
    """
    requested = parse_entity_name(company_name)
    aggregated: Dict[str, Dict[str, Dict[str, Any]]] = {spec.identifier_type: {} for spec in specs}
    satisfied: set[str] = set()
    all_types = {spec.identifier_type for spec in specs}

    for site, site_documents in iter_gst_documents(company_name):
        print("\n" + "=" * 80)
        print(f"Checking site: {site} (still looking for: {sorted(all_types - satisfied)})")

        for document in site_documents:
            print(f"  - {document.title} ({document.url})")
            mention_positions = find_entity_mentions(document.cleaned_text, requested)
            if not mention_positions:
                print("    ❌ company not detected on this page")
                continue

            contributed = False
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
                print(f"    ✅ {spec.identifier_type} candidate: {candidate}")

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
                print(f"✅ {spec.identifier_type} reached its corroboration threshold")

        if satisfied == all_types:
            print("✅ all identifiers satisfied, stopping the fallback chain early")
            break

    return {
        spec.identifier_type: _finalize_best_record(company_name, spec, aggregated[spec.identifier_type])
        for spec in specs
    }
```
with:
```python
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
                print("    ❌ company not detected on this page")
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
                print(f"    ✅ {spec.identifier_type} candidate: {candidate}")

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
                print(f"✅ {spec.identifier_type} reached its corroboration threshold")

        if satisfied == all_types:
            print("✅ all identifiers satisfied, stopping the fallback chain early")
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
```

Replace `get_company_identifiers()`:
```python
def get_company_identifiers(company_name: str) -> Dict[str, List[IdentifierRecord] | str]:
    """
    The main entry point: retrieves and extracts BOTH GST and CIN for
    `company_name` in a single retrieval pass, returning
    `{"GST": [...] | "GST not verified", "CIN": [...] | "CIN not verified"}`.

    Checks each identifier's configured API override first
    (GST_API_URL / CIN_API_URL); whichever identifiers aren't resolved
    that way go through the shared retrieval fallback chain together, so
    a company needing both still costs one retrieval pass, not two.
    """
    if not company_name or not company_name.strip():
        return {spec.identifier_type: f"{spec.identifier_type} not verified" for spec in _ALL_SPECS}

    results: Dict[str, List[IdentifierRecord] | str] = {}
    remaining_specs: List[_IdentifierSpec] = []

    for spec in _ALL_SPECS:
        api_records = _lookup_via_configured_api(company_name, spec)
        if api_records:
            results[spec.identifier_type] = api_records
        else:
            remaining_specs.append(spec)

    if remaining_specs:
        chain_results = _lookup_via_retrieval_chain(company_name, remaining_specs)
        for identifier_type, records in chain_results.items():
            results[identifier_type] = records if records else f"{identifier_type} not verified"

    return results
```
with:
```python
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
```

Update `get_gst_details()` and `get_cin_details()`:
```python
def get_gst_details(company_name: str) -> List[GSTRecord] | str:
    """
    Backward-compatible GST-only entry point. NOTE: if you also need CIN
    for the same company, prefer calling `get_company_identifiers()`
    directly and reading both keys -- calling this alongside
    `get_cin_details()` separately runs the retrieval chain twice.
    """
    result = get_company_identifiers(company_name).get("GST", "GST not verified")
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
    result = get_company_identifiers(company_name).get("CIN", "CIN not verified")
    if isinstance(result, str):
        return result
    return [_to_cin_record(record) for record in result]
```
with:
```python
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
```

- [ ] **Step 4: Update `discovery.py`'s call site**

Change:
```python
        identifiers = get_company_identifiers(entry["company_name"])
        gst_result = identifiers.get("GST", "GST not verified")
        cin_result = identifiers.get("CIN", "CIN not verified")
```
to:
```python
        identifiers, _identifier_trace = get_company_identifiers(entry["company_name"])
        gst_result = identifiers.get("GST", "GST not verified")
        cin_result = identifiers.get("CIN", "CIN not verified")
```

- [ ] **Step 4a: Fix `test_gst.py`, the interactive console tool**

This script isn't part of the automated test suite (it reads from stdin), so it won't surface as a failure when running the other tests, but it calls `get_company_identifiers()` directly and would break silently otherwise. Change:
```python
    result = get_company_identifiers(company)

    print("=" * 80)


    gst = result.get("GST", "GST not found")
    cin = result.get("CIN", "CIN not found")

    print("GST")
    print(gst)

    print("\nCIN")
    print(cin)

    print("=" * 80)
```
to:
```python
    result, trace = get_company_identifiers(company)

    print("=" * 80)


    gst = result.get("GST", "GST not found")
    cin = result.get("CIN", "CIN not found")

    print("GST")
    print(gst)

    print("\nCIN")
    print(cin)

    print("=" * 80)
    print("Trace")
    print(f"Corroboration counts: {trace.corroboration_counts}")
    print(f"Validation notes: {trace.validation_notes}")
    print(f"Sites checked: {len(trace.site_checks)}")
    print("=" * 80)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 test_entity_matching.py`
Expected: `ALL TESTS PASSED`, exit code 0.

Also run: `python3 test_discovery_structured_output.py` to confirm Task 1's work still passes with this additional discovery.py edit.
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add identifier_lookup.py discovery.py test_entity_matching.py test_discovery_structured_output.py test_gst.py
git commit -m "Add IdentifierTrace; get_company_identifiers now returns (results, trace)"
```

---

### Task 3: `evidence_extractor.py` — add `EvidenceTrace`, change return type

**Files:**
- Modify: `evidence_extractor.py`, `discovery.py`
- Test: `test_evidence_extractor_structured_output.py` (extend), `test_discovery_structured_output.py` (extend -- its `extract_evidence` stub must also return a tuple, or `discover()`'s new unpacking will raise `ValueError` once this task's `discovery.py` edit lands)

**Interfaces:**
- Produces: `EvidenceRejection(point: str, source_url: str, reason: str)`, `EvidenceTrace(selected: List[str], rejected: List[EvidenceRejection])`. `extract(company_name, user_query, discovery_reason, documents) -> Tuple[List[EvidenceItem], EvidenceTrace]`.
- `discovery.py`'s `discover()` unpacks the new tuple but does not yet store the trace (that's Task 4) — prefixed `_evidence_trace`.

- [ ] **Step 1: Write the failing test**

In `test_evidence_extractor_structured_output.py`, change:
```python
result = evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", documents)

check("Only the entry citing a supplied URL survives", len(result) == 1, len(result))
check(
    "The surviving entry's fields match the supplied document's URL",
    bool(result) and result[0].source_url == "https://example.com/about",
)
```
to:
```python
result, trace = evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", documents)

check("Only the entry citing a supplied URL survives", len(result) == 1, len(result))
check(
    "The surviving entry's fields match the supplied document's URL",
    bool(result) and result[0].source_url == "https://example.com/about",
)
check("The trace records the selected point", trace.selected == ["Manufactures precision forgings"], trace.selected)
check(
    "The trace records the hallucinated entry as rejected, with a reason",
    len(trace.rejected) == 1
    and trace.rejected[0].point == "Hallucinated fact"
    and "not in the supplied documents" in trace.rejected[0].reason,
    trace.rejected,
)
```

Then change:
```python
evidence_extractor.get_provider = _fail_if_called
check(
    "extract() with no documents returns [] without calling the LLM",
    evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", []) == [],
)
```
to:
```python
evidence_extractor.get_provider = _fail_if_called
no_docs_result, no_docs_trace = evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", [])
check("extract() with no documents returns [] without calling the LLM", no_docs_result == [])
check("extract() with no documents returns an empty trace", no_docs_trace.selected == [] and no_docs_trace.rejected == [])
```

Then change:
```python
evidence_extractor.get_provider = lambda **kwargs: _FakeChatModel(RuntimeError("simulated failure"))
check(
    "A provider/invoke failure returns [] rather than raising",
    evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", documents) == [],
)
```
to:
```python
evidence_extractor.get_provider = lambda **kwargs: _FakeChatModel(RuntimeError("simulated failure"))
failure_result, failure_trace = evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", documents)
check("A provider/invoke failure returns [] rather than raising", failure_result == [])
check("A provider/invoke failure returns an empty trace", failure_trace.selected == [] and failure_trace.rejected == [])
```

Also, in `test_discovery_structured_output.py`, change the import line:
```python
from identifier_lookup import IdentifierRecord, IdentifierTrace
```
to:
```python
from evidence_extractor import EvidenceTrace
from identifier_lookup import IdentifierRecord, IdentifierTrace
```
and change:
```python
    discovery.extract_evidence = lambda *args, **kwargs: []
```
to:
```python
    discovery.extract_evidence = lambda *args, **kwargs: ([], EvidenceTrace(selected=[], rejected=[]))
```
(Also inside `_stub_downstream_pipeline()`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 test_evidence_extractor_structured_output.py`
Expected: `ValueError: too many values to unpack` (or similar), since `extract()` still returns a bare list.

Run: `python3 test_discovery_structured_output.py`
Expected: passes for now (nothing in `discovery.py` has changed yet in this task) -- a baseline run, re-run in Step 5 after `discovery.py`'s call site changes.

- [ ] **Step 3: Update `evidence_extractor.py`**

Change the typing import:
```python
from typing import List
```
to:
```python
from typing import List, Tuple
```

Add after the `EvidenceItem` dataclass:
```python
@dataclass(frozen=True)
class EvidenceItem:
    """One sourced, LLM-summarized fact about a company."""
    point: str
    source_title: str
    source_url: str
```
becomes:
```python
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
```

Replace `_validate_evidence_entry()`:
```python
def _validate_evidence_entry(entry: LLMEvidenceItem, valid_urls: set[str]) -> EvidenceItem | None:
    """
    Validates one structured LLMEvidenceItem against the documents actually
    supplied to the LLM. Pydantic has already guaranteed point/source_title/
    source_url are strings; this is the deterministic backstop against a
    hallucinated citation, dropping any entry whose source_url isn't one of
    the documents actually supplied to the LLM.
    """
    point = entry.point.strip()
    source_title = entry.source_title.strip()
    source_url = entry.source_url.strip()

    if not point or not source_url:
        return None
    if source_url not in valid_urls:
        return None

    return EvidenceItem(
        point=point,
        source_title=source_title or source_url,
        source_url=source_url,
    )
```
with:
```python
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
```

Replace `extract()`:
```python
def extract(company_name: str, user_query: str, discovery_reason: str, documents: List[Document]) -> List[EvidenceItem]:
    """
    Asks the LLM to summarize `documents` into short, sourced bullet points
    about `company_name`.

    Returns an empty list -- rather than raising -- if there are no
    documents to summarize, the LLM call fails, or its response can't be
    parsed. Evidence is an enrichment on top of a company result, not a
    required field, so a failure here should never block the rest of a
    company's result from being returned.
    """
    if not documents:
        return []

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
        return []

    evidence: List[EvidenceItem] = []
    for entry in response.items:
        item = _validate_evidence_entry(entry, valid_urls)
        if item is not None:
            evidence.append(item)

    return evidence
```
with:
```python
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
```

- [ ] **Step 4: Update `discovery.py`'s call site**

Change:
```python
        evidence_documents, _evidence_retrieval_trace = retriever.retrieve_for_evidence(entry["company_name"])
        evidence = extract_evidence(entry["company_name"], query, entry["reason"], evidence_documents)
```
to:
```python
        evidence_documents, _evidence_retrieval_trace = retriever.retrieve_for_evidence(entry["company_name"])
        evidence, _evidence_trace = extract_evidence(entry["company_name"], query, entry["reason"], evidence_documents)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 test_evidence_extractor_structured_output.py && python3 test_discovery_structured_output.py`
Expected: both print `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add evidence_extractor.py discovery.py test_evidence_extractor_structured_output.py test_discovery_structured_output.py
git commit -m "Add EvidenceTrace; extract() now returns (evidence, trace)"
```

---

### Task 4: `discovery.py` — attach traces to `CompanyResult`/`RejectedCompany`

**Files:**
- Modify: `discovery.py`
- Test: `test_discovery_structured_output.py` (extend)

**Interfaces:**
- Produces: `CompanyResult` gains `retrieval_trace: Optional[retriever.RetrievalTrace] = None`, `identifier_trace: Optional[IdentifierTrace] = None`, `evidence_trace: Optional[EvidenceTrace] = None`. `RejectedCompany` gains `identifier_trace: Optional[IdentifierTrace] = None` (populated only for `rejection_type="verification"`).

- [ ] **Step 1: Write the failing test**

In `test_discovery_structured_output.py`, change the import line:
```python
from evidence_extractor import EvidenceTrace
from identifier_lookup import IdentifierRecord, IdentifierTrace
```
(this is already the state after Task 3 -- no change needed to this specific line).

Change the `_stub_downstream_pipeline()` fixtures so the trace objects carry distinguishable content (not just empty placeholders), so the new assertions below can verify they actually flow through to `CompanyResult`/`RejectedCompany`. Change:
```python
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
        IdentifierTrace(site_checks=[], corroboration_counts={"GST": 2}, validation_notes={"GST": "stub"}),
    )
    discovery.retriever.retrieve_for_evidence = lambda name: (
        [_FIXTURE_DOCUMENT],
        retriever.RetrievalTrace(queries=[], attempts=[]),
    )
    discovery.extract_evidence = lambda *args, **kwargs: ([], EvidenceTrace(selected=[], rejected=[]))
```
to:
```python
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
```

Then, immediately after the existing check:
```python
check(
    "Non-grounded: accepted company's documents are populated from retrieve_for_evidence, not discarded",
    bool(result.accepted) and result.accepted[0].documents == [_FIXTURE_DOCUMENT],
)
```
insert:
```python
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
```

Now add a new scenario testing the verification-rejection path's `identifier_trace`. Immediately before the file's final block (the one starting with `print()` / `print("=" * 70)` / the `_FAILURES` check), insert:
```python
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

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_discovery_structured_output.py`
Expected: `AttributeError: 'CompanyResult' object has no attribute 'retrieval_trace'` (or similar), since `CompanyResult`/`RejectedCompany` don't have these fields yet.

- [ ] **Step 3: Update `discovery.py`**

Change the imports:
```python
import retriever
from evidence_extractor import EvidenceItem, extract as extract_evidence
from identifier_lookup import IdentifierRecord, get_company_identifiers
```
to:
```python
import retriever
from evidence_extractor import EvidenceItem, EvidenceTrace, extract as extract_evidence
from identifier_lookup import IdentifierRecord, IdentifierTrace, get_company_identifiers
```

Change the `CompanyResult` dataclass:
```python
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
```
to:
```python
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
```

Change the `RejectedCompany` dataclass:
```python
    company_name: str
    reason: str
    confidence: str  # the LLM's original confidence -- never overwritten, even for verification rejections
    rejection_type: str  # "llm" | "verification"
    constraint_evaluation: Dict[str, ConstraintEvaluation] = field(default_factory=dict)
    decision: str = "REJECT"
    gst: Optional[str] = None  # always None here; present so the UI can display "GST: Not found" for verification rejections
    cin: Optional[str] = None  # always None here; present so the UI can display "CIN: Not found" for verification rejections
```
to:
```python
    company_name: str
    reason: str
    confidence: str  # the LLM's original confidence -- never overwritten, even for verification rejections
    rejection_type: str  # "llm" | "verification"
    constraint_evaluation: Dict[str, ConstraintEvaluation] = field(default_factory=dict)
    decision: str = "REJECT"
    gst: Optional[str] = None  # always None here; present so the UI can display "GST: Not found" for verification rejections
    cin: Optional[str] = None  # always None here; present so the UI can display "CIN: Not found" for verification rejections
    identifier_trace: Optional[IdentifierTrace] = None  # populated only for rejection_type="verification" -- LLM rejections never reach identifier lookup (Phase 4 debug mode)
```

Change the identifier-lookup call and its surrounding block:
```python
        identifiers, _identifier_trace = get_company_identifiers(entry["company_name"])
        gst_result = identifiers.get("GST", "GST not verified")
        cin_result = identifiers.get("CIN", "CIN not verified")
```
to:
```python
        identifiers, identifier_trace = get_company_identifiers(entry["company_name"])
        gst_result = identifiers.get("GST", "GST not verified")
        cin_result = identifiers.get("CIN", "CIN not verified")
```

Change the verification-rejection `RejectedCompany` construction:
```python
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
                )
            )
            continue
```
to:
```python
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
```

Change the verified-path retrieval/extraction and `CompanyResult` construction:
```python
        # STEP 3b (verified): proceed exactly as before -- retrieve, then
        # extract, evidence for this company.
        evidence_documents, _evidence_retrieval_trace = retriever.retrieve_for_evidence(entry["company_name"])
        evidence, _evidence_trace = extract_evidence(entry["company_name"], query, entry["reason"], evidence_documents)

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
            )
        )
```
to:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_discovery_structured_output.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Run the full existing regression to confirm nothing else broke**

Run: `python3 test_llm_schemas.py && python3 test_llm_provider.py && python3 test_prompts_chat_templates.py && python3 test_evidence_extractor_structured_output.py && python3 test_retriever.py && python3 test_qa.py && python3 test_entity_matching.py`
Expected: every script prints `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add discovery.py test_discovery_structured_output.py
git commit -m "Attach retrieval/identifier/evidence traces to CompanyResult and RejectedCompany"
```

---

### Task 5: `conversation.py` — capture intent/QA reasoning onto `ChatTurn`

**Files:**
- Modify: `conversation.py`
- Test: `test_conversation.py` (extend)

**Interfaces:**
- Produces: `ChatTurn` gains `intent_reasoning: str = ""`, `intent_confidence: str = ""`, `qa_sources: List[str] = field(default_factory=list)`, `qa_used_new_retrieval: bool = False`. `AssistantReply` gains `qa_sources`/`qa_used_new_retrieval` (same shape), populated only by `_handle_company_question`. `handle_message`'s public signature is unchanged.

- [ ] **Step 1: Write the failing test**

In `test_conversation.py`, change:
```python
check("NEW_DISCOVERY: reply mentions a match", "found" in reply.text.lower())
check("NEW_DISCOVERY: reply carries the accepted company", len(reply.companies) == 1)
check("NEW_DISCOVERY: discovery_history grew by one", len(state.discovery_history) == 1)
check("NEW_DISCOVERY: current_company reset to None", state.current_company is None)
check("NEW_DISCOVERY: turn recorded", len(state.turns) == 1 and state.turns[0].intent == "NEW_DISCOVERY")
```
to:
```python
check("NEW_DISCOVERY: reply mentions a match", "found" in reply.text.lower())
check("NEW_DISCOVERY: reply carries the accepted company", len(reply.companies) == 1)
check("NEW_DISCOVERY: discovery_history grew by one", len(state.discovery_history) == 1)
check("NEW_DISCOVERY: current_company reset to None", state.current_company is None)
check("NEW_DISCOVERY: turn recorded", len(state.turns) == 1 and state.turns[0].intent == "NEW_DISCOVERY")
check("NEW_DISCOVERY: turn records intent_reasoning", state.turns[0].intent_reasoning == "stub")
check("NEW_DISCOVERY: turn records intent_confidence", state.turns[0].intent_confidence == "High")
check("NEW_DISCOVERY: turn's qa_sources stays empty (not a COMPANY_QUESTION turn)", state.turns[0].qa_sources == [])
check(
    "NEW_DISCOVERY: turn's qa_used_new_retrieval stays False (not a COMPANY_QUESTION turn)",
    state.turns[0].qa_used_new_retrieval is False,
)
```

Then change:
```python
check("Classification failure yields UNRECOGNIZED, not a crash", reply.intent == "UNRECOGNIZED")
check("Classification failure's turn is still recorded", state.turns[-1].intent == "UNRECOGNIZED")
```
to:
```python
check("Classification failure yields UNRECOGNIZED, not a crash", reply.intent == "UNRECOGNIZED")
check("Classification failure's turn is still recorded", state.turns[-1].intent == "UNRECOGNIZED")
check(
    "Classification failure's turn has no intent reasoning (classification never completed)",
    state.turns[-1].intent_reasoning == "",
)
```

Then change:
```python
check("COMPANY_QUESTION: reply uses the QA answer text", reply.text == "The directors are Jane Doe.")
check(
    "COMPANY_QUESTION: reply carries the resolved company",
    len(reply.companies) == 1 and reply.companies[0].company_name == "Beta Industries Limited",
)
check(
    "COMPANY_QUESTION: the resolved company's documents are updated in place",
    reply.companies[0].documents == [new_document],
)

_beta_result = next(r for r in state.discovery_history if any(c.company_name == "Beta Industries Limited" for c in r.accepted))
_beta_company = next(c for c in _beta_result.accepted if c.company_name == "Beta Industries Limited")
check(
    "COMPANY_QUESTION: the SAME CompanyResult object in discovery_history reflects the update",
    _beta_company.documents == [new_document],
)
```
to:
```python
check("COMPANY_QUESTION: reply uses the QA answer text", reply.text == "The directors are Jane Doe.")
check(
    "COMPANY_QUESTION: reply carries the resolved company",
    len(reply.companies) == 1 and reply.companies[0].company_name == "Beta Industries Limited",
)
check(
    "COMPANY_QUESTION: the resolved company's documents are updated in place",
    reply.companies[0].documents == [new_document],
)
check("COMPANY_QUESTION: reply carries QA sources", reply.qa_sources == ["https://example.com/directors"])
check("COMPANY_QUESTION: reply carries qa_used_new_retrieval", reply.qa_used_new_retrieval is True)
check(
    "COMPANY_QUESTION: the recorded turn carries QA sources",
    state.turns[-1].qa_sources == ["https://example.com/directors"],
)
check("COMPANY_QUESTION: the recorded turn carries qa_used_new_retrieval", state.turns[-1].qa_used_new_retrieval is True)

_beta_result = next(r for r in state.discovery_history if any(c.company_name == "Beta Industries Limited" for c in r.accepted))
_beta_company = next(c for c in _beta_result.accepted if c.company_name == "Beta Industries Limited")
check(
    "COMPANY_QUESTION: the SAME CompanyResult object in discovery_history reflects the update",
    _beta_company.documents == [new_document],
)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_conversation.py`
Expected: `AttributeError: 'ChatTurn' object has no attribute 'intent_reasoning'` (or similar).

- [ ] **Step 3: Update `conversation.py`**

Change the `ChatTurn` dataclass:
```python
@dataclass
class ChatTurn:
    """One past turn: what the user asked, what was classified, and what came back."""
    user_message: str
    intent: str
    assistant_response: str
    companies: List[CompanyResult] = field(default_factory=list)
    discovery_result: Optional[DiscoveryResult] = None
```
to:
```python
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
```

Change the `AssistantReply` dataclass:
```python
@dataclass
class AssistantReply:
    """What ui.py renders for one turn: the reply text, plus any companies to show as cards."""
    text: str
    intent: str
    companies: List[CompanyResult] = field(default_factory=list)
    discovery_result: Optional[DiscoveryResult] = None
```
to:
```python
@dataclass
class AssistantReply:
    """What ui.py renders for one turn: the reply text, plus any companies to show as cards."""
    text: str
    intent: str
    companies: List[CompanyResult] = field(default_factory=list)
    discovery_result: Optional[DiscoveryResult] = None
    qa_sources: List[str] = field(default_factory=list)  # only for COMPANY_QUESTION turns (Phase 4 debug mode)
    qa_used_new_retrieval: bool = False  # only for COMPANY_QUESTION turns (Phase 4 debug mode)
```

Change `_handle_company_question`'s final return:
```python
    state.current_company = company.company_name
    qa_answer = answer_question(company, user_message)
    company.documents = qa_answer.documents

    return AssistantReply(text=qa_answer.answer, intent="COMPANY_QUESTION", companies=[company])
```
to:
```python
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
```

Change `handle_message`'s final `ChatTurn` construction:
```python
    state.turns.append(
        ChatTurn(
            user_message=user_message,
            intent=reply.intent,
            assistant_response=reply.text,
            companies=reply.companies,
            discovery_result=reply.discovery_result,
        )
    )
    return state, reply
```
to:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_conversation.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add conversation.py test_conversation.py
git commit -m "Capture intent-classification reasoning and QA sources onto ChatTurn"
```

---

### Task 6: `ui.py` — render all six debug sections

**Files:**
- Modify: `ui.py`

**Interfaces:**
- No new imports needed beyond what's already there -- `company.retrieval_trace`/`.identifier_trace`/`.evidence_trace` and `turn.intent_reasoning`/`.intent_confidence`/`.qa_sources`/`.qa_used_new_retrieval` are read via plain attribute access, no new type imports required since nothing here constructs those objects.

This task has no automated test -- same as every prior `ui.py` task. Verification is `py_compile` plus a manual smoke test with the debug toggle on, per Task 7.

- [ ] **Step 1: Extend `_render_debug_html(company)`**

Change:
```python
    decision_class = "accept" if company.decision == "ACCEPT" else "reject"

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
    )
```
to:
```python
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
```

- [ ] **Step 2: Extend `_render_rejected_card_html(rejected)`**

Change:
```python
    identifiers_html = (
        f'<div class="gst-line not-found">GST: Not found</div>'
        f'<div class="gst-line not-found">CIN: Not found</div>'
        if is_verification
        else ""
    )

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
        '</div>'
    )
```
to:
```python
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
```

- [ ] **Step 3: Add `_render_intent_debug_html` and use it in `_render_turn`**

Change:
```python
def _render_turn(turn: ChatTurn, show_debug: bool) -> None:
    """Renders one past turn: the user's message, the assistant's reply, and any company cards."""
    with st.chat_message("user"):
        st.write(turn.user_message)

    with st.chat_message("assistant"):
        st.write(turn.assistant_response)

        for company in turn.companies:
```
to:
```python
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
```

- [ ] **Step 4: Syntax/import sanity check**

Run: `python3 -m py_compile ui.py`
Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add ui.py
git commit -m "Render all six debug sections in ui.py (Phase 4 debug mode)"
```

---

### Task 7: Full regression pass

**Files:**
- None created or modified — this task only runs verification.

- [ ] **Step 1: Run every automated test script**

Run:
```bash
python3 test_llm_schemas.py && \
python3 test_llm_provider.py && \
python3 test_prompts_chat_templates.py && \
python3 test_discovery_structured_output.py && \
python3 test_evidence_extractor_structured_output.py && \
python3 test_retriever.py && \
python3 test_qa.py && \
python3 test_conversation.py && \
python3 test_entity_matching.py
```
Expected: every script prints `ALL TESTS PASSED`; the whole chain exits 0.

- [ ] **Step 2: Confirm `ui.py` still compiles**

Run: `python3 -m py_compile ui.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Confirm no stale bare (non-tuple) call sites remain**

Run:
```bash
grep -rn "= retriever\.retrieve_for_evidence(\|= retrieve_for_question(\|= get_company_identifiers(\|= extract_evidence(\|= evidence_extractor\.extract(" --include="*.py" . | grep -v "test_\|, .*trace\|, .*_trace"
```
Expected: no output (empty) -- every real call site should already be unpacking a tuple (containing the word "trace" in the unpacked variable name) after Tasks 1-3. If anything unexpected matches, it's a call site that was missed and needs the same fix pattern applied.

- [ ] **Step 4: Manual smoke test with real credentials**

This step needs real API keys and can't be scripted here. With a working `.env`:

```bash
streamlit run ui.py
```

Verify by hand, with "Show Debug Information" turned ON:
- Run a discovery search and confirm each company card still shows everything it did before this phase (constraints, decision, confidence explanation, assumptions) **plus** three new sections: Retrieval (queries issued, pages included/discarded with reasons), Identifier Lookup (site checks, corroboration, validation notes), Evidence Extraction (selected vs. rejected points).
- Confirm the "Rejected Candidates" panel still works, and that verification-type rejections now also show an Identifier Lookup section.
- Confirm every turn (not just discovery turns) shows a new "Intent Classification" block above the reply, with a detected intent, confidence, and reasoning that actually makes sense for what you asked.
- Ask a company question (e.g. "who are its competitors") and confirm the turn's debug block also shows a "Final Answer Generation" section with whether new retrieval was used and which sources were consulted.
- Turn "Show Debug Information" OFF and confirm the view looks exactly as it did before this phase -- no new sections, no layout changes.

- [ ] **Step 5: Final commit (only if Step 4 surfaced any fixes)**

If the manual smoke test in Step 4 required any code changes, commit them now with a message describing what was fixed. If everything worked as-is, there is nothing to commit for this task.
