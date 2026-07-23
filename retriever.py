"""
retriever.py

Web retriever for the Company Discovery Agent.

This module is the ONLY place in the project that performs web searches,
downloads pages, or parses raw HTML. Every other module -- identifier_lookup.py
(the deterministic GST/CIN extractor) and evidence_extractor.py (the
LLM-based evidence extractor) -- consumes its output (`Document` objects)
and never touches the network or raw HTML directly.

Design notes:

- GST/CIN lookup and evidence extraction have different objectives and
  different retrieval shapes:

  * `retrieve_for_evidence()` is eager/batch: it runs a small set of broad
    queries, merges and dedupes the results, downloads up to a fixed
    number of pages, and returns them all at once. It delegates that
    downloading/parsing/dedup/ranking logic to the shared `_retrieve()`
    primitive.

  * `iter_gst_documents()` is lazy/incremental by design, NOT built on top
    of `_retrieve()`. Identifier lookup runs an ordered fallback chain
    across registry sites (see identifier_lookup.py) and stops as soon as
    it has enough corroboration for everything it's looking for -- so
    nothing about site 4 or 5 should ever be searched or downloaded if
    earlier sites already answered the question, and nothing about a
    site's 2nd/3rd page should be downloaded once its 1st page already
    produced a usable candidate. `_retrieve()`'s contract is "fetch
    everything, then return," which is fundamentally incompatible with
    that early-stopping requirement, so this path does not use it. It
    DOES reuse `_search_company_pages()`, `_download_page()`, and
    `_parse_page()` -- the actual network/parsing work is still shared,
    just not the eager batch-orchestration wrapper.

  This module is intentionally ignorant of WHAT identifiers a caller is
  looking for -- it just fetches pages from registry sites, in priority
  order, lazily. It has no notion of "this site gives GST" or "this site
  gives CIN"; that's an extraction-layer concern, not a retrieval-layer
  one, and keeping it that way means this module never needs to change
  just because identifier_lookup.py starts looking for something new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

_USER_AGENT = "Mozilla/5.0 (compatible; CompanyDiscoveryBot/1.0)"

# Ordered fallback chain for registry-site retrieval: each entry is
# queried via a site:-restricted search, in this order, until
# identifier_lookup.py's corroboration thresholds are met for everything
# it's looking for. Reorder or edit this list to tune behavior
# empirically -- nothing else needs to change.
GST_SITE_PRIORITY: Tuple[str, ...] = (
    "thecompanycheck.com",
    "zaubacorp.com",
    "tofler.in",
    "knowyourgst.com",
    "indiafilings.com",
)

# Evidence extraction wants general business information (products,
# location, customers, "about us" pages), not registry data, so it has no
# domain bias -- results are used in whatever order the search backend
# returns them.
_EVIDENCE_PREFERRED_DOMAINS: Tuple[str, ...] = ()

_MAX_RESULTS_PER_QUERY = 10
_MAX_RESULTS_PER_SITE_QUERY = 3
_MAX_DOCUMENTS_PER_COMPANY = 10
_PAGE_DOWNLOAD_TIMEOUT_SECONDS = 10


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


def _search_company_pages(query: str, max_results: int = _MAX_RESULTS_PER_QUERY) -> List[str]:
    """
    Performs a web search via the `ddgs` library (the maintained successor
    to `duckduckgo-search`) and returns up to `max_results` distinct
    result URLs, in the order the search backend returned them.

    Results missing an 'href' are skipped, and duplicate URLs are dropped
    while preserving ordering. If the search library raises for any reason
    (rate limiting, network failure, backend changes, etc.), this returns
    an empty list instead of propagating the exception, so a search hiccup
    degrades to "no results" rather than crashing the caller.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception:
        return []

    urls: List[str] = []
    seen: set[str] = set()
    for result in results:
        href = (result.get("href") or "").strip()
        if not href or href in seen:
            continue
        seen.add(href)
        urls.append(href)
        if len(urls) >= max_results:
            break

    return urls


def _merge_and_dedupe_urls(*url_lists: List[str]) -> List[str]:
    """
    Merges multiple ordered URL lists into one, dropping duplicates while
    preserving the order URLs were first seen in -- the first list's URLs
    come first, then any new URLs from subsequent lists.
    """
    merged: List[str] = []
    seen: set[str] = set()
    for urls in url_lists:
        for url in urls:
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _prioritize_urls(urls: List[str], preferred_domains: Sequence[str]) -> List[str]:
    """
    Moves URLs on any of `preferred_domains` to the front of the list,
    preserving relative order within each group. Works fine if
    `preferred_domains` is empty or none of them are present -- the list is
    simply returned in its original order.
    """
    def is_preferred(url: str) -> bool:
        return any(domain in url.lower() for domain in preferred_domains)

    preferred = [u for u in urls if is_preferred(u)]
    other = [u for u in urls if not is_preferred(u)]
    return preferred + other


def _download_page(url: str, timeout: int = _PAGE_DOWNLOAD_TIMEOUT_SECONDS) -> str | None:
    """
    Downloads a single page's raw HTML. Returns None (rather than raising)
    on any network failure, timeout, or non-2xx response, so the caller can
    simply skip this page and move on to the next candidate URL.
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None
    return response.text


def _parse_page(html: str) -> tuple[str, str]:
    """
    Parses a downloaded page once, returning both its title and its cleaned
    visible text.
    """
    soup = BeautifulSoup(html, "html.parser")

    title = "Untitled"
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    cleaned_text = soup.get_text(separator=" ")
    return title, cleaned_text


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


def _iter_site_documents(company_name: str, site: str) -> Iterator[Document]:
    """
    Lazily searches a single registry `site` for `company_name` (via a
    site:-restricted query, unquoted so exact punctuation/casing
    differences between the LLM-generated name and the registry's own
    rendering don't cause false misses -- name verification happens
    downstream in identifier_lookup.py's fuzzy matcher instead) and yields
    up to `_MAX_RESULTS_PER_SITE_QUERY` downloaded, cleaned Documents ONE
    AT A TIME.

    The search itself happens once, when the first item is requested.
    Each page is only downloaded when the caller actually advances the
    iterator to it -- a caller that stops after the first document never
    triggers the second or third page's download at all.
    """
    query = f"site:{site} {company_name}"
    urls = _search_company_pages(query, max_results=_MAX_RESULTS_PER_SITE_QUERY)
    for url in urls:
        html = _download_page(url)
        if not html:
            continue
        title, cleaned_text = _parse_page(html)
        yield Document(url=url, title=title, cleaned_text=cleaned_text)


def iter_gst_documents(company_name: str) -> Iterator[Tuple[str, Iterator[Document]]]:
    """
    The registry-retrieval entry point (named for its original GST-only
    purpose; it now serves any identifier extraction that wants to walk
    the same site-priority chain -- see identifier_lookup.py). Yields
    `(site, documents)` pairs, one per site in `GST_SITE_PRIORITY` order,
    where `documents` is itself a lazy per-page iterator (see
    `_iter_site_documents`).

    Both levels are lazy: nothing is searched or downloaded for site N+1
    until the caller pulls the next pair from this generator, and nothing
    is downloaded for a site's 2nd/3rd page until the caller advances that
    site's inner iterator. This lets identifier_lookup.py implement an
    early-stopping fallback chain without this module knowing anything
    about GST, CIN, extraction, or corroboration. It only knows how to
    search one site at a time.
    """
    for site in GST_SITE_PRIORITY:
        yield site, _iter_site_documents(company_name, site)


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
