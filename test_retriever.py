"""
test_retriever.py

Unit tests for retriever.py's retrieve_for_question(), verifying its
query construction without making real network calls --
retriever._search_company_pages is monkeypatched to capture the queries
it's called with and return no results, so _retrieve() never attempts a
real download.

Run with: python3 test_retriever.py
"""

from __future__ import annotations

import retriever
from retriever import retrieve_for_question

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


_captured_queries: list[str] = []


def _fake_search_company_pages(query, max_results=10):
    _captured_queries.append(query)
    return []


retriever._search_company_pages = _fake_search_company_pages

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
