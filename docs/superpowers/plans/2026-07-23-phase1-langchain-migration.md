# Phase 1: LangChain Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Company Discovery Agent's hand-rolled LLM provider abstraction and manual JSON parsing with LangChain chat models, `ChatPromptTemplate`, and Pydantic-validated structured output, with zero behavior change visible to `ui.py` or any other downstream consumer.

**Architecture:** `llm_provider.py` becomes a factory returning a configured LangChain `BaseChatModel` (`ChatOpenAI` for OpenAI/GLM/ZAI, `ChatGoogleGenerativeAI` for Gemini). New `llm_schemas.py` holds Pydantic models used only at the LLM I/O boundary. `discovery.py` and `evidence_extractor.py` call `.with_structured_output(...)` themselves, except discovery.py's one special case: when Gemini Search grounding is active, it falls back to a raw `.invoke()` + manual JSON parse (still validated through the same Pydantic schema), because Gemini's API rejects combining the `google_search` tool with a structured-output schema constraint in one call.

**Tech Stack:** `langchain`, `langchain-openai`, `langchain-google-genai`, `google-genai`, `pydantic` (all newly added); existing `requests`, `beautifulsoup4`, `ddgs`, `streamlit` untouched.

## Global Constraints

- Testing stays in the existing script + `check(label, condition)` harness convention (see `test_gst.py`, `test_entity_matching.py`) — no pytest introduced.
- Env var names are unchanged from before the migration: `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_GROUNDING`.
- ZAI's raw `web_search` tool block and its `LLM_WEB_SEARCH` env var are removed entirely (confirmed unused in practice).
- Gemini Search grounding is preserved, but only reachable by a caller that passes `get_provider(allow_grounding=True)` — currently only `discovery.py`. `evidence_extractor.py` never requests it.
- New Pydantic models (`llm_schemas.py`) exist only at the LLM I/O boundary. The existing dataclasses `CompanyResult`, `RejectedCompany`, `EvidenceItem`, `DiscoveryResult` (all in `discovery.py`/`evidence_extractor.py`) keep their exact current shape.
- `get_provider()` returns a raw LangChain `BaseChatModel`. Callers build their own `ChatPromptTemplate` and call `.with_structured_output(...)` themselves — no `generate()`-style wrapper.
- `ui.py` requires zero changes. `streamlit run ui.py` must behave identically to before this migration.
- Verified compatible library versions (installed and exercised during design research): `langchain==1.3.14`, `langchain-openai==1.4.0`, `langchain-google-genai==4.3.1`, `google-genai==2.13.0`, `pydantic==2.13.4`.
- Full design rationale lives in `docs/superpowers/specs/2026-07-23-phase1-langchain-migration-design.md` — consult it for the "why" behind any of the above.

---

### Task 1: `llm_schemas.py` and dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `llm_schemas.py`
- Test: `test_llm_schemas.py`

**Interfaces:**
- Produces: `LLMConstraintEvaluation(name: str, status: Literal["PASS","FAIL","UNKNOWN"], reason: str)`, `LLMCompanyEntry(company_name: str, constraint_evaluation: List[LLMConstraintEvaluation], decision: Literal["ACCEPT","REJECT"], reason: str, confidence: Literal["High","Medium","Low"])`, `LLMDiscoveryResponse(companies: List[LLMCompanyEntry])`, `LLMEvidenceItem(point: str, source_title: str, source_url: str)`, `LLMEvidenceResponse(items: List[LLMEvidenceItem])` — all in `llm_schemas.py`, all `pydantic.BaseModel` subclasses.

- [ ] **Step 1: Update `requirements.txt`**

Replace the full file contents with:

```
google-genai>=2.13.0
openai>=1.30.0
python-dotenv>=1.0.1
requests>=2.31.0
beautifulsoup4>=4.12.3
flask>=3.0.0
streamlit>=1.35.0
rapidfuzz>=3.0.0
ddgs>=9.0.0
langchain>=1.3.0
langchain-openai>=1.4.0
langchain-google-genai>=4.3.0
pydantic>=2.13.0
```

(This drops `google-generativeai`, the deprecated SDK that was declared but never actually imported — `llm_provider.py` already imports the newer `google-genai` package — and adds the LangChain/Pydantic dependencies this phase needs.)

- [ ] **Step 2: Install the updated dependencies**

Run: `python3 -m pip install -r requirements.txt`
Expected: no errors. (Use whichever `python3` the project actually runs under — check with `which python3`/`which streamlit` if unsure; this codebase has previously had a `pip` vs `python3 -m pip` environment mismatch.)

- [ ] **Step 3: Write the failing test**

Create `test_llm_schemas.py`:

```python
"""
test_llm_schemas.py

Unit tests for llm_schemas.py's Pydantic models, especially the
constraint_evaluation dict/list normalization that lets discovery.py's
grounded and non-grounded paths share one schema.

Run with: python3 test_llm_schemas.py
"""

from __future__ import annotations

from pydantic import ValidationError

from llm_schemas import (
    LLMCompanyEntry,
    LLMConstraintEvaluation,
    LLMDiscoveryResponse,
    LLMEvidenceItem,
    LLMEvidenceResponse,
)

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


# ---------------------------------------------------------------------
# LLMConstraintEvaluation / LLMCompanyEntry
# ---------------------------------------------------------------------

list_shape_entry = LLMCompanyEntry.model_validate(
    {
        "company_name": "Acme Forgings Private Limited",
        "constraint_evaluation": [
            {"name": "location", "status": "PASS", "reason": "HQ in Thane."}
        ],
        "decision": "ACCEPT",
        "reason": "Matches all constraints.",
        "confidence": "High",
    }
)
check(
    "List-shape constraint_evaluation (with_structured_output path) validates",
    list_shape_entry.constraint_evaluation == [
        LLMConstraintEvaluation(name="location", status="PASS", reason="HQ in Thane.")
    ],
)

dict_shape_entry = LLMCompanyEntry.model_validate(
    {
        "company_name": "Acme Forgings Private Limited",
        "constraint_evaluation": {
            "location": {"status": "PASS", "reason": "HQ in Thane."}
        },
        "decision": "ACCEPT",
        "reason": "Matches all constraints.",
        "confidence": "High",
    }
)
check(
    "Dict-shape constraint_evaluation (grounded-fallback prompt shape) normalizes to a list",
    dict_shape_entry.constraint_evaluation == [
        LLMConstraintEvaluation(name="location", status="PASS", reason="HQ in Thane.")
    ],
)

check(
    "Empty constraint_evaluation defaults to an empty list",
    LLMCompanyEntry.model_validate(
        {
            "company_name": "Acme Forgings Private Limited",
            "decision": "ACCEPT",
            "reason": "No explicit constraints in this query.",
            "confidence": "Medium",
        }
    ).constraint_evaluation
    == [],
)

try:
    LLMCompanyEntry.model_validate(
        {
            "company_name": "Acme Forgings Private Limited",
            "decision": "MAYBE",
            "reason": "x",
            "confidence": "High",
        }
    )
    check("Invalid decision value is rejected by validation", False, "no exception raised")
except ValidationError:
    check("Invalid decision value is rejected by validation", True)

# ---------------------------------------------------------------------
# LLMDiscoveryResponse
# ---------------------------------------------------------------------

discovery_response = LLMDiscoveryResponse.model_validate(
    {
        "companies": [
            {
                "company_name": "Acme Forgings Private Limited",
                "constraint_evaluation": [],
                "decision": "ACCEPT",
                "reason": "Fits.",
                "confidence": "High",
            }
        ]
    }
)
check(
    "LLMDiscoveryResponse wraps companies under a 'companies' key",
    len(discovery_response.companies) == 1
    and discovery_response.companies[0].company_name == "Acme Forgings Private Limited",
)

check(
    "LLMDiscoveryResponse defaults to an empty companies list",
    LLMDiscoveryResponse.model_validate({}).companies == [],
)

# ---------------------------------------------------------------------
# LLMEvidenceItem / LLMEvidenceResponse
# ---------------------------------------------------------------------

evidence_response = LLMEvidenceResponse.model_validate(
    {
        "items": [
            {
                "point": "Manufactures precision automotive components",
                "source_title": "Official Website",
                "source_url": "https://example.com/about",
            }
        ]
    }
)
check(
    "LLMEvidenceResponse wraps items under an 'items' key",
    len(evidence_response.items) == 1
    and evidence_response.items[0].source_url == "https://example.com/about",
)

try:
    LLMEvidenceItem.model_validate({"point": "x", "source_title": "y"})
    check("LLMEvidenceItem requires source_url", False, "no exception raised")
except ValidationError:
    check("LLMEvidenceItem requires source_url", True)


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

- [ ] **Step 4: Run the test to verify it fails**

Run: `python3 test_llm_schemas.py`
Expected: `ModuleNotFoundError: No module named 'llm_schemas'`

- [ ] **Step 5: Create `llm_schemas.py`**

```python
"""
llm_schemas.py

Pydantic models describing the LLM's structured output shape for the
Company Discovery Agent. These models exist only at the LLM I/O boundary:
callers (discovery.py, evidence_extractor.py) convert instances of these
into the project's existing plain dataclasses
(CompanyResult/RejectedCompany/EvidenceItem/DiscoveryResult) before
returning to their own callers. Nothing outside discovery.py and
evidence_extractor.py should import from this module.

Responses are wrapped in a top-level object (`companies` / `items`)
rather than a bare array, since structured-output/function-calling JSON
schemas require an object root.
"""

from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, Field, field_validator


class LLMConstraintEvaluation(BaseModel):
    """One named constraint's PASS/FAIL/UNKNOWN judgement, as returned by the LLM."""

    name: str = Field(description="Short snake_case constraint name, e.g. 'location', 'turnover'.")
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    reason: str = Field(description="One or two sentence explanation for this status.")


class LLMCompanyEntry(BaseModel):
    """One company entry (accepted or rejected) as returned by the discovery LLM call."""

    company_name: str = Field(description="The full legal or commonly used company name.")
    constraint_evaluation: List[LLMConstraintEvaluation] = Field(
        default_factory=list,
        description=(
            "Per-constraint PASS/FAIL/UNKNOWN evaluation, one entry per "
            "explicit constraint relevant to the query."
        ),
    )
    decision: Literal["ACCEPT", "REJECT"]
    reason: str = Field(description="Executive summary of why this company was accepted or rejected.")
    confidence: Literal["High", "Medium", "Low"]

    @field_validator("constraint_evaluation", mode="before")
    @classmethod
    def _normalize_constraint_evaluation(cls, value: Any) -> Any:
        """
        Accepts either the schema-enforced shape (a list of
        {name, status, reason} objects -- what with_structured_output
        forces the model into, regardless of prompt wording) or a dict
        keyed by constraint name (the shape prompts.py's Rule 9 describes
        in prose, which is what discovery.py's grounded fallback path
        actually receives, since no JSON-schema constraint is enforced
        there). Normalizes a dict into the list shape; a list (or a list
        of already-built LLMConstraintEvaluation instances) passes through
        untouched.
        """
        if isinstance(value, dict):
            return [
                {"name": name, **item}
                for name, item in value.items()
                if isinstance(item, dict)
            ]
        return value


class LLMDiscoveryResponse(BaseModel):
    """Top-level structured response for a discovery request."""

    companies: List[LLMCompanyEntry] = Field(default_factory=list)


class LLMEvidenceItem(BaseModel):
    """One sourced fact extracted from already-downloaded documents."""

    point: str = Field(description="One short, concrete fact (roughly 3-12 words).")
    source_title: str = Field(description="Short human-readable label for the source.")
    source_url: str = Field(description="Copied exactly from one of the supplied document URLs.")


class LLMEvidenceResponse(BaseModel):
    """Top-level structured response for an evidence-extraction request."""

    items: List[LLMEvidenceItem] = Field(default_factory=list)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python3 test_llm_schemas.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt llm_schemas.py test_llm_schemas.py
git commit -m "Add llm_schemas.py Pydantic models for the LangChain migration"
```

---

### Task 2: Rewrite `llm_provider.py`

**Files:**
- Modify: `llm_provider.py` (full rewrite)
- Test: `test_llm_provider.py`

**Interfaces:**
- Consumes: nothing from Task 1's `llm_schemas.py` (this module is schema-agnostic).
- Produces: `get_provider(*, allow_grounding: bool = False) -> BaseChatModel`, `is_grounded() -> bool`, `LLMProviderError(Exception)`. `discovery.py` and `evidence_extractor.py` (Tasks 4-5) import all three (well — `evidence_extractor.py` only imports `get_provider`).

- [ ] **Step 1: Write the failing test**

Create `test_llm_provider.py`:

```python
"""
test_llm_provider.py

Unit tests for llm_provider.py's provider factory and grounding logic.
No network calls are made -- these only check that get_provider() selects
and configures the right LangChain chat model class based on environment
variables, and that is_grounded() agrees with what get_provider() actually
does.

Run with: python3 test_llm_provider.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from langchain_core.runnables import RunnableBinding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from llm_provider import LLMProviderError, get_provider, is_grounded

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


_ENV_KEYS = ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_GROUNDING")


@contextmanager
def env(**overrides: str):
    """Temporarily sets env vars in _ENV_KEYS, restoring the previous state afterward."""
    previous = {key: os.environ.get(key) for key in _ENV_KEYS}
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in overrides.items():
        os.environ[key] = value
    try:
        yield
    finally:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in previous.items():
            if value is not None:
                os.environ[key] = value


# ---------------------------------------------------------------------
# Config errors
# ---------------------------------------------------------------------

with env(LLM_PROVIDER="OPENAI", LLM_MODEL="gpt-4o-mini"):
    try:
        get_provider()
        check("Missing LLM_API_KEY raises LLMProviderError", False, "no exception raised")
    except LLMProviderError:
        check("Missing LLM_API_KEY raises LLMProviderError", True)

with env(LLM_PROVIDER="OPENAI", LLM_API_KEY="sk-test"):
    try:
        get_provider()
        check("Missing LLM_MODEL raises LLMProviderError", False, "no exception raised")
    except LLMProviderError:
        check("Missing LLM_MODEL raises LLMProviderError", True)

with env(LLM_PROVIDER="BOGUS", LLM_API_KEY="sk-test", LLM_MODEL="whatever"):
    try:
        get_provider()
        check("Unsupported LLM_PROVIDER raises LLMProviderError", False, "no exception raised")
    except LLMProviderError:
        check("Unsupported LLM_PROVIDER raises LLMProviderError", True)

# ---------------------------------------------------------------------
# OpenAI-compatible providers
# ---------------------------------------------------------------------

with env(LLM_PROVIDER="OPENAI", LLM_API_KEY="sk-test", LLM_MODEL="gpt-4o-mini"):
    model = get_provider()
    check("LLM_PROVIDER=OPENAI returns a ChatOpenAI instance", isinstance(model, ChatOpenAI))
    check("OpenAI base_url is left as the SDK default (None)", model.openai_api_base is None)

with env(LLM_PROVIDER="GLM", LLM_API_KEY="sk-test", LLM_MODEL="glm-x"):
    model = get_provider()
    check("LLM_PROVIDER=GLM returns a ChatOpenAI instance", isinstance(model, ChatOpenAI))
    check(
        "GLM defaults to the NVIDIA base_url when LLM_BASE_URL is unset",
        model.openai_api_base == "https://integrate.api.nvidia.com/v1",
        model.openai_api_base,
    )

with env(
    LLM_PROVIDER="ZAI",
    LLM_API_KEY="sk-test",
    LLM_MODEL="zai-x",
    LLM_BASE_URL="https://custom.example.com/v1",
):
    model = get_provider()
    check(
        "LLM_BASE_URL overrides ZAI's default base_url",
        model.openai_api_base == "https://custom.example.com/v1",
        model.openai_api_base,
    )

# ---------------------------------------------------------------------
# Gemini + grounding
# ---------------------------------------------------------------------

with env(LLM_PROVIDER="GEMINI", LLM_API_KEY="fake-key", LLM_MODEL="gemini-2.0-flash"):
    check("is_grounded() is False when LLM_GROUNDING is unset", is_grounded() is False)
    model = get_provider(allow_grounding=True)
    check(
        "get_provider(allow_grounding=True) with no LLM_GROUNDING returns a plain ChatGoogleGenerativeAI",
        isinstance(model, ChatGoogleGenerativeAI),
    )

with env(
    LLM_PROVIDER="GEMINI",
    LLM_API_KEY="fake-key",
    LLM_MODEL="gemini-2.0-flash",
    LLM_GROUNDING="true",
):
    check(
        "is_grounded() is True when LLM_PROVIDER=GEMINI and LLM_GROUNDING=true",
        is_grounded() is True,
    )

    grounded_model = get_provider(allow_grounding=True)
    check(
        "get_provider(allow_grounding=True) with LLM_GROUNDING=true binds the search tool",
        isinstance(grounded_model, RunnableBinding)
        and not isinstance(grounded_model, ChatGoogleGenerativeAI),
    )
    bound_tools = grounded_model.kwargs.get("tools", [])
    check(
        "The bound tool is google_search",
        len(bound_tools) == 1 and bound_tools[0].get("google_search") is not None,
        bound_tools,
    )

    not_opted_in_model = get_provider(allow_grounding=False)
    check(
        "get_provider(allow_grounding=False) never binds the search tool, even with LLM_GROUNDING=true",
        isinstance(not_opted_in_model, ChatGoogleGenerativeAI),
    )

with env(LLM_PROVIDER="OPENAI", LLM_API_KEY="sk-test", LLM_MODEL="gpt-4o-mini", LLM_GROUNDING="true"):
    check(
        "is_grounded() is False for non-Gemini providers even with LLM_GROUNDING=true",
        is_grounded() is False,
    )


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

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_llm_provider.py`
Expected: `ImportError: cannot import name 'is_grounded' from 'llm_provider'` (the old `llm_provider.py` has no `is_grounded`, and `get_provider()` returns a custom `LLMProvider` object, not a LangChain `BaseChatModel`).

- [ ] **Step 3: Rewrite `llm_provider.py`**

Replace the full file contents with:

```python
"""
llm_provider.py

Provider abstraction layer for the Company Discovery Agent, built on
LangChain chat models.

get_provider() returns a configured LangChain BaseChatModel (ChatOpenAI or
ChatGoogleGenerativeAI). Callers build their own ChatPromptTemplate and
call `.with_structured_output(Schema)` themselves -- this module's only
job is picking and configuring the right underlying chat model.

Currently supports:
    - OpenAI                 (ChatOpenAI)
    - NVIDIA GLM / Z.ai       (ChatOpenAI, OpenAI-compatible endpoint)
    - Google Gemini          (ChatGoogleGenerativeAI)

Switching providers requires only changing environment variables.

GEMINI SEARCH GROUNDING:

discovery.py is the one caller in this project that needs Gemini's
google_search grounding tool (real-time web recall beyond the model's
training data, needed to discover companies at all). Every other LLM call
in this project needs with_structured_output, and Gemini's API does not
allow combining the google_search tool with a structured-output schema
constraint in the same request (verified against langchain-google-genai's
with_structured_output implementation and its own documented example of
binding google_search). So grounding is opt-in per call
(allow_grounding=True) rather than a blanket per-provider setting, and
is_grounded() lets a caller check, before invoking, whether it must take a
manual-parse fallback instead of with_structured_output (see discovery.py).
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class LLMProviderError(Exception):
    """Raised when an LLM provider is misconfigured."""


def _grounding_requested() -> bool:
    """True iff LLM_PROVIDER=GEMINI and LLM_GROUNDING=true are both set."""
    provider = os.getenv("LLM_PROVIDER", "OPENAI").strip().upper()
    grounding_flag = os.getenv("LLM_GROUNDING", "false").strip().lower() == "true"
    return provider == "GEMINI" and grounding_flag


def is_grounded() -> bool:
    """
    True iff get_provider(allow_grounding=True) will return a model with
    Google Search grounding bound. Lets a caller decide, without
    duplicating provider/env-var selection logic, whether it must take a
    manual-parse fallback instead of with_structured_output (Gemini
    rejects combining the google_search tool with a structured-output
    schema constraint in the same request).
    """
    return _grounding_requested()


def get_provider(*, allow_grounding: bool = False) -> BaseChatModel:
    """
    Reads environment variables and returns a configured LangChain chat model.

    Required:

        LLM_PROVIDER
        LLM_API_KEY
        LLM_MODEL

    Optional:

        LLM_BASE_URL   (OpenAI-compatible providers only)
        LLM_GROUNDING  ("true" / "false", default "false")
                       Only takes effect when LLM_PROVIDER=GEMINI AND the
                       caller passes allow_grounding=True.

    Args:
        allow_grounding: Whether this call site is willing to receive a
            model with Gemini's google_search grounding tool bound (see
            module docstring). Callers that need with_structured_output on
            every call (e.g. evidence_extractor.py) must leave this False.
    """
    provider = os.getenv("LLM_PROVIDER", "OPENAI").strip().upper()
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")

    if not api_key:
        raise LLMProviderError("LLM_API_KEY is not configured.")
    if not model:
        raise LLMProviderError("LLM_MODEL is not configured.")

    if provider == "GEMINI":
        llm: BaseChatModel = ChatGoogleGenerativeAI(
            model=model, api_key=api_key, temperature=0.3
        )
        if allow_grounding and _grounding_requested():
            llm = llm.bind_tools([{"google_search": {}}])
        return llm

    base_url = os.getenv("LLM_BASE_URL", "").strip()

    if provider == "OPENAI":
        base_url = None
    elif provider == "GLM":
        if not base_url:
            base_url = "https://integrate.api.nvidia.com/v1"
    elif provider == "ZAI":
        if not base_url:
            base_url = "https://api.z.ai/api/paas/v4"
    else:
        raise LLMProviderError(f"Unsupported LLM_PROVIDER: {provider}")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0.3,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_llm_provider.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add llm_provider.py test_llm_provider.py
git commit -m "Rewrite llm_provider.py on LangChain chat models"
```

---

### Task 3: Wrap prompts in `ChatPromptTemplate`

**Files:**
- Modify: `prompts.py`
- Test: `test_prompts_chat_templates.py`

**Interfaces:**
- Produces: `DISCOVERY_PROMPT: ChatPromptTemplate`, `EVIDENCE_PROMPT: ChatPromptTemplate` (both take a `user_prompt` variable via `.format_messages(user_prompt=...)`). `discovery.py` (Task 4) uses `DISCOVERY_PROMPT`; `evidence_extractor.py` (Task 5) uses `EVIDENCE_PROMPT`.
- `SYSTEM_PROMPT`, `EVIDENCE_SYSTEM_PROMPT`, `build_user_prompt()`, `build_evidence_prompt()` are unchanged — no existing export is removed or altered.

- [ ] **Step 1: Write the failing test**

Create `test_prompts_chat_templates.py`:

```python
"""
test_prompts_chat_templates.py

Unit tests confirming DISCOVERY_PROMPT / EVIDENCE_PROMPT format correctly,
in particular that literal curly braces (the JSON example in each system
prompt's OUTPUT FORMAT section, and potentially arbitrary braces in
scraped webpage text substituted into the human turn) are never mistaken
for template variables.

Run with: python3 test_prompts_chat_templates.py
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from prompts import DISCOVERY_PROMPT, EVIDENCE_PROMPT, EVIDENCE_SYSTEM_PROMPT, SYSTEM_PROMPT

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


discovery_messages = DISCOVERY_PROMPT.format_messages(user_prompt="Find pharma companies in Powai")

check("DISCOVERY_PROMPT produces exactly 2 messages", len(discovery_messages) == 2)
check("DISCOVERY_PROMPT's first message is a SystemMessage", isinstance(discovery_messages[0], SystemMessage))
check(
    "DISCOVERY_PROMPT's system message content is SYSTEM_PROMPT verbatim, braces included",
    discovery_messages[0].content == SYSTEM_PROMPT,
)
check("DISCOVERY_PROMPT's second message is a HumanMessage", isinstance(discovery_messages[1], HumanMessage))
check(
    "DISCOVERY_PROMPT's human message content is exactly the supplied user_prompt",
    discovery_messages[1].content == "Find pharma companies in Powai",
)

brace_heavy_query = 'Find companies similar to "{Acme}" with turnover {>100cr}'
discovery_messages_with_braces = DISCOVERY_PROMPT.format_messages(user_prompt=brace_heavy_query)
check(
    "A user_prompt containing literal braces passes through DISCOVERY_PROMPT unchanged",
    discovery_messages_with_braces[1].content == brace_heavy_query,
)

evidence_messages = EVIDENCE_PROMPT.format_messages(user_prompt="Document text with a { curly brace } in it")

check("EVIDENCE_PROMPT produces exactly 2 messages", len(evidence_messages) == 2)
check(
    "EVIDENCE_PROMPT's system message content is EVIDENCE_SYSTEM_PROMPT verbatim",
    evidence_messages[0].content == EVIDENCE_SYSTEM_PROMPT,
)
check(
    "EVIDENCE_PROMPT's human message passes through literal braces unchanged",
    evidence_messages[1].content == "Document text with a { curly brace } in it",
)


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

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_prompts_chat_templates.py`
Expected: `ImportError: cannot import name 'DISCOVERY_PROMPT' from 'prompts'`

- [ ] **Step 3: Add the `ChatPromptTemplate` imports and wrapping to `prompts.py`**

In `prompts.py`, replace the file's opening docstring block:

```python
"""
prompts.py

Holds the system prompt and prompt-building helpers used to instruct the LLM
for the Company Discovery Agent. Keeping prompts in one place makes it easy
to tune behavior without touching application logic.
"""

# ---------------------------------------------------------------------------
# Rule 2 variants
```

with:

```python
"""
prompts.py

Holds the system prompt and prompt-building helpers used to instruct the LLM
for the Company Discovery Agent. Keeping prompts in one place makes it easy
to tune behavior without touching application logic.

DISCOVERY_PROMPT / EVIDENCE_PROMPT wrap SYSTEM_PROMPT / EVIDENCE_SYSTEM_PROMPT
into LangChain ChatPromptTemplates for discovery.py / evidence_extractor.py
to call .format_messages(user_prompt=...) on. The prompt text itself is
unchanged from before the LangChain migration.
"""

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Rule 2 variants
```

Then, replace:

```python
def build_user_prompt(query: str) -> str:
    """Build the user-turn prompt sent to the LLM for a given discovery query."""
    return (
        f"User request: {query}\n\n"
        "Identify real companies that best satisfy this request and respond "
        "using only the JSON array format described in your instructions."
    )


EVIDENCE_SYSTEM_PROMPT = """You are an Evidence Extraction Analyst for the Company
```

with:

```python
def build_user_prompt(query: str) -> str:
    """Build the user-turn prompt sent to the LLM for a given discovery query."""
    return (
        f"User request: {query}\n\n"
        "Identify real companies that best satisfy this request and respond "
        "using only the JSON array format described in your instructions."
    )


# SYSTEM_PROMPT is wrapped in a SystemMessage (not a ("system", SYSTEM_PROMPT)
# tuple) so ChatPromptTemplate never treats its literal curly braces (the
# JSON example in Rule 9's OUTPUT FORMAT section) as template variables --
# SystemMessage content is passed through unparsed. Only the human turn
# ("{user_prompt}") is templated, and a template consisting of nothing but
# one placeholder never re-parses whatever string ends up substituted into
# it, however many literal braces that string itself contains.
DISCOVERY_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=SYSTEM_PROMPT), ("human", "{user_prompt}")]
)


EVIDENCE_SYSTEM_PROMPT = """You are an Evidence Extraction Analyst for the Company
```

Finally, append to the very end of the file (after the existing `build_evidence_prompt` function's closing `)`):

```python


EVIDENCE_PROMPT = ChatPromptTemplate.from_messages(
    [SystemMessage(content=EVIDENCE_SYSTEM_PROMPT), ("human", "{user_prompt}")]
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_prompts_chat_templates.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add prompts.py test_prompts_chat_templates.py
git commit -m "Wrap discovery/evidence prompts in ChatPromptTemplate"
```

---

### Task 4: Migrate `discovery.py` to structured output (+ grounded fallback)

**Files:**
- Modify: `discovery.py`
- Test: `test_discovery_structured_output.py`

**Interfaces:**
- Consumes: `get_provider(*, allow_grounding=False) -> BaseChatModel`, `is_grounded() -> bool`, `LLMProviderError` (from `llm_provider.py`, Task 2); `LLMCompanyEntry`, `LLMConstraintEvaluation`, `LLMDiscoveryResponse` (from `llm_schemas.py`, Task 1); `DISCOVERY_PROMPT` (from `prompts.py`, Task 3).
- Produces: `discover(query, history=None) -> DiscoveryResult` — **signature and return type unchanged**; `CompanyResult`, `RejectedCompany`, `DiscoveryResult`, `ConstraintEvaluation`, `DiscoveryError` all keep their current shape (defined earlier in this same file, untouched by this task).

- [ ] **Step 1: Write the failing test**

Create `test_discovery_structured_output.py`:

```python
"""
test_discovery_structured_output.py

Unit tests for discovery.py's LangChain-based LLM call, covering both
branches: the plain with_structured_output path, and the grounded
fallback path used when Gemini Search grounding is active (which cannot
be combined with with_structured_output -- see llm_provider.py). No
network calls are made -- discovery.get_provider, discovery.is_grounded,
discovery.get_company_identifiers, discovery.retriever.retrieve_for_evidence,
and discovery.extract_evidence are all monkeypatched.

Run with: python3 test_discovery_structured_output.py
"""

from __future__ import annotations

from types import SimpleNamespace

import discovery
from identifier_lookup import IdentifierRecord
from llm_schemas import LLMCompanyEntry, LLMConstraintEvaluation, LLMDiscoveryResponse

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


def _stub_downstream_pipeline() -> None:
    """
    Stubs everything discover() calls after parsing the LLM response, so
    these tests only exercise the LLM-calling/parsing change itself.
    """
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
    discovery.retriever.retrieve_for_evidence = lambda name: []
    discovery.extract_evidence = lambda *args, **kwargs: []


class _FakeStructuredRunnable:
    def __init__(self, response_or_exception):
        self._response_or_exception = response_or_exception

    def invoke(self, messages):
        if isinstance(self._response_or_exception, Exception):
            raise self._response_or_exception
        return self._response_or_exception


class _FakeChatModel:
    def __init__(self, structured_response=None, raw_text=None):
        self._structured_response = structured_response
        self._raw_text = raw_text

    def with_structured_output(self, schema):
        return _FakeStructuredRunnable(self._structured_response)

    def invoke(self, messages):
        return SimpleNamespace(text=self._raw_text)


# ---------------------------------------------------------------------
# Non-grounded branch (with_structured_output)
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: False
discovery.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=LLMDiscoveryResponse(
        companies=[
            LLMCompanyEntry(
                company_name="Acme Forgings Private Limited",
                constraint_evaluation=[
                    LLMConstraintEvaluation(name="location", status="PASS", reason="HQ in Thane.")
                ],
                decision="ACCEPT",
                reason="Matches all constraints.",
                confidence="High",
            ),
            LLMCompanyEntry(
                company_name="Bad Corp Limited",
                constraint_evaluation=[
                    LLMConstraintEvaluation(name="location", status="FAIL", reason="HQ in Pune, not Thane.")
                ],
                decision="REJECT",
                reason="Location fails.",
                confidence="Medium",
            ),
        ]
    )
)

result = discovery.discover("Find manufacturing companies in Thane")

check("Non-grounded: one company accepted", len(result.accepted) == 1, len(result.accepted))
check(
    "Non-grounded: accepted company name matches",
    bool(result.accepted) and result.accepted[0].company_name == "Acme Forgings Private Limited",
)
check(
    "Non-grounded: accepted company's constraint_evaluation dict is rebuilt correctly",
    bool(result.accepted)
    and result.accepted[0].constraint_evaluation.get("location") is not None
    and result.accepted[0].constraint_evaluation["location"].status == "PASS",
)
check("Non-grounded: one LLM rejection recorded", len(result.rejected) == 1, len(result.rejected))
check(
    "Non-grounded: rejected company is tagged rejection_type='llm'",
    bool(result.rejected) and result.rejected[0].rejection_type == "llm",
)

# ---------------------------------------------------------------------
# Non-grounded branch: invoke() failure surfaces as DiscoveryError
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: False
discovery.get_provider = lambda **kwargs: _FakeChatModel(
    structured_response=RuntimeError("simulated provider failure")
)

try:
    discovery.discover("Find manufacturing companies in Thane")
    check("Non-grounded: provider failure raises DiscoveryError", False, "no exception raised")
except discovery.DiscoveryError:
    check("Non-grounded: provider failure raises DiscoveryError", True)

# ---------------------------------------------------------------------
# Grounded fallback branch (manual JSON parse + Pydantic validation)
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: True
discovery.get_provider = lambda **kwargs: _FakeChatModel(
    raw_text="""
    [
      {
        "company_name": "Acme Forgings Private Limited",
        "constraint_evaluation": {
          "location": {"status": "PASS", "reason": "HQ in Thane."}
        },
        "decision": "ACCEPT",
        "reason": "Matches all constraints.",
        "confidence": "High"
      }
    ]
    """
)

grounded_result = discovery.discover("Find manufacturing companies in Thane")

check(
    "Grounded: dict-shaped constraint_evaluation from the raw prompt path is accepted",
    len(grounded_result.accepted) == 1,
    len(grounded_result.accepted),
)
check(
    "Grounded: accepted company's constraint_evaluation dict is rebuilt correctly",
    bool(grounded_result.accepted)
    and grounded_result.accepted[0].constraint_evaluation.get("location") is not None
    and grounded_result.accepted[0].constraint_evaluation["location"].status == "PASS",
)

# ---------------------------------------------------------------------
# Grounded fallback branch: malformed JSON surfaces as DiscoveryError
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: True
discovery.get_provider = lambda **kwargs: _FakeChatModel(raw_text="not valid json at all")

try:
    discovery.discover("Find manufacturing companies in Thane")
    check("Grounded: malformed JSON raises DiscoveryError", False, "no exception raised")
except discovery.DiscoveryError:
    check("Grounded: malformed JSON raises DiscoveryError", True)

# ---------------------------------------------------------------------
# Grounded fallback branch: schema-violating entry surfaces as DiscoveryError
# ---------------------------------------------------------------------

_stub_downstream_pipeline()
discovery.is_grounded = lambda: True
discovery.get_provider = lambda **kwargs: _FakeChatModel(
    raw_text='[{"company_name": "Acme Ltd", "decision": "MAYBE", "reason": "x", "confidence": "High"}]'
)

try:
    discovery.discover("Find manufacturing companies in Thane")
    check("Grounded: schema-violating entry raises DiscoveryError", False, "no exception raised")
except discovery.DiscoveryError:
    check("Grounded: schema-violating entry raises DiscoveryError", True)


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

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_discovery_structured_output.py`
Expected: `AttributeError: module 'discovery' has no attribute 'is_grounded'` (or an `ImportError` while `discovery.py` still tries to import `SYSTEM_PROMPT`/`extract_json_array` in the old way — either way, a clear failure, not a pass).

- [ ] **Step 3: Update `discovery.py`'s imports**

Replace:

```python
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import retriever
from evidence_extractor import EvidenceItem, extract as extract_evidence
from identifier_lookup import IdentifierRecord, get_company_identifiers
from json_utils import JsonArrayParseError, extract_json_array
from llm_provider import LLMProviderError, get_provider
from prompts import SYSTEM_PROMPT, build_user_prompt
```

with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic import ValidationError

import retriever
from evidence_extractor import EvidenceItem, extract as extract_evidence
from identifier_lookup import IdentifierRecord, get_company_identifiers
from json_utils import JsonArrayParseError, extract_json_array
from llm_provider import LLMProviderError, get_provider, is_grounded
from llm_schemas import LLMCompanyEntry, LLMConstraintEvaluation, LLMDiscoveryResponse
from prompts import DISCOVERY_PROMPT, build_user_prompt
```

(`sys` is dropped — it was only used by the per-entry "skipping malformed entry" warning print, which Step 4 below removes, since Pydantic validation now handles that at the whole-response level.)

- [ ] **Step 4: Replace `_validate_constraint_evaluation` / `_validate_company_entry`**

Replace:

```python
def _validate_constraint_evaluation(raw) -> Dict[str, ConstraintEvaluation]:
    """
    Validates and normalizes the "constraint_evaluation" object from the LLM
    response into a dict of constraint_name -> ConstraintEvaluation.

    Defensive by design: malformed or missing input simply yields an empty
    dict rather than raising, since this is debug metadata and should never
    take down a discovery request.
    """
    if not isinstance(raw, dict):
        return {}

    normalized: Dict[str, ConstraintEvaluation] = {}
    for constraint_name, value in raw.items():
        constraint_name = str(constraint_name).strip()
        if not constraint_name or not isinstance(value, dict):
            continue
        status = str(value.get("status", "")).strip().upper()
        if status not in {"PASS", "FAIL", "UNKNOWN"}:
            status = "UNKNOWN"  # Safe default if the model returns something unexpected.
        reason = str(value.get("reason", "")).strip()
        normalized[constraint_name] = ConstraintEvaluation(status=status, reason=reason)

    return normalized


def _validate_company_entry(entry: dict) -> dict:
    """Validates and normalizes a single company entry from the LLM response."""
    name = str(entry.get("company_name", "")).strip()
    reason = str(entry.get("reason", "")).strip()
    confidence = str(entry.get("confidence", "")).strip().title()

    if not name:
        raise ValueError(f"Company entry missing 'company_name': {entry}")
    if confidence not in {"High", "Medium", "Low"}:
        confidence = "Low"  # Safe default if the model returns something unexpected.

    decision = str(entry.get("decision", "ACCEPT")).strip().upper()
    if decision not in {"ACCEPT", "REJECT"}:
        decision = "ACCEPT"  # Safe default; this is debug metadata, not a filter.

    constraint_evaluation = _validate_constraint_evaluation(entry.get("constraint_evaluation", {}))

    return {
        "company_name": name,
        "reason": reason,
        "confidence": confidence,
        "decision": decision,
        "constraint_evaluation": constraint_evaluation,
    }
```

with:

```python
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
```

- [ ] **Step 5: Replace the LLM-calling section of `discover()`**

Replace:

```python
    if not query or not query.strip():
        raise DiscoveryError("Query cannot be empty.")

    try:
        provider = get_provider()
        raw_response = provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_prompt(query, history),
        )
    except LLMProviderError as exc:
        raise DiscoveryError(f"LLM provider error: {exc}") from exc

    try:
        parsed_entries = extract_json_array(raw_response)
    except JsonArrayParseError as exc:
        raise DiscoveryError(str(exc)) from exc

    validated_entries = []
    for entry in parsed_entries:
        try:
            validated_entries.append(_validate_company_entry(entry))
        except ValueError as exc:
            print(f"Warning: skipping malformed entry ({exc})", file=sys.stderr)

    # Split by the LLM's explicit decision. ACCEPT entries are candidates
    # for recommendation and go through identifier lookup + verification
    # below -- but verification, not the LLM, has the final say on whether
    # any of them actually end up in `accepted` (see the loop below). REJECT
    # entries are the LLM's own rejections and never reach identifier
    # lookup or evidence retrieval at all.
    accepted_entries = [e for e in validated_entries if e["decision"] == "ACCEPT"]
    llm_rejected_entries = [e for e in validated_entries if e["decision"] == "REJECT"][:_MAX_REJECTED_CANDIDATES]
```

with:

```python
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
            response = llm.with_structured_output(LLMDiscoveryResponse).invoke(prompt_messages)
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
```

Nothing below this point in `discover()` (identifier lookup, verification gating, evidence retrieval, `CompanyResult`/`RejectedCompany` assembly, the final `return DiscoveryResult(...)`) changes.

- [ ] **Step 6: Run the test to verify it passes**

Run: `python3 test_discovery_structured_output.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 7: Commit**

```bash
git add discovery.py test_discovery_structured_output.py
git commit -m "Migrate discovery.py to LangChain structured output with grounded fallback"
```

---

### Task 5: Migrate `evidence_extractor.py` to structured output

**Files:**
- Modify: `evidence_extractor.py`
- Test: `test_evidence_extractor_structured_output.py`

**Interfaces:**
- Consumes: `get_provider() -> BaseChatModel` (Task 2, called with no arguments — evidence extraction never requests grounding); `LLMEvidenceItem`, `LLMEvidenceResponse` (Task 1); `EVIDENCE_PROMPT` (Task 3).
- Produces: `extract(company_name, user_query, discovery_reason, documents) -> List[EvidenceItem]` — **signature and return type unchanged**; `EvidenceItem` dataclass unchanged.

- [ ] **Step 1: Write the failing test**

Create `test_evidence_extractor_structured_output.py`:

```python
"""
test_evidence_extractor_structured_output.py

Unit tests for evidence_extractor.py's LangChain-based structured-output
call. No network calls are made -- evidence_extractor.get_provider is
monkeypatched.

Run with: python3 test_evidence_extractor_structured_output.py
"""

from __future__ import annotations

import evidence_extractor
from llm_schemas import LLMEvidenceItem, LLMEvidenceResponse
from retriever import Document

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


class _FakeStructuredRunnable:
    def __init__(self, response_or_exception):
        self._response_or_exception = response_or_exception

    def invoke(self, messages):
        if isinstance(self._response_or_exception, Exception):
            raise self._response_or_exception
        return self._response_or_exception


class _FakeChatModel:
    def __init__(self, structured_response):
        self._structured_response = structured_response

    def with_structured_output(self, schema):
        return _FakeStructuredRunnable(self._structured_response)


documents = [
    Document(url="https://example.com/about", title="About Acme", cleaned_text="Acme makes forgings."),
]

# ---------------------------------------------------------------------
# Valid response, one entry citing a real document URL, one hallucinated
# ---------------------------------------------------------------------

evidence_extractor.get_provider = lambda **kwargs: _FakeChatModel(
    LLMEvidenceResponse(
        items=[
            LLMEvidenceItem(
                point="Manufactures precision forgings",
                source_title="Official Website",
                source_url="https://example.com/about",
            ),
            LLMEvidenceItem(
                point="Hallucinated fact",
                source_title="Nowhere",
                source_url="https://not-supplied.example.com/",
            ),
        ]
    )
)

result = evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", documents)

check("Only the entry citing a supplied URL survives", len(result) == 1, len(result))
check(
    "The surviving entry's fields match the supplied document's URL",
    bool(result) and result[0].source_url == "https://example.com/about",
)

# ---------------------------------------------------------------------
# No documents -> empty list, no LLM call attempted
# ---------------------------------------------------------------------


def _fail_if_called(**kwargs):
    raise AssertionError("get_provider() should not be called when there are no documents")


evidence_extractor.get_provider = _fail_if_called
check(
    "extract() with no documents returns [] without calling the LLM",
    evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", []) == [],
)

# ---------------------------------------------------------------------
# Provider/invoke failure degrades to an empty list, not an exception
# ---------------------------------------------------------------------

evidence_extractor.get_provider = lambda **kwargs: _FakeChatModel(RuntimeError("simulated failure"))
check(
    "A provider/invoke failure returns [] rather than raising",
    evidence_extractor.extract("Acme Forgings", "find forging companies", "fits", documents) == [],
)


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

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 test_evidence_extractor_structured_output.py`
Expected: `ImportError` or `AttributeError` — the old `evidence_extractor.py` calls `provider.generate(...)`, not `.with_structured_output(...)`, so `_FakeChatModel` (which only implements `with_structured_output`) won't satisfy it.

- [ ] **Step 3: Rewrite `evidence_extractor.py`**

Replace the full file contents with:

```python
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

from dataclasses import dataclass
from typing import List

from llm_provider import get_provider
from llm_schemas import LLMEvidenceItem, LLMEvidenceResponse
from prompts import EVIDENCE_PROMPT, build_evidence_prompt
from retriever import Document


@dataclass(frozen=True)
class EvidenceItem:
    """One sourced, LLM-summarized fact about a company."""
    point: str
    source_title: str
    source_url: str


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
        response = llm.with_structured_output(LLMEvidenceResponse).invoke(prompt_messages)
    except Exception:
        return []

    evidence: List[EvidenceItem] = []
    for entry in response.items:
        item = _validate_evidence_entry(entry, valid_urls)
        if item is not None:
            evidence.append(item)

    return evidence
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 test_evidence_extractor_structured_output.py`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add evidence_extractor.py test_evidence_extractor_structured_output.py
git commit -m "Migrate evidence_extractor.py to LangChain structured output"
```

---

### Task 6: Full regression pass and manual smoke test

**Files:**
- None created or modified — this task only runs verification.

**Interfaces:**
- Consumes: every test file from Tasks 1-5, plus the pre-existing `test_entity_matching.py`.

- [ ] **Step 1: Run every automated test script**

Run:
```bash
python3 test_llm_schemas.py && \
python3 test_llm_provider.py && \
python3 test_prompts_chat_templates.py && \
python3 test_discovery_structured_output.py && \
python3 test_evidence_extractor_structured_output.py && \
python3 test_entity_matching.py
```
Expected: every script prints `ALL TESTS PASSED`; the whole chain exits 0. (`test_gst.py` is interactive — `input()`-driven — and isn't exercised here; it's untouched by this migration.)

- [ ] **Step 2: Confirm no stale references to the removed provider classes/flags**

Run:
```bash
grep -rn "provider\.generate(\|OpenAICompatibleProvider\|GeminiProvider\|LLM_WEB_SEARCH\|enable_web_search\|enable_search_grounding" --include="*.py" .
```
Expected: no output (empty). If anything matches, it's leftover code from before this migration that needs updating to match the new `get_provider()`/`is_grounded()` interface from Task 2.

- [ ] **Step 3: Confirm `json_utils.py` is still imported exactly where expected**

Run:
```bash
grep -rln "json_utils" --include="*.py" .
```
Expected output: `discovery.py` and `json_utils.py` itself (its own docstring/self-reference) — `evidence_extractor.py` should **not** appear, since Task 5 removed its `json_utils` import.

- [ ] **Step 4: Manual smoke test with real credentials**

This step needs real API keys and can't be scripted here. With a working `.env` (or exported env vars) for at least one provider:

```bash
streamlit run ui.py
```

Verify by hand:
- A plain-language query (e.g. "Find manufacturing companies in Mumbai with turnover between 20 and 100 crore") returns company cards exactly as it did before this migration — names, confidence badges, GST/CIN lines, evidence bullets.
- The "Show Debug Information" sidebar toggle still expands each card with constraint evaluation, decision badge, and the "Rejected Candidates" panel.
- If you have `LLM_PROVIDER=GEMINI` and `LLM_GROUNDING=true` configured, run one query and confirm discovery still succeeds (this exercises the grounded fallback path from Task 4, which no automated test can invoke against a real model).

- [ ] **Step 5: Final commit (only if Step 4 surfaced any fixes)**

If the manual smoke test in Step 4 required any code changes, commit them now with a message describing what was fixed. If everything worked as-is, there is nothing to commit for this task.
