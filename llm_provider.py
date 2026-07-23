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


def structured_output_kwargs() -> dict:
    """
    Optional override for with_structured_output()'s method parameter, via
    LLM_STRUCTURED_OUTPUT_METHOD ("function_calling" | "json_mode" |
    "json_schema"). Both ChatOpenAI and ChatGoogleGenerativeAI default
    with_structured_output() to method="json_schema", which requires the
    endpoint to implement OpenAI's/Google's native structured-output
    protocol. Native OpenAI and Gemini support this well; NVIDIA GLM and
    Z.ai's OpenAI-compatible endpoints may not, since json_schema-based
    structured outputs are a newer, less universally-implemented protocol
    extension than plain tool/function calling. Returns {} (i.e. use the
    library's own default) when unset, so behavior is unchanged unless a
    deployment explicitly opts into overriding it.
    """
    method = os.getenv("LLM_STRUCTURED_OUTPUT_METHOD", "").strip()
    return {"method": method} if method else {}


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
