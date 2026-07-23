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

from llm_provider import LLMProviderError, get_provider, is_grounded, structured_output_kwargs

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(label)


_ENV_KEYS = ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_GROUNDING", "LLM_STRUCTURED_OUTPUT_METHOD")


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

# ---------------------------------------------------------------------
# structured_output_kwargs()
# ---------------------------------------------------------------------

with env():
    check(
        "structured_output_kwargs() is empty when LLM_STRUCTURED_OUTPUT_METHOD is unset",
        structured_output_kwargs() == {},
    )

with env(LLM_STRUCTURED_OUTPUT_METHOD="function_calling"):
    check(
        "structured_output_kwargs() returns the override when set",
        structured_output_kwargs() == {"method": "function_calling"},
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
