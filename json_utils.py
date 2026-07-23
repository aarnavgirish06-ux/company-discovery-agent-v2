"""
json_utils.py

Shared helper for parsing a JSON array out of raw LLM text output.

Both discovery.py (parsing the company-discovery response) and
evidence_extractor.py (parsing the evidence-extraction response) need to
turn "the LLM's raw text reply" into "a JSON array", including tolerating
markdown code fences and stray leading/trailing prose the model sometimes
adds despite being told not to. That parsing logic previously lived only
inside discovery.py; it's factored out here so it exists in exactly one
place instead of being duplicated as the project grows more LLM-calling
modules.
"""

from __future__ import annotations

import json
import re


class JsonArrayParseError(Exception):
    """Raised when a JSON array cannot be extracted from raw text."""


def extract_json_array(raw_text: str) -> list:
    """
    Extracts a JSON array from raw LLM response text.

    Handles two common deviations from "respond with only a JSON array":
    the array wrapped in a ```json ... ``` markdown fence, and the array
    preceded/followed by stray prose. Raises JsonArrayParseError if no
    valid JSON array can be recovered.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present.
    fence_match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # If there's leading/trailing prose, try to isolate the outermost array.
    if not text.startswith("["):
        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if array_match:
            text = array_match.group(0)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JsonArrayParseError(
            f"Could not parse LLM response as JSON: {exc}"
        ) from exc

    if not isinstance(parsed, list):
        raise JsonArrayParseError("Expected a JSON array from the LLM.")

    return parsed
