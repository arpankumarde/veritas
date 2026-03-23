"""Shared JSON extraction utilities for parsing LLM responses.

LLM outputs often contain JSON wrapped in markdown code blocks or surrounded
by explanatory text.  These helpers use a 3-strategy fallback to robustly
extract JSON arrays or objects:

1. Strip markdown fences, try direct ``json.loads``
2. Greedy regex (first ``[`` to last ``]``, or ``{`` to ``}``)
3. Balanced-bracket / brace depth tracking
"""

import json
import re


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code-block fences (```json ... ```) from *text*."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    return re.sub(r"```\s*$", "", cleaned).strip()


def _find_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced substring delimited by *open_ch* / *close_ch*."""
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_array(response: str) -> list | None:
    """Extract a JSON array from an LLM response string.

    Returns the parsed ``list`` on success, or ``None`` if no valid array is
    found.
    """
    if not response or not isinstance(response, str):
        return None

    cleaned = _strip_markdown_fences(response)

    # Strategy 1: direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: greedy regex
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: balanced brackets
    balanced = _find_balanced(cleaned, "[", "]")
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass

    return None


def parse_json_object(response: str) -> dict | None:
    """Extract a JSON object from an LLM response string.

    Returns the parsed ``dict`` on success, or ``None`` if no valid object is
    found.
    """
    if not response or not isinstance(response, str):
        return None

    cleaned = _strip_markdown_fences(response)

    # Strategy 1: direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: greedy regex
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: balanced braces
    balanced = _find_balanced(cleaned, "{", "}")
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass

    return None
