from __future__ import annotations

import re
from typing import Any

_WHITESPACE = re.compile(r"\s+")
_FORMAT_WRAPPER = re.compile(r"\\(?:text|mathrm)\s*\{([^{}]*)\}")
_LEFT_RIGHT = re.compile(r"\\(?:left|right)(?![A-Za-z])")
_BOXED = re.compile(r"\\boxed(?![A-Za-z])\s*")


def matching_braced_content(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return None


def probe_completion(text: str) -> tuple[str, bool]:
    """Parse a probe completion whose suffix already ended at \\boxed.

    Returns (answer, done). Leading whitespace is allowed. The first non-space
    character must be the outer '{'. done is False only while that argument is
    still open. An empty pair or any other shape is done with an empty answer.
    """

    body = text.lstrip()
    if not body:
        return "", False
    if not body.startswith("{"):
        return "", True
    content = matching_braced_content(body)
    if content is None:
        return "", False
    return content, True


def boxed_close_keep_length(texts: list[str]) -> int | None:
    """Tokens to keep through the outer closing brace, or None while it is open.

    An empty pair does not stop. Nested braces do not stop. The token that
    produces the matching brace is kept, including when the whole pair is one token.
    """

    depth = 0
    opened_at: int | None = None
    content = False
    for index, text in enumerate(texts):
        for char in text:
            if char == "{":
                if depth == 0:
                    opened_at = index
                    content = False
                depth += 1
            elif char == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and opened_at is not None and opened_at <= index and content:
                    return index + 1
                if depth == 0:
                    opened_at = None
                    content = False
            elif depth > 0:
                content = True
    return None


def extract_last_boxed(text: str) -> str:
    for match in reversed(list(_BOXED.finditer(text))):
        rest = text[match.end() :]
        if not rest.startswith("{"):
            continue
        content = matching_braced_content(rest)
        if content:
            return content
    return ""


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = _WHITESPACE.sub("", str(value).strip().lower())
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    while True:
        updated = _FORMAT_WRAPPER.sub(r"\1", text)
        if updated == text:
            break
        text = updated
    return _LEFT_RIGHT.sub("", text)
