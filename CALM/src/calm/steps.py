from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")


@dataclass(frozen=True, slots=True)
class Paragraph:
    text: str
    snapshot: Any


@dataclass(frozen=True, slots=True)
class StepBoundary:
    step: int
    text: str
    snapshot: Any


@dataclass(frozen=True, slots=True)
class _Chunk:
    text: str
    enumerated: bool
    kind: str


class ParagraphStream:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []

    def feed(self, token_id: int) -> str | None:
        self.token_ids.append(int(token_id))
        text = self.text()
        if _PARAGRAPH_BREAK.search(text) is None:
            return None
        self.token_ids.clear()
        return text

    def text(self) -> str:
        if not self.token_ids:
            return ""
        return str(
            self.tokenizer.decode(
                self.token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def flush(self, remove: str | None = None) -> str:
        text = self.text()
        self.token_ids.clear()
        if remove and remove in text:
            text = text.split(remove, 1)[0]
        elif remove:
            for width in range(len(remove) - 1, 0, -1):
                if text.endswith(remove[:width]):
                    text = text[:-width]
                    break
        return text


class OnlineStepSegmenter:
    def __init__(self, min_chars: int = 200, max_chars: int = 1000) -> None:
        if min_chars < 1 or max_chars < min_chars:
            raise ValueError("invalid step character limits")
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.current_text = ""
        self.current_kind = ""
        self.current_snapshot: Any = None
        self.step_count = 0
        self.held: StepBoundary | None = None

    def add_paragraph(self, text: str, snapshot: Any) -> list[StepBoundary]:
        cleaned = text.strip()
        if not cleaned:
            return []
        chunk = _analyze(cleaned)
        if not self.current_text:
            self.current_text = chunk.text
            self.current_kind = chunk.kind
            self.current_snapshot = snapshot
            return self._release_if_stable()
        if self._should_merge(chunk):
            self.current_text = self.current_text + "\n\n" + chunk.text
            self.current_snapshot = snapshot
            return self._release_if_stable()
        closed = self._emit()
        self.current_text = chunk.text
        self.current_kind = chunk.kind
        self.current_snapshot = snapshot
        ready = self._take_held()
        if len(chunk.text) < self.min_chars * 0.5:
            self.held = closed
            return ready
        ready.append(closed)
        return ready

    def flush(self) -> list[StepBoundary]:
        if (
            self.current_text
            and self.held is not None
            and len(self.current_text) < self.min_chars * 0.5
        ):
            merged = StepBoundary(
                step=self.held.step,
                text=self.held.text + "\n\n" + self.current_text,
                snapshot=self.current_snapshot,
            )
            self.held = None
            self.current_text = ""
            self.current_kind = ""
            self.current_snapshot = None
            return [merged]
        ready = self._take_held()
        if self.current_text:
            ready.append(self._emit())
        return ready

    def _release_if_stable(self) -> list[StepBoundary]:
        if self.held is not None and len(self.current_text) >= self.min_chars * 0.5:
            return self._take_held()
        return []

    def _take_held(self) -> list[StepBoundary]:
        if self.held is None:
            return []
        ready = [self.held]
        self.held = None
        return ready

    def _emit(self) -> StepBoundary:
        self.step_count += 1
        boundary = StepBoundary(
            step=self.step_count,
            text=self.current_text,
            snapshot=self.current_snapshot,
        )
        self.current_text = ""
        self.current_kind = ""
        self.current_snapshot = None
        return boundary

    def _should_merge(self, chunk: _Chunk) -> bool:
        previous_length = len(self.current_text)
        incoming = len(chunk.text)
        if chunk.enumerated and previous_length >= self.min_chars:
            return False
        if chunk.kind == self.current_kind:
            if self.current_kind in {"doubt", "verify"}:
                return previous_length + incoming <= self.max_chars * 1.2
            if self.current_kind == "calculate":
                return (
                    previous_length < self.min_chars
                    and previous_length + incoming <= self.max_chars
                )
            return previous_length < self.min_chars * 1.5
        if (self.current_kind, chunk.kind) in {
            ("setup", "calculate"),
            ("calculate", "verify"),
            ("doubt", "doubt"),
            ("verify", "verify"),
        }:
            return previous_length + incoming <= self.max_chars
        if (self.current_kind, chunk.kind) in {
            ("doubt", "calculate"),
            ("verify", "conclude"),
            ("calculate", "doubt"),
            ("setup", "doubt"),
        }:
            return previous_length < self.min_chars * 0.5
        if previous_length < self.min_chars * 0.6:
            return True
        return (
            incoming < self.min_chars * 0.4
            and not chunk.enumerated
            and previous_length + incoming <= self.max_chars
        )


def _analyze(text: str) -> _Chunk:
    enumerated = bool(
        re.match(
            r"^\s*(?:(?:\d+[\.\)])\s+|step\s+\d+\b\s*[:.]?\s*|first\b|second\b|third\b|finally\b)",
            text,
            re.IGNORECASE,
        )
    )
    lowered = text.lower()
    if any(
        phrase in lowered[:100]
        for phrase in (
            "need to solve",
            "problem states",
            "given that",
            "first,",
            "the problem says",
            "let me understand",
        )
    ):
        kind = "setup"
    elif any(
        phrase in lowered
        for phrase in (
            "calculate",
            "compute",
            " = ",
            "divided by",
            "multiply",
            "adding",
            "subtract",
            "total",
            "sum",
        )
    ) and not any(phrase in lowered for phrase in ("let me", "should i", "wait")):
        kind = "calculate"
    elif any(
        phrase in lowered
        for phrase in (
            "wait",
            "but ",
            "however",
            "that doesn't make sense",
            "let me reconsider",
            "or maybe",
            "perhaps",
            "could it be",
            "i'm not sure",
            "ambiguous",
            "confusing",
            "mistake",
        )
    ):
        kind = "doubt"
    elif any(
        phrase in lowered
        for phrase in (
            "let me check",
            "double-check",
            "verify",
            "confirm",
            "seems correct",
            "looks right",
            "to make sure",
        )
    ):
        kind = "verify"
    elif any(
        phrase in lowered
        for phrase in (
            "therefore",
            "so the answer",
            "final answer",
            "in conclusion",
            "thus",
            "hence",
            "this gives us",
        )
    ):
        kind = "conclude"
    else:
        kind = "general"
    return _Chunk(text=text, enumerated=enumerated, kind=kind)
