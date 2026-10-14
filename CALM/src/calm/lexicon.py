from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def atp_surfaces() -> tuple[str, ...]:
    base = [
        "Wait",
        "wait",
        "WAIT",
        "Alternatively",
        "alternatively",
        "ALTERNATIVELY",
    ]
    for count in range(1, 13):
        base.extend(("H" + "m" * count, "h" + "m" * count))
    return tuple(form for word in base for form in (word, " " + word))


def compile_token_sequences(
    tokenizer: Any,
    surfaces: Sequence[str] | None = None,
) -> tuple[tuple[int, ...], ...]:
    seen: set[tuple[int, ...]] = set()
    sequences: list[tuple[int, ...]] = []
    for surface in surfaces or atp_surfaces():
        token_ids = tuple(
            int(token_id) for token_id in tokenizer.encode(surface, add_special_tokens=False)
        )
        if token_ids and token_ids not in seen:
            seen.add(token_ids)
            sequences.append(token_ids)
    return tuple(sequences)


def completing_token_ids(
    history: Sequence[int],
    sequences: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    found: dict[int, None] = {}
    previous = tuple(int(token_id) for token_id in history)
    for sequence_value in sequences:
        sequence = tuple(int(token_id) for token_id in sequence_value)
        if not sequence:
            continue
        if len(sequence) == 1:
            found.setdefault(sequence[0], None)
            continue
        prefix = sequence[:-1]
        if len(prefix) <= len(previous) and previous[-len(prefix) :] == prefix:
            found.setdefault(sequence[-1], None)
    return tuple(found)


def completes_atp(
    history: Sequence[int],
    token_id: int,
    sequences: Sequence[Sequence[int]],
) -> bool:
    return int(token_id) in completing_token_ids(history, sequences)
