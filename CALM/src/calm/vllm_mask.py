from __future__ import annotations

from collections.abc import Sequence

import torch

from calm.lexicon import completing_token_ids


def completed_atp_count(
    token_ids: Sequence[int],
    sequences: Sequence[Sequence[int]],
    start: int,
) -> int:
    ends: set[int] = set()
    size = len(token_ids)
    for sequence_value in sequences:
        sequence = tuple(int(token_id) for token_id in sequence_value)
        width = len(sequence)
        if width <= 0 or width > size:
            continue
        for begin in range(size - width + 1):
            end = begin + width - 1
            if end < start:
                continue
            if tuple(token_ids[begin : begin + width]) == sequence:
                ends.add(end)
    return len(ends)


def apply_calm_mask(
    logits: torch.Tensor,
    history: Sequence[int],
    sequences: Sequence[Sequence[int]],
    mode: str,
    lock_index: int,
    rho: float,
    peak: float,
) -> torch.Tensor:
    token_ids = completing_token_ids(history, sequences)
    if not token_ids:
        return logits
    index = list(token_ids)
    if mode == "hard":
        logits[index] = float("-inf")
        return logits
    if mode != "soft":
        raise ValueError(f"unknown CALM mode {mode!r}")
    count = completed_atp_count(history, sequences, lock_index)
    if count == 0:
        return logits
    logits[index] += -float(peak) * (1.0 - float(rho) ** count)
    return logits


def calm_logits_processor_class():
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

    class CalmMaskLogitsProcessor(AdapterLogitsProcessor):
        @classmethod
        def validate_params(cls, params: SamplingParams):
            extra = params.extra_args or {}
            mode = extra.get("calm_mode")
            if mode is None:
                return
            if mode not in {"hard", "soft"}:
                raise ValueError(f"unknown calm_mode {mode!r}")
            lock_index = extra.get("calm_lock")
            sequences = extra.get("calm_sequences")
            if isinstance(lock_index, bool) or not isinstance(lock_index, int) or lock_index < 0:
                raise ValueError(f"calm_lock must be an int >= 0, got {lock_index!r}")
            if not isinstance(sequences, list) or not sequences:
                raise ValueError("calm_sequences must be a non-empty list")
            if mode == "soft":
                rho = extra.get("calm_rho")
                peak = extra.get("calm_peak")
                if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not 0 < float(rho) < 1:
                    raise ValueError(f"calm_rho must be in (0, 1), got {rho!r}")
                if isinstance(peak, bool) or not isinstance(peak, (int, float)) or float(peak) <= 0:
                    raise ValueError(f"calm_peak must be positive, got {peak!r}")

        def is_argmax_invariant(self) -> bool:
            return False

        def new_req_logits_processor(self, params: SamplingParams):
            extra = params.extra_args or {}
            mode = extra.get("calm_mode")
            if mode is None:
                return None
            sequences = extra["calm_sequences"]
            lock_index = int(extra["calm_lock"])
            rho = float(extra.get("calm_rho", 0.98))
            peak = float(extra.get("calm_peak", 10.0))

            def processor(prompt_ids, output_ids, logits):
                history = list(prompt_ids) + list(output_ids)
                return apply_calm_mask(
                    logits,
                    history,
                    sequences,
                    mode,
                    lock_index,
                    rho,
                    peak,
                )

            return processor

    return CalmMaskLogitsProcessor
