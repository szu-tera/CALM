from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from calm.lexicon import completes_atp, completing_token_ids


class LogitPolicy(Protocol):
    def apply(self, logits: torch.Tensor, history: Sequence[int]) -> torch.Tensor: ...

    def observe(self, history: Sequence[int], token_id: int) -> None: ...


class NoPolicy:
    def apply(self, logits: torch.Tensor, history: Sequence[int]) -> torch.Tensor:
        del history
        return logits

    def observe(self, history: Sequence[int], token_id: int) -> None:
        del history, token_id


class HardPolicy:
    def __init__(self, sequences: Sequence[Sequence[int]]) -> None:
        self.sequences = tuple(tuple(value) for value in sequences)

    def apply(self, logits: torch.Tensor, history: Sequence[int]) -> torch.Tensor:
        token_ids = completing_token_ids(history, self.sequences)
        if not token_ids:
            return logits
        adjusted = logits.clone()
        adjusted[list(token_ids)] = float("-inf")
        return adjusted

    def observe(self, history: Sequence[int], token_id: int) -> None:
        del history, token_id


class SoftPolicy:
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        rho: float = 0.98,
        peak: float = 10.0,
    ) -> None:
        if not 0.0 < rho < 1.0:
            raise ValueError("rho must be in (0, 1)")
        if peak <= 0.0:
            raise ValueError("peak must be positive")
        self.sequences = tuple(tuple(value) for value in sequences)
        self.rho = float(rho)
        self.peak = float(peak)
        self.count = 0

    @property
    def bias(self) -> float:
        return -self.peak * (1.0 - self.rho**self.count)

    def apply(self, logits: torch.Tensor, history: Sequence[int]) -> torch.Tensor:
        bias = self.bias
        if bias == 0.0:
            return logits
        token_ids = completing_token_ids(history, self.sequences)
        if not token_ids:
            return logits
        adjusted = logits.clone()
        adjusted[list(token_ids)] += bias
        return adjusted

    def observe(self, history: Sequence[int], token_id: int) -> None:
        if completes_atp(history, token_id, self.sequences):
            self.count += 1
