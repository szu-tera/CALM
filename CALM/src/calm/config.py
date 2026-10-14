from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Method = Literal["hard", "soft"]


@dataclass(frozen=True, slots=True)
class OnlineConfig:
    method: Method
    k: int = 4
    mss: int = 10
    rho: float = 0.98
    peak: float = 10.0
    probe_max_tokens: int = 30
    generation_tokens: int = 32768
    answer_fix_tokens: int = 2048
    prompt_reserve_tokens: int = 3072
    max_model_len: int = 37888
    min_step_chars: int = 200
    max_step_chars: int = 1000

    def __post_init__(self) -> None:
        if self.method not in {"hard", "soft"}:
            raise ValueError("method must be 'hard' or 'soft'")
        if self.k < 1 or self.mss < 1:
            raise ValueError("k and mss must be positive")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must be in (0, 1)")
        if self.peak <= 0:
            raise ValueError("peak must be positive")
        if (
            min(
                self.probe_max_tokens,
                self.generation_tokens,
                self.answer_fix_tokens,
                self.prompt_reserve_tokens,
                self.max_model_len,
                self.min_step_chars,
                self.max_step_chars,
            )
            < 1
        ):
            raise ValueError("token and step limits must be positive")
        expected = self.generation_tokens + self.answer_fix_tokens + self.prompt_reserve_tokens
        if self.max_model_len != expected:
            raise ValueError(
                "max_model_len must equal generation_tokens + "
                "answer_fix_tokens + prompt_reserve_tokens"
            )
        if self.min_step_chars > self.max_step_chars:
            raise ValueError("min_step_chars cannot exceed max_step_chars")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
