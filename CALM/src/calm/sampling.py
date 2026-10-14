from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

    @classmethod
    def from_generation_config(cls, config: Any) -> SamplingConfig:
        temperature = getattr(config, "temperature", None)
        top_p = getattr(config, "top_p", None)
        top_k = getattr(config, "top_k", None)
        repetition_penalty = getattr(config, "repetition_penalty", None)
        return cls(
            do_sample=bool(getattr(config, "do_sample", True)),
            temperature=float(temperature if temperature is not None else 0.6),
            top_p=float(top_p if top_p is not None else 0.95),
            top_k=int(top_k if top_k is not None and top_k > 0 else 0),
            repetition_penalty=float(repetition_penalty if repetition_penalty is not None else 1.0),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sample_token(
    logits: torch.Tensor,
    history: list[int],
    config: SamplingConfig,
    generator: torch.Generator,
) -> int:
    scores = logits.float().clone()
    if config.repetition_penalty != 1.0 and history:
        indices = torch.tensor(
            sorted(set(history)),
            dtype=torch.long,
            device=scores.device,
        )
        selected = scores[indices]
        scores[indices] = torch.where(
            selected < 0,
            selected * config.repetition_penalty,
            selected / config.repetition_penalty,
        )
    if not config.do_sample:
        return int(torch.argmax(scores).item())
    scores /= config.temperature
    if config.top_k > 0 and config.top_k < scores.numel():
        threshold = torch.topk(scores, config.top_k).values[-1]
        scores[scores < threshold] = float("-inf")
    if config.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
        remove = cumulative > config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = float("-inf")
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())
