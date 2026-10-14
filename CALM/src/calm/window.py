from __future__ import annotations

from dataclasses import dataclass

from calm.answers import normalize_answer


@dataclass(frozen=True, slots=True)
class WindowHit:
    step: int
    answer: str


class FirstSameAnswerWindow:
    def __init__(self, k: int = 4, mss: int = 10) -> None:
        if k < 1 or mss < 1:
            raise ValueError("k and mss must be positive")
        self.k = int(k)
        self.mss = int(mss)
        self.previous_step = 0
        self.answer = ""
        self.run_length = 0
        self.hit: WindowHit | None = None

    def observe(self, step: int, answer: str | None) -> WindowHit | None:
        if self.hit is not None:
            return self.hit
        if step <= self.previous_step:
            raise ValueError("steps must be strictly increasing")
        normalized = normalize_answer(answer)
        contiguous = step == self.previous_step + 1
        if normalized and contiguous and normalized == self.answer:
            self.run_length += 1
        elif normalized:
            self.answer = normalized
            self.run_length = 1
        else:
            self.answer = ""
            self.run_length = 0
        self.previous_step = step
        if step >= self.mss and self.run_length >= self.k:
            self.hit = WindowHit(step=step, answer=self.answer)
        return self.hit
