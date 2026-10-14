from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from calm.answers import boxed_close_keep_length, extract_last_boxed, probe_completion
from calm.backend import Backend, DecodeSnapshot, Session
from calm.config import OnlineConfig
from calm.lexicon import compile_token_sequences
from calm.policy import HardPolicy, LogitPolicy, SoftPolicy
from calm.prompts import build_main_prompt_ids, probe_suffix
from calm.steps import OnlineStepSegmenter, ParagraphStream, StepBoundary
from calm.window import FirstSameAnswerWindow

_THINK_CLOSE = "</think>"
_CANONICAL_CLOSE = "\n</think>\n\n"
_ANSWER_CUE = "The final answer is \\boxed"


@dataclass(frozen=True, slots=True)
class ProbeRecord:
    step: int
    answer: str
    prefix_tokens: int
    probe_tokens: int


def derived_seed(base_seed: int, uid: str, branch: str) -> int:
    payload = f"{int(base_seed)}\0{uid}\0{branch}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


class OnlineCALM:
    def __init__(self, backend: Backend, config: OnlineConfig) -> None:
        self.backend = backend
        self.config = config
        self.atp_sequences = compile_token_sequences(backend.tokenizer)
        if not self.atp_sequences:
            raise ValueError("ATP lexicon produced no token sequences")

    def run(
        self,
        question: str,
        dataset: str,
        uid: str,
        seed: int,
    ) -> dict[str, Any]:
        prompt_ids = build_main_prompt_ids(
            self.backend.tokenizer,
            question,
            dataset,
            self.backend.model_name,
        )
        if len(prompt_ids) > self.config.prompt_reserve_tokens:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens; limit is {self.config.prompt_reserve_tokens}"
            )
        main_seed = derived_seed(seed, uid, "main")
        main = self.backend.create_session(prompt_ids, main_seed)
        suffix_ids = [
            int(token_id)
            for token_id in self.backend.tokenizer.encode(
                probe_suffix(dataset),
                add_special_tokens=False,
            )
        ]
        paragraph_stream = ParagraphStream(self.backend.tokenizer)
        segmenter = OnlineStepSegmenter(
            min_chars=self.config.min_step_chars,
            max_chars=self.config.max_step_chars,
        )
        window = FirstSameAnswerWindow(k=self.config.k, mss=self.config.mss)
        probes: list[ProbeRecord] = []
        policy: LogitPolicy | None = None
        locked = False
        in_thought = True
        natural_close = False
        reached_eos = False
        sampled_main_tokens = 0
        probe_tokens = 0
        b_star_step: int | None = None
        b_star_token: int | None = None
        recent_snapshots: dict[int, DecodeSnapshot] = {}

        while True:
            while len(main.generated_ids) < self.config.generation_tokens and not reached_eos:
                before_token = main.snapshot()
                recent_snapshots[before_token.token_count] = before_token
                minimum_snapshot = len(main.token_ids) - 64
                recent_snapshots = {
                    count: snapshot
                    for count, snapshot in recent_snapshots.items()
                    if count >= minimum_snapshot
                }
                token_id = main.sample(policy if locked and in_thought else None)
                sampled_main_tokens += 1
                if token_id in main.eos_token_ids:
                    reached_eos = True
                if in_thought and self._has_think_close(main):
                    natural_close = True
                    if not locked:
                        marker_snapshot = self._marker_start_snapshot(
                            main,
                            recent_snapshots,
                            before_token,
                        )
                        boundaries = self._flush_boundaries(
                            paragraph_stream,
                            segmenter,
                            marker_snapshot,
                            remove=_THINK_CLOSE,
                        )
                        for boundary in boundaries:
                            hit, used = self._probe_boundary(
                                main,
                                boundary,
                                suffix_ids,
                                window,
                                probes,
                                uid,
                                seed,
                            )
                            probe_tokens += used
                            if hit:
                                locked = True
                                natural_close = False
                                reached_eos = False
                                b_star_step = boundary.step
                                b_star_token = boundary.snapshot.token_count - len(prompt_ids)
                                policy = self._new_policy()
                                paragraph_stream = ParagraphStream(self.backend.tokenizer)
                                recent_snapshots.clear()
                                break
                        if locked:
                            continue
                    in_thought = False
                    policy = None
                    if locked:
                        break
                elif in_thought and not locked and not reached_eos:
                    paragraph = paragraph_stream.feed(token_id)
                    if paragraph is not None:
                        for boundary in segmenter.add_paragraph(
                            paragraph,
                            main.snapshot(),
                        ):
                            hit, used = self._probe_boundary(
                                main,
                                boundary,
                                suffix_ids,
                                window,
                                probes,
                                uid,
                                seed,
                            )
                            probe_tokens += used
                            if hit:
                                locked = True
                                b_star_step = boundary.step
                                b_star_token = boundary.snapshot.token_count - len(prompt_ids)
                                policy = self._new_policy()
                                paragraph_stream = ParagraphStream(self.backend.tokenizer)
                                recent_snapshots.clear()
                                break
            if not in_thought or locked:
                break
            final_snapshot = before_token if reached_eos else main.snapshot()
            boundaries = self._flush_boundaries(
                paragraph_stream,
                segmenter,
                final_snapshot,
            )
            hit_boundary: StepBoundary | None = None
            for boundary in boundaries:
                hit, used = self._probe_boundary(
                    main,
                    boundary,
                    suffix_ids,
                    window,
                    probes,
                    uid,
                    seed,
                )
                probe_tokens += used
                if hit:
                    hit_boundary = boundary
                    break
            if hit_boundary is None:
                break
            locked = True
            reached_eos = False
            b_star_step = hit_boundary.step
            b_star_token = hit_boundary.snapshot.token_count - len(prompt_ids)
            policy = self._new_policy()
            paragraph_stream = ParagraphStream(self.backend.tokenizer)
            recent_snapshots.clear()

        hit_generation_cap = (
            len(main.generated_ids) >= self.config.generation_tokens and not reached_eos
        )
        forced_close = hit_generation_cap and not natural_close
        thinking_sampled = sampled_main_tokens
        forced_marker_tokens = 0
        answer_fix_generated_tokens = 0
        cue_tokens = 0
        if locked and natural_close:
            head_tokens = len(main.generated_ids)
            forced_marker_tokens, cue_tokens, answer_tokens, hit_eos = self._emit_boxed_answer(
                main,
                self.config.generation_tokens - head_tokens,
                write_close=False,
            )
            delivered_generation_tokens = len(main.generated_ids)
            sampled_main_tokens += answer_tokens + hit_eos
            if hit_eos:
                reached_eos = True
        elif forced_close and locked:
            head_tokens = len(main.generated_ids)
            forced_marker_tokens, cue_tokens, answer_fix_generated_tokens, hit_eos = (
                self._emit_boxed_answer(
                    main,
                    self.config.answer_fix_tokens,
                    write_close=True,
                )
            )
            delivered_generation_tokens = head_tokens
            sampled_main_tokens += answer_fix_generated_tokens + hit_eos
            if hit_eos:
                reached_eos = True
        else:
            head_tokens = len(main.generated_ids)
            delivered_generation_tokens = head_tokens
            if forced_close:
                close_ids = self._encode(_CANONICAL_CLOSE)
                if len(main.token_ids) + len(close_ids) <= self.config.max_model_len:
                    forced_marker_tokens = len(close_ids)
                    main.append_forced(close_ids)
                for _ in range(self.config.answer_fix_tokens):
                    if len(main.token_ids) >= self.config.max_model_len:
                        break
                    token_id = main.sample()
                    sampled_main_tokens += 1
                    answer_fix_generated_tokens += 1
                    if token_id in main.eos_token_ids:
                        reached_eos = True
                        break
        discarded_lookahead_tokens = max(0, thinking_sampled - head_tokens)

        response_ids = list(main.generated_ids)
        if response_ids and response_ids[-1] in main.eos_token_ids:
            response_ids.pop()
        response = self.backend.decode(response_ids)
        reasoning, answer_text = _split_response(response)
        return {
            "uid": uid,
            "dataset": dataset,
            "method": self.config.method,
            "seed": seed,
            "response": response,
            "reasoning": reasoning,
            "answer_text": answer_text,
            "final_answer": extract_last_boxed(answer_text),
            "b_star_step": b_star_step,
            "b_star_token": b_star_token,
            "natural_close": natural_close,
            "forced_close": forced_close,
            "reached_eos": reached_eos,
            "prompt_tokens": len(prompt_ids),
            "delivered_generation_tokens": delivered_generation_tokens,
            "discarded_lookahead_tokens": discarded_lookahead_tokens,
            "forced_marker_tokens": forced_marker_tokens,
            "answer_fix_tokens": answer_fix_generated_tokens,
            "cue_tokens": cue_tokens,
            "main_tokens": len(main.generated_ids),
            "sampled_main_tokens": sampled_main_tokens,
            "probe_tokens": probe_tokens,
            "probes": [asdict(record) for record in probes],
        }

    def _probe_boundary(
        self,
        main: Session,
        boundary: StepBoundary,
        suffix_ids: list[int],
        window: FirstSameAnswerWindow,
        records: list[ProbeRecord],
        uid: str,
        seed: int,
    ) -> tuple[bool, int]:
        snapshot = boundary.snapshot
        if not isinstance(snapshot, DecodeSnapshot):
            raise TypeError("step boundary has an invalid decoder snapshot")
        prefix_ids = list(main.token_ids[: snapshot.token_count])
        probe_seed = derived_seed(seed, uid, f"probe:{boundary.step}")
        text, generated = self.backend.run_probe(
            main,
            prefix_ids,
            suffix_ids,
            self.config.probe_max_tokens,
            probe_seed,
        )
        answer, done = probe_completion(text)
        if not done:
            answer = ""
        records.append(
            ProbeRecord(
                step=boundary.step,
                answer=answer,
                prefix_tokens=snapshot.token_count,
                probe_tokens=generated,
            )
        )
        hit = window.observe(boundary.step, answer)
        if hit is None:
            return False, generated
        main.restore(snapshot)
        return True, generated

    def _encode(self, text: str) -> list[int]:
        return [
            int(token_id)
            for token_id in self.backend.tokenizer.encode(text, add_special_tokens=False)
        ]

    def _emit_boxed_answer(
        self,
        main: Session,
        answer_budget: int,
        *,
        write_close: bool,
    ) -> tuple[int, int, int, int]:
        close_ids = self._encode(_CANONICAL_CLOSE) if write_close else []
        cue_ids = self._encode(_ANSWER_CUE)
        if len(main.token_ids) + len(close_ids) > self.config.max_model_len:
            return 0, 0, 0, 0
        if close_ids:
            main.append_forced(close_ids)
        room = self.config.max_model_len - len(main.token_ids)
        body_limit = min(answer_budget - len(cue_ids), room - len(cue_ids))
        if body_limit < 1 or len(cue_ids) > room:
            return len(close_ids), 0, 0, 0
        main.append_forced(cue_ids)
        generated: list[int] = []
        hit_eos = 0
        for _ in range(body_limit):
            if len(main.token_ids) >= self.config.max_model_len:
                break
            before = main.snapshot()
            token_id = main.sample()
            if token_id in main.eos_token_ids:
                main.restore(before)
                hit_eos = 1
                break
            generated.append(token_id)
            pieces = [self.backend.decode([item]) for item in generated]
            if boxed_close_keep_length(pieces) is not None:
                break
        return len(close_ids), len(cue_ids), len(generated), hit_eos

    def _new_policy(self) -> LogitPolicy:
        if self.config.method == "hard":
            return HardPolicy(self.atp_sequences)
        return SoftPolicy(
            self.atp_sequences,
            rho=self.config.rho,
            peak=self.config.peak,
        )

    def _has_think_close(self, main: Session) -> bool:
        tail = main.generated_ids[-32:]
        return _THINK_CLOSE in self.backend.decode(tail)

    def _marker_start_snapshot(
        self,
        main: Session,
        recent_snapshots: dict[int, DecodeSnapshot],
        fallback: DecodeSnapshot,
    ) -> DecodeSnapshot:
        generated = main.generated_ids
        tail_start = max(0, len(generated) - 32)
        marker_start = tail_start
        found = False
        for start in range(tail_start, len(generated)):
            if _THINK_CLOSE in self.backend.decode(generated[start:]):
                marker_start = start
                found = True
            elif found:
                break
        token_count = main.prompt_length + marker_start
        return recent_snapshots.get(token_count, fallback)

    @staticmethod
    def _flush_boundaries(
        paragraph_stream: ParagraphStream,
        segmenter: OnlineStepSegmenter,
        snapshot: DecodeSnapshot,
        remove: str | None = None,
    ) -> list[StepBoundary]:
        boundaries: list[StepBoundary] = []
        paragraph_text = paragraph_stream.flush(remove=remove)
        if paragraph_text.strip():
            boundaries.extend(segmenter.add_paragraph(paragraph_text, snapshot))
        boundaries.extend(segmenter.flush())
        return boundaries


def _split_response(response: str) -> tuple[str, str]:
    text = response
    if text.lstrip().startswith("<think>"):
        text = text.lstrip()[len("<think>") :]
    if _THINK_CLOSE not in text:
        return text.strip(), ""
    reasoning, answer = text.split(_THINK_CLOSE, 1)
    return reasoning.strip(), answer.strip()
