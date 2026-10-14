from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from transformers import AutoTokenizer, GenerationConfig

from calm.answers import probe_completion
from calm.policy import LogitPolicy
from calm.sampling import SamplingConfig


@dataclass(frozen=True, slots=True)
class DecodeSnapshot:
    token_count: int
    rng_state: torch.Tensor


class Session(Protocol):
    token_ids: list[int]
    prompt_length: int
    eos_token_ids: frozenset[int]

    @property
    def generated_ids(self) -> list[int]: ...

    def sample(self, policy: LogitPolicy | None = None) -> int: ...

    def snapshot(self) -> DecodeSnapshot: ...

    def restore(self, snapshot: DecodeSnapshot) -> None: ...

    def append_forced(self, token_ids: list[int]) -> None: ...

    def release_cache(self) -> None: ...

    def suspend_cache(self) -> tuple[Any, torch.Tensor]: ...

    def resume_cache(self, state: tuple[Any, torch.Tensor]) -> None: ...


class Backend(Protocol):
    tokenizer: Any
    model_name: str
    sampling: SamplingConfig

    def create_session(self, prompt_ids: list[int], seed: int) -> Session: ...

    def run_probe(
        self,
        main: Session,
        prefix_ids: list[int],
        suffix_ids: list[int],
        max_tokens: int,
        seed: int,
    ) -> tuple[str, int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


def prepare_runtime_libs() -> None:
    root = Path(sys.prefix) / "lib"
    libs = [str(path) for path in sorted(root.glob("**/nvidia/*/lib")) if path.is_dir()]
    torch_lib = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "torch"
        / "lib"
    )
    if torch_lib.is_dir():
        libs.append(str(torch_lib))
    if not libs:
        return
    if os.environ.get("CALM_LIBS_READY") == "1":
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs) if not current else ":".join([*libs, current])
    os.environ["CALM_LIBS_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def position_seed(base_seed: int, position: int) -> int:
    payload = f"{int(base_seed)}\0{int(position)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _policy_extra(policy: LogitPolicy, lock_index: int) -> dict[str, object]:
    sequences = [list(sequence) for sequence in policy.sequences]
    if hasattr(policy, "count"):
        return {
            "calm_mode": "soft",
            "calm_lock": int(lock_index),
            "calm_sequences": sequences,
            "calm_rho": float(policy.rho),
            "calm_peak": float(policy.peak),
        }
    return {
        "calm_mode": "hard",
        "calm_lock": int(lock_index),
        "calm_sequences": sequences,
    }


class VLLMSession:
    def __init__(self, backend: VLLMBackend, prompt_ids: list[int], seed: int) -> None:
        self.backend = backend
        self.token_ids = list(prompt_ids)
        self.prompt_length = len(prompt_ids)
        self.eos_token_ids = backend.eos_token_ids
        self._seed = int(seed)
        self.mask_from: int | None = None

    @property
    def generated_ids(self) -> list[int]:
        return self.token_ids[self.prompt_length :]

    def sample(self, policy: LogitPolicy | None = None) -> int:
        extra = None
        if policy is not None:
            if self.mask_from is None:
                self.mask_from = len(self.token_ids)
            extra = _policy_extra(policy, self.mask_from)
        token_id = self.backend.generate_one(
            self.token_ids,
            position_seed(self._seed, len(self.generated_ids)),
            extra,
        )
        if policy is not None:
            policy.observe(self.token_ids, token_id)
        self.token_ids.append(token_id)
        return token_id

    def snapshot(self) -> DecodeSnapshot:
        return DecodeSnapshot(
            token_count=len(self.token_ids),
            rng_state=torch.tensor([len(self.generated_ids)], dtype=torch.int64),
        )

    def restore(self, snapshot: DecodeSnapshot) -> None:
        if snapshot.token_count < self.prompt_length:
            raise ValueError("cannot restore before the prompt")
        if snapshot.token_count > len(self.token_ids):
            raise ValueError("snapshot extends beyond the current sequence")
        self.token_ids = self.token_ids[: snapshot.token_count]
        if self.mask_from is not None and self.mask_from > len(self.token_ids):
            self.mask_from = None

    def append_forced(self, token_ids: list[int]) -> None:
        self.token_ids.extend(int(token_id) for token_id in token_ids)

    def release_cache(self) -> None:
        return None

    def suspend_cache(self) -> tuple[Any, torch.Tensor]:
        return (None, torch.empty(0))

    def resume_cache(self, state: tuple[Any, torch.Tensor]) -> None:
        del state


class VLLMBackend:
    def __init__(
        self,
        model_name: str,
        max_model_len: int = 37888,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        from vllm import LLM, SamplingParams, TokensPrompt

        from calm.vllm_mask import calm_logits_processor_class

        self.model_name = model_name
        self.max_model_len = int(max_model_len)
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        generation_config = GenerationConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.sampling = SamplingConfig.from_generation_config(generation_config)
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=int(tensor_parallel_size),
            max_model_len=self.max_model_len,
            gpu_memory_utilization=float(gpu_memory_utilization),
            enable_prefix_caching=True,
            logits_processors=[calm_logits_processor_class()],
        )
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        eos = getattr(generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos_values = eos if isinstance(eos, (list, tuple)) else [eos]
        self.eos_token_ids = frozenset(
            int(token_id) for token_id in eos_values if token_id is not None
        )

    def create_session(self, prompt_ids: list[int], seed: int) -> VLLMSession:
        return VLLMSession(self, prompt_ids, seed)

    def generate_one(
        self,
        token_ids: list[int],
        seed: int,
        extra_args: dict[str, object] | None,
    ) -> int:
        if len(token_ids) + 1 > self.max_model_len:
            raise ValueError("sequence exceeds max_model_len")
        sampling = self.sampling
        temperature = 0.0 if not sampling.do_sample else sampling.temperature
        params = self._SamplingParams(
            temperature=temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            repetition_penalty=sampling.repetition_penalty,
            max_tokens=1,
            seed=int(seed),
            ignore_eos=True,
            extra_args=extra_args,
            skip_special_tokens=False,
        )
        outputs = self.llm.generate(
            [self._TokensPrompt(prompt_token_ids=list(token_ids))],
            params,
            use_tqdm=False,
        )
        generated = list(outputs[0].outputs[0].token_ids)
        if len(generated) != 1:
            raise RuntimeError(f"vLLM returned {len(generated)} tokens for a one-token step")
        return int(generated[0])

    def run_probe(
        self,
        main: Session,
        prefix_ids: list[int],
        suffix_ids: list[int],
        max_tokens: int,
        seed: int,
    ) -> tuple[str, int]:
        del main
        token_ids = list(prefix_ids) + list(suffix_ids)
        generated: list[int] = []
        for step in range(max_tokens):
            token_id = self.generate_one(token_ids, position_seed(seed, step), None)
            generated.append(token_id)
            token_ids.append(token_id)
            text = self.decode(generated)
            if probe_completion(text)[1] or token_id in self.eos_token_ids:
                break
        return self.decode(generated), len(generated)

    def decode(self, token_ids: list[int]) -> str:
        return str(
            self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
