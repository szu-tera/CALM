from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calm.backend import VLLMBackend, prepare_runtime_libs
from calm.config import OnlineConfig
from calm.lexicon import atp_surfaces
from calm.online import OnlineCALM

PROTOCOL_ID = "calm-online-32k-v2"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("JSON input must be an array")
        return [dict(row) for row in value]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("hard", "soft"), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.98)
    parser.add_argument("--peak", type=float, default=10.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    prepare_runtime_libs()
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}")
    rows = _load_rows(args.input)
    config = OnlineConfig(
        method=args.method,
        rho=args.rho,
        peak=args.peak,
    )
    backend = VLLMBackend(
        args.model,
        max_model_len=config.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    runner = OnlineCALM(backend, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "model": args.model,
        "seed": args.seed,
        "prompt_version": "default",
        "config": config.to_dict(),
        "host": "vllm",
        "sampling": backend.sampling.to_dict(),
        "atp_surfaces": list(atp_surfaces()),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            question = str(row.get("question") or "")
            dataset = str(row.get("dataset") or args.dataset or "")
            if not question or not dataset:
                raise ValueError(f"row {index} requires question and dataset")
            uid = str(row.get("uid") or row.get("id") or index)
            result = runner.run(
                question=question,
                dataset=dataset,
                uid=uid,
                seed=args.seed,
            )
            for field in ("id", "question_idx", "ground_truth"):
                if field in row:
                    result[field] = row[field]
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
