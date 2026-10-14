from __future__ import annotations

from typing import Any

MATH_DATASETS = {
    "aime24",
    "aime25",
    "aime26",
    "amc23",
    "brumo25",
    "gsm8k",
    "hmmt25",
    "math-500",
    "olympiadbench",
}

INSTRUCTIONS = {
    "math": (
        "Solve the following math problem. Please reason step by step, "
        "and put your final answer within \\boxed{}."
    ),
    "gpqa": (
        "Answer the following multiple choice question. Please reason step by step, "
        "and put your choice letter (A, B, C, or D) without any other text "
        "within \\boxed{} in the end."
    ),
    "nq": (
        "Answer the following question. Please reason step by step, "
        "and put your final answer within \\boxed{}."
    ),
}


def task_type(dataset: str) -> str:
    name = dataset.lower()
    if "gpqa" in name:
        return "gpqa"
    if name in MATH_DATASETS:
        return "math"
    return "nq"


def model_type(model_name: str) -> str:
    name = model_name.lower()
    if "deepseek" in name and "r1" in name and "qwen3" in name:
        return "qwen3"
    if "deepseek" in name and "r1" in name:
        return "deepseek_r1"
    if "qwen3" in name and "thinking" in name:
        return "qwen3_thinking"
    if "qwq" in name:
        return "qwq"
    if "qwen3" in name:
        return "qwen3"
    if "nemotron" in name:
        return "nemotron"
    return "other"


def build_main_prompt_ids(
    tokenizer: Any,
    question: str,
    dataset: str,
    model_name: str,
) -> list[int]:
    kind = task_type(dataset)
    user_content = f"{INSTRUCTIONS[kind]}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": user_content}]
    family = model_type(model_name)
    if family == "nemotron":
        messages.insert(0, {"role": "system", "content": "detailed thinking on"})
    kwargs: dict[str, object] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if family == "qwen3":
        kwargs["enable_thinking"] = True
    try:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    return [int(token_id) for token_id in token_ids]


def probe_suffix(dataset: str) -> str:
    if task_type(dataset) == "gpqa":
        return "\n**Final Answer**\n\nThe answer choice is \\boxed"
    return "\n**Final Answer**\n\nThe final answer is \\boxed"
