# CALM

CALM applies an online intervention after answer convergence is detected during
reasoning. The repository provides two policies over the same online generation
engine:

- CALM Hard blocks every ATP completion after the lock point.
- CALM Soft applies a count-adaptive logit bias to ATP completions.

## Online procedure

For each question, CALM:

1. Builds the model input with the model chat template and the dataset instruction.
2. Starts the main generation on a vLLM engine with the model `generation_config.json`.
3. Forms semantic reasoning steps online with bounded paragraph lookahead.
4. Starts an independent probe at each committed step boundary.
5. Generates at most 30 probe tokens and stops at the matching outer right brace.
6. Normalizes the boxed answer and updates the consecutive-answer state.
7. Locks at the first step `b*` where `b* >= 10` and four consecutive answers agree.
8. Restores the token prefix saved at `b*`.
9. Applies CALM from the first token after `b*` until the model writes `</think>`.
10. For a locked question that emits `</think>` inside the main budget, keep that marker and append `The final answer is \boxed`. Generation then continues one token at a time until the outer `}` closes. A locked question that exhausts the main budget before `</think>` appends `\n</think>\n\nThe final answer is \boxed` inside the 2048-token answer fix. An unlocked question keeps the answer the model writes itself.

Each draw uses a seed fixed by the request seed and the token position, so
returning to `b*` resamples that position under the logit mask. Probe requests
are separate vLLM generations and do not advance the main branch.

The math probe suffix is:

```text
\n**Final Answer**\n\nThe final answer is \boxed
```

The GPQA probe suffix is:

```text
\n**Final Answer**\n\nThe answer choice is \boxed
```

## ATP lexicon

The ATP surface forms are:

```text
Wait / wait / WAIT
Alternatively / alternatively / ALTERNATIVELY
Hm / hm
Hmm / hmm
Hmmm / hmmm
...
H + 12 m / h + 12 m
```

Every form also has a single-leading-space variant. Bare and spaced forms are
tokenized independently.

For a multi-token form, only a candidate token that completes the whole form is
modified. Shared prefix tokens are unchanged.

## Policies

CALM Hard sets the logit of every ATP-completing candidate to negative infinity
after `b*`.

CALM Soft uses:

```text
c(t) = number of ATP strings completed after b* and before position t
eta(c) = -10 * (1 - rho^c)
rho = 0.98
```

The first post-lock ATP sees `c=0` and receives zero bias. If it is generated,
the count becomes one and the next ATP receives the first non-zero penalty.
Multiple matches ending at the same token position increase the count once.

## Generation budget

```text
main generation: 32768 tokens
truncated answer fix: 2048 tokens
prompt reserve: 3072 tokens
max model length: 37888 tokens
```

The saved prefix consumes the original 32768-token main budget. After a lock, a
natural `</think>` stays in the trace. The remainder of that budget is used for
the cue `The final answer is \boxed` and the boxed answer. The cue counts
against the remainder. If the main budget is exhausted before `</think>`, a
locked question appends `\n</think>\n\nThe final answer is \boxed` and writes the
boxed answer inside the separate 2048-token answer fix. An unlocked question
that exhausts the main budget appends `</think>` and then generates at most
2048 tokens without the cue.

## Installation

From the directory that contains `pyproject.toml`:

```bash
uv sync
```

## Input

The input can be a JSON array or JSONL. Each row requires `question` and
`dataset`. `uid` is optional.

```json
{"uid":"math-500:0","dataset":"math-500","question":"Find ..."}
```

## Run

CALM Hard:

```bash
uv run calm-online \
  --model /path/to/model \
  --input questions.jsonl \
  --output hard.jsonl \
  --method hard
```

CALM Soft:

```bash
uv run calm-online \
  --model /path/to/model \
  --input questions.jsonl \
  --output soft.jsonl \
  --method soft \
  --rho 0.98 \
  --peak 10
```

Use `--dataset` when all input rows share a dataset and omit the field.

Each output row contains the complete response, reasoning, final-answer text,
boxed answer, `b*`, probe records, and token counts. A manifest is written next
to the output JSONL.

## Contanct
- Shengjing Liu: 18105891723@163.com
