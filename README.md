# KV Transition

CLI-based, reproducible evaluation harness for measuring KV-cache compression tradeoffs and instability transitions across context lengths.

---

## Overview

KV Transition evaluates how a single KV-cache compression policy (e.g., SnapKV) affects:

- Accuracy
- Latency
- Memory usage
- Failure rate
- Variance across context length bins

The system is config-driven, OpenAI-compatible, and logs experiment artifacts to SQLite for reproducibility.

---

## Installation (dev)

Using uv:

    uv venv
    uv pip install -e .

Set `BASE_URL`, `MODEL_NAME`, and `OPENAI_API_KEY` in `.env` (or environment) for inference.

---

## Core Goals

- Length-binned evaluation (5 bins by default)
- KV budget comparison (e.g. 1.0 / 0.5 / 0.2)
- Automatic scoring (EM / F1) and failure taxonomy
- Confidence intervals (bootstrap) and instability / transition zone detection
- Plots and Markdown report generation

---

## Quickstart (5 commands)

Deterministic lifecycle: **prepare → run → score → analyze → report**. Use the same config for all steps.

**Submission profile** (copy-paste; uses `config/experiments/submission.yaml`):

    uv run python -m kv_transition prepare -c config/experiments/submission.yaml
    uv run python -m kv_transition run -c config/experiments/submission.yaml
    uv run python -m kv_transition score -c config/experiments/submission.yaml
    uv run python -m kv_transition analyze -c config/experiments/submission.yaml
    uv run python -m kv_transition report -c config/experiments/submission.yaml

Or run the full pipeline in one shot:

    uv run python -m kv_transition all -c config/experiments/submission.yaml

**Smoke test** (smaller set, faster):

    uv run python -m kv_transition all -c config/experiments/sanity.yaml

---

## Artifacts Produced

The pipeline writes under **`runs/<exp_group_id>/`** and **`data/processed/<dataset_id>/`**. Exact tree:

```
data/
  processed/
    <dataset_id>/           # e.g. longbench__narrativeqa
      manifest.json          # portable evaluation set (entries, bin_edges, examples)

runs/
  <exp_group_id>/           # e.g. submission_longbench_narrativeqa_v1
    kv_transition.sqlite     # canonical DB: runs, requests, responses, scores, failures, bin_stats, etc.
    plots/
      acc_by_bin.png
      fail_by_bin.png
      latency_p50_by_bin.png
    report.md               # final markdown report (references plots + aggregates)
```

- **`<dataset_id>`** = `{dataset.name}__{dataset.task}` (e.g. `longbench__narrativeqa`).
- **`<exp_group_id>`** = from config `output.exp_group_id` (e.g. `submission_longbench_narrativeqa_v1`).

SQLite is the source of truth for prompts, responses, scores, telemetry, and aggregated statistics.

---

## Compatibility / Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | 3.13+ (see `pyproject.toml`) |
| **uv** | Recommended for install and running (`uv run python -m kv_transition ...`) |
| **Engines** | **OpenAI API** (`https://api.openai.com/v1`) and **OpenAI-compatible** servers (e.g. vLLM, SGLang at a custom `base_url`) |
| **GPU / JupyterHub** | Optional; only needed when running inference workloads (e.g. local vLLM). The CLI and report generation run without GPU. |

---

## How to Read Failures

Failures are classified and stored in the `failures` table with an `error_type` from the failure taxonomy. Use this to tell infrastructure/rate issues from model-quality issues.

| `error_type` | Meaning | What to do |
|--------------|---------|------------|
| **RATE_LIMITED** | HTTP 429 or rate-limit message from the API. | **Rerun** the affected run or **reduce request rate** (e.g. `rate_limit.max_rpm` in config, or pacing in the engine). Not a model/compression bug. |
| **EMPTY_OUTPUT** | Model returned no text or whitespace only. | May indicate KV compression or refusal; compare failure rate vs accuracy. |
| **TRUNCATED** | Response cut off (e.g. `finish_reason: length`). | Strong signal of compression limits; often precedes accuracy drop. |
| **TIMEOUT** / **ENGINE_ERROR** | Timeouts, server/connection errors. | Retry or fix infrastructure; exclude from analysis if persistent. |
| **REFUSAL** | Model explicitly refused (safety/policy). | Can inflate failure rate; separate from compression-induced failures. |
| **FORMAT_ERROR** | Response didn’t match expected structure. | May indicate model confusion under compression; check variance. |

**In practice**: If you see many **RATE_LIMITED** rows, adjust pacing or rerun; then re-score/analyze/report. Other types inform whether the drop is “system failure” vs “wrong answer.”

---

## Config Discoverability

| Config | Path | Purpose |
|--------|------|---------|
| **Submission** | `config/experiments/submission.yaml` | Pinned grading config: LongBench NarrativeQA, 5 bins, 2 per bin, budgets [1.0, 0.5, 0.2]. Use for final runs. |
| **Sanity** | `config/experiments/sanity.yaml` | Fast smoke test: 1 bin, 10 pinned examples, budgets [1.0, 0.2]. No `n_per_bin` scaling. |
| **Focus transition** | Add `sampling.strategy: focus_transition` and `sampling.focus` (e.g. `focus_bins`, `base_n_per_bin`, `focus_n_per_bin`) to any config | Adaptive sampling: more examples in selected bins for transition-zone analysis. See manifest builder docs. |

---

## Usage

Individual pipeline steps (using `uv`):

    uv run python -m kv_transition prepare -c config/experiments/submission.yaml
    uv run python -m kv_transition run -c config/experiments/submission.yaml
    uv run python -m kv_transition score -c config/experiments/submission.yaml
    uv run python -m kv_transition analyze -c config/experiments/submission.yaml
    uv run python -m kv_transition report -c config/experiments/submission.yaml

Or run the full pipeline:

    uv run python -m kv_transition all -c config/experiments/submission.yaml

---

## Project Structure

    config/         Experiment configs
    src/            Core package
      data/         Dataset loading + binning + manifest
      engines/      OpenAI-compatible inference
      run/          Execution orchestration
      eval/         Scoring + failure classification
      analysis/     Aggregation + transition detection
      report/       Markdown report generation
    runs/           Experiment artifacts (per exp_group_id)

---

## Definition of Done: Blueprint

A submission-grade run must satisfy:

### Required CLI Flow
- [ ] `prepare` → produces bins and writes `manifest.json` + DB tables (`datasets`, `examples`, `bins`, `manifest_entries`)
- [ ] `run` → completes for budgets `[1.0, 0.5, 0.2]` and logs `runs`, `requests`, `responses`, `telemetry` to SQLite
- [ ] `score` → writes EM/F1 scores + failure taxonomy (`scores`, `failures` tables)
- [ ] `analyze` → writes bin-level aggregates (`bin_stats`, `transition_summary` tables) + generates plots
- [ ] `report` → renders `report.md` referencing plots and aggregated stats
- [ ] `all` → executes prepare→run→score→analyze→report with skip-if-exists logic

### Exit Codes & Error Messaging
- Commands should exit with code `0` on success, non-zero on failure
- Errors should be logged with clear context (config path, phase, specific failure)
- Partial progress should be preserved (SQLite writes are transactional)

### Required Artifacts (Fresh Run Group)
- `data/processed/<dataset_id>/manifest.json` — portable manifest snapshot
- `runs/<exp_group_id>/kv_transition.sqlite` — canonical experiment database
- `runs/<exp_group_id>/plots/*.png` — analysis plots (`acc_by_bin.png`, `fail_by_bin.png`, `latency_p50_by_bin.png`)
- `runs/<exp_group_id>/report.md` — final markdown report

---

## Scope (Current)

- 1 model
- 1 task
- 1 KV policy
- 3 KV budgets
- 5 context-length bins

---

## License

TBD