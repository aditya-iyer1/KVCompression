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

## Status

In active development. This README is temporary and will be expanded as core functionality lands.

---

## Core Goals

- Length-binned evaluation (5 bins by default)
- KV budget comparison (e.g., 1.0 / 0.5 / 0.2)
- Automatic scoring (EM / F1)
- Failure taxonomy
- Confidence intervals (bootstrap)
- Instability / transition zone detection
- Plots + Markdown report generation

---

## Project Structure

    config/         Experiment configs
    src/            Core package
      data/         Dataset loading + binning
      engines/      OpenAI-compatible inference
      run/          Execution orchestration
      eval/         Scoring + failure classification
      analysis/     Aggregation + transition detection
      report/       Markdown report generation
    runs/           Experiment artifacts

---

## Installation (dev)

Using uv:

    uv venv
    uv pip install -e .

---

## Quickstart

Run the complete pipeline for the submission experiment:

    uv run python -m kv_transition all -c config/experiments/submission.yaml

Or run a quick sanity check:

    uv run python -m kv_transition all -c config/experiments/sanity.yaml

**Expected artifacts** (for submission config):
- `data/processed/longbench__narrativeqa/manifest.json`
- `runs/submission_longbench_narrativeqa_v1/kv_transition.sqlite`
- `runs/submission_longbench_narrativeqa_v1/plots/*.png`
- `runs/submission_longbench_narrativeqa_v1/report.md`

For step-by-step execution, see [Usage](#usage) below.

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

## Expected Output Tree

After a fresh run group, the artifact directory structure is:

```
data/
  processed/
    <dataset_id>/
      manifest.json

runs/
  <exp_group_id>/
    manifest.json
    kv_transition.sqlite
    plots/
      acc_by_bin.png
      fail_by_bin.png
      latency_p50_by_bin.png
    report.md
```

Where:
- `<dataset_id>` = `{dataset.name}__{dataset.task}` (e.g., `longbench__narrativeqa`)
- `<exp_group_id>` = value from config `output.exp_group_id` (e.g., `submission_longbench_narrativeqa_v1`)

SQLite is the canonical source of truth for prompts, responses, scores, telemetry, and aggregated statistics.

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
- `runs/<exp_group_id>/manifest.json` — run-specific manifest copy (if applicable)
- `runs/<exp_group_id>/kv_transition.sqlite` — canonical experiment database
- `runs/<exp_group_id>/plots/*.png` — analysis plots (at minimum: `acc_by_bin.png`, `fail_by_bin.png`, `latency_p50_by_bin.png`)
- `runs/<exp_group_id>/report.md` — final markdown report

---

## Compatibility Matrix

### Validated Datasets/Tasks

| Dataset | Task | Status | Notes |
|---------|------|--------|-------|
| LongBench | NarrativeQA | ✅ Validated | Submission config (`config/experiments/submission.yaml`) |
| LongBench | *TBD* | 🔄 Placeholder | Additional task to be validated |
| *TBD* | *TBD* | 🔄 Placeholder | Additional dataset/task to be validated |

### Validated Engines

| Engine Type | Base URL Pattern | Status | Notes |
|-------------|------------------|--------|-------|
| OpenAI API | `https://api.openai.com/v1` | ✅ Validated | Standard OpenAI API endpoint |
| OpenAI-compatible Server | `http://localhost:8000/v1` (vLLM/SGLang) | ✅ Validated | Local or remote OpenAI-compatible server |

### Config Variability Tested

| Parameter | Tested Values | Status |
|-----------|---------------|--------|
| `binning.n_bins` | `1`, `5` | ✅ Validated |
| `dataset.n_per_bin` | `2`, `5`, `10` | ✅ Validated |
| `decoding.max_tokens` | `256` | ✅ Validated |
| `kv.budgets` | `[1.0, 0.5, 0.2]`, `[1.0, 0.2]` | ✅ Validated |

---

## Known Failure Modes / How to Interpret Failures

### Integrity Signals

The system classifies failures into categories that can masquerade as accuracy collapse:

1. **Empty Outputs** (`EMPTY_OUTPUT`)
   - Model returns no text or whitespace-only response
   - **Interpretation**: May indicate KV compression truncation or model refusal; check failure rate vs accuracy drop

2. **Refusals** (`REFUSAL`)
   - Model explicitly refuses to answer (safety filters, policy violations)
   - **Interpretation**: Can inflate failure rate; distinguish from compression-induced failures

3. **Timeouts / Engine Errors** (`TIMEOUT`, `ENGINE_ERROR`)
   - HTTP timeouts, server errors, connection failures
   - **Interpretation**: Infrastructure issues, not compression effects; should be retried or excluded from analysis

4. **Format Errors** (`FORMAT_ERROR`)
   - Response doesn't match expected structure (e.g., missing answer markers)
   - **Interpretation**: May indicate model confusion under compression; correlate with variance spikes

5. **Truncation** (`TRUNCATED`)
   - Response cut off mid-generation (token limit or KV cache exhaustion)
   - **Interpretation**: Strong signal of compression limits; often precedes accuracy collapse

### Failure Rate vs Accuracy Collapse

**Critical distinction**: A high failure rate can masquerade as accuracy collapse if not properly separated:

- **True accuracy collapse**: Model produces answers, but they are incorrect (low EM/F1)
- **Failure-induced collapse**: Model fails to produce answers (high failure rate), leading to artificially low accuracy when failures are counted as incorrect

**How to interpret**:
- Check `failures` table for failure type distribution per bin
- Compare `fail_rate` vs `acc_mean` curves: if failure rate spikes before accuracy drops, compression is causing system failures, not just quality degradation
- Transition zone detection should account for both failure rate and accuracy variance, not just mean accuracy

---

## Configs Index

Key experiment configurations:

| Config | Path | Description |
|--------|------|-------------|
| **Submission** | `config/experiments/submission.yaml` | Pinned submission/grading config: LongBench NarrativeQA, 5 bins, 2 per bin, budgets [1.0, 0.5, 0.2] |
| **Sanity** | `config/experiments/sanity.yaml` | Quick sanity check: 1 bin, 10 per bin, budgets [1.0, 0.2] for fast validation |
| **SnapKV LongBench** | `config/experiments/snapkv_longbench.yaml` | Full experiment config: 5 bins, 5 per bin, budgets [1.0, 0.5, 0.2] |

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