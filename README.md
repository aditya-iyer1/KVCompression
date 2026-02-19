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

## Usage

Prepare dataset:

    python -m kv_transition prepare -c config/experiments/snapkv_longbench.yaml

Run inference:

    python -m kv_transition run -c config/experiments/snapkv_longbench.yaml

Score:

    python -m kv_transition score -c config/experiments/snapkv_longbench.yaml

Analyze:

    python -m kv_transition analyze -c config/experiments/snapkv_longbench.yaml

Generate report:

    python -m kv_transition report -c config/experiments/snapkv_longbench.yaml

Run full pipeline:

    python -m kv_transition all -c config/experiments/snapkv_longbench.yaml

---

## Output Artifacts

Each experiment writes to:

    runs/<exp_group_id>/
      kv_transition.sqlite
      plots/
      report.md

SQLite is the canonical source of truth for prompts, responses, scores, telemetry, and aggregated statistics.

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