# KV Transition: KV-Cache Compression Stability & Tradeoffs Harness (Blueprint)

> **Goal (1 sentence):** Build a reproducible CLI-first evaluation harness that measures how a single KV-cache compression policy (e.g., SnapKV/H2O) shifts the **accuracy–latency–memory** tradeoff as **context length increases**, and whether there is an **instability transition zone** where **variance and failure rates spike before mean accuracy collapses**.

This blueprint is written to match your existing repo scaffold exactly and to keep everything **engine-agnostic via OpenAI-compatible HTTP**, **experiment-reproducible via configs + SQLite logging**, and **scope-safe for 4–5 weeks**.

---

## 1) High-level overview: what the project achieves

### What you will produce
1. **A CLI tool** (`python -m kv_transition ...`) that can:
   - prepare a long-context dataset into 5 context-length bins,
   - run inference across KV budgets (1.0 / 0.5 / 0.2) on an OpenAI-compatible endpoint,
   - score outputs automatically (EM/F1),
   - compute bin-level statistics (mean/CI, std/variance, failure rate, latency),
   - detect and summarize a “transition zone”,
   - generate a polished Markdown report with plots.

2. **A reproducible experiment artifact set** per experiment group:
   - SQLite DB containing prompts, outputs, timings, KV settings, scores, and derived stats
   - Plot images (accuracy/std/failure/latency vs length bin)
   - A report that tells a clear systems story

### The scientific claim you’re testing (class + industry aligned)
For a fixed model and task:
- As KV budget decreases, the performance-vs-length curve shifts and degrades.
- Degradation is not necessarily smooth: there may exist a **transition region** where:
  - failures increase (format breaks / empty answers / truncation),
  - output quality becomes less reliable (variance spikes),
  - *before* mean accuracy fully collapses.

### Why this is high-signal
- **Industry signal:** inference systems measurement, reproducibility, telemetry, robust evaluation, SQLite experiment tracking.
- **Efficient AI signal:** quantifying the memory/latency gains vs reliability/accuracy cost of KV compression.
- **Lab signal (optional):** “transition zone” framing + rigorous statistics (bootstrap CI) + extensible harness.

---

## 2) Modular overview: phases, responsibilities, and interactions

This project is intentionally broken into rerunnable phases, each producing explicit artifacts (mostly in SQLite + derived files), so debugging is local and incremental.

### Phase A — Configuration & Reproducibility (settings layer)
**Purpose:** Make every run reproducible, comparable, and easily restarted.
- Inputs:
  - `config/default.yaml` or `config/experiments/*.yaml`
  - environment variables (API keys, base URLs)
- Outputs:
  - config snapshot stored in SQLite `experiments` table
  - metadata about the code version (git hash if available)

**Key interaction:** Every other phase loads settings through `settings.py` and uses canonical paths via `paths.py`.

---

### Phase B — Dataset Ingest + Normalization + Tokenization + Binning (data layer)
**Purpose:** Construct a stable evaluation set binned by context length.
- Inputs:
  - LongBench task (or cached local copy)
  - tokenizer definition tied to the chosen model
  - binning config: `bins=5`, sampling `n_per_bin`
- Outputs:
  - `data/processed/<dataset_id>/manifest.json`  (portable contract)
  - SQLite tables populated:
    - `datasets`, `examples`, `bins`, `manifest_entries`

**Key interaction:** Runner never touches raw data; it reads the manifest from DB (and/or `manifest.json` as a secondary portable artifact).

---

### Phase C — Inference Runs (engines + run layer)
**Purpose:** Run the same manifest through different KV budgets and log everything.
- Inputs:
  - manifest entries (example ids, bin assignment, prompt spec)
  - engine config:
    - OpenAI API or vLLM/SGLang OpenAI-compatible server
  - KV settings:
    - `kv_policy` (e.g., snapkv)
    - `kv_budget` (1.0 / 0.5 / 0.2)
- Outputs:
  - SQLite tables populated:
    - `runs`, `requests`, `responses`, `telemetry`
  - optional derived JSONL export for debugging (not required)

**Key interaction:** The only thing that changes across settings is the *engine endpoint and/or server instance*. The runner code remains constant.

---

### Phase D — Scoring + Failure Taxonomy (eval layer)
**Purpose:** Convert raw model text into measurable quality + failure signals.
- Inputs:
  - `responses` and corresponding ground truth answers from `examples`
- Outputs (SQLite):
  - `scores` table (EM, F1, normalized answer strings)
  - `failures` table (failure class, message)

**Key interaction:** Analysis consumes `scores + telemetry + bins` to generate aggregated curves.

---

### Phase E — Aggregation + Statistics + Transition Detection (analysis layer)
**Purpose:** Produce bin-level curves and detect the “transition zone.”
- Inputs:
  - `scores`, `telemetry`, `bins`, `runs` metadata
- Outputs:
  - SQLite:
    - `bin_stats` table (mean, std, CI, failure rate, latency percentiles)
    - `transition_summary` table (detected zone, supporting metrics)
  - Files:
    - `runs/<exp_group_id>/plots/*.png`
    - `runs/<exp_group_id>/tables/*.csv` (optional but useful)
    - `runs/<exp_group_id>/summary.json` (optional)

**Key interaction:** Report builder uses plots + aggregated stats queries to write a final narrative.

---

### Phase F — Report Generation (report layer)
**Purpose:** Produce a single artifact suitable for submission and future extension.
- Inputs:
  - aggregated stats + plots
- Outputs:
  - `runs/<exp_group_id>/report.md` (optionally renderable to PDF later)

---

## 3) Architecture: how phases map to your file layout (critical role of each file)

Below is your exact scaffold, annotated with **role**, **core functions**, and **I/O** expectations.

---

### Repo root

- `pyproject.toml`
  - **Role:** packaging + dependencies + entrypoints.
  - **I/O:** none.
  - **Notes:** define console script (optional) and enable `python -m kv_transition`.

- `README.md`
  - **Role:** how to run the pipeline; minimal “quickstart”.
  - **I/O:** none.

- `.env.example`
  - **Role:** document environment variables.
  - **I/O:** none.
  - **Typical vars:**
    - `OPENAI_API_KEY`
    - `BASE_URL` (OpenAI or vLLM/SGLang server)
    - optional `MODEL_NAME`

---

### `config/`
- `config/default.yaml`
  - **Role:** default experiment settings (safe baseline).
  - **I/O:** input to CLI.
- `config/experiments/snapkv_longbench.yaml`
  - **Role:** pinned experiment config for submission/repro.
  - **I/O:** input to CLI.

**Recommended config keys (schema enforced in `settings.py`):**
- `dataset.name` (e.g., longbench)
- `dataset.task` (single task/subset)
- `dataset.n_per_bin`
- `binning.n_bins = 5`
- `model.name`
- `engine.base_url`
- `engine.api_key_env`
- `engine.request.timeout_s`
- `kv.policy`
- `kv.budgets: [1.0, 0.5, 0.2]`
- `decoding.temperature = 0`
- `decoding.max_tokens`
- `output.exp_group_id` (or auto-generated)
- `db.path` (default: `runs/<exp_group_id>/kv_transition.sqlite`)

---

## `src/kv_transition/` (package)

### Top-level CLI + settings + paths

- `__main__.py`
  - **Role:** enables `python -m kv_transition ...`.
  - **I/O:** none (delegates to `cli.py`).

- `cli.py`
  - **Role:** CLI entrypoint and command routing.
  - **Commands (recommended):**
    - `prepare` → Phase B
    - `run` → Phase C
    - `score` → Phase D
    - `analyze` → Phase E
    - `report` → Phase F
    - `all` → A→F pipeline (with skip-if-exists)
  - **I/O:**
    - reads config YAML
    - writes/reads SQLite at canonical run path
    - writes plots + report into `runs/<exp_group_id>/`

- `settings.py`
  - **Role:** config schema, load/validate/merge defaults, resolve env vars.
  - **I/O:** reads YAML + env; outputs validated settings object used everywhere.

- `paths.py`
  - **Role:** canonical path builders to ensure stable I/O locations.
  - **I/O:** none directly; provides functions like:
    - `processed_dir(dataset_id)`
    - `manifest_path(dataset_id)`
    - `run_dir(exp_group_id)`
    - `db_path(exp_group_id)`

---

## `data/` (Phase B)

- `longbench_loader.py`
  - **Role:** fetch/load a single LongBench task subset (robust mode: cache locally).
  - **I/O:**
    - input: remote dataset OR cached copy in `data/raw/` (optional)
    - output: in-memory examples list (to normalization)

- `normalize.py`
  - **Role:** canonicalize example schema.
  - **Canonical example fields:**
    - `example_id`
    - `question`
    - `context`
    - `answers` (list or string)
    - `meta` (task name, source id, etc.)
  - **I/O:** outputs normalized objects inserted to DB + optionally `examples.jsonl` in processed dir.

- `tokenizer.py`
  - **Role:** token counting for binning consistency (preferrably same tokenizer family as model).
  - **I/O:** none directly; returns `token_count(prompt_parts)`.

- `binning.py`
  - **Role:** compute bin edges (quantiles) and assign bin indices.
  - **I/O:** none directly; writes bin assignments via `manifest.py`.

- `manifest.py`
  - **Role:** define the “contract” between dataset prep and runner.
  - **Outputs:**
    - writes `data/processed/<dataset_id>/manifest.json` (portable snapshot)
    - populates DB tables: `datasets`, `examples`, `bins`, `manifest_entries`
  - **Manifest contents (recommended):**
    - dataset/task identifiers
    - tokenizer/model used for length computation
    - bin edges (token ranges)
    - selected examples per bin
    - prompt template version hash

---

## `db/` (Cross-cutting: storage backbone)

> SQLite is your single source of truth. Keep it simple: one DB per experiment group.

- `connect.py`
  - **Role:** open SQLite connection with correct pragmas (WAL recommended).
  - **I/O:** creates/opens DB file at `runs/<exp_group_id>/kv_transition.sqlite`.

- `schema.py`
  - **Role:** defines tables and indexes.
  - **I/O:** executed on init.

  **Minimum tables (recommended):**
  - `experiments(exp_group_id, created_at, config_yaml, git_hash, notes)`
  - `datasets(dataset_id, name, task, created_at)`
  - `examples(example_id, dataset_id, question, context, answers_json, meta_json, token_len)`
  - `bins(bin_id, dataset_id, bin_idx, token_min, token_max, n_examples)`
  - `manifest_entries(entry_id, dataset_id, example_id, bin_idx)`
  - `runs(run_id, exp_group_id, kv_policy, kv_budget, engine_name, base_url, model_name, created_at)`
  - `requests(request_id, run_id, entry_id, prompt_json, prompt_tokens, max_tokens, temperature)`
  - `responses(response_id, request_id, text, finish_reason, completion_tokens, raw_json)`
  - `telemetry(request_id, latency_s, ttfb_s, prefill_s, decode_s, peak_mem_mb, notes_json)`
  - `scores(request_id, em, f1, pred_norm, gold_norm)`
  - `failures(request_id, failure_type, message)`
  - `bin_stats(run_id, bin_idx, n, acc_mean, acc_ci_low, acc_ci_high, acc_std, fail_rate, lat_p50, lat_p95, mem_p95)`
  - `transition_summary(run_id, transition_bin_start, transition_bin_end, rationale_json)`

- `dao.py`
  - **Role:** typed DB operations:
    - insert examples
    - create run rows
    - upsert request/response/telemetry
    - query bins + stats
  - **I/O:** reads/writes SQLite.

- `migrations.py` (optional)
  - **Role:** schema versioning. Can be a no-op initially with a `schema_version` table.

---

## `engines/` (Phase C: engine abstraction)

- `base.py`
  - **Role:** defines the engine interface, e.g.:
    - `generate(messages, **params) -> EngineResult(text, usage, timings, raw)`
  - **I/O:** none.

- `openai_compat.py`
  - **Role:** a unified OpenAI-compatible HTTP client used for both:
    - OpenAI API
    - vLLM/SGLang OpenAI-server
  - **I/O:** network requests; returns structured response.

- `endpoints.py`
  - **Role:** resolves endpoint details from settings:
    - base_url
    - api key env var
    - model name
  - **I/O:** reads env; no file output.

**KV toggles note:** In this architecture, KV settings are ideally applied by:
- selecting the correct server instance/port per KV budget (Option A),
- or passing supported request-time parameters if your serving stack supports it (rarely robust).
Your runner just records `kv_policy/kv_budget`; it does not implement compression itself.

---

## `run/` (Phase C: orchestration + runner + telemetry)

- `orchestrate.py`
  - **Role:** run-grid controller:
    - loops over `kv_budgets`
    - creates a `run_id` per budget
    - calls `runner.run_one_setting(...)`
  - **I/O:** writes `runs`, then runner writes requests/responses/telemetry.

- `runner.py`
  - **Role:** core execution loop:
    1. query manifest entries
    2. build prompt/messages (via `prompts/` if you add it later; for now embed in runner or a helper)
    3. call engine
    4. persist request/response/telemetry to SQLite
  - **I/O:** writes to SQLite; optional debug exports.

- `retries.py`
  - **Role:** retry policy + backoff for transient HTTP errors/timeouts.
  - **I/O:** none (used by openai_compat + runner).

- `telemetry.py`
  - **Role:** measure:
    - latency (total)
    - optional TTFB
    - token usage
    - optional GPU memory via NVML if available on server
  - **I/O:** writes telemetry rows to SQLite.

---

## `eval/` (Phase D: scoring + failure classification)

- `metrics.py`
  - **Role:** EM/F1 implementation (normalize → token overlap).
  - **I/O:** none; pure functions.

- `failure_taxonomy.py`
  - **Role:** classify failures, e.g.:
    - `EMPTY_OUTPUT`
    - `TRUNCATED`
    - `FORMAT_ERROR`
    - `REFUSAL`
    - `TIMEOUT`
  - **I/O:** none; returns labels.

- `score.py`
  - **Role:** join predictions with gold answers and write:
    - `scores`
    - `failures` (if applicable)
  - **I/O:** reads/writes SQLite.

---

## `analysis/` (Phase E: aggregation + transition detection + plots)

- `queries.py`
  - **Role:** SQL query helpers for pulling run-level and bin-level data from SQLite.
  - **I/O:** reads SQLite.

- `aggregate.py`
  - **Role:** compute bin-level aggregates:
    - mean accuracy
    - std
    - failure rate
    - latency percentiles
    - memory percentiles (if available)
  - **I/O:** writes `bin_stats` table, and optional CSV exports.

- `bootstrap.py`
  - **Role:** bootstrap confidence intervals for:
    - mean accuracy per bin
    - (optionally) std or difference vs baseline
  - **I/O:** writes CI fields into `bin_stats` and/or separate table.

- `transition.py`
  - **Role:** detect instability region.
  - **Recommended minimal detector (interpretable + robust):**
    - compute z-scored curves across bins:
      - `z_std(bin)` and `z_fail(bin)` and `z_acc_drop(bin)`
    - define transition bins where:
      - `z_std` is high AND/OR `z_fail` rises sharply,
      - while mean accuracy has not yet fully collapsed.
    - output:
      - `[start_bin, end_bin]` + rationale JSON.
  - **I/O:** writes `transition_summary`.

- `plots.py`
  - **Role:** save plots to `runs/<exp_group_id>/plots/`:
    - `acc_vs_len.png` (mean ± CI)
    - `std_vs_len.png`
    - `failrate_vs_len.png`
    - `latency_vs_len.png`
  - **I/O:** writes PNG files.

---

## `report/` (Phase F: report builder)

- `build.py`
  - **Role:** generate `runs/<exp_group_id>/report.md` using Jinja template.
  - **I/O:**
    - reads SQLite + plot paths
    - writes `report.md`

- `templates/report.md.jinja`
  - **Role:** report skeleton:
    - setup table
    - figures
    - key findings bullets
    - transition zone summary
    - limitations + future work

---

## `tests/` (Guardrails)

- `test_cli_smoke.py`
  - **Role:** ensure CLI commands import and run basic no-op flows.

- `test_manifest.py`
  - **Role:** ensure binning + manifest schema stable.

- `test_metrics.py`
  - **Role:** correctness for EM/F1, normalization.

---

## `data/processed/` and `runs/`

- `data/processed/<dataset_id>/manifest.json`
  - **Role:** portable manifest snapshot (helpful for debugging and non-DB portability).
  - **I/O:** written by Phase B; optionally read by Phase C if DB not present.

- `runs/<exp_group_id>/kv_transition.sqlite`
  - **Role:** canonical experiment store (single source of truth).
  - **I/O:** written/read by all phases.

- `runs/<exp_group_id>/plots/*.png`
  - **Role:** analysis artifacts.
  - **I/O:** written by Phase E; embedded by report.

- `runs/<exp_group_id>/report.md`
  - **Role:** final deliverable.
  - **I/O:** written by Phase F.

---

## 4) Artifact flow: what consumes/produces what (explicit I/O map)

### Inputs (human-controlled)
- `config/*.yaml`
- environment vars (`OPENAI_API_KEY`, `BASE_URL`, etc.)
- optionally cached dataset in `data/raw/` (not required)

### Outputs (machine-generated)
- `data/processed/<dataset_id>/manifest.json`
- `runs/<exp_group_id>/kv_transition.sqlite`
- `runs/<exp_group_id>/plots/*.png`
- `runs/<exp_group_id>/report.md`

### Phase-by-phase I/O summary
- **prepare**
  - reads: config + dataset source
  - writes: manifest.json + DB tables (`datasets/examples/bins/manifest_entries`)
- **run**
  - reads: manifest entries (DB)
  - writes: (`runs/requests/responses/telemetry`)
- **score**
  - reads: responses + gold answers (DB)
  - writes: (`scores/failures`)
- **analyze**
  - reads: scores + telemetry + bins
  - writes: (`bin_stats/transition_summary`) + plots
- **report**
  - reads: bin_stats + transition_summary + plots
  - writes: report.md

---

## 5) End goal/output of the project (tool definition)

### The CLI tool (primary deliverable)
A CLI that supports “one-command reproducibility”:

- `python -m kv_transition prepare -c config/experiments/snapkv_longbench.yaml`
- `python -m kv_transition run -c config/experiments/snapkv_longbench.yaml`
- `python -m kv_transition score -c config/experiments/snapkv_longbench.yaml`
- `python -m kv_transition analyze -c config/experiments/snapkv_longbench.yaml`
- `python -m kv_transition report -c config/experiments/snapkv_longbench.yaml`
- `python -m kv_transition all -c config/experiments/snapkv_longbench.yaml`

### What the tool “does” in plain terms
Given a dataset and a KV-compression setting, it:
1. builds a length-binned evaluation set,
2. runs the model across KV budgets,
3. measures quality + latency + failures,
4. identifies where compression introduces instability,
5. generates plots + a report that summarizes the tradeoff.

---

## Practical scope locks (to prevent explosion)

**Hard scope (do not exceed):**
- 1 model
- 1 LongBench task (subset)
- 1 KV policy
- 3 budgets: {1.0, 0.5, 0.2}
- 5 length bins
- n_per_bin: start 5; increase only near transition bins

**Explicit future work section (allowed but not implemented):**
- multiple KV policies
- multiple models
- more tasks
- adaptive budget policy
- UI/dashboard

---

## Compatibility notes with ContextCliff (light-touch)
Your scaffold already matches “manifest → runner → scorer → profiler/report” separation.
To keep compatibility easy later:
- keep `manifest.json` schema stable and versioned
- keep engine abstraction strictly OpenAI-compatible
- keep metrics + binning modular
- keep experiment artifacts self-contained in `runs/<exp_group_id>/`

If you share your existing ContextCliff modules later, the likely mapping is:
- CC manifest/binning → `data/*`
- CC runner clients → `engines/*` + `run/runner.py`
- CC scoring/profiling → `eval/*` + `analysis/*`
- CC report → `report/*`

---

## Minimal “definition of done” checklist (submission-ready)
- [ ] `prepare` produces bins and writes manifest + DB
- [ ] `run` completes for budgets 1.0/0.5/0.2 and logs requests/responses/telemetry
- [ ] `score` writes EM/F1 + failure taxonomy
- [ ] `analyze` writes bin_stats + transition_summary + 4 plots
- [ ] `report` renders one clean report.md referencing plots
- [ ] `tests` pass (metrics, manifest schema, CLI smoke)

--- 

## Reference: course project topic alignment
This blueprint is a scoped version of **Topic 2 → Sparse Attention → [Analytical] Analysis of KV Cache Compression Policies** (single method, single task, tight grid) while still producing a clear efficiency–accuracy tradeoff story.