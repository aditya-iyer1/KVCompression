# Phase 1

FTM:

DONE:
- `config/default.yaml` 
- `config/experiements/snapkv_longbench.yaml`
- `src/kv_transition/settings.py`
- `src/kv_transition/paths.py`
- `src/kv_transition/cli.py` (only enough to load/validate config + resolve run/db paths)
- `.env.example`
- `/src/kv_transition/__main__.py` (if not already delegating cleanly to `cli.py`)

# Phase 1 Completion Report — Configuration & Reproducibility Layer

## Files Completed
- config/default.yaml  
- config/experiments/snapkv_longbench.yaml  
- src/kv_transition/settings.py  
- src/kv_transition/paths.py  
- src/kv_transition/cli.py  
- src/kv_transition/__main__.py  
- .env.example  

---

## Responsibilities Implemented

### 1. Config Schema & Baseline
- Established canonical config structure:
  - dataset, binning, model, engine, kv, decoding, output, db
- Implemented base → experiment YAML merge.
- Added environment variable resolution (${VAR}).
- Enforced validation for required keys + types.
- Supported dot-notation CLI overrides.

---

### 2. Canonical Path Resolution
- Implemented repository root detection.
- Canonical builders:
  - data_dir()
  - processed_dir(dataset_id)
  - runs_dir()
  - run_dir(exp_group_id)
  - db_path(exp_group_id, db_path_cfg)
- Enforced consistent DB location:
  - auto → runs/<exp_group_id>/kv_transition.sqlite
- No side effects on import.

---

### 3. Minimal Phase-A CLI
- Commands:
  - validate
  - resolve
- Required flags:
  - -c / --config
- Optional:
  - --override (dot-notation)
  - --print-config
- Exit codes:
  - 0 success
  - 2 validation/argument error
- No Phase B–F logic included.

---

### 4. Module Delegation
- __main__.py cleanly delegates to cli.main().
- Proper exit code propagation via SystemExit.
- No runtime side effects on import.

---

## Architectural Integrity Check

Phase A now satisfies blueprint requirements:

- Reproducible experiment configuration via YAML.
- Deterministic run directory + SQLite path resolution.
- Engine-agnostic (OpenAI-compatible base_url abstraction).
- CLI-first execution model.
- No leakage into dataset / engine / evaluation layers.
- One DB per experiment group enforced structurally.

No structural drift detected.

---

## Public API (Phase A Surface)

- load_settings(config_path, overrides=None) -> dict
- paths.run_dir(exp_group_id) -> Path
- paths.db_path(exp_group_id, db_path_cfg) -> Path
- CLI:
  - python -m kv_transition validate -c <config>
  - python -m kv_transition resolve -c <config>

---

## Phase A Definition of Done Status

- [x] Default config defined
- [x] Experiment config defined
- [x] Schema validation enforced
- [x] Env resolution implemented
- [x] Canonical DB path logic implemented
- [x] CLI validation + resolution commands operational
- [x] .env.example documented

Phase 1 complete.




# Phase 2

FTM: 

DONE:
- `src/kv_transition/data/longbench_loader.py`
- `src/kv_transition/data/normalize.py`
- `src/kv_transition/data/tokenizer.py`
- `src/kv_transition/data/binning.py`
- `src/kv_transition/data/manifest.py`

- `db/connect.py`
- `db/schema.py`
- `db/dao.py`

# Phase 2 Completion Report — Dataset + Manifest Layer

## Files Completed

### Data Layer
- data/longbench_loader.py  
- data/normalize.py  
- data/tokenizer.py  
- data/binning.py  
- data/manifest.py  

### DB Layer (Phase B scope)
- db/connect.py  
- db/schema.py  
- db/dao.py  

---

## Responsibilities Implemented

### 1. Dataset Loading (Authoritative Task Binding)
- LongBench loader fetches task subset using exact, case-sensitive dataset key.
- `list_longbench_tasks()` dynamically queries dataset metadata (no hardcoding).
- Raw examples returned untouched (no implicit normalization).
- Repo-local cache under `data/raw/longbench/`.

This preserves:
- Task immutability per experiment group.
- Alignment between config.dataset.task and actual dataset key.

---

### 2. Canonical Example Normalization

Canonical schema enforced:

{
example_id: str,
question: str,
context: str,
answers: list[str],
meta: dict
}

- Deterministic ID generation (stable across runs).
- Answers always normalized to list[str].
- JSON-serializable output.
- Exact task string preserved in meta.

No tokenization or binning logic embedded here.

---

### 3. Tokenization Layer (Configurable + Deterministic)

- `tokenizer.name` explicitly supported.
- Defaults to `model.name` if unspecified.
- `count_tokens()`:
  - Uses tiktoken when available.
  - Falls back to deterministic approximation.
- Stable token counts across runs.

This locks in reproducibility for binning.

---

### 4. Length-Based Binning

- Quantile-style bin edge computation.
- Deterministic bin assignment.
- No randomness.
- Guaranteed:
  - Exactly `n_bins` edges.
  - Every example assigned to `[0, n_bins-1]`.

This establishes the structural backbone for transition analysis later.

---

### 5. Manifest Contract (Portable Artifact)

`manifest.json` now required and generated at:

data/processed/<dataset_id>/manifest.json

Dataset ID convention:

{dataset_name}__{task}

Manifest contains:
- dataset metadata (name, task)
- tokenizer_name
- n_bins, n_per_bin
- bin_edges
- entries (example_id, bin_idx, token_len)
- examples (full canonical examples)

Selection logic:
- Deterministic.
- Up to `n_bins * n_per_bin`.
- Sorted by (token_len, example_id).

Manifest is now the durable contract between Phase B and Phase C.

---

### 6. SQLite Backbone (Phase B Scope Only)

#### Tables Created
- experiments
- datasets
- examples
- bins
- manifest_entries

#### Properties
- Foreign keys enforced.
- Idempotent schema creation.
- Indexed for expected queries.
- JSON fields stored as TEXT.

DAO supports:
- Upserting datasets
- Bulk inserting examples
- Inserting bins
- Inserting manifest entries
- Reading manifest entries + examples

DB is now the canonical source of truth.
manifest.json is the portable snapshot.

---

## Architectural Integrity Check

Phase B satisfies blueprint constraints:

- DB as source of truth.
- manifest.json required and portable.
- Tokenizer explicitly configurable.
- Task string exact and preserved.
- No leakage into Phase C (no engine logic).
- No Phase C schema tables prematurely introduced.
- Deterministic artifact generation.

No structural drift detected.

---

## Phase B Definition of Done Status

- [x] LongBench loader implemented
- [x] Canonical normalization schema enforced
- [x] Configurable tokenizer abstraction
- [x] Deterministic binning
- [x] Deterministic per-bin sampling
- [x] manifest.json written
- [x] SQLite schema (Phase B subset) initialized
- [x] DAO persistence implemented
- [x] DB + manifest parity achieved

Phase 2 complete.


# Phase 3

FTM:

DONE:

Engines:
- `engines/base.py`
- `engines/openai_compat.py`
- `engines/endpoints.py`

Run Layer:
- `run/orchestrate.py`
- `run/runner.py`
- `run/retries.py`
- `run/telemetry.py`


DB updates (Phase C schema + DAO surface)
- `db/schema.py` (extend with Phase C tables/indexes)
- `db/dao.py` (extend with run/request/response/telemetry ops)

# Phase 3 Completion Report — Inference & Logging Layer

## Files Completed

### Engines
- engines/base.py  
- engines/openai_compat.py  
- engines/endpoints.py  

### Run Layer
- run/orchestrate.py  
- run/runner.py  
- run/retries.py  
- run/telemetry.py  

### DB (Phase C Extension)
- db/schema.py (extended)
- db/dao.py (extended)

---

## Responsibilities Implemented

## 1. Engine Abstraction (Provider-Agnostic)

### Base Interface
- `BaseEngine.generate(...)` defines the stable contract.
- `EngineResult` encapsulates:
  - text
  - usage
  - finish_reason
  - timings
  - raw response

Runner is now completely provider-agnostic.

### OpenAI-Compatible Engine
- Plain HTTP client (no SDK dependency).
- Works with:
  - OpenAI API
  - vLLM OpenAI server
  - SGLang OpenAI server
- Configurable via:
  - base_url
  - api_key (optional)
  - model name
- Returns structured `EngineResult`.

### Endpoint Resolution
- `build_engine(settings)` resolves:
  - base_url
  - API key from env
  - model name
- Runner does not read environment directly.

---

## 2. Orchestration Layer

### orchestrate(settings)

Responsibilities:
- Resolve run paths + DB.
- Initialize schema (idempotent).
- Build engine.
- Loop over `kv.budgets`.
- Create one run per budget.
- Call `run_one_setting(...)`.

Design properties:
- Sequential (no concurrency yet).
- No scoring or analysis logic.
- Clean separation of control flow vs execution.

---

## 3. Execution Layer (Runner)

### run_one_setting(...)

For a single `(run_id, kv_budget)`:

1. Resolve dataset_id.
2. Load manifest entries from DB.
3. Fetch canonical examples.
4. Build deterministic messages:
   - system: instruction
   - user: context + question
5. Call engine.
6. Persist:
   - run
   - request
   - response
   - telemetry
   - failure (if exception)

Error handling:
- Failures logged.
- Loop continues.
- No global crash.

Runner now produces structured inference logs.

---

## 4. Retry Utility

- Exponential backoff with jitter.
- Retry on:
  - timeouts
  - connection errors
  - HTTP 429
  - HTTP 5xx
- Re-raises non-retryable errors immediately.

Ready for integration in engine calls (optional hook).

---

## 5. Telemetry Normalization

`extract_telemetry()` produces normalized JSON-safe telemetry:

- latency_s
- ttfb_s
- prompt_tokens
- completion_tokens
- total_tokens
- finish_reason
- notes_json

Stable and provider-agnostic.

---

## 6. Database Schema — Phase C Extension

### New Tables Added

- runs
- requests
- responses
- telemetry
- failures

All:
- Idempotent creation
- Foreign keys enforced
- Indexed for run-level queries

Phase B tables preserved intact.

---

## 7. DAO — Phase C Surface

New write operations:

- upsert_run
- insert_request
- insert_response
- upsert_telemetry
- upsert_failure

New read helpers:

- run_exists
- count_requests_for_run

Properties:
- Transaction-safe
- Deterministic IDs supplied externally
- JSON fields consistently serialized
- Phase B APIs untouched

---

## End-to-End Capability Achieved

Given:
- Config
- Prepared manifest (Phase B)

The system can now:

1. Open DB
2. Create run for each KV budget
3. Execute all manifest entries
4. Persist:
   - inputs
   - outputs
   - timing
   - token usage
   - errors
5. Complete without crashing on single failure

The harness now produces structured experiment logs suitable for scoring and analysis.

---

## Architectural Integrity Check

- No modification to Phase B data pipeline.
- Schema extended in-place (idempotent).
- Engine fully abstracted.
- Runner isolated from scoring.
- No leakage into analysis/report layers.
- DB remains single source of truth.

No structural drift detected.

---

## Phase 3 Definition of Done Status

- [x] Engine abstraction defined
- [x] OpenAI-compatible engine implemented
- [x] Endpoint resolution isolated
- [x] Orchestration grid loop implemented
- [x] Runner executes manifest entries
- [x] Retry utility implemented
- [x] Telemetry normalization implemented
- [x] Schema extended for inference logging
- [x] DAO extended for Phase C persistence
- [x] End-to-end run persistence validated

Phase 3 complete.

# Phase 4

Eval Layer: (Phase D)
- `eval/metrics.py` (DONE)
- `eval/failure_taxonomy.py` (DONE)
- `eval/score.py` (DONE)


# Phase 4 Completion Report — Scoring & Failure Taxonomy (Phase D)

## Files Completed

- eval/metrics.py  
- eval/failure_taxonomy.py  
- eval/score.py  

---

## Responsibilities Implemented

## 1. Core Metrics (Pure Functions)

### Text Normalization
- Lowercase
- Trim + collapse whitespace
- Remove punctuation
- Deterministic output

### Exact Match (EM)
- 1.0 if normalized strings identical
- 0.0 otherwise
- Both empty → 1.0
- One empty → 0.0

### F1 Score
- Token-overlap F1
- Whitespace tokenization
- Deterministic
- Handles empty strings safely

### Multi-Gold Handling
- `best_exact_match(pred, golds)`
- `best_f1(pred, golds)`
- Max over gold answers

Metrics are:
- Pure
- Dependency-free
- Deterministic
- Model-agnostic

---

## 2. Stable Failure Taxonomy

Defined stable failure labels:

- EMPTY_OUTPUT  
- TRUNCATED  
- FORMAT_ERROR  
- REFUSAL  
- TIMEOUT  
- ENGINE_ERROR  

Classifier characteristics:

- Conservative (minimal pattern matching)
- Deterministic
- Priority-ordered
- Returns `None` for normal responses
- Compatible with Phase C raw error logs

This prevents unstable failure rate measurement later.

---

## 3. Run-Scoped Scoring Stage

### score_run(conn, run_id)

Responsibilities:

1. Read Phase C logs:
   - requests
   - responses
   - failures
   - examples (gold answers)

2. Compute:
   - EM
   - F1
   - normalized prediction
   - normalized gold reference

3. Classify failures via taxonomy.

4. Persist:
   - scores(request_id, em, f1, pred_norm, gold_norm)
   - Update failures table with taxonomy labels.

No model calls.
No inference logic.
Purely post-hoc scoring.

---

## 4. Schema Handling

- `scores` table created idempotently (IF NOT EXISTS).
- Does not modify Phase C schema.
- Does not require schema migrations.
- CSV fallback included (defensive only).

Scoring remains cleanly separated from inference.

---

## Architectural Integrity Check

- Phase C remains pure execution + logging.
- Phase D reads from DB only.
- No re-execution of model.
- Metrics are isolated.
- Failure taxonomy stable and centralized.
- No schema drift introduced.
- DB remains canonical source of truth.

Separation between:
- Inference (Phase C)
- Evaluation (Phase D)
is preserved.

No architectural drift detected.

---

## Phase D Definition of Done

- [x] Deterministic EM implementation
- [x] Deterministic F1 implementation
- [x] Multi-gold handling
- [x] Failure taxonomy defined
- [x] Failure classifier implemented
- [x] score_run(run_id) implemented
- [x] Scores written to DB
- [x] Failures normalized into taxonomy
- [x] No inference reruns required
- [x] Fully DB-driven scoring

Phase 4 complete.

# Phase 5


Analysis Layer: (Phase E, immediately after D)
- `analysis/queries.py` (DONE)
- `analysis/aggregate.py` (DONE)
- `analysis/bootstrap.py` (DONE)
- `analysis/transition.py` (DONE)
- `analysis/plots.py` (DONE)

- `db/schema.py` (extend with bin_stats, transition_summary tables + indexes) (DONE)
- `db/dao.py` (extend with aggregation read/write helpers) (DONE)

CLI Wiring: (To expose D/E/F as stages; still consistent with blueprint)
- `src/kv_transition/cli.py` (add score, analyze, report, all routing)


# Phase 5 Completion Report — Aggregation, Transition Detection, and Analysis Artifacts (Phase E)

## Files Completed

### Analysis Layer
- analysis/queries.py  
- analysis/aggregate.py  
- analysis/bootstrap.py  
- analysis/transition.py  
- analysis/plots.py  

### DB Persistence (Phase E Extension)
- db/schema.py (extended with Phase E tables)
- db/dao.py (extended with Phase E helpers)

### CLI Exposure (Phase D/E routing only)
- src/kv_transition/cli.py (extended)

---

## Responsibilities Implemented

## 1. Read-Only Analysis Queries (DB → Rows)

`analysis/queries.py` provides explicit joins to produce per-request, per-bin rows for a run:

- Joins:
  - requests
  - scores
  - telemetry
  - manifest_entries (bin assignment)
  - failures (failure indicator)
- Outputs stable fields required for aggregation:
  - bin_idx, em, f1, latency_s, token counts, failure flag
- Deterministic ordering (bin_idx, request_id)

This is the single source for aggregation inputs (no inference loop dependency).

---

## 2. Bin-Level Aggregation (Per Run)

`analysis/aggregate.py` computes per-bin stats for a single run:

- n
- acc_mean (mean F1; primary “accuracy”)
- acc_std (population std)
- em_mean
- fail_rate
- latency percentiles: p50/p95
- token usage percentiles: p50/p95

Pure computation, no DB writes, deterministic.

---

## 3. Deterministic Bootstrap Confidence Intervals

`analysis/bootstrap.py` implements percentile bootstrap for mean accuracy:

- Deterministic seed (default 1337; CLI reads settings.analysis.seed if present)
- Adds:
  - acc_ci_low
  - acc_ci_high
per bin

Standard library only.

---

## 4. Transition Detection (Aggregates → Transition Summary)

`analysis/transition.py` detects the instability transition zone from aggregated curves:

- Input: list of runs with kv_budget + bin acc_mean values
- Rule: first budget where overall mean accuracy drops by >= drop_threshold vs prior higher budget
- Outputs:
  - transition_budget, pre_budget
  - acc_pre, acc_post
  - drop
  - method
  - transition_bin_idx (largest per-bin drop between pre/post)

Consumes only aggregated/bin-level stats, not raw request loops.

---

## 5. Minimal Plotting Artifacts

`analysis/plots.py` generates reproducible matplotlib plots:

- acc_by_bin.png (one curve per kv_budget)
- fail_by_bin.png
- latency_p50_by_bin.png (only if data present)

Saved under:
- runs/<exp_group_id>/plots/

Deterministic ordering:
- kv_budget sorted ascending
- bin_idx ascending

---

## 6. DB Persistence for Aggregates + Transition

### Schema extension (idempotent)

Added:
- bin_stats (PK: run_id, bin_idx)
- transition_summary (PK: exp_group_id)

Indexes:
- bin_stats(run_id)
- bin_stats(dataset_id, bin_idx)

Reserved keyword handled:
- `"drop"` column quoted in SQL.

### DAO extension

Added:
- upsert_bin_stats(run_id, dataset_id, bin_stats)
- upsert_transition_summary(summary)  (handles `"drop"` + created_at)
- get_bin_stats_for_run(run_id)
- get_transition_summary(exp_group_id)

Phase B/C functions preserved.

---

## 7. CLI Stage Exposure (Score + Analyze)

`src/kv_transition/cli.py` now exposes:

- `score` (Phase D)
  - DB-only scoring stage
  - run-scoped (single run_id or all runs for exp_group_id)

- `analyze` (Phase E)
  - DB-only aggregation + bootstrap + transition detection + plots
  - Writes:
    - bin_stats rows per run
    - transition_summary per exp_group_id
    - plots under runs/<exp_group_id>/plots/

Run discovery:
- if `--run-id` omitted, queries runs for exp_group_id ordered by kv_budget DESC.

No Phase F report wiring added (by design).

---

## Architectural Integrity Check

- Phase E consumes:
  - Phase C logs + Phase D scores only
- No model calls.
- Transition detection depends only on aggregated/bin-level outputs.
- Aggregates persisted to DB as canonical analysis products.
- Plots are derived artifacts.

No structural drift detected.

---

## Phase E Definition of Done

- [x] Per-run bin-level query surface implemented
- [x] Bin-level aggregates computed deterministically
- [x] Percentile bootstrap CIs (deterministic seed)
- [x] Transition detection implemented (aggregate-driven)
- [x] Matplotlib plots generated as reproducible artifacts
- [x] bin_stats + transition_summary tables added
- [x] DAO supports aggregate persistence + retrieval
- [x] CLI exposes analyze stage (DB-only)

Phase 5 complete.

# Phase F

Report layer
- `report/build.py` (DONE)
- `report/templates/report.md.jinja` (DONE)

CLI exposure (Phase F routing + “all” pipeline)
- `src/kv_transition/cli.py` (add report and all commands) (DONE)

Optional DB touch (only if missing from earlier phases) (SKIPPED)
- `db/dao.py` (read helpers needed by report builder if not already present: experiment metadata, run list w/ kv_budget, bin_stats, transition_summary, plot paths) (Skip_)
- `db/schema.py` (only if you want a tiny report_artifacts table; not required by the blueprint)


# Phase 6 Completion Report — Report Generation & Pipeline Finalization (Phase F)

## Files Completed

### Report Layer
- report/build.py  
- report/templates/report.md.jinja  

### CLI Exposure
- src/kv_transition/cli.py (extended with `report` and `all`)

No required schema changes for Phase F.

---

## Responsibilities Implemented

## 1. Report Builder (DB → Markdown)

### build_report(conn, settings)

Produces:

runs/<exp_group_id>/report.md

Inputs:
- SQLite:
  - experiments
  - runs
  - bin_stats
  - transition_summary
- Filesystem:
  - plots under `runs/<exp_group_id>/plots/`

Properties:
- No scoring.
- No aggregation.
- No transition recomputation.
- No inference.
- Purely reads persisted Phase E outputs.

Template rendered via Jinja2 using a single structured context:
- exp_group_id
- experiment metadata (if present)
- runs (with embedded bin_stats)
- transition summary
- plot paths
- generated_at timestamp

Deterministic given same DB + plot files.

---

## 2. Markdown Template

`report.md.jinja` renders:

1. Title
2. Overview (metadata + timestamp)
3. Transition Summary (if present)
4. Plots (only those that exist)
5. Runs Summary Table
6. Per-Run Bin Stats tables

Features:
- Graceful handling of missing metadata
- Graceful handling of missing plots
- Graceful handling of missing transition
- Consistent numeric formatting
- Markdown-only (no HTML)

---

## 3. CLI Exposure — Phase F Routing

### `report`
DB-only command:

python -m kv_transition report -c config.yaml

- Opens DB
- Calls build_report()
- Writes report.md
- Prints report path

---

### `all`
Full pipeline (DB-only; no inference):

python -m kv_transition all -c config.yaml

Sequentially runs:
1. Phase D — score
2. Phase E — analyze
3. Phase F — report

Optional:

–run-id

applies to score + analyze; report remains experiment-group scoped.

Single DB open.
Single schema init.
Clean stage boundaries.

---

## Architectural Integrity Check

Separation boundaries preserved:

| Phase | Responsibility | Recomputed? |
|--------|----------------|------------|
| C | Inference + logging | No |
| D | Scoring | No |
| E | Aggregation + transition | No |
| F | Report rendering | No |

Report builder:
- Reads DB only.
- Reads filesystem plots only.
- Does not depend on inference logic.
- Does not depend on scoring logic.
- Does not mutate experiment state (except writing report file).

DB remains the canonical source of structured experiment truth.

No structural drift detected.

---

## Phase F Definition of Done

- [x] Markdown report generated from DB
- [x] Jinja template created and validated
- [x] No recomputation inside report layer
- [x] Plots embedded conditionally
- [x] Transition summary rendered conditionally
- [x] CLI exposes `report`
- [x] CLI exposes `all` (D→E→F)
- [x] Stage isolation maintained
- [x] Deterministic output given fixed DB + plots

Phase 6 complete.

---

# System Status — End-to-End Capability

The system can now:

1. Prepare dataset + manifest (Phase B)
2. Execute inference across KV budgets (Phase C)
3. Persist structured logs (requests/responses/telemetry)
4. Compute EM/F1 + normalize failures (Phase D)
5. Aggregate bin-level statistics + bootstrap CIs (Phase E)
6. Detect transition zone (Phase E)
7. Generate plots (Phase E)
8. Render full experiment report (Phase F)

All stages are:
- Modular
- DB-driven
- Reproducible
- Deterministic (with fixed seeds)
- Cleanly separated

The blueprint is fully realized.













# Pass 2

Next steps (architecture-owned)

1) Lock the “Reproducible Experiment Contract”

Add/verify these invariants are persisted for every exp_group_id:
	•	config snapshot (already in experiments.config_yaml) (COMPLETE)
	•	model + base_url + engine name (in runs)  (COMPLETE)
	•	prompt_template_version (either runs or experiments)
	•	dataset_id (COMPLETE) + tokenizer_name (COMPLETE) + bin_edges (COMPLETE) (already in manifest/DB)

This is the minimum to guarantee future comparability.

2) Define the Phase-0 “Submission Run” profile

Create one pinned experiment config intended for grading/submission:
	•	1 task
	•	n_bins=5
	•	budgets=[1.0, 0.5, 0.2]
	•	n_per_bin small enough for runtime
	•	temperature=0
This becomes the canonical run you can cite in the report.

3) Run-to-run comparability & guardrails

Architectural guardrails to prevent silent mismatch:
	•	refuse to run if dataset_id/task/tokenizer differs from what’s already in DB for that exp_group_id (unless explicit --force) (COMPLETE)
	•	refuse to score/analyze/report if required tables are missing for the target run/exp_group_id (COMPLETE)

4) Minimal “Definition of Done” validation suite

Add a small test surface (even if just smoke-level) that asserts:
	•	prepare → manifest exists + DB tables populated (COMPLETE)
	•	run → requests/responses counts match manifest entries (COMPLETE)
	•	score → scores count matches requests (COMPLETE)
	•	analyze → bin_stats exists for all (run_id, bin_idx) (COMPLETE)
	•	report → report.md exists and references plot paths that exist (COMPLETE)

(Keep these as smoke tests; no performance assertions.)
