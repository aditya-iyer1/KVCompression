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
- `analysis/queries.py`
- `analysis/aggregate.py`
- `analysis/bootstrap.py`
- `analysis/transition.py`
- `analysis/plots.py`


# Phase 6

Report Layer: (Phase F)
- `report/build.py`
- `report/templates/report.md.jinja`

CLI Wiring: (To expose D/E/F as stages; still consistent with blueprint)
- `src/kv_transition/cli.py` (add analyze command routing only; no report yet unless you’re merging Phase F next)

DB Updated needed for D/E/F tables:
- `db/schema.py` (extend with bin_stats, transition_summary tables + indexes)
- `db/dao.py` (extend with aggregation read/write helpers)