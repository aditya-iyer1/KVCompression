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


NOT DONE:

Run layer


- `run/retries.py`
- `run/telemetry.py`

DB updates (Phase C schema + DAO surface)
- `db/schema.py` (extend with Phase C tables/indexes)
- `db/dao.py` (extend with run/request/response/telemetry ops)

(You should not need to touch Phase B data files for Phase C.)

