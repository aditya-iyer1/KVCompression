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

---

## Information Required Before Phase 2

To proceed to Phase B (Dataset + Manifest Layer), confirm:

1. Which exact LongBench task string should be canonical for snapkv_longbench.yaml?
2. Are we pinning tokenizer to a specific model family (e.g., same as model.name), or do we want tokenizer explicitly configurable?
3. Should manifest.json be required alongside DB, or DB-only sufficient?

Once confirmed, we will select the first Phase B file.






# Phase 2

FTM: 

DONE:
- `src/kv_transition/data/longbench_loader.py`
- `src/kv_transition/data/normalize.py`
- `src/kv_transition/data/tokenizer.py`
- `src/kv_transition/data/binning.py`
- `src/kv_transition/data/manifest.py`

- `db/connect.py`

NOT DONE:





Also Phase B requires DB support, so you’ll need these in the same phase boundary if they don’t exist yet (still structural Phase B dependency, even though they live under db/):

- `db/schema.py`
- `db/dao.py`


