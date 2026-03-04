# PASS Y — KV Budget Realization via KVCache-Factory

## Goal
Replace the **non-functional `kv_budget` label** in the harness with a backend that **actually performs KV-cache compression** so budget sweeps produce real behavioral differences.

---

# Current Phase
PASS Y — Backend swap + verification of real KV compression.

Backend chosen: **KVCache-Factory**

Integration mode: **Mode 2 — LocalEngine adapter (planned)**

---

# Relevant Dependencies

Backend
- KVCache-Factory (SnapKV / PyramidKV / H2O / StreamingLLM implementations)

Harness
- kv_transition evaluation harness
- SQLite experiment DB
- pipeline:  
  `prepare → run → score → analyze → report`

Key harness modules

src/kv_transition/run/orchestrate.py
src/kv_transition/run/runner.py
src/kv_transition/engines/base.py
src/kv_transition/engines/openai_compat.py
src/kv_transition/engines/endpoints.py

Execution environment
- GPU notebook for KVCache-Factory runs
- LOCAL machine for harness edits

---

# What Was Done

## 1. Identified Root Cause of Previous Invalid Results
Earlier experiments used:

vLLM OpenAI server
kv_budget parameter

However:

- vLLM does **not implement SnapKV**
- OpenAI server **ignores `kv_budget`**

Observed symptoms:

- identical outputs across budgets
- identical latency
- identical metrics

Conclusion:

**`kv_budget` knob was a no-op.**

PASS Y exists to fix this.

---

# 2. Selected Compression Backend

Chosen backend:

**KVCache-Factory**

Reason:

- supports real KV eviction algorithms
- LongBench evaluation scripts already exist
- budget control exposed via CLI parameters

Example invocation:

python run_longbench.py 
–dataset trec 
–model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 
–eval_batch_size 1 
–max_num_examples 20 
–method snapkv 
–attn_implementation sdpa 
–max_capacity_prompts -1 
–max_capacity_prompts_ratio 1.0 
–pruning_ratio 0.0 
–save_dir runs_ratiofix_1p0

Budget 0.2 run:

–max_capacity_prompts_ratio 0.2
–pruning_ratio 0.8

---

# 3. Verified Budget is Real (Hard Evidence)

Two runs executed:

runs_ratiofix_1p0
runs_ratiofix_0p2

Files generated:

runs_ratiofix_1p0/…/snapkv.json
runs_ratiofix_0p2/…/snapkv.json

Verification:

n = 20 examples
diff_pred = 20

Meaning:

- every prediction changed between budgets

Example difference:

pred_1p0:
Other location
Question: What is the name of the first American president…

pred_0p2:
Other location
Question: What is the name of the largest city in the United States…

Interpretation:

- compression changes retained attention context
- output generation diverges

This confirms:

**KVCache-Factory is applying compression.**

---

# 4. Budget Semantics Mapping

Harness budget:

kv_budget ∈ {1.0, 0.2}

KVCache-Factory parameters:

| Harness Budget | KVCache-Factory |
|---|---|
| 1.0 | `max_capacity_prompts_ratio=1.0`, `pruning_ratio=0.0` |
| 0.2 | `max_capacity_prompts_ratio=0.2`, `pruning_ratio=0.8` |

Meaning:

retained KV tokens ≈ budget × prompt_length

---

# 5. Confirmed Harness Engine Architecture

Key discovery:

Runner calls engines through **BaseEngine interface**

engine.generate(…)

Source:

src/kv_transition/run/runner.py

Call site:

result = engine.generate(
messages=messages_to_send,
model=model_name,
temperature=temperature,
max_tokens=max_tokens,
)

Engine object constructed in:

src/kv_transition/run/orchestrate.py

via:

engine, model_name = endpoints.build_engine(settings)

---

# 6. Current Engine Builder Behavior

File:

src/kv_transition/engines/endpoints.py

Current behavior:

build_engine() → OpenAICompatEngine

Hardcoded:

engine_name = “openai_compat”

Implication:

Harness currently **cannot run local engines**.

---

# 7. Integration Strategy (Chosen)

Integration mode:

**Mode 2 — LocalEngine adapter**

Reason:

- KVCache-Factory exposes CLI runner
- no HTTP server exists
- minimal harness modification

Planned adapter:

KVFactoryEngine(BaseEngine)

Responsibilities:

generate()
↓
convert messages → prompt
↓
invoke run_longbench.py subprocess
↓
parse snapkv.json output
↓
return EngineResult

No changes required in:

runner.py
orchestrate.py

Only modifications required:

src/kv_transition/engines/endpoints.py

---

# PASS Y Progress

Completed:

✔ backend chosen  
✔ real compression verified  
✔ budget semantics mapped  
✔ harness architecture analyzed  
✔ engine integration design finalized  

---

# What Remains

## Step 4 — Harness Integration

Implement:

KVFactoryEngine

and modify:

build_engine()

to support:

engine:
type: kvfactory

---

## Step 4 Smoke Test

Run tiny harness experiment:

dataset: NarrativeQA
n_bins: 2
n_per_bin: 2
budgets: [1.0, 0.2]

Expected outputs:

runs/<exp_group_id>/
kv_transition.sqlite
plots/
report.md

Verification checks:

- requests recorded
- responses recorded
- scoring pipeline runs
- plots generated

---

## Step 5 — PASSY_VERDICT.md

Document:

1. backend + commit
2. integration mode
3. budget semantics
4. proof of compression
5. smoke experiment outputs

---

# Handoff State for Next Chat

Backend verified:

KVCache-Factory

Compression method:

SnapKV

Harness integration target:

KVFactoryEngine adapter

Required code changes:

src/kv_transition/engines/endpoints.py
src/kv_transition/engines/kvfactory_engine.py (new)

Next task:

**Implement KVFactoryEngine and run Step-4 smoke test.**