# PASS Y — Make KV Budget *Real* on GPU (Backend Swap + Evidence)

**Theme:** Replace “kv_budget label” with a **backend that actually performs KV token dropping/eviction** so budget sweeps become meaningful.  
**Scope lock:** No new metrics, no new plots, no presentation polish. Only: **compression engine integration + proof it’s applied**.  
**Why this pass exists:** Stock vLLM OpenAI server does **not** expose SnapKV/H2O/PyramidKV per-request budget controls; `kv_budget` is ignored/rejected depending on schema.  [oai_citation:0‡deep-research-report.md](sediment://file_000000006fbc722f85ba7cfc6c2c30fb)

---

## PASS Y Output Artifacts (minimal)
1. One pinned GPU config:
   - `config/experiments/passy_gpu_backend.yaml`
2. One GPU run bundle proving budgets are meaningful:
   - `runs/<exp_group_id>/{kv_transition.sqlite, plots/, report.md}`
3. One short note:
   - `PASSY_VERDICT.md` (proof budget is applied + how)

---

## PASS Y Success Definition (binary)
PASS Y succeeds if **all** are true:

1. **Backend implements real KV compression** (SnapKV/H2O/PyramidKV/StreamingLLM/KVPress-style), i.e., not just KV dtype, memory pool size, or prompt truncation.  [oai_citation:1‡deep-research-report.md](sediment://file_000000006fbc722f85ba7cfc6c2c30fb)  
2. **Budget is a real control**: changing budget changes *some measurable runtime state* (KV retained tokens, window size, eviction count, etc.) OR produces consistent behavioral differences on a controlled stress prompt.
3. **Harness can run multi-budget** on GPU using the same pipeline stages you already have (`prepare→run→score→analyze→report`) with budgets `[1.0, 0.2]`.
4. **Evidence captured** in DB and/or logs and summarized in `PASSY_VERDICT.md`.

---

# Lane Split (Hard)
- **[GPU]** Everything in PASS Y is GPU-only.
- **[NO-GPU]** Only allowed for repo edits + config prep + reading docs. No OpenAI API runs.

---

# 0) Choose the Compression Backend (one choice, no hedging) — [NO-GPU]
**Goal:** Pick the fastest “works this week” backend.

**Allowed choices (pick ONE):** SELECTED
- **A. KVCache-Factory** (supports PyramidKV/SnapKV/H2O/StreamingLLM; LongBench-oriented)  

---

# 1) Define the Integration Mode (minimal change to harness) — [NO-GPU]
**Goal:** Make your harness talk to the backend without rewriting evaluation logic.

**Integration modes (choose ONE):**
- **Mode 2: “LocalEngine” adapter**   (SELECTED)
  Add a new engine implementation that calls the backend directly (Python API / subprocess), but still returns `EngineResult`.

**Constraint:** Prefer the smallest diff:
- If backend already has CLI scripts for LongBench → Mode 2 is usually fastest.
- If you already rely on HTTP infra → Mode 1 is cleaner.

**Check:** You can run a single prompt end-to-end through the harness with this mode (even before budgets).

---

# 2) Budget Semantics Mapping (normalize what “1.0” and “0.2” mean) — [NO-GPU]
**Goal:** Align your harness `kv_budget ∈ {1.0, 0.2}` to backend’s budget primitive.

**Rules:**
- If backend uses **retained KV tokens** `K`: map  
  `budget=1.0 → K = full` (or very large), `budget=0.2 → K = 0.2 * prompt_len` (or configured cap).
- If backend uses **compression_ratio r**: map directly  
  `budget = r`.
- If backend uses **(sink_tokens + window)**: map `budget` to window size (document mapping).

**Deliverable:** A 5-line mapping block included in `PASSY_VERDICT.md`.

**Check:** For a fixed long prompt, the backend’s effective retention/window differs substantially between 1.0 and 0.2.


We’ll normalize your harness kv_budget ∈ {1.0, 0.2} onto KVCache-Factory’s two knobs used in its LongBench runner:
	•	max_capacity_prompts_ratio → sets max_capacity_prompt ≈ round(prompt_len * ratio)  ￼
	•	pruning_ratio → “pruning ratio of Key Cache” (their ratio field wired into attention config)  ￼

Mapping block (this goes into PASSY_VERDICT.md)
	•	budget = 1.0
	•	max_capacity_prompts_ratio = 1.0 (cap ≈ full prompt len)  ￼
	•	pruning_ratio = 0.0 (prune nothing; retain ~100%)  ￼
	•	budget = 0.2
	•	max_capacity_prompts_ratio = 0.2 (cap ≈ 20% of prompt len)  ￼
	•	pruning_ratio = 0.8 (prune ~80%; retain ~20%)  ￼


---

# 3) Hard Proof the Budget is Applied (before LongBench) — [GPU]
**Goal:** Prevent another “ignored knob” situation.

**Steps**
3.1 Run a single long prompt twice (same text), once with budget=1.0 and once with budget=0.2.

3.2 Capture at least one of the following “proof signals”:
- backend prints/logs retained KV length / eviction count / window size
- response raw metadata includes a method/budget field
- GPU memory footprint or prefill/decode timing changes meaningfully *and consistently* (supporting, not sole proof)

**Checks (must satisfy ≥1 hard signal):**
- **Hard signal:** explicit logged/returned evidence of different KV retention/window/eviction for the two budgets
- **Soft signal:** consistent difference in latency/memory *plus* qualitative output change on the stress prompt (only as supporting evidence)

> If you cannot produce a hard signal, stop. PASS X and Pass 4 will not be valid.

---

# 4) Plug Into the Harness and Run a Tiny Multi-budget Smoke — [GPU]
**Goal:** Confirm your pipeline works with the real compression backend.

**Config:** `passy_gpu_backend.yaml`
- one task only (NarrativeQA recommended)
- `n_bins=2`, `n_per_bin=2` (tiny)
- budgets `[1.0, 0.2]`
- temp=0, fixed max_tokens

**Checks:**
- requests/responses recorded for both budgets
- score/analyze/report succeed
- DB integrity invariants hold
- plot files exist (even if uninformative at tiny scale)

---

# 5) PASS Y Verdict (required) — [NO-GPU]
**Goal:** Leave a clean artifact that unblocks PASS X Step 2-end.

`PASSY_VERDICT.md` must contain:
1) Which backend + version/commit  
2) Integration mode (Mode 1 or 2)  
3) Budget mapping semantics (Step 2)  
4) Proof budget applied (Step 3 evidence)  
5) Smoke run exp_group_id and artifact paths

**Check:** Anyone can read it and conclude “budget is real” in <60 seconds.
