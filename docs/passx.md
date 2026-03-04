# PASS X — GPU-First Core Verification (KV Budget → Context Degradation is Real)

**Theme:** Prove the *actual project thesis* works on a controllable GPU backend: **KV budget compression causes measurable context-length degradation**.  
**Scope lock:** GPU lane only. No OpenAI comparisons, no presentation polish, no broad robustness sweep.  
**Deliverable deadline:** “class presentation in a week” → prioritize a single, undeniable empirical result.

---

## PASS X Output Artifacts (minimum)
1. One pinned GPU experiment config:
   - `config/experiments/passx_gpu_core.yaml`
2. One results bundle:
   - `runs/<exp_group_id>/{kv_transition.sqlite, plots/, report.md}`
3. One “PASS X Verdict” note (markdown) containing:
   - evidence KV params were *sent + accepted*
   - per-bin ΔF1 table (0.2 vs 1.0)
   - failure-attribution bound
   - PASS/FAIL checklist outcome

---

## PASS X Success Definition (binary)
PASS X succeeds if **all** are true on GPU/vLLM:

1. **KV knobs are real:** request payload contains kv_budget and server accepts it.
2. **Degradation detected:** On the chosen task, budget **0.2 vs 1.0** shows **≥10 absolute F1 drop** in **≥2 of 4 bins**.
3. **Not failure-driven:** ≤5% of the observed drop is attributable to failure-rate inflation (conservative correction).
4. **Directionality sanity:** In ≥3/4 bins, ΔF1 ≤ 0 (compression is not “helping” systematically).

---

# 0) Preconditions Gate — GPU-only (Hard stop if fail)

### 0.1 GPU server contract (one-time per GPU session) **(COMPLETE + VERIFIED)**
**Goal:** Ensure the serving stack can accept request-time KV params.

**Steps**
- Start vLLM OpenAI-compatible server with SnapKV (or target compression policy) enabled.
- Confirm the server is the one you intend to test (correct model, correct endpoint).

**Check:** A single completion request succeeds and is logged to DB (non-empty output).

### 0.2 KV param acceptance proof (must be DB-evidenced) **(COMPLETE + VERIFIED)**
**Goal:** Verify budgets are not “just labels.”

**Steps**
- Send **two** requests with identical prompt/messages:
  - budget=1.0 and budget=0.2
- Persist raw request payload and raw response JSON.

**Check (hard):**
- DB shows **payload differs** between the two requests with kv_budget present (extra_body or equivalent).
- Server does not reject/ignore unknown params (no 4xx due to params).
- If no explicit echo exists, acceptance is evidenced by: request logged + response returned successfully for both budgets with distinct payloads.

> If 0.2 is not provably sent/accepted, PASS X is blocked. Do not proceed.

---

# 1) Pin a “Degradation-Revealing” GPU Experiment — [GPU]

### 1.1 Choose a single task that should show context sensitivity **(COMPLETE + VERIFIED)**
**Goal:** Maximize chance of a clear effect quickly.

**Constraint:** Use **one** task for PASS X (NarrativeQA preferred unless you have a better known-stressor).

**Check:** Task is pinned and immutable in the config.

### 1.2 Config lock (4 bins, 2 budgets, deterministic)
Create `passx_gpu_core.yaml`:
- `n_bins = 4`
- `budgets = [1.0, 0.2]`
- `n_per_bin = 8` (increase only if underpowered; see Step 4)
- `temperature = 0`
- fixed `max_tokens`
- exp_group_id pinned (e.g., `passx_gpu_core_<task>_v1`)

**Check:** `prepare` yields exactly 4 bins and `entries = 4 * n_per_bin`.

---

# 2) Run the Minimal Core Experiment — [GPU]

### 2.1 Execute full pipeline
For PASS X, do exactly:
- `prepare → run → score → analyze → report`

**Check (cardinality):**
- For each budget: `requests = 4 * n_per_bin`
- `responses + failures == requests`
- `failures ∩ responses = ∅`
- `bin_stats` exists for both budgets and all 4 bins (8 rows total).

---

# 3) Core Result Extraction (the “thesis”)

### 3.1 Per-bin degradation table
Compute for each bin:
- `ΔF1(bin) = F1_mean(0.2, bin) - F1_mean(1.0, bin)`

**Checks:**
- At least **2 of 4 bins** have `ΔF1 ≤ -0.10`
- In at least **3 of 4 bins**, `ΔF1 ≤ 0`

### 3.2 Failure inflation attribution bound (≤5%)
For the bins used to satisfy the ≥10-drop criterion:
- Let `fail_rate_0.2(bin)` be failure fraction under 0.2
- Conservative correction: assume failures would have scored F1=1.0
- Confirm the corrected drop still meets the threshold and failure inflation explains ≤5% of drop.

**Check:** ≤5% attribution bound holds for the bins you’re claiming.

> If degradation appears but is mostly failures: record it, but PASS X fails (the thesis is “quality degradation with context,” not just instability).

---

# 4) If PASS X Fails: One Bounded Amplification Loop (No scope creep)

### 4.1 Underpowered bins (variance too high)
**Trigger:** ΔF1 is negative but not crossing -0.10 reliably.

**Action:** Increase only the longest bin sample size:
- set `n_per_bin_long = 20` (or equivalent focused sampling if supported)
- keep other bins unchanged

**Check:** Re-run budgets and re-evaluate Step 3.1/3.2.

### 4.2 Length signal too weak (bins not separated)
**Trigger:** longest-bin token lengths are not much larger than shortest bin.

**Action:** Adjust binning/sampling to widen separation (without adding tasks):
- ensure longest bin is truly “long context” (top-quantile by token_len)

**Check:** median token_len(bin 3) is meaningfully larger than bin 0 (target: ~2×).

**Constraint:** Only one amplification loop allowed in PASS X. If still failing, stop and write a clear failure diagnosis.

---

# 5) PASS X Verdict Note (required deliverable)

Create `PASSX_VERDICT.md` with exactly:
1. **Server + model + endpoint** used
2. Evidence KV budget was **sent + accepted** (DB pointers: request_id snippets acceptable)
3. Table: per-bin `F1(1.0)`, `F1(0.2)`, `ΔF1`, `fail_rate(0.2)`
4. Failure-attribution result for the claimed bins
5. Final PASS/FAIL against the PASS X checklist

**Check:** A reader can see the thesis result in <60 seconds without reading code.

---

## Constraints (enforced in PASS X)
- GPU-only. No OpenAI API runs in this pass.
- Only 1 task in the main attempt.
- Only budgets [1.0, 0.2].
- Only 1 amplification loop (Step 4) permitted.
- No new detectors, no new plotting beyond existing Phase E artifacts.
- No presentation polish; only “evidence artifacts.”

---

## What PASS X Enables Next
If PASS X passes, you are unblocked to proceed to:
- Pass 4 (multi-task replication + directionality across tasks + stronger statistical confidence)
- Any optional OpenAI lane as a “reference harness” only (not compression)