

# Phase H — Architecture Pass 4 Blueprint (Core Functionality: Detect Context Degradation)

**Theme:** Prove the harness can *reliably detect true context-length degradation* under KV budget compression.  
**Scope lock:** No presentation polish, no broad robustness sweep. Only what improves *signal detection validity*.  
**Success target (given):**  
> On LongBench **TREC** and **NarrativeQA**, KV budget **0.2** shows **≥10 absolute F1 drop** vs **1.0** in **≥2 of 4 length bins**, with consistent directionality across **OpenAI** and **[GPU] vLLM** runs, and **≤5%** of the observed drop attributable to **failure-rate inflation**.

---

## Pass 4 Output Artifacts (minimal)
- Two pinned configs:
  - `config/experiments/pass4_trec_4bins.yaml`
  - `config/experiments/pass4_narrativeqa_4bins.yaml`
- One results bundle per engine:
  - `runs/<exp_group_id>/{kv_transition.sqlite, plots/, report.md}` (report can be minimal; must include required stats below)
- A single “Pass4 Verdict” markdown note (can live in repo root or runs dir) containing:
  - the bin-level deltas + attribution check + whether success criteria met.

---

## 0) Budget-Application Sanity (Request-Time KV Params) — [NO-GPU] + [GPU]
**Goal:** Ensure the server is actually receiving/applying the KV budget parameter; otherwise you can’t expect degradation.

### Steps
0.1 **Echo-proof check via raw response**
- Run 2 requests (same prompt) with budget=1.0 and 0.2 against each engine.
- Inspect stored `requests.prompt_json` + `runs.kv_budget` and confirm the KV budget parameter is present in the request payload for *both* budgets.
- **Check:** For each engine, DB contains two requests with *different* KV budget values passed in payload (not just logged locally).

0.2 **Server acknowledgement (best-effort)**
- If the server returns any recognizable indicator in `responses.raw_json` (or headers) that references KV settings, confirm it differs across budgets.
- **Check:** Either (A) server explicitly acknowledges KV budget difference, OR (B) you document “no explicit ack exists” and proceed to Step 1 as the empirical validation.

> If 0.1 fails, stop Pass 4: you are not testing compression.

---

## 1) Build a “Degradation-Revealing” Measurement Setup — [NO-GPU]
**Goal:** Choose bins/tasks/settings that *should* show degradation if KV compression works.

### Steps
1.1 **Pin two tasks, 4 bins, deterministic sampling**
- Use your two tasks: **TREC** and **NarrativeQA**.
- Set `n_bins = 4`, `n_per_bin = 8` (enough for stable bin means without blowing cost).
- `temperature=0`, keep `max_tokens` fixed across runs.
- Budgets: `[1.0, 0.2]` only (don’t dilute with 0.5 in Pass 4).
- **Check:** `prepare` produces exactly 4 bins, and per run you get `requests = 32` (4 bins × 8) for each budget.

1.2 **Minimize failure-driven artifacts**
- Ensure prompts are consistent and avoid formatting requirements that induce FORMAT_ERROR.
- Keep response length sufficient to answer (avoid artificially low `max_tokens`).
- **Check:** Baseline (budget=1.0) failure rate ≤2% across both tasks (OpenAI engine).

---

## 2) Establish Degradation on OpenAI (Reference Engine) — [NO-GPU]
**Goal:** Show the phenomenon exists at least once in a stable environment before moving to GPU/vLLM.

### Steps
2.1 **Run full pipeline for each task**
For each config (TREC, NarrativeQA):
- `prepare → run → score → analyze → report`
- **Check:** `bin_stats` exists for both budgets and all 4 bins (8 rows per task: 2 budgets × 4 bins).

2.2 **Primary success metric: per-bin F1 drop**
Compute per bin:  
`ΔF1(bin) = F1_mean(budget=0.2, bin) - F1_mean(budget=1.0, bin)`
- **Check:** For each task, at least **2 of 4 bins** have `ΔF1 ≤ -0.10` (absolute 10-point drop).

2.3 **Failure inflation attribution (≤5% of drop)**
For bins where the F1 drop is large, estimate how much of the drop could be “explained” by failures:
- Let `fail_rate` be fraction of requests in failures for that (budget, bin).
- Conservative bound: assume every failure would have been perfect (F1=1) under 0.2 (best case counterfactual).
- Compute:  
  `F1_corrected = (F1_mean + fail_rate * 1.0) / (1.0)`  (i.e., add back max possible)
- Then compute `ΔF1_corrected` vs baseline and attribute:
  `attribution = (ΔF1 - ΔF1_corrected) / ΔF1` (clamped)
- **Check:** For each task, in the bins used to satisfy 2.2, **≤5%** of the observed drop is attributable to failure inflation (i.e., the drop remains even under generous correction).

> If OpenAI fails to show degradation: do not proceed to GPU. The issue is likely measurement/task/prompt/metric mismatch, not hardware.

---

## 3) Replicate Directionality on vLLM (GPU) — [GPU]
**Goal:** Same detection on the GPU serving stack; directionality must match OpenAI.

### Steps
3.1 **[GPU] One-time server readiness**
- Start vLLM OpenAI-compatible server with the same model family you used for tokenization/binning assumptions.
- Confirm a single “ping” completion works.
- **Check:** One successful completion stored in DB with expected `model_name`, non-empty output.

3.2 **[GPU] Run the same two configs**
For each task (TREC, NarrativeQA):
- `prepare` can be skipped if manifest/DB already exist and are identical; otherwise run it.
- `run → score → analyze → report`
- **Check:** Same cardinalities as OpenAI: `requests=32` per budget and 4 bins populated.

3.3 **Directionality replication check**
- Compare the sign of `ΔF1(bin)` between engines.
- **Check:** For each task, at least **3 of 4 bins** match direction (both engines show ΔF1 ≤ 0, and at least 2 bins satisfy ≤ -0.10 on GPU as well OR are within a small tolerance if variance is high).

3.4 **Failure inflation attribution on GPU**
- Repeat Step 2.3 on GPU runs.
- **Check:** Same ≤5% attribution bound holds for the bins used to claim degradation.

---

## 4) Diagnose If Degradation Still Doesn’t Appear (Single-threaded, bounded) — [NO-GPU] then [GPU]
**Goal:** If Pass 4 fails, isolate *one* root cause without exploding scope.

### Steps (execute in order; stop when resolved)
4.1 **Confirm KV param is not ignored**
- Run a single long-context example (top bin) repeatedly with budget=1.0 vs 0.2 (n=5 each).
- **Check:** If outputs are statistically indistinguishable across budgets (scores + length + behavior), suspect KV param is being ignored by server.

4.2 **Amplify the stress test without changing the task**
- Increase `n_per_bin` only for the longest bin (e.g., bin 3 from 8 → 20) for one task only (NarrativeQA).
- **Check:** If degradation emerges only with more samples, it was variance/underpowered bins; keep Pass 4 success tied to updated sample size and document it.

4.3 **Lock “effective context length”**
- Verify token_len distribution per bin is meaningfully separated (bin edges not too close).
- **Check:** Longest bin median token_len is at least ~2× shortest bin median token_len (rough heuristic). If not, binning isn’t creating a length signal.

(Do not add new tasks, new detectors, or new plotting in Pass 4. Only these bounded diagnostics.)

---

## Pass 4 Completion Checklist (binary)
Pass 4 is complete when all are true:

- [ ] KV budget request-time parameter presence proven in DB for both engines (Step 0.1)
- [ ] OpenAI: both tasks meet `≥10 F1 drop` in `≥2/4 bins` (Step 2.2)
- [ ] OpenAI: failure inflation accounts for `≤5%` of drop in those bins (Step 2.3)
- [ ] [GPU] vLLM: directionality consistent with OpenAI and degradation visible (Step 3.3)
- [ ] [GPU] vLLM: failure attribution bound holds (Step 3.4)

**Pass 4 deliverable:** A short “Pass4 Verdict” note containing the bin tables (ΔF1, fail_rate, corrected ΔF1) for both engines and both tasks, and a clear PASS/FAIL against the checklist above.