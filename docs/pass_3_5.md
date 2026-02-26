# Phase H½ — Architecture Pass 3.5 Blueprint (OpenAI-only Validation + GPU-only Compression Lane)

**Purpose:** Unblock progress by validating the harness *as a measurement/diagnostics system* on OpenAI **without pretending compression exists**, while cleanly isolating the **GPU-only** lane where KV compression is actually testable.

**Canonical truth:** OpenAI API cannot apply SnapKV/KV compression controls; request-time KV params must not be sent to `api.openai.com`. OpenAI runs are *reference plumbing + stability validation only*. Compression validation happens only on controllable OpenAI-compatible servers (vLLM/SGLang) in GPU environment.

---

## Lanes (Hard Split)

### Lane A — [NO-GPU] OpenAI-only Harness Validation
**Goal:** Prove dataset prep → inference → scoring → analysis → report + diagnostics are correct and stable under real-world rate limits, with **one budget only** (or budgets as labels without payload KV params).

### Lane B — [GPU] Compression Validation Only
**Goal:** Prove request-time KV params are accepted and **budget causes measurable degradation** on a controllable server, with explicit payload presence/echo.

---

# Lane A — OpenAI-only Validation (NO compression)

## A0) Pin “OpenAI Reference” configs (single-budget)
**Goal:** Make runs meaningful and rate-limit-friendly.

**Steps**
A0.1 Create/confirm pinned configs per task (TREC, NarrativeQA):
- `budgets: [1.0]` (single run)
- `n_bins: 4`, `n_per_bin: small` (e.g., 5–8)
- `temperature: 0`, conservative `max_tokens`
- Ensure prompts do not exceed model context (cap context or choose model accordingly)

**Check:** Each config produces exactly `n_bins * n_per_bin` requests and no budget comparison is implied anywhere.

---

## A1) Prepare correctness (binning + manifest + DB parity)
**Goal:** Ensure the “length signal” is real and deterministic.

**Steps**
A1.1 `prepare` for each task config.

A1.2 Verify bin separation using DB/manifest stats:
- bin edges monotonically increasing
- token_len distribution increases with bin_idx

**Check (deterministic):**
- 4 bins exist
- each bin has `n_per_bin` entries (or documented shortfall if dataset constrained)
- median token_len(bin 3) > median token_len(bin 0)

---

## A2) Run stability (rate-limit friendly, low failures)
**Goal:** OpenAI run produces clean responses with low operational noise.

**Steps**
A2.1 Run inference once per task config with pacing enabled (whatever your harness supports: fixed sleep / max RPM / exponential backoff on 429).

A2.2 Record rate-limit behavior:
- count of 429 retries
- effective throughput (requests/min)

**Checks:**
- Failure rate ≤ 2% overall (excluding known “prompt too long” cases which should be prevented by config)
- No systematic EMPTY_OUTPUT
- Retries happen but do not corrupt DB integrity (`responses ∩ failures = ∅` and totals match)

---

## A3) Scoring correctness (spot-check validity)
**Goal:** Metrics reflect reality; taxonomy is stable.

**Steps**
A3.1 Run `score` for the OpenAI run(s).

A3.2 Manual spot-check (bounded):
- sample 10 rows across bins
- compare `pred_norm` vs golds (human eyeball)

**Checks:**
- EM/F1 aligns directionally with human judgment in the sample
- failure taxonomy does not mislabel normal answers as failures

---

## A4) Analysis + diagnostics correctness (bootstrap, plots, attribution math)
**Goal:** Phase E outputs are sane and stable even without compression comparisons.

**Steps**
A4.1 Run `analyze` and confirm:
- `bin_stats` exists for all 4 bins
- bootstrap CI bounds are valid (`low ≤ mean ≤ high`)

A4.2 Validate failure attribution computation *as a standalone diagnostic*:
- run the attribution calculation on the OpenAI run (it should generally be near-zero effect if failures are low)
- verify the math is stable (no negative/NaN/inf outputs)

**Checks:**
- Plots generated and load (paths exist)
- Attribution diagnostic produces finite values and does not exceed logical bounds

---

## A5) Report generation sanity (no polish requirements)
**Goal:** Report is a faithful reflection of DB artifacts, not a recomputation.

**Steps**
A5.1 Run `report`.

**Checks:**
- `report.md` exists
- references only existing plots
- clearly shows single-budget nature (no budget delta claims)

---

## Lane A Deliverable: “Pass 3.5 Verdict — OpenAI”
A single markdown note stating (with numbers):
- tasks run, config ids, request counts, failure rate, retry summary
- confirmation that **compression is NOT tested on OpenAI**
- confirmation that prepare/run/score/analyze/report + diagnostics are operational and trustworthy

---

# Lane B — GPU-only Compression Validation (controllable server)

## B0) GPU prerequisites gate (must pass before any multi-budget test)
**Goal:** Avoid wasting GPU setup cycles.

**Steps**
B0.1 Start vLLM (or equivalent) OpenAI-compatible server with SnapKV enabled and request-time knobs accepted.

B0.2 Single “echo proof” request:
- send budget=1.0 then 0.2 for the same prompt
- confirm request payload contains KV params (extra_body or equivalent)
- confirm server accepts them (no 4xx) and ideally echoes in raw response/logs

**Checks (hard):**
- Payload contains kv_budget for GPU lane
- No server rejection of unknown params
- Two requests logged with distinct kv settings

---

## B1) Minimal compression experiment (single task, 4 bins, 2 budgets)
**Goal:** Show measurable degradation exists *somewhere* before expanding.

**Steps**
B1.1 Use one task first (NarrativeQA preferred for long context sensitivity):
- `budgets: [1.0, 0.2]`
- `n_bins: 4`, `n_per_bin: 8` (or smaller if GPU unstable)

B1.2 Run full pipeline: `run → score → analyze → report`

**Checks:**
- Both budgets complete with matching request counts
- `ΔF1 ≤ -0.10` in at least 2/4 bins (your Pass 4 target shape, but only for 1 task here)
- Failure inflation attribution ≤ 5% in the bins used to claim degradation

---

## B2) Second task replication (TREC) — only if B1 passes
**Goal:** Confirm it’s not a one-off.

**Steps**
B2.1 Repeat B1 on TREC with identical structure.

**Checks:**
- Directionality consistent (0.2 worse than 1.0 in most bins)
- Same attribution bound holds

---

## Lane B Deliverable: “Pass 3.5 Verdict — GPU”
A single markdown note stating:
- server + model id, how KV budget is applied, evidence payload is present/accepted
- the bin-level ΔF1 table and attribution bound
- whether compression degradation is detected

---

# Transition Back to Pass 4 (Compression-Centric)

## Pass 4 minimal prerequisites (must be true)
1. **Stable controllable backend exists** (vLLM/SGLang) with SnapKV enabled.
2. **Base URL is non-OpenAI** for compression tests.
3. **Payload KV params are present and accepted** (Lane B0 passes).
4. **One task shows degradation with attribution bound** (Lane B1 passes).
5. **GPU run completes without high failure noise** (failures low enough that attribution remains meaningful).

Only after these are true does “Pass 4: detect degradation across tasks and engines” become valid.

---

# PLAN Question — Multi-provider abstraction (LiteLLM / OpenRouter / similar)

## Decision: Out-of-scope for Pass 3.5; do not adopt now.

### Pros (for this project)
- Simplifies switching between providers/models with a single interface
- Potentially reduces custom endpoint plumbing and auth handling
- Makes “OpenAI-like” testing easier across multiple hosted backends

### Cons / Non-benefits (critical)
- **Does not solve the core need:** providers still won’t expose or honor KV compression internals
- Compression validation still requires a **controllable server** (vLLM/SGLang) where KV knobs are real
- Adds a dependency + extra abstraction layer that can complicate telemetry/raw response handling and failure modes

### Architectural stance
- **Pass 3.5 focus is measurement correctness + controllable compression lane.**
- Revisit multi-provider tooling **after** Pass 4 succeeds on GPU lane, as an optional convenience layer (not a capability unlock).

---

## Pass 3.5 Completion Criteria (Binary)
Pass 3.5 is complete when:

### Lane A (OpenAI-only)
- [ ] Single-budget runs completed for TREC and NarrativeQA
- [ ] Failure rate low and DB integrity invariants hold
- [ ] Scoring spot-check passes
- [ ] Analyze + bootstrap + plots + report generated
- [ ] “Pass 3.5 Verdict — OpenAI” written with explicit limitation statement

### Lane B (GPU-only)
- [ ] KV budget params proven present + accepted in payload
- [ ] At least 1 task shows ≥10 F1 drop in ≥2/4 bins at budget 0.2 vs 1.0
- [ ] Failure attribution ≤5% for the bins used
- [ ] “Pass 3.5 Verdict — GPU” written with evidence

Once both lanes pass, resume Pass 4 as originally intended (compression-centric, multi-task, directionality checks).