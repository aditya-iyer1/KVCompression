# Root-Cause Report: KV params missing from outbound `/v1/chat/completions` body

## Traced Code Path

| Step | File : Function : Lines | What happens |
|---|---|---|
| 1 | `settings.py:load_settings()` | Deep-merges `default.yaml` + `pass4_trec_4bins.yaml`. `"{kv_budget}"` and `"{kv.policy}"` have no `${}` pattern, so `_resolve_env_vars` passes them through unchanged. `settings["engine"]["request"]["extra_body"]` = `{"kv_budget": "{kv_budget}", "kv_policy": "{kv.policy}"}` ✓ |
| 2 | `cli.py:cmd_run()` → `orchestrate.py:orchestrate()` | Calls `endpoints.build_engine(settings)` which creates `OpenAICompatEngine(base_url=<resolved ${BASE_URL}>)`. Then loops over `kv.budgets` calling `runner.run_one_setting()`. |
| 3 | **`runner.py:run_one_setting()` lines 303–309** | **THE CRITICAL GATE:** <br>`base_url = getattr(engine, "base_url", "")` <br>`is_openai = _is_openai_base_url(base_url)` <br>`extra_kwargs: Dict[str, Any] = {}` <br>`if not is_openai:` <br>&nbsp;&nbsp;`extra_body = settings...get("extra_body")` <br>&nbsp;&nbsp;`extra_kwargs = _resolve_extra_body(extra_body, kv_budget)` |
| 4 | `runner.py:_resolve_extra_body()` lines 93–102 | Only replaces `v == "{kv_budget}"` → float. The value `"{kv.policy}"` does **not** match, so it passes through as the **literal string** `"{kv.policy}"`. |
| 5 | `runner.py` line 482 | `engine.generate(..., **extra_kwargs)` — if `extra_kwargs` is `{}`, no kv params are sent. |
| 6 | `openai_compat.py:generate()` lines 155–161 | `body = {"model": ..., "messages": ..., **kwargs}` — kwargs would include kv params IF they were passed. Snapshot `outbound_for_notes = _truncate_for_notes(body)` is taken **after** merge, so snapshot is faithful to the actual body. |

## Answer: Which scenario is true?

**Scenario C + a secondary value bug** — but the primary symptom is most likely caused by **Scenario A or B** depending on the user's environment:

## Three Hypotheses, Ranked by Likelihood

### 1. (MOST LIKELY) `is_openai` gate is `True` → Scenario C (runner resolves it but doesn't pass to engine)

- **File:** `runner.py:run_one_setting()`, lines 305–309
- `_is_openai_base_url()` (line 78) checks if the hostname is `api.openai.com`. If `${BASE_URL}` resolved to an OpenAI URL (or was left as the literal placeholder `${BASE_URL}` and happened to match), the guard `if not is_openai:` **skips the entire `extra_body` block**. `extra_kwargs` stays `{}`, so zero kv params reach the engine or the telemetry snapshot.
- **Evidence:** The telemetry *does* contain an `outbound_request` snapshot (latency_s + outbound_request structure), which means `openai_compat.py` ran successfully. The snapshot has `model` and `messages` but no `kv_budget`/`kv_policy` — exactly what you'd see if `**kwargs` was empty.
- **Verification:** Add a temporary `print(f"DEBUG is_openai={is_openai} base_url={base_url!r}")` at line 306.

### 2. Config/timing mismatch → Scenario A (config extra_body never loads)

- **File:** `config/experiments/pass4_trec_4bins.yaml`
- The `extra_body` addition is currently an **unstaged git change**. If the run was launched from a stale config (e.g., the file was saved after the run started, or a cached config was used), `extra_body` would be absent from `settings["engine"]["request"]`.
- **Verification:** `print(settings["engine"]["request"])` at the top of `orchestrate()`.

### 3. `_resolve_extra_body` value bug → partial Scenario B

- **File:** `runner.py:_resolve_extra_body()`, lines 93–102
- Even when the gate passes, `_resolve_extra_body` only recognizes the **exact** string `"{kv_budget}"`. The config value `"{kv.policy}"` (with a dot) does **not** match any template, so `kv_policy` would be sent as the **literal string** `"{kv.policy}"` instead of the actual policy name (e.g., `"snapkv"`).
- This is a secondary bug — it means even if Hypothesis 1 is fixed, the `kv_policy` value would be wrong.

## Recommended Fix Location

### Primary fix: `runner.py:run_one_setting()`

Around line 305 — rethink the `is_openai` gate. The intent is presumably to avoid sending non-standard fields to the real OpenAI API, but it also blocks sending them to vLLM/custom servers that happen to share some URL characteristics.

### Secondary fix: `runner.py:_resolve_extra_body()`

Around line 93 — generalize the template resolution to also handle `"{kv.policy}"` (or any `{key.subkey}` pattern) by accepting a substitution context dict (e.g., `{"kv_budget": float, "kv.policy": str}`).

## Recommended Debug Prints (no code changes needed yet)

```python
# runner.py, right after line 305:
print(f"DEBUG: base_url={base_url!r}, is_openai={is_openai}")
print(f"DEBUG: extra_body from config = {settings.get('engine', {}).get('request', {}).get('extra_body')!r}")
print(f"DEBUG: extra_kwargs = {extra_kwargs!r}")
```

These three prints will immediately confirm which hypothesis is active.
