"""Core execution loop for running inference on manifest entries.

Executes manifest entries for a single KV setting and persists results to DB.
"""

import hashlib
import inspect
import json
import random
import sqlite3
import time
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, Optional

from ..db import dao
from ..engines.base import BaseEngine
from ..eval.failure_taxonomy import classify_failure
from ..data import tokenizer

# Global pacer: monotonic time of last request start (persists across budgets in same process)
_last_request_monotonic: Optional[float] = None


def _compute_dataset_id(dataset_name: str, task: str) -> str:
    """Compute dataset_id from dataset name and task.
    
    Matches Phase B convention: {dataset.name}__{dataset.task}
    
    Args:
        dataset_name: Dataset name.
        task: Task identifier.
    
    Returns:
        Dataset ID string.
    """
    return f"{dataset_name}__{task}"


def _compute_prompt_template_version() -> str:
    """Compute deterministic hash of prompt template builder function.
    
    Uses SHA-256 hash of the source code of _build_messages to create
    a stable version identifier for the prompt template.
    
    Returns:
        First 12 characters of SHA-256 hex digest (deterministic prefix).
    """
    source = inspect.getsource(_build_messages)
    hash_obj = hashlib.sha256(source.encode('utf-8'))
    return hash_obj.hexdigest()[:12]


def _build_messages(context: str, question: str) -> list[Dict[str, str]]:
    """Build OpenAI-style messages for inference.
    
    Args:
        context: Context text.
        question: Question text.
    
    Returns:
        List of message dicts in OpenAI format.
    """
    return [
        {
            "role": "system",
            "content": "Answer the question using the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True if exception looks like HTTP 429 / rate limit."""
    msg = (str(exc) or "").lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "rate limit reached" in msg
        or "tokens per min" in msg
        or " tpm" in msg
        or "tpm " in msg
    )


def _get_http_status(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of HTTP status code from exception."""
    # Common attributes: status, status_code, response.status, response.status_code
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    resp = getattr(exc, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status", None) or getattr(resp, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _get_retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Parse numeric Retry-After header (seconds) if present."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
    if not isinstance(headers, dict):
        return None
    val = headers.get("Retry-After") or headers.get("retry-after")
    if val is None:
        return None
    try:
        secs = float(val)
        if secs < 0:
            return None
        return secs
    except (TypeError, ValueError):
        # Ignore HTTP-date format for now; caller falls back to exponential backoff
        return None


def _is_tpm_429(msg: str) -> bool:
    """True if error message indicates TPM-style 429 (needs longer backoff)."""
    m = (msg or "").lower()
    return "tokens per min" in m or " tpm" in m or "tpm " in m


def _backoff_seconds(
    attempt: int,
    is_tpm: bool,
    base_s: float,
    max_s: float,
) -> float:
    """Exponential backoff with jitter. TPM uses larger effective base (min 10s)."""
    effective_base = max(base_s * 2, 10.0) if is_tpm else base_s
    raw = min(effective_base * (2 ** (attempt - 1)), max_s)
    jitter = 0.5 * raw * (random.random() * 2 - 1)  # ±50%
    return max(0.0, raw + jitter)


def run_one_setting(
    conn: sqlite3.Connection,
    settings: Dict[str, Any],
    engine: BaseEngine,
    model_name: str,
    run_id: str,
    kv_budget: float
) -> None:
    """Run inference for a single KV budget setting.
    
    Reads manifest entries, executes inference, and persists results to DB.
    
    Args:
        conn: SQLite connection.
        settings: Settings dict (from load_settings).
        engine: Engine instance (from endpoints.build_engine).
        model_name: Model name string.
        run_id: Run identifier.
        kv_budget: KV budget value for this run.
    """
    # Optional local debug flag for extra logging
    debug = False
    
    # Check if run_id already has requests (prevent duplicate runs)
    cursor = conn.execute(
        "SELECT COUNT(*) FROM requests WHERE run_id = ?",
        (run_id,)
    )
    request_count = cursor.fetchone()[0]
    
    if request_count > 0:
        raise ValueError(
            f"Run {run_id} already has {request_count} request(s) persisted. "
            f"Refusing to re-run to avoid mixing data. "
            f"To rerun, use a new exp_group_id or delete existing rows for this run_id."
        )
    
    # Optional RPM pacing: run.pacing.requests_per_minute or engine.rate_limit.max_rpm (backward compat)
    pacing_rpm: Optional[int] = None
    run_pacing = settings.get("run", {}).get("pacing", {})
    rpm_candidate = run_pacing.get("requests_per_minute")
    if rpm_candidate is not None:
        try:
            v = int(rpm_candidate)
            if v > 0:
                pacing_rpm = v
        except (TypeError, ValueError):
            pass
    if pacing_rpm is None:
        max_rpm = settings.get("engine", {}).get("rate_limit", {}).get("max_rpm")
        if max_rpm is not None:
            try:
                v = int(max_rpm)
                if v > 0:
                    pacing_rpm = v
            except (TypeError, ValueError):
                pass
    if pacing_rpm is not None:
        pacing_interval = 60.0 / pacing_rpm
        print(f"  Pacing enabled: {pacing_rpm} RPM ({pacing_interval:.2f}s interval)")
    else:
        pacing_interval = None
    
    # 429 retry config (conservative defaults when run.retries not set)
    retries_cfg = settings.get("run", {}).get("retries", {})
    retry_max_attempts = retries_cfg.get("max_attempts")
    if retry_max_attempts is None:
        retry_max_attempts = 3
    else:
        try:
            retry_max_attempts = int(retry_max_attempts)
            retry_max_attempts = max(1, min(retry_max_attempts, 10))
        except (TypeError, ValueError):
            retry_max_attempts = 3
    retry_backoff_base = 2.0
    try:
        retry_backoff_base = float(retries_cfg.get("backoff_base_s", retry_backoff_base))
        retry_backoff_base = max(0.5, min(retry_backoff_base, 60.0))
    except (TypeError, ValueError):
        pass
    retry_backoff_max = 30.0
    try:
        retry_backoff_max = float(retries_cfg.get("backoff_max_s", retry_backoff_max))
        retry_backoff_max = max(retry_backoff_base, min(retry_backoff_max, 300.0))
    except (TypeError, ValueError):
        pass
    
    # Compute prompt template version hash
    prompt_template_version = _compute_prompt_template_version()
    
    # Ensure run row exists once per run
    exp_group_id = settings["output"]["exp_group_id"]
    kv_policy = settings.get("kv", {}).get("policy", "unknown")
    engine_name = "openai_compat"
    base_url = getattr(engine, "base_url", "")
    
    # Check if run already exists and validate prompt_template_version
    cursor = conn.execute(
        "SELECT prompt_template_version FROM runs WHERE run_id = ?",
        (run_id,)
    )
    existing_row = cursor.fetchone()
    
    if existing_row:
        existing_version = existing_row[0]
        if existing_version is not None and existing_version != prompt_template_version:
            raise ValueError(
                f"Run {run_id} already exists with prompt_template_version={existing_version}, "
                f"but current code computes version={prompt_template_version}. "
                f"Prompt template has changed; aborting to prevent inconsistent data."
            )
    
    # Upsert run with prompt_template_version (using direct SQL since dao.upsert_run doesn't support it)
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO runs
            (run_id, exp_group_id, kv_policy, kv_budget, engine_name, base_url, model_name, prompt_template_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            exp_group_id,
            kv_policy,
            kv_budget,
            engine_name,
            base_url,
            model_name,
            prompt_template_version,
            datetime.utcnow().isoformat()
        ))
    
    if debug:
        print(f"  Debug: run_id={run_id}, kv_budget={kv_budget}, model={model_name}, base_url={base_url}")
    
    # Identify dataset_id
    dataset_name = settings["dataset"]["name"]
    task = settings["dataset"]["task"]
    dataset_id = _compute_dataset_id(dataset_name, task)
    # Tokenizer name for prompt length guard (reuse Phase B logic)
    tokenizer_name = tokenizer.get_tokenizer_name(settings)
    
    # Read manifest entries from DB
    manifest_entries = dao.get_manifest_entries(conn, dataset_id)
    
    if not manifest_entries:
        print(f"  Warning: No manifest entries found for dataset_id={dataset_id}")
        return
    
    print(f"  Processing {len(manifest_entries)} manifest entries")
    
    # Get example IDs
    example_ids = [entry["example_id"] for entry in manifest_entries]
    
    # Fetch examples from DB
    examples = dao.get_examples_by_ids(conn, example_ids)
    
    if len(examples) != len(example_ids):
        missing = set(example_ids) - set(examples.keys())
        print(f"  Warning: {len(missing)} examples not found in DB")
    
    # Extract decoding / generation parameters
    decoding = settings.get("decoding", {})
    temperature = decoding.get("temperature", 0.0)
    max_tokens = decoding.get("max_tokens", 256)
    generation = settings.get("generation", {})
    gen_max_tokens = generation.get("max_tokens", max_tokens)
    try:
        gen_max_tokens_int = int(gen_max_tokens)
    except (TypeError, ValueError):
        gen_max_tokens_int = max_tokens if isinstance(max_tokens, int) else 256
    if gen_max_tokens_int < 0:
        gen_max_tokens_int = 0
    # Model max context length (fallback to 8192)
    model_max_len = settings.get("model", {}).get("max_model_len", 8192)
    try:
        model_max_len_int = int(model_max_len)
    except (TypeError, ValueError):
        model_max_len_int = 8192
    if model_max_len_int <= 0:
        model_max_len_int = 8192
    # Runtime prompt cap: model_max_len - max_new_tokens - safety buffer
    runtime_prompt_cap: Optional[int] = None
    cap_candidate = model_max_len_int - gen_max_tokens_int - 256
    if cap_candidate > 0:
        runtime_prompt_cap = cap_candidate
    
    # Extract timeout
    timeout_s = None
    engine_settings = settings.get("engine", {})
    request_settings = engine_settings.get("request", {})
    if request_settings:
        timeout_s = request_settings.get("timeout_s")
    
    # Process each manifest entry
    for idx, entry in enumerate(manifest_entries):
        example_id = entry["example_id"]
        entry_idx = entry["entry_idx"]
        
        if example_id not in examples:
            print(f"  Skipping entry {idx}: example_id {example_id} not found")
            continue
        
        example = examples[example_id]
        
        # Build messages
        context = example["context"]
        question = example["question"]
        messages = _build_messages(context, question)
        
        # Generate stable unique IDs
        request_id = str(uuid4())
        
        # Persist request row (what we are about to send to the engine)
        dao.insert_request(
            conn=conn,
            request_id=request_id,
            run_id=run_id,
            dataset_id=dataset_id,
            entry_idx=entry_idx,
            example_id=example_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        # Optional runtime prompt length guard (model context window)
        if runtime_prompt_cap is not None:
            # Use same prompt construction as Phase B for length counting
            prompt_text = context + "\n\n" + question
            prompt_token_count = tokenizer.count_tokens(
                prompt_text,
                tokenizer_name if tokenizer_name else None,
            )
            if prompt_token_count > runtime_prompt_cap:
                print(
                    f"Skipped example {example_id}: runtime prompt too long "
                    f"({prompt_token_count} > {runtime_prompt_cap})"
                )
                message = (
                    f"runtime prompt too long: {prompt_token_count} > {runtime_prompt_cap}"
                )
                dao.upsert_failure(
                    conn=conn,
                    request_id=request_id,
                    error_type="runtime_prompt_too_long",
                    message=message,
                )
                continue
        
        # Optional RPM pacing: sleep if needed so interval between request starts >= pacing_interval
        global _last_request_monotonic
        if pacing_interval is not None:
            now = time.monotonic()
            if _last_request_monotonic is not None:
                elapsed = now - _last_request_monotonic
                if elapsed < pacing_interval:
                    time.sleep(pacing_interval - elapsed)
            _last_request_monotonic = time.monotonic()
        
        # Call engine with 429 retry and backoff
        result = None
        last_error = None
        for attempt in range(1, retry_max_attempts + 1):
            try:
                result = engine.generate(
                    messages=messages,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )
                break
            except Exception as e:
                status = _get_http_status(e)
                # Explicit HTTP 429 handling with Retry-After / exponential backoff
                if status == 429 and attempt < retry_max_attempts:
                    retry_after = _get_retry_after_seconds(e)
                    if retry_after is not None:
                        sleep_s = min(retry_after, 60.0)
                    else:
                        base_delay = retry_backoff_base
                        sleep_s = base_delay * (2 ** (attempt - 1))
                        sleep_s = min(sleep_s, retry_backoff_max, 60.0)
                    print(
                        f"    Rate limited (HTTP 429): attempt {attempt}/{retry_max_attempts}, "
                        f"sleeping {sleep_s:.1f}s"
                    )
                    time.sleep(sleep_s)
                    continue
                # Legacy heuristic rate-limit handling (non-HTTP or unknown status)
                if _is_rate_limit_error(e) and attempt < retry_max_attempts:
                    is_tpm = _is_tpm_429(str(e))
                    kind = "TPM" if is_tpm else "RPM"
                    sleep_s = _backoff_seconds(attempt, is_tpm, retry_backoff_base, retry_backoff_max)
                    print(f"    Rate limited (429, {kind}): attempt {attempt}/{retry_max_attempts}, sleeping {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                else:
                    last_error = e
                    break
        
        if result is not None:
            # Success: extract and persist
            text = result.text
            finish_reason = result.finish_reason
            raw = result.raw or {}
            usage = result.usage or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            timings = result.timings or {}
            latency_s = timings.get("latency_s")
            response_id = str(uuid4())
            dao.insert_response(
                conn=conn,
                response_id=response_id,
                request_id=request_id,
                text=text,
                finish_reason=finish_reason,
                usage=usage,
                raw=raw,
            )
            telemetry = {
                "latency_s": latency_s,
                "ttfb_s": timings.get("ttfb_s"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "notes": None,
            }
            dao.upsert_telemetry(conn, request_id=request_id, telemetry=telemetry)
            conn.execute("DELETE FROM failures WHERE request_id = ?", (request_id,))
            if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest_entries):
                print(f"    Processed {idx + 1}/{len(manifest_entries)} entries")
            continue
        
        # All retries failed or non-429 error: record failure
        e = last_error
        print(f"    Error processing entry {idx} (example_id={example_id}): {e}")
        cursor = conn.execute(
            "SELECT 1 FROM responses WHERE request_id = ?",
            (request_id,)
        )
        response_exists = cursor.fetchone() is not None
        if not response_exists:
            message = str(e)
            label = classify_failure(text="", finish_reason=None, error_message=message)
            error_type = label if label is not None else type(e).__name__
            dao.upsert_failure(conn, request_id=request_id, error_type=error_type, message=message)
        continue
    
    print(f"  Completed processing for run_id={run_id}")
