"""Core execution loop for running inference on manifest entries.

Executes manifest entries for a single KV setting and persists results to DB.
"""

import hashlib
import inspect
import json
import sqlite3
import time
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, Optional

from ..db import dao
from ..engines.base import BaseEngine

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
    
    # Extract decoding parameters
    decoding = settings.get("decoding", {})
    temperature = decoding.get("temperature", 0.0)
    max_tokens = decoding.get("max_tokens", 256)
    
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
        
        # Global rate-limit pacing (engine.rate_limit.max_rpm)
        global _last_request_monotonic
        rate_limit = settings.get("engine", {}).get("rate_limit", {})
        max_rpm = rate_limit.get("max_rpm")
        if max_rpm is not None and max_rpm != 0:
            try:
                max_rpm_f = float(max_rpm)
            except (TypeError, ValueError):
                max_rpm_f = 0.0
            if max_rpm_f > 0:
                min_interval = 60.0 / max_rpm_f
                now = time.monotonic()
                if _last_request_monotonic is not None:
                    elapsed = now - _last_request_monotonic
                    if elapsed < min_interval:
                        sleep_duration = min_interval - elapsed
                        time.sleep(sleep_duration)
                        print(f"  Rate limit: slept {sleep_duration:.2f}s (max_rpm={max_rpm_f:.0f})")
                _last_request_monotonic = time.monotonic()
        
        # Call engine
        try:
            result = engine.generate(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            
            # Extract result data
            text = result.text
            finish_reason = result.finish_reason
            raw = result.raw or {}
            
            # Extract usage
            usage = result.usage or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            
            # Extract timings
            timings = result.timings or {}
            latency_s = timings.get("latency_s")
            
            # Persist response row
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
            
            # Persist telemetry row
            telemetry = {
                "latency_s": latency_s,
                "ttfb_s": timings.get("ttfb_s"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "notes": None,
            }
            dao.upsert_telemetry(conn, request_id=request_id, telemetry=telemetry)
            
            # Clean up any existing failure row for this request_id (success overwrites failure)
            conn.execute("DELETE FROM failures WHERE request_id = ?", (request_id,))
            
            # Progress logging
            if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest_entries):
                print(f"    Processed {idx + 1}/{len(manifest_entries)} entries")
        
        except Exception as e:
            # Error handling: record failure and continue
            print(f"    Error processing entry {idx} (example_id={example_id}): {e}")
            
            # Check if response already exists for this request_id
            # If it does, skip recording failure to avoid inconsistent state
            cursor = conn.execute(
                "SELECT 1 FROM responses WHERE request_id = ?",
                (request_id,)
            )
            response_exists = cursor.fetchone() is not None
            
            if not response_exists:
                # Only record failure if no response was persisted
                error_type = type(e).__name__
                message = str(e)
                dao.upsert_failure(conn, request_id=request_id, error_type=error_type, message=message)
            
            continue
    
    print(f"  Completed processing for run_id={run_id}")
