"""Data Access Object for Phase B and Phase C database operations.

Provides functions to insert and query datasets, examples, bins, manifest entries,
runs, requests, responses, telemetry, and failures.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Union


def upsert_dataset(
    conn: sqlite3.Connection,
    dataset_id: str,
    name: str,
    task: str,
    tokenizer_name: Optional[str] = None
) -> None:
    """Insert or replace a dataset record.
    
    Args:
        conn: SQLite connection.
        dataset_id: Dataset identifier.
        name: Dataset name (e.g., "longbench").
        task: Task identifier.
        tokenizer_name: Optional tokenizer name.
    """
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO datasets 
            (dataset_id, name, task, tokenizer_name, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (dataset_id, name, task, tokenizer_name, datetime.utcnow().isoformat()))


def insert_examples(
    conn: sqlite3.Connection,
    dataset_id: str,
    examples: List[Dict]
) -> None:
    """Bulk insert examples into the database.
    
    Args:
        conn: SQLite connection.
        dataset_id: Dataset identifier.
        examples: List of normalized example dicts with fields:
            - example_id
            - question
            - context
            - answers (list, will be JSON-encoded)
            - meta (dict, will be JSON-encoded)
            - token_len (optional, defaults to 0)
    """
    with conn:
        for ex in examples:
            example_id = ex["example_id"]
            question = ex["question"]
            context = ex["context"]
            answers_json = json.dumps(ex["answers"], ensure_ascii=False)
            meta_json = json.dumps(ex["meta"], ensure_ascii=False)
            token_len = ex.get("token_len", 0)
            
            conn.execute("""
                INSERT OR REPLACE INTO examples
                (example_id, dataset_id, question, context, answers_json, meta_json, token_len)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (example_id, dataset_id, question, context, answers_json, meta_json, token_len))


def insert_bins(
    conn: sqlite3.Connection,
    dataset_id: str,
    bin_edges: Union[List[Dict], List[tuple]],
    n_examples_per_bin: Optional[Dict[int, int]] = None
) -> None:
    """Insert bin definitions into the database.
    
    Args:
        conn: SQLite connection.
        dataset_id: Dataset identifier.
        bin_edges: List of bin edge definitions. Can be:
            - List of dicts with keys: bin_idx, token_min, token_max
            - List of tuples: (bin_idx, token_min, token_max)
        n_examples_per_bin: Optional dict mapping bin_idx to example count.
                          If not provided, counts are set to 0.
    """
    with conn:
        for edge in bin_edges:
            if isinstance(edge, dict):
                bin_idx = edge["bin_idx"]
                token_min = edge["token_min"]
                token_max = edge["token_max"]
            else:
                # Tuple format: (bin_idx, token_min, token_max)
                bin_idx, token_min, token_max = edge
            
            n_examples = 0
            if n_examples_per_bin is not None:
                n_examples = n_examples_per_bin.get(bin_idx, 0)
            
            conn.execute("""
                INSERT OR REPLACE INTO bins
                (dataset_id, bin_idx, token_min, token_max, n_examples)
                VALUES (?, ?, ?, ?, ?)
            """, (dataset_id, bin_idx, token_min, token_max, n_examples))


def insert_manifest_entries(
    conn: sqlite3.Connection,
    dataset_id: str,
    entries: List[Dict]
) -> None:
    """Insert manifest entries into the database.
    
    Args:
        conn: SQLite connection.
        dataset_id: Dataset identifier.
        entries: List of entry dicts with fields:
            - example_id
            - bin_idx
            - token_len
        entry_idx is assigned in insertion order (0..n-1).
    """
    with conn:
        for entry_idx, entry in enumerate(entries):
            example_id = entry["example_id"]
            bin_idx = entry["bin_idx"]
            token_len = entry["token_len"]
            
            conn.execute("""
                INSERT OR REPLACE INTO manifest_entries
                (dataset_id, entry_idx, example_id, bin_idx, token_len)
                VALUES (?, ?, ?, ?, ?)
            """, (dataset_id, entry_idx, example_id, bin_idx, token_len))


def get_manifest_entries(
    conn: sqlite3.Connection,
    dataset_id: str
) -> List[sqlite3.Row]:
    """Get all manifest entries for a dataset.
    
    Args:
        conn: SQLite connection.
        dataset_id: Dataset identifier.
    
    Returns:
        List of Row objects, ordered by entry_idx.
    """
    return conn.execute("""
        SELECT * FROM manifest_entries
        WHERE dataset_id = ?
        ORDER BY entry_idx
    """, (dataset_id,)).fetchall()


def get_examples_by_ids(
    conn: sqlite3.Connection,
    example_ids: List[str]
) -> Dict[str, sqlite3.Row]:
    """Get examples by their IDs.
    
    Args:
        conn: SQLite connection.
        example_ids: List of example IDs to fetch.
    
    Returns:
        Dict mapping example_id to Row object.
    """
    if not example_ids:
        return {}
    
    placeholders = ",".join("?" * len(example_ids))
    rows = conn.execute(f"""
        SELECT * FROM examples
        WHERE example_id IN ({placeholders})
    """, example_ids).fetchall()
    
    return {row["example_id"]: row for row in rows}


# ===== Phase C: Inference logging operations =====

def upsert_run(
    conn: sqlite3.Connection,
    run_id: str,
    exp_group_id: str,
    kv_policy: str,
    kv_budget: float,
    engine_name: str,
    base_url: str,
    model_name: str
) -> None:
    """Insert or replace a run record.
    
    Args:
        conn: SQLite connection.
        run_id: Run identifier.
        exp_group_id: Experiment group identifier.
        kv_policy: KV compression policy name.
        kv_budget: KV budget ratio (float).
        engine_name: Engine name (e.g., "openai_compat").
        base_url: Base URL for the engine.
        model_name: Model name.
    """
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO runs
            (run_id, exp_group_id, kv_policy, kv_budget, engine_name, base_url, model_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            exp_group_id,
            kv_policy,
            kv_budget,
            engine_name,
            base_url,
            model_name,
            datetime.utcnow().isoformat()
        ))


def insert_request(
    conn: sqlite3.Connection,
    request_id: str,
    run_id: str,
    dataset_id: str,
    entry_idx: int,
    example_id: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
    timeout_s: Optional[float] = None
) -> None:
    """Insert a request record.
    
    Args:
        conn: SQLite connection.
        request_id: Request identifier.
        run_id: Run identifier.
        dataset_id: Dataset identifier.
        entry_idx: Manifest entry index.
        example_id: Example identifier.
        messages: List of message dicts (OpenAI-style).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout_s: Optional request timeout in seconds.
    """
    with conn:
        messages_json = json.dumps(messages, ensure_ascii=False)
        conn.execute("""
            INSERT INTO requests
            (request_id, run_id, dataset_id, entry_idx, example_id, messages_json,
             temperature, max_tokens, timeout_s, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            run_id,
            dataset_id,
            entry_idx,
            example_id,
            messages_json,
            temperature,
            max_tokens,
            timeout_s,
            datetime.utcnow().isoformat()
        ))


def insert_response(
    conn: sqlite3.Connection,
    response_id: str,
    request_id: str,
    text: str,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict] = None,
    raw: Optional[Dict] = None
) -> None:
    """Insert a response record.
    
    Args:
        conn: SQLite connection.
        response_id: Response identifier.
        request_id: Request identifier.
        text: Generated text.
        finish_reason: Finish reason (e.g., "stop", "length").
        usage: Optional token usage dict (will be JSON-encoded).
        raw: Optional raw response dict (will be JSON-encoded).
    """
    with conn:
        usage_json = json.dumps(usage, ensure_ascii=False) if usage else None
        raw_json = json.dumps(raw, ensure_ascii=False) if raw else None
        conn.execute("""
            INSERT INTO responses
            (response_id, request_id, text, finish_reason, usage_json, raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            response_id,
            request_id,
            text,
            finish_reason,
            usage_json,
            raw_json,
            datetime.utcnow().isoformat()
        ))


def upsert_telemetry(
    conn: sqlite3.Connection,
    request_id: str,
    telemetry: Dict
) -> None:
    """Insert or replace telemetry record.
    
    Args:
        conn: SQLite connection.
        request_id: Request identifier.
        telemetry: Telemetry dict with fields:
            - latency_s (float | None)
            - ttfb_s (float | None)
            - prompt_tokens (int | None)
            - completion_tokens (int | None)
            - total_tokens (int | None)
            - notes (dict | None) - will be stored as notes_json
    """
    with conn:
        notes = telemetry.get("notes")
        notes_json = json.dumps(notes, ensure_ascii=False) if notes else None
        conn.execute("""
            INSERT OR REPLACE INTO telemetry
            (request_id, latency_s, ttfb_s, prompt_tokens, completion_tokens, total_tokens, notes_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            telemetry.get("latency_s"),
            telemetry.get("ttfb_s"),
            telemetry.get("prompt_tokens"),
            telemetry.get("completion_tokens"),
            telemetry.get("total_tokens"),
            notes_json
        ))


def upsert_failure(
    conn: sqlite3.Connection,
    request_id: str,
    error_type: str,
    message: str
) -> None:
    """Insert or replace a failure record.
    
    Args:
        conn: SQLite connection.
        request_id: Request identifier.
        error_type: Error type (e.g., "RuntimeError", "TimeoutError").
        message: Error message.
    """
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO failures
            (request_id, error_type, message)
            VALUES (?, ?, ?)
        """, (request_id, error_type, message))


def run_exists(conn: sqlite3.Connection, run_id: str) -> bool:
    """Check if a run exists.
    
    Args:
        conn: SQLite connection.
        run_id: Run identifier.
    
    Returns:
        True if run exists, False otherwise.
    """
    cursor = conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))
    return cursor.fetchone() is not None


def count_requests_for_run(conn: sqlite3.Connection, run_id: str) -> int:
    """Count requests for a run.
    
    Args:
        conn: SQLite connection.
        run_id: Run identifier.
    
    Returns:
        Number of requests for the run.
    """
    cursor = conn.execute("SELECT COUNT(*) FROM requests WHERE run_id = ?", (run_id,))
    result = cursor.fetchone()
    return result[0] if result else 0
