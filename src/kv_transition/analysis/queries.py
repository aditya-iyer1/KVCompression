"""Read-only SQL query helpers for Phase E aggregation.

Provides functions to fetch per-run, per-bin data needed for statistics computation.
"""

import sqlite3
from typing import List, Optional


def get_run_metadata(conn: sqlite3.Connection, run_id: str) -> Optional[sqlite3.Row]:
    """Get metadata for a run.
    
    Args:
        conn: SQLite connection.
        run_id: Run identifier.
    
    Returns:
        Row with run metadata (run_id, exp_group_id, kv_policy, kv_budget, etc.)
        or None if run not found.
    """
    cursor = conn.execute("""
        SELECT *
        FROM runs
        WHERE run_id = ?
    """, (run_id,))
    
    return cursor.fetchone()


def get_bin_structure(conn: sqlite3.Connection, dataset_id: str) -> List[sqlite3.Row]:
    """Get bin structure for a dataset.
    
    Args:
        conn: SQLite connection.
        dataset_id: Dataset identifier.
    
    Returns:
        List of rows with bin_idx, token_min, token_max, ordered by bin_idx.
    """
    cursor = conn.execute("""
        SELECT bin_idx, token_min, token_max, n_examples
        FROM bins
        WHERE dataset_id = ?
        ORDER BY bin_idx
    """, (dataset_id,))
    
    return cursor.fetchall()


def get_bin_level_rows(conn: sqlite3.Connection, run_id: str) -> List[sqlite3.Row]:
    """Get bin-level data for all requests in a run.
    
    Joins requests, scores, telemetry, manifest_entries, and failures
    to provide per-request data with bin assignments.
    
    Args:
        conn: SQLite connection.
        run_id: Run identifier.
    
    Returns:
        List of rows with:
        - bin_idx
        - em (from scores)
        - f1 (from scores)
        - latency_s (from telemetry)
        - prompt_tokens (from telemetry)
        - completion_tokens (from telemetry)
        - total_tokens (from telemetry)
        - failure (0 or 1, indicating if failure exists)
        Ordered by bin_idx.
    """
    cursor = conn.execute("""
        SELECT 
            me.bin_idx,
            s.em,
            s.f1,
            t.latency_s,
            t.prompt_tokens,
            t.completion_tokens,
            t.total_tokens,
            CASE WHEN f.request_id IS NOT NULL THEN 1 ELSE 0 END AS failure
        FROM requests r
        LEFT JOIN scores s ON r.request_id = s.request_id
        LEFT JOIN telemetry t ON r.request_id = t.request_id
        LEFT JOIN manifest_entries me ON r.dataset_id = me.dataset_id 
            AND r.entry_idx = me.entry_idx
        LEFT JOIN failures f ON r.request_id = f.request_id
        WHERE r.run_id = ?
        ORDER BY me.bin_idx, r.request_id
    """, (run_id,))
    
    return cursor.fetchall()
