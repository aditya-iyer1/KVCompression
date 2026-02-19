"""Data Access Object for Phase B database operations.

Provides functions to insert and query datasets, examples, bins, and manifest entries.
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
