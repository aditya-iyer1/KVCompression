"""Phase D scoring: compute EM/F1 and classify failures.

Joins predictions with gold answers and writes scores and failure taxonomy.
"""

import csv
import json
import sqlite3
from pathlib import Path
from typing import Optional

from .. import paths
from .failure_taxonomy import classify_failure
from .metrics import best_exact_match, best_f1, normalize_text


def _ensure_scores_table(conn: sqlite3.Connection) -> bool:
    """Ensure scores table exists (create if not present).
    
    Uses IF NOT EXISTS to avoid errors if table already exists.
    This is safe and doesn't modify the schema.py file.
    
    Args:
        conn: SQLite connection.
    
    Returns:
        True if table exists or was created, False if creation failed.
    """
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                request_id TEXT PRIMARY KEY,
                em REAL NOT NULL,
                f1 REAL NOT NULL,
                pred_norm TEXT,
                gold_norm TEXT,
                FOREIGN KEY (request_id) REFERENCES requests(request_id)
            )
        """)
        conn.commit()
        return True
    except Exception:
        return False


def _get_run_exp_group_id(conn: sqlite3.Connection, run_id: str) -> Optional[str]:
    """Get exp_group_id for a run.
    
    Args:
        conn: SQLite connection.
        run_id: Run identifier.
    
    Returns:
        exp_group_id or None if run not found.
    """
    cursor = conn.execute("SELECT exp_group_id FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    return row["exp_group_id"] if row else None


def _write_scores_csv(exp_group_id: str, scores_data: list) -> Path:
    """Write scores to CSV as fallback.
    
    Args:
        exp_group_id: Experiment group identifier.
        scores_data: List of dicts with score data.
    
    Returns:
        Path to written CSV file.
    """
    run_dir = paths.run_dir(exp_group_id)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = tables_dir / "scores.csv"
    
    with open(csv_path, 'w', newline='') as f:
        if not scores_data:
            return csv_path
        
        writer = csv.DictWriter(f, fieldnames=scores_data[0].keys())
        writer.writeheader()
        writer.writerows(scores_data)
    
    return csv_path


def score_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Score all requests in a run.
    
    Computes EM/F1 for each request by comparing response text with gold answers,
    classifies failures, and persists results to SQLite.
    
    Args:
        conn: SQLite connection.
        run_id: Run identifier.
    
    Raises:
        ValueError: If run_id not found or no requests found.
    """
    # Verify run exists
    cursor = conn.execute("SELECT exp_group_id FROM runs WHERE run_id = ?", (run_id,))
    run_row = cursor.fetchone()
    if not run_row:
        raise ValueError(f"Run {run_id} not found")
    
    exp_group_id = run_row["exp_group_id"]
    
    # Query all requests with their responses and failure info
    cursor = conn.execute("""
        SELECT 
            r.request_id,
            r.example_id,
            resp.text,
            resp.finish_reason,
            fail.error_type,
            fail.message as error_message
        FROM requests r
        LEFT JOIN responses resp ON r.request_id = resp.request_id
        LEFT JOIN failures fail ON r.request_id = fail.request_id
        WHERE r.run_id = ?
        ORDER BY r.request_id
    """, (run_id,))
    
    request_rows = cursor.fetchall()
    if not request_rows:
        raise ValueError(f"No requests found for run {run_id}")
    
    # Load gold answers for all examples
    example_ids = [row["example_id"] for row in request_rows]
    placeholders = ",".join("?" * len(example_ids))
    cursor = conn.execute(f"""
        SELECT example_id, answers_json
        FROM examples
        WHERE example_id IN ({placeholders})
    """, example_ids)
    
    example_rows = cursor.fetchall()
    gold_answers_map = {}
    for row in example_rows:
        answers_json = row["answers_json"]
        answers = json.loads(answers_json) if answers_json else []
        gold_answers_map[row["example_id"]] = answers
    
    # Ensure scores table exists
    scores_table_exists = _ensure_scores_table(conn)
    
    # Prepare scores data
    scores_to_insert = []
    csv_fallback_data = []
    
    # Process each request
    for req_row in request_rows:
        request_id = req_row["request_id"]
        example_id = req_row["example_id"]
        pred_text = req_row["text"] if req_row["text"] else ""
        finish_reason = req_row["finish_reason"]
        error_type = req_row["error_type"]
        error_message = req_row["error_message"]
        
        # Get gold answers
        gold_answers = gold_answers_map.get(example_id, [])
        if not gold_answers:
            # Skip if no gold answers found
            continue
        
        # Compute scores (use empty string if pred_text is None)
        pred_text_for_scoring = pred_text if pred_text else ""
        em = best_exact_match(pred_text_for_scoring, gold_answers)
        f1 = best_f1(pred_text_for_scoring, gold_answers)
        
        # Normalize texts
        pred_norm = normalize_text(pred_text_for_scoring)
        # Use first gold answer for gold_norm (consistent approach)
        gold_norm = normalize_text(gold_answers[0]) if gold_answers else ""
        
        # Classify failure
        failure_label = classify_failure(pred_text, finish_reason, error_message)
        
        # Update failures table with taxonomy label if needed
        if failure_label:
            # Update error_type with taxonomy label (only if not already set or overwrite)
            # Use taxonomy label as prefix in message if error_type already exists
            if error_type:
                # If error_type exists, append taxonomy to message
                new_message = f"[TAXONOMY: {failure_label}] {error_message}" if error_message else f"[TAXONOMY: {failure_label}]"
                with conn:
                    conn.execute("""
                        UPDATE failures
                        SET message = ?
                        WHERE request_id = ?
                    """, (new_message, request_id))
            else:
                # If no error_type, set it to taxonomy label
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO failures (request_id, error_type, message)
                        VALUES (?, ?, ?)
                    """, (request_id, failure_label, error_message or ""))
        
        # Prepare score data
        score_data = {
            "request_id": request_id,
            "em": em,
            "f1": f1,
            "pred_norm": pred_norm,
            "gold_norm": gold_norm
        }
        scores_to_insert.append(score_data)
        csv_fallback_data.append(score_data)
    
    # Write scores to DB if table exists
    if scores_table_exists:
        with conn:
            for score_data in scores_to_insert:
                conn.execute("""
                    INSERT OR REPLACE INTO scores
                    (request_id, em, f1, pred_norm, gold_norm)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    score_data["request_id"],
                    score_data["em"],
                    score_data["f1"],
                    score_data["pred_norm"],
                    score_data["gold_norm"]
                ))
        print(f"  Scored {len(scores_to_insert)} requests, wrote to scores table")
    else:
        # Fallback: write to CSV
        csv_path = _write_scores_csv(exp_group_id, csv_fallback_data)
        print(f"  Scored {len(scores_to_insert)} requests, wrote to CSV: {csv_path}")
        print(f"  TODO: Add scores table to schema.py for proper DB storage")
