"""Unit tests for report generation prerequisites (0-scored vs scored run groups)."""

import sqlite3
import pytest
from pathlib import Path

from kv_transition.db import connect, schema
from kv_transition.report.build import validate_report_prerequisites


def _create_minimal_db(db_path: Path, exp_group_id: str, run_id: str, dataset_id: str) -> str:
    """Create minimal DB with experiments, runs, requests, bin_stats. Returns request_id."""
    conn = connect.connect(db_path)
    schema.init_schema(conn)
    request_id = "req-001"
    example_id = "ex-1"
    conn.execute(
        "INSERT OR IGNORE INTO experiments (exp_group_id, created_at) VALUES (?, ?)",
        (exp_group_id, "2025-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO datasets (dataset_id, name, task, created_at) VALUES (?, ?, ?, ?)",
        (dataset_id, "test-ds", "qa", "2025-01-01T00:00:00"),
    )
    conn.execute(
        """INSERT OR IGNORE INTO examples (example_id, dataset_id, question, context, answers_json, meta_json, token_len)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (example_id, dataset_id, "q", "c", "[]", "{}", 100),
    )
    conn.execute(
        """INSERT OR IGNORE INTO runs (run_id, exp_group_id, kv_policy, kv_budget, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, exp_group_id, "snapkv", 1.0, "2025-01-01T00:00:00"),
    )
    conn.execute(
        """INSERT OR IGNORE INTO requests (request_id, run_id, dataset_id, entry_idx, example_id, messages_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (request_id, run_id, dataset_id, 0, example_id, "[]"),
    )
    conn.execute(
        """INSERT OR IGNORE INTO bin_stats (run_id, dataset_id, bin_idx, n, acc_mean, fail_rate)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, dataset_id, 0, 1, 0.5, 0.0),
    )
    conn.commit()
    conn.close()
    return request_id


def test_validate_prereqs_zero_scored_allows_missing_acc_latency_plots(tmp_path: Path) -> None:
    """When scored_count == 0, validation passes without acc_by_bin or latency_p50_by_bin."""
    exp_group_id = "edge_zero_scored"
    run_id = f"{exp_group_id}_budget_1.0"
    dataset_id = "longbench__narrativeqa"
    db_path = tmp_path / "kv_transition.sqlite"
    run_dir = tmp_path / "runs" / exp_group_id
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True)

    _create_minimal_db(db_path, exp_group_id, run_id, dataset_id)
    # No scores table rows -> scored_count == 0

    is_valid, missing = validate_report_prerequisites(
        db_path, run_dir, exp_group_id, allow_partial=False
    )
    assert is_valid is True, f"Expected valid for 0-scored run group; missing={missing}"
    assert not any("acc_by_bin" in m or "latency_p50" in m for m in missing)


def test_validate_prereqs_with_scores_requires_plots(tmp_path: Path) -> None:
    """When scored_count > 0, missing acc/latency plots make validation fail (strictness)."""
    exp_group_id = "edge_has_scores"
    run_id = f"{exp_group_id}_budget_1.0"
    dataset_id = "longbench__narrativeqa"
    db_path = tmp_path / "kv_transition.sqlite"
    run_dir = tmp_path / "runs" / exp_group_id
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True)

    request_id = _create_minimal_db(db_path, exp_group_id, run_id, dataset_id)

    # Create scores table and insert one row so scored_count > 0
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            request_id TEXT PRIMARY KEY,
            em REAL NOT NULL,
            f1 REAL NOT NULL,
            acc REAL,
            pred_norm TEXT,
            gold_norm TEXT,
            FOREIGN KEY (request_id) REFERENCES requests(request_id)
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO scores (request_id, em, f1, acc) VALUES (?, ?, ?, ?)",
        (request_id, 0.0, 0.5, 0.5),
    )
    conn.commit()
    conn.close()

    # Only fail_by_bin present (or none) -> missing acc and latency should be reported
    (fail_plot := plots_dir / "fail_by_bin.png").write_text("")
    # Do not create acc_by_bin.png or latency_p50_by_bin.png

    is_valid, missing = validate_report_prerequisites(
        db_path, run_dir, exp_group_id, allow_partial=False
    )
    assert is_valid is False
    assert any("acc_by_bin" in m for m in missing)
    assert any("latency_p50" in m for m in missing)
