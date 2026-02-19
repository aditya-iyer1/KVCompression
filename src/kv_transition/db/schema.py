"""SQLite schema definition for Phase B dataset preparation, Phase C inference logging, and Phase E aggregation.

Creates tables for storing datasets, examples, bins, manifest entries, runs, requests,
responses, telemetry, failures, bin_stats, and transition_summary.
"""

import sqlite3


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize database schema with Phase B, Phase C, and Phase E tables.
    
    Creates tables for experiments, datasets, examples, bins, manifest_entries,
    runs, requests, responses, telemetry, failures, bin_stats, and transition_summary.
    Uses IF NOT EXISTS for idempotency. Foreign keys are enabled by connect.py.
    
    Args:
        conn: SQLite connection (from connect.connect()).
    """
    # Experiments table (minimal, for Phase A+ metadata)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            exp_group_id TEXT PRIMARY KEY,
            created_at TEXT,
            config_yaml TEXT,
            git_hash TEXT,
            notes TEXT
        )
    """)
    
    # Datasets table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            task TEXT NOT NULL,
            tokenizer_name TEXT,
            created_at TEXT
        )
    """)
    
    # Examples table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS examples (
            example_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            question TEXT NOT NULL,
            context TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            token_len INTEGER NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        )
    """)
    
    # Bins table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bins (
            dataset_id TEXT NOT NULL,
            bin_idx INTEGER NOT NULL,
            token_min INTEGER NOT NULL,
            token_max INTEGER NOT NULL,
            n_examples INTEGER NOT NULL,
            PRIMARY KEY (dataset_id, bin_idx),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        )
    """)
    
    # Manifest entries table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS manifest_entries (
            dataset_id TEXT NOT NULL,
            entry_idx INTEGER NOT NULL,
            example_id TEXT NOT NULL,
            bin_idx INTEGER NOT NULL,
            token_len INTEGER NOT NULL,
            PRIMARY KEY (dataset_id, entry_idx),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            FOREIGN KEY (example_id) REFERENCES examples(example_id)
        )
    """)
    
    # Indexes for common Phase B queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_examples_dataset_id
        ON examples(dataset_id)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_manifest_entries_dataset_bin
        ON manifest_entries(dataset_id, bin_idx)
    """)
    
    # ===== Phase C: Inference logging tables =====
    
    # Runs table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            exp_group_id TEXT,
            kv_policy TEXT,
            kv_budget REAL,
            engine_name TEXT,
            base_url TEXT,
            model_name TEXT,
            created_at TEXT,
            FOREIGN KEY (exp_group_id) REFERENCES experiments(exp_group_id)
        )
    """)
    
    # Requests table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            entry_idx INTEGER NOT NULL,
            example_id TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            temperature REAL,
            max_tokens INTEGER,
            timeout_s REAL,
            created_at TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            FOREIGN KEY (example_id) REFERENCES examples(example_id)
        )
    """)
    
    # Responses table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            response_id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL,
            text TEXT,
            finish_reason TEXT,
            usage_json TEXT,
            raw_json TEXT,
            created_at TEXT,
            FOREIGN KEY (request_id) REFERENCES requests(request_id)
        )
    """)
    
    # Telemetry table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            request_id TEXT PRIMARY KEY,
            latency_s REAL,
            ttfb_s REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            notes_json TEXT,
            FOREIGN KEY (request_id) REFERENCES requests(request_id)
        )
    """)
    
    # Failures table (for error tracking)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS failures (
            request_id TEXT PRIMARY KEY,
            error_type TEXT,
            message TEXT,
            FOREIGN KEY (request_id) REFERENCES requests(request_id)
        )
    """)
    
    # Indexes for common Phase C queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_requests_run_id
        ON requests(run_id)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_requests_dataset_entry
        ON requests(dataset_id, entry_idx)
    """)
    
    # ===== Phase E: Aggregation and transition detection tables =====
    
    # Bin stats table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bin_stats (
            run_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            bin_idx INTEGER NOT NULL,
            token_min INTEGER,
            token_max INTEGER,
            n INTEGER,
            acc_mean REAL,
            acc_std REAL,
            acc_ci_low REAL,
            acc_ci_high REAL,
            em_mean REAL,
            fail_rate REAL,
            lat_p50 REAL,
            lat_p95 REAL,
            tok_p50 REAL,
            tok_p95 REAL,
            PRIMARY KEY (run_id, bin_idx),
            FOREIGN KEY (run_id) REFERENCES runs(run_id),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        )
    """)
    
    # Transition summary table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transition_summary (
            exp_group_id TEXT PRIMARY KEY,
            kv_policy TEXT,
            method TEXT,
            drop_threshold REAL,
            pre_budget REAL,
            transition_budget REAL,
            acc_pre REAL,
            acc_post REAL,
            "drop" REAL,
            transition_bin_idx INTEGER,
            created_at TEXT,
            FOREIGN KEY (exp_group_id) REFERENCES experiments(exp_group_id)
        )
    """)
    
    # Indexes for common Phase E queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bin_stats_run_id
        ON bin_stats(run_id)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bin_stats_dataset_bin
        ON bin_stats(dataset_id, bin_idx)
    """)
    
    # Commit changes
    conn.commit()
