"""SQLite schema definition for Phase B dataset preparation.

Creates tables for storing datasets, examples, bins, and manifest entries.
"""

import sqlite3


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize database schema with Phase B tables.
    
    Creates tables for experiments, datasets, examples, bins, and manifest_entries.
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
    
    # Commit changes
    conn.commit()
