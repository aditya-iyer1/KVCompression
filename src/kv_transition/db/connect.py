"""SQLite connection helper with safe pragmas.

Provides minimal connection utilities for database operations.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Union


def connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Open SQLite connection with safe pragmas.
    
    Creates parent directory if needed and applies recommended pragmas
    for concurrent access and data integrity.
    
    Args:
        db_path: Path to SQLite database file.
    
    Returns:
        sqlite3.Connection with pragmas applied and row_factory set.
    """
    db_path = Path(db_path)
    
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open connection
    conn = sqlite3.connect(str(db_path))
    
    # Apply pragmas
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    # Set row_factory for ergonomic queries (returns Row objects)
    conn.row_factory = sqlite3.Row
    
    return conn


@contextmanager
def connect_ctx(db_path: Union[str, Path]):
    """Context manager for SQLite connection.
    
    Automatically closes the connection when exiting the context.
    
    Args:
        db_path: Path to SQLite database file.
    
    Yields:
        sqlite3.Connection with pragmas applied.
    
    Example:
        with connect_ctx("path/to/db.sqlite") as conn:
            conn.execute("SELECT * FROM table")
    """
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
