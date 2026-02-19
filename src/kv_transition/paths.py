"""Canonical path builders for KV Transition evaluation harness.

Phase A: Provides stable, OS-agnostic path resolution for data, runs, and database locations.
"""

import os
from pathlib import Path
from typing import Optional, Union


def repo_root() -> Path:
    """Find repository root directory robustly.
    
    Works when package is installed or run from repo by searching for
    marker files (pyproject.toml, .git) starting from this file's location.
    
    Returns:
        Path to repository root.
    
    Raises:
        RuntimeError: If repository root cannot be determined.
    """
    # Start from this file's location
    current = Path(__file__).resolve()
    
    # Walk up the directory tree looking for markers
    for parent in [current.parent] + list(current.parents):
        # Check for common repo markers
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
        if (parent / "config" / "default.yaml").exists():
            return parent
    
    # Fallback: assume we're in src/kv_transition/, go up 2 levels
    # This handles the case where markers aren't present but structure is correct
    fallback = current.parent.parent.parent
    if (fallback / "config").exists() or (fallback / "pyproject.toml").exists():
        return fallback
    
    # Last resort: use current working directory
    # This is less reliable but better than crashing
    cwd = Path.cwd()
    if (cwd / "config" / "default.yaml").exists():
        return cwd
    
    raise RuntimeError(
        "Could not determine repository root. "
        "Expected to find pyproject.toml, .git, or config/default.yaml in parent directories."
    )


def data_dir() -> Path:
    """Get the data directory path.
    
    Returns:
        Path to data/ directory in repository root.
    """
    return repo_root() / "data"


def processed_dir(dataset_id: str) -> Path:
    """Get the processed data directory for a dataset.
    
    Args:
        dataset_id: Dataset identifier (e.g., "longbench_narrativeqa").
    
    Returns:
        Path to data/processed/<dataset_id>/ directory.
    
    Raises:
        ValueError: If dataset_id is empty.
    """
    if not dataset_id or not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")
    return data_dir() / "processed" / dataset_id.strip()


def manifest_path(dataset_id: str) -> Path:
    """Get the manifest.json path for a dataset.
    
    Args:
        dataset_id: Dataset identifier (e.g., "longbench_narrativeqa").
    
    Returns:
        Path to data/processed/<dataset_id>/manifest.json file.
    
    Raises:
        ValueError: If dataset_id is empty.
    """
    return processed_dir(dataset_id) / "manifest.json"


def runs_dir() -> Path:
    """Get the runs directory path.
    
    Returns:
        Path to runs/ directory in repository root.
    """
    return repo_root() / "runs"


def run_dir(exp_group_id: str) -> Path:
    """Get the run directory for an experiment group.
    
    Args:
        exp_group_id: Experiment group identifier.
    
    Returns:
        Path to runs/<exp_group_id>/ directory.
    
    Raises:
        ValueError: If exp_group_id is empty.
    """
    if not exp_group_id or not exp_group_id.strip():
        raise ValueError("exp_group_id must be non-empty")
    return runs_dir() / exp_group_id.strip()


def db_path(exp_group_id: str, db_path_cfg: Optional[str] = None) -> Path:
    """Get the database path for an experiment group.
    
    Behavior:
        - If db_path_cfg is None or "auto" (case-insensitive), returns
          runs/<exp_group_id>/kv_transition.sqlite
        - If db_path_cfg is a relative path, interprets it relative to repo_root()
        - If db_path_cfg is an absolute path, uses it as-is
    
    Args:
        exp_group_id: Experiment group identifier.
        db_path_cfg: Database path from config (None, "auto", relative, or absolute).
    
    Returns:
        Path to database file.
    
    Raises:
        ValueError: If exp_group_id is empty.
    """
    if not exp_group_id or not exp_group_id.strip():
        raise ValueError("exp_group_id must be non-empty")
    
    exp_group_id = exp_group_id.strip()
    
    # Default/auto behavior: use canonical location
    if db_path_cfg is None or (isinstance(db_path_cfg, str) and db_path_cfg.strip().lower() == "auto"):
        return run_dir(exp_group_id) / "kv_transition.sqlite"
    
    db_path_cfg = db_path_cfg.strip()
    
    # Absolute path: use as-is
    if os.path.isabs(db_path_cfg):
        return Path(db_path_cfg)
    
    # Relative path: interpret relative to repo_root()
    return repo_root() / db_path_cfg


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist (with parents).
    
    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def ensure_run_dir(exp_group_id: str) -> Path:
    """Create and return the run directory for an experiment group.
    
    Args:
        exp_group_id: Experiment group identifier.
    
    Returns:
        Path to runs/<exp_group_id>/ directory (created if needed).
    
    Raises:
        ValueError: If exp_group_id is empty.
    """
    path = run_dir(exp_group_id)
    ensure_dir(path)
    return path
