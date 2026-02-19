"""LongBench dataset loader with local caching.

Phase B: Fetches and loads LongBench task subsets using HuggingFace datasets.
"""

from pathlib import Path
from typing import Optional, Union

try:
    from datasets import load_dataset, get_dataset_config_names
except ImportError:
    raise ImportError(
        "The 'datasets' library is required. Install with: pip install datasets"
    ) from None

from ..paths import data_dir


# LongBench dataset identifier on HuggingFace
LONGBENCH_DATASET_ID = "THUDM/LongBench"


def _get_default_cache_dir() -> Path:
    """Get default cache directory for LongBench data."""
    return data_dir() / "raw" / "longbench"


def list_longbench_tasks(cache_dir: Optional[Union[Path, str]] = None) -> list[str]:
    """List available LongBench task keys from dataset metadata.
    
    Queries the dataset repository to get authoritative, case-sensitive task names.
    
    Args:
        cache_dir: Optional cache directory. If None, uses default repo-local cache.
    
    Returns:
        List of available task keys (case-sensitive, as exposed by the dataset).
    
    Raises:
        ImportError: If datasets library is not installed.
        RuntimeError: If dataset cannot be accessed or has no configs.
    """
    # Resolve cache directory
    if cache_dir is None:
        cache_dir_path = _get_default_cache_dir()
    else:
        cache_dir_path = Path(cache_dir)
    
    try:
        # Get available configuration names (these are the task keys)
        config_names = get_dataset_config_names(
            LONGBENCH_DATASET_ID,
            trust_remote_code=True
        )
        
        if not config_names:
            raise RuntimeError(
                f"Dataset {LONGBENCH_DATASET_ID} has no available configurations. "
                "This may indicate the dataset is not accessible or has changed."
            )
        
        # Return as sorted list for stable output
        return sorted(config_names)
    
    except Exception as e:
        if "datasets" in str(type(e).__module__):
            # Re-raise dataset-specific errors with context
            raise RuntimeError(
                f"Failed to list tasks from {LONGBENCH_DATASET_ID}: {e}. "
                "Ensure the dataset is accessible and you have network connectivity."
            ) from e
        raise


def load_longbench(
    task: str,
    cache_dir: Optional[Union[Path, str]] = None,
    split: str = "test"
) -> list[dict]:
    """Load a single LongBench task subset with local caching.
    
    Args:
        task: Exact task/subset key (case-sensitive, as returned by list_longbench_tasks()).
        cache_dir: Optional cache directory. If None, uses default repo-local cache.
        split: Dataset split to load (default: "test").
    
    Returns:
        List of raw example dicts exactly as provided by the dataset rows.
        No normalization is performed here.
    
    Raises:
        ValueError: If task is empty or invalid.
        ImportError: If datasets library is not installed.
        RuntimeError: If task is not available or dataset cannot be loaded.
    """
    # Validate task
    if not task or not task.strip():
        raise ValueError("task must be non-empty")
    
    task = task.strip()
    
    # Resolve cache directory
    if cache_dir is None:
        cache_dir_path = _get_default_cache_dir()
    else:
        cache_dir_path = Path(cache_dir)
    
    # Create cache directory if needed (only when function is called)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the dataset with the specified task/config
        dataset = load_dataset(
            LONGBENCH_DATASET_ID,
            name=task,
            split=split,
            cache_dir=str(cache_dir_path),
            trust_remote_code=True
        )
        
        # Convert to list of dicts (raw examples as provided by dataset)
        examples = [dict(row) for row in dataset]
        
        if not examples:
            raise RuntimeError(
                f"Task '{task}' split '{split}' returned no examples. "
                f"Available splits may differ. Try calling list_longbench_tasks() to see available tasks."
            )
        
        return examples
    
    except Exception as e:
        # Provide helpful error messages
        error_msg = str(e).lower()
        
        if "config name" in error_msg or "not found" in error_msg or "doesn't exist" in error_msg:
            # Task not found - suggest checking available tasks
            try:
                available = list_longbench_tasks(cache_dir)
                available_str = ", ".join(available[:10])  # Show first 10
                if len(available) > 10:
                    available_str += f", ... ({len(available)} total)"
                suggestion = f"Available tasks: {available_str}. Call list_longbench_tasks() for full list."
            except Exception:
                suggestion = "Call list_longbench_tasks() to see available task keys."
            
            raise ValueError(
                f"Task '{task}' is not available in {LONGBENCH_DATASET_ID}. {suggestion}"
            ) from e
        
        if "split" in error_msg and ("not found" in error_msg or "doesn't exist" in error_msg):
            raise ValueError(
                f"Split '{split}' is not available for task '{task}'. "
                "Common splits: 'test', 'train', 'validation'. "
                "Check the dataset documentation for available splits."
            ) from e
        
        # Generic error with context
        raise RuntimeError(
            f"Failed to load task '{task}' from {LONGBENCH_DATASET_ID}: {e}"
        ) from e
