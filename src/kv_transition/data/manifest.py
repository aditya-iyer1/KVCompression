"""Manifest builder for Phase B dataset preparation.

Creates the portable manifest.json artifact that defines the evaluation set.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ..paths import processed_dir, manifest_path

from . import binning
from . import normalize
from . import tokenizer


def _compute_dataset_id(dataset_name: str, task: str) -> str:
    """Compute stable dataset_id from dataset name and task.
    
    Args:
        dataset_name: Dataset name (e.g., "longbench").
        task: Task identifier (exact string, case-sensitive).
    
    Returns:
        Dataset ID string (e.g., "longbench__narrativeqa").
    """
    return f"{dataset_name}__{task}"


def _compute_token_length(example: Dict[str, Any], tokenizer_name: str) -> int:
    """Compute token length for an example's full prompt material.
    
    For binning purposes, uses context + question as the prompt.
    
    Args:
        example: Normalized example dict with 'question' and 'context'.
        tokenizer_name: Tokenizer name for counting.
    
    Returns:
        Token count for the prompt.
    """
    context = example.get("context", "")
    question = example.get("question", "")
    
    # Simple, stable prompt construction: context + separator + question
    prompt = context + "\n\n" + question
    
    return tokenizer.count_tokens(prompt, tokenizer_name if tokenizer_name else None)


def _select_examples_per_bin(
    examples: List[Dict[str, Any]],
    bin_idxs: List[int],
    token_lens: List[int],
    n_bins: int,
    n_per_bin: int
) -> List[int]:
    """Select up to n_per_bin examples per bin deterministically.
    
    Selection is deterministic: within each bin, examples are sorted by
    token length (then by example_id for stability), and first n_per_bin are selected.
    
    Args:
        examples: List of normalized examples.
        bin_idxs: Bin index for each example (aligned with examples).
        token_lens: Token length for each example (aligned with examples).
        n_bins: Number of bins.
        n_per_bin: Maximum examples to select per bin.
    
    Returns:
        List of selected example indices (into examples list).
    """
    # Group examples by bin
    bin_groups: Dict[int, List[int]] = {i: [] for i in range(n_bins)}
    
    for idx, bin_idx in enumerate(bin_idxs):
        bin_groups[bin_idx].append(idx)
    
    # Select examples per bin
    selected_indices = []
    
    for bin_idx in range(n_bins):
        bin_example_indices = bin_groups[bin_idx]
        
        if not bin_example_indices:
            continue
        
        # Sort deterministically: by token length, then by example_id
        bin_example_indices.sort(
            key=lambda idx: (token_lens[idx], examples[idx]["example_id"])
        )
        
        # Take first n_per_bin
        selected_indices.extend(bin_example_indices[:n_per_bin])
    
    return selected_indices


def build_manifest(settings: Dict[str, Any], raw_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build manifest from settings and raw examples.
    
    Args:
        settings: Settings dict (from load_settings).
        raw_examples: List of raw example dicts from dataset loader.
    
    Returns:
        Manifest dict with all required fields.
    
    Raises:
        ValueError: If required settings are missing or invalid.
    """
    # Extract settings
    dataset_name = settings.get("dataset", {}).get("name")
    task = settings.get("dataset", {}).get("task")
    n_per_bin = settings.get("dataset", {}).get("n_per_bin")
    n_bins = settings.get("binning", {}).get("n_bins")
    
    if not dataset_name:
        raise ValueError("settings.dataset.name is required")
    if not task:
        raise ValueError("settings.dataset.task is required")
    if n_per_bin is None or n_per_bin < 1:
        raise ValueError("settings.dataset.n_per_bin must be >= 1")
    if n_bins is None or n_bins < 1:
        raise ValueError("settings.binning.n_bins must be >= 1")
    
    # Compute dataset_id
    dataset_id = _compute_dataset_id(dataset_name, task)
    
    # Normalize examples
    normalized_examples = normalize.normalize_longbench_examples(raw_examples, task)
    
    # Get tokenizer name
    tokenizer_name = tokenizer.get_tokenizer_name(settings)
    
    # Compute token lengths for each example
    token_lens = [
        _compute_token_length(ex, tokenizer_name)
        for ex in normalized_examples
    ]
    
    # Bin examples
    bin_idxs, bin_edges = binning.bin_examples(token_lens, n_bins)
    
    # Select examples per bin
    selected_indices = _select_examples_per_bin(
        normalized_examples, bin_idxs, token_lens, n_bins, n_per_bin
    )
    
    # Build bin_edges structure
    bin_edges_list = [
        {
            "bin_idx": i,
            "token_min": min_len,
            "token_max": max_len
        }
        for i, (min_len, max_len) in enumerate(bin_edges)
    ]
    
    # Build entries (only for selected examples)
    entries = []
    examples_dict = {}
    
    for idx in selected_indices:
        ex = normalized_examples[idx]
        example_id = ex["example_id"]
        
        entries.append({
            "example_id": example_id,
            "bin_idx": bin_idxs[idx],
            "token_len": token_lens[idx]
        })
        
        # Store example data (without example_id, as it's the key)
        examples_dict[example_id] = {
            "question": ex["question"],
            "context": ex["context"],
            "answers": ex["answers"],
            "meta": ex["meta"]
        }
    
    # Build manifest
    manifest = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task": task,  # Preserve exact task string
        "tokenizer_name": tokenizer_name,
        "n_bins": n_bins,
        "n_per_bin": n_per_bin,
        "bin_edges": bin_edges_list,
        "entries": entries,
        "examples": examples_dict  # Dict keyed by example_id
    }
    
    return manifest


def write_manifest(manifest: Dict[str, Any], dataset_id: str) -> Path:
    """Write manifest to JSON file.
    
    Args:
        manifest: Manifest dict from build_manifest.
        dataset_id: Dataset identifier.
    
    Returns:
        Path to written manifest.json file.
    """
    # Get manifest path
    manifest_file = manifest_path(dataset_id)
    
    # Create directory if needed
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest_file
