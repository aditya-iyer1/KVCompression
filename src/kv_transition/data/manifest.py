"""Manifest builder for Phase B dataset preparation.

Creates the portable manifest.json artifact that defines the evaluation set.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..paths import processed_dir, manifest_path

from . import binning
from . import normalize
from . import tokenizer

try:  # Optional dependency for chat-template-based token counting
    from transformers import AutoTokenizer  # type: ignore[import]
except Exception:  # pragma: no cover - absence is an expected fallback
    AutoTokenizer = None  # type: ignore[assignment]


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


def _seeded_order_key(seed: int, example_id: str) -> str:
    """Deterministic sort key for (seed, example_id); same seed + id → same key."""
    return hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()


def _select_examples_per_bin(
    examples: List[Dict[str, Any]],
    bin_idxs: List[int],
    token_lens: List[int],
    n_bins: int,
    n_per_bin: int,
    seed: Optional[int] = None,
) -> List[int]:
    """Select up to n_per_bin examples per bin deterministically.
    
    When seed is set, within each bin examples are sorted by seeded key (seed + example_id)
    then example_id, so selection varies with seed. When seed is None, sort by token length
    then example_id (legacy behavior).
    
    Args:
        examples: List of normalized examples.
        bin_idxs: Bin index for each example (aligned with examples).
        token_lens: Token length for each example (aligned with examples).
        n_bins: Number of bins.
        n_per_bin: Maximum examples to select per bin.
        seed: Optional dataset seed for deterministic per-bin ordering; None = legacy (shortest first).
    
    Returns:
        List of selected example indices (into examples list).
    """
    # Group examples by bin
    bin_groups: Dict[int, List[int]] = {i: [] for i in range(n_bins)}
    
    for idx, bin_idx in enumerate(bin_idxs):
        bin_groups[bin_idx].append(idx)
    
    selected_indices = []
    for bin_idx in range(n_bins):
        bin_example_indices = bin_groups[bin_idx]
        if not bin_example_indices:
            continue
        if seed is not None:
            bin_example_indices.sort(
                key=lambda idx: (_seeded_order_key(seed, examples[idx]["example_id"]), examples[idx]["example_id"])
            )
        else:
            bin_example_indices.sort(
                key=lambda idx: (token_lens[idx], examples[idx]["example_id"])
            )
        selected_indices.extend(bin_example_indices[:n_per_bin])
    
    return selected_indices


def _select_examples_per_bin_adaptive(
    examples: List[Dict[str, Any]],
    bin_idxs: List[int],
    token_lens: List[int],
    n_bins: int,
    n_per_bin_by_bin: List[int],
    seed: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Select examples per bin with per-bin caps (deterministic).
    
    When seed is set, within each bin sort by seeded key then example_id; when None,
    sort by token_len then example_id. Take first n_per_bin_by_bin[bin_idx] per bin.
    
    Returns:
        (selected_indices, per_bin_n) where per_bin_n[i] is count selected from bin i.
    """
    if len(n_per_bin_by_bin) != n_bins:
        raise ValueError(f"n_per_bin_by_bin length must be n_bins ({n_bins}), got {len(n_per_bin_by_bin)}")
    
    bin_groups: Dict[int, List[int]] = {i: [] for i in range(n_bins)}
    for idx, bin_idx in enumerate(bin_idxs):
        bin_groups[bin_idx].append(idx)
    
    selected_indices = []
    per_bin_n = [0] * n_bins
    
    for bin_idx in range(n_bins):
        cap = n_per_bin_by_bin[bin_idx]
        bin_example_indices = bin_groups[bin_idx]
        if not bin_example_indices:
            continue
        if seed is not None:
            bin_example_indices.sort(
                key=lambda idx: (_seeded_order_key(seed, examples[idx]["example_id"]), examples[idx]["example_id"])
            )
        else:
            bin_example_indices.sort(
                key=lambda idx: (token_lens[idx], examples[idx]["example_id"])
            )
        take = bin_example_indices[:cap]
        selected_indices.extend(take)
        per_bin_n[bin_idx] = len(take)
    
    return selected_indices, per_bin_n


def _select_examples_sanity_slices(
    examples: List[Dict[str, Any]],
    token_lens: List[int],
    sanity_slices: List[Dict[str, Any]],
) -> List[int]:
    """Select examples by sanity slice specs (deterministic).
    
    Each slice defines a window (by percentile or token bounds) and n examples
    to take. Candidates are sorted by (token_len ASC, example_id ASC). Selection
    is global over the entire pool.
    
    Args:
        examples: List of normalized examples.
        token_lens: Token length per example (aligned with examples).
        sanity_slices: List of slice specs; each has "n", and either
            ("min_pct", "max_pct") or ("min_tokens", "max_tokens"); optional "name".
    
    Returns:
        List of selected example indices.
    
    Raises:
        ValueError: If a slice cannot yield enough examples.
    """
    n_examples = len(examples)
    if n_examples == 0:
        return []
    
    # Sort all by (token_len, example_id) for deterministic selection
    sorted_indices = sorted(
        range(n_examples),
        key=lambda idx: (token_lens[idx], examples[idx]["example_id"])
    )
    sorted_tokens = sorted(token_lens)
    
    def percentile_token(pct: float) -> int:
        """Token value at given percentile (0..100)."""
        if pct <= 0:
            return sorted_tokens[0]
        if pct >= 100:
            return sorted_tokens[-1]
        idx = min(n_examples - 1, int((n_examples - 1) * pct / 100))
        return sorted_tokens[idx]
    
    selected: List[int] = []
    selected_set = set()
    slice_counts: List[int] = []
    
    for i, spec in enumerate(sanity_slices):
        n = spec.get("n")
        if n is None or not isinstance(n, int) or n < 1:
            raise ValueError(f"sanity_slices[{i}]: 'n' must be a positive integer")
        
        has_pct = "min_pct" in spec or "max_pct" in spec
        has_tokens = "min_tokens" in spec or "max_tokens" in spec
        if has_pct and has_tokens:
            raise ValueError(f"sanity_slices[{i}]: use either (min_pct, max_pct) or (min_tokens, max_tokens), not both")
        if not has_pct and not has_tokens:
            raise ValueError(f"sanity_slices[{i}]: specify either (min_pct, max_pct) or (min_tokens, max_tokens)")
        
        if has_pct:
            min_pct = spec.get("min_pct", 0)
            max_pct = spec.get("max_pct", 100)
            min_tok = percentile_token(float(min_pct))
            max_tok = percentile_token(float(max_pct))
        else:
            min_tok = spec.get("min_tokens", 0)
            max_tok = spec.get("max_tokens", sorted_tokens[-1] if sorted_tokens else 0)
        
        name = spec.get("name", f"slice_{i}")
        count = 0
        for idx in sorted_indices:
            if idx in selected_set:
                continue
            tl = token_lens[idx]
            if min_tok <= tl <= max_tok:
                selected.append(idx)
                selected_set.add(idx)
                count += 1
                if count >= n:
                    break
        if count < n:
            raise ValueError(
                f"sanity_slices[{i}] ({name}): requested n={n}, only {count} examples in token range [{min_tok}, {max_tok}]"
            )
        slice_counts.append(count)
    
    total = len(selected)
    # Debug: concise print when sanity_slices is used
    parts = [f"{spec.get('name', f'slice_{i}')}={c}" for i, (spec, c) in enumerate(zip(sanity_slices, slice_counts))]
    print(f"  sanity_slices: {' '.join(parts)} total={total}")
    
    return selected


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
    sanity_slices = settings.get("dataset", {}).get("sanity_slices")
    pinned_entry_indices = settings.get("dataset", {}).get("pinned_entry_indices")
    dataset_seed_raw = settings.get("dataset", {}).get("seed")
    dataset_seed: Optional[int] = None
    if dataset_seed_raw is not None:
        try:
            dataset_seed = int(dataset_seed_raw)
        except (TypeError, ValueError):
            dataset_seed = 0
    
    if not dataset_name:
        raise ValueError("settings.dataset.name is required")
    if not task:
        raise ValueError("settings.dataset.task is required")
    if n_bins is None or n_bins < 1:
        raise ValueError("settings.binning.n_bins must be >= 1")
    use_pinned = isinstance(pinned_entry_indices, list) and len(pinned_entry_indices) > 0
    use_sanity_slices = isinstance(sanity_slices, list) and len(sanity_slices) > 0
    sampling_strategy = (settings.get("sampling") or {}).get("strategy", "uniform")
    if sampling_strategy is None:
        sampling_strategy = "uniform"
    if not use_pinned and not use_sanity_slices and sampling_strategy == "uniform" and (n_per_bin is None or n_per_bin < 1):
        raise ValueError("settings.dataset.n_per_bin must be >= 1 when sanity_slices and pinned_entry_indices are not set and sampling.strategy is uniform")
    
    # Compute dataset_id
    dataset_id = _compute_dataset_id(dataset_name, task)
    
    # Normalize examples
    normalized_examples = normalize.normalize_longbench_examples(raw_examples, task)
    
    # Get tokenizer name (fallback path)
    tokenizer_name = tokenizer.get_tokenizer_name(settings)
    
    # Optional chat-template-based token counting for chat models (vLLM/OpenAI-compatible)
    chat_tokenizer = None
    chat_tokenizer_model: Optional[str] = None
    chat_warned = False
    # Priority: env HF_TOKENIZER_MODEL, settings.run.hf_tokenizer_model, settings.run.model_name (if repo id)
    env_model = os.getenv("HF_TOKENIZER_MODEL")
    run_cfg = settings.get("run", {}) or {}
    if isinstance(env_model, str) and env_model.strip():
        chat_tokenizer_model = env_model.strip()
    else:
        cfg_model = run_cfg.get("hf_tokenizer_model")
        if isinstance(cfg_model, str) and cfg_model.strip():
            chat_tokenizer_model = cfg_model.strip()
        else:
            run_model_name = run_cfg.get("model_name")
            if isinstance(run_model_name, str) and "/" in run_model_name:
                chat_tokenizer_model = run_model_name.strip()
    if chat_tokenizer_model and AutoTokenizer is not None:
        try:
            chat_tokenizer = AutoTokenizer.from_pretrained(
                chat_tokenizer_model,
                trust_remote_code=True,
            )
            if not hasattr(chat_tokenizer, "apply_chat_template"):
                raise AttributeError("tokenizer has no apply_chat_template")
        except Exception as e:  # pragma: no cover - best effort, falls back
            print(
                f"  WARN: chat-template token counting disabled "
                f"for model={chat_tokenizer_model!r}; falling back to base tokenizer ({e})"
            )
            chat_tokenizer = None
            chat_warned = True
    
    def _chat_token_len(example: Dict[str, Any]) -> Optional[int]:
        nonlocal chat_tokenizer, chat_warned
        if chat_tokenizer is None:
            return None
        try:
            context = example.get("context", "")
            question = example.get("question", "")
            messages = [
                {
                    "role": "system",
                    "content": "Answer the question using the provided context.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                },
            ]
            # Prefer tokenize=True if supported (returns list[int] or similar)
            try:
                tokens = chat_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                )
                if isinstance(tokens, list):
                    return len(tokens)
                if isinstance(tokens, dict) and "input_ids" in tokens:
                    ids = tokens["input_ids"]
                    # ids can be list[int] or list[list[int]]
                    if isinstance(ids, list) and ids and isinstance(ids[0], list):
                        return len(ids[0])
                    if isinstance(ids, list):
                        return len(ids)
            except TypeError:
                # Older signature without tokenize kwarg – fall through to string path
                pass
            # Fallback: get rendered prompt string, then tokenize
            rendered = chat_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            enc = chat_tokenizer(
                rendered,
                add_special_tokens=False,
            )
            ids = enc.get("input_ids", [])
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                return len(ids[0])
            if isinstance(ids, list):
                return len(ids)
            return None
        except Exception as e:  # pragma: no cover - defensive; fall back once
            if not chat_warned:
                print(
                    f"  WARN: chat-template token counting failed; "
                    f"falling back to base tokenizer ({e})"
                )
                chat_warned = True
            chat_tokenizer = None
            return None
    
    # Compute token lengths for each example (chat-template path if available)
    token_lens: List[int] = []
    for ex in normalized_examples:
        chat_len = _chat_token_len(ex) if chat_tokenizer is not None else None
        if isinstance(chat_len, int) and chat_len >= 0:
            token_lens.append(chat_len)
        else:
            token_lens.append(_compute_token_length(ex, tokenizer_name))
    
    # Bin examples (used for bin_edges and bin_idx assignment)
    bin_idxs, bin_edges = binning.bin_examples(token_lens, n_bins)
    
    # Optional prompt-length cap (dataset.max_prompt_tokens)
    max_prompt_tokens = settings.get("dataset", {}).get("max_prompt_tokens")
    if max_prompt_tokens is not None:
        max_prompt_tokens_raw = max_prompt_tokens
        try:
            max_prompt_tokens = int(max_prompt_tokens_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"settings.dataset.max_prompt_tokens must be a positive integer, "
                f"got {max_prompt_tokens_raw!r}"
            )
        if max_prompt_tokens <= 0:
            raise ValueError(
                f"settings.dataset.max_prompt_tokens must be a positive integer, "
                f"got {max_prompt_tokens_raw!r}"
            )
        filtered_examples: List[Dict[str, Any]] = []
        filtered_token_lens: List[int] = []
        filtered_bin_idxs: List[int] = []
        removed = 0
        for ex, tl, b in zip(normalized_examples, token_lens, bin_idxs):
            if tl <= max_prompt_tokens:
                filtered_examples.append(ex)
                filtered_token_lens.append(tl)
                filtered_bin_idxs.append(b)
            else:
                removed += 1
        normalized_examples = filtered_examples
        token_lens = filtered_token_lens
        bin_idxs = filtered_bin_idxs
        n_candidates = len(normalized_examples)
        print(f"  max_prompt_tokens: {max_prompt_tokens} (filtered {removed} examples)")
        if n_candidates == 0:
            raise ValueError(
                f"settings.dataset.max_prompt_tokens={max_prompt_tokens} filtered out all "
                f"examples; no candidates remain for dataset '{dataset_id}'"
            )
    else:
        n_candidates = len(normalized_examples)
    
    # Select examples: pinned indices, sanity slices, or per-bin selection
    if use_pinned:
        # Validate: all integers, unique, and in range for this dataset
        seen = set()
        resolved = []
        for i, raw in enumerate(pinned_entry_indices):
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"pinned_entry_indices[{i}] must be an integer, got {type(raw).__name__}: {raw}"
                )
            if idx in seen:
                raise ValueError(f"pinned_entry_indices: duplicate index {idx}")
            seen.add(idx)
            if idx < 0 or idx >= n_candidates:
                raise ValueError(
                    f"pinned_entry_indices: entry_idx {idx} out of range [0, {n_candidates}) for dataset (total examples={n_candidates})"
                )
            resolved.append(idx)
        # Use stable sorted order so manifest order is deterministic
        selected_indices = sorted(resolved)
        n_per_bin = len(selected_indices)
        per_bin_n = None
        print(f"  pinned_entry_indices: {len(selected_indices)} entries")
    elif use_sanity_slices:
        selected_indices = _select_examples_sanity_slices(
            normalized_examples, token_lens, sanity_slices
        )
        n_per_bin = len(selected_indices)  # for manifest metadata
        per_bin_n = None
    else:
        # Per-bin selection: uniform or focus_transition
        sampling = settings.get("sampling", {})
        strategy = sampling.get("strategy", "uniform")
        if strategy is None:
            strategy = "uniform"
        if strategy == "uniform":
            selected_indices = _select_examples_per_bin(
                normalized_examples, bin_idxs, token_lens, n_bins, n_per_bin, seed=dataset_seed
            )
            per_bin_n = None  # no per-bin breakdown for uniform
        elif strategy == "focus_transition":
            focus = sampling.get("focus")
            if not focus or not isinstance(focus, dict):
                raise ValueError(
                    "sampling.strategy is focus_transition but sampling.focus is missing or not a dict"
                )
            focus_bins = focus.get("focus_bins")
            if not isinstance(focus_bins, list) or len(focus_bins) == 0:
                raise ValueError("sampling.focus.focus_bins must be a non-empty list")
            base_n_per_bin = focus.get("base_n_per_bin")
            if base_n_per_bin is None or not isinstance(base_n_per_bin, int) or base_n_per_bin < 1:
                raise ValueError("sampling.focus.base_n_per_bin must be an integer >= 1")
            focus_n_per_bin = focus.get("focus_n_per_bin")
            if focus_n_per_bin is None or not isinstance(focus_n_per_bin, int) or focus_n_per_bin < 1:
                raise ValueError("sampling.focus.focus_n_per_bin must be an integer >= 1")
            focus_radius_bins = focus.get("focus_radius_bins", 0)
            if focus_radius_bins is None:
                focus_radius_bins = 0
            if not isinstance(focus_radius_bins, int) or focus_radius_bins < 0:
                raise ValueError("sampling.focus.focus_radius_bins must be a non-negative integer")
            for i, b in enumerate(focus_bins):
                if not isinstance(b, int):
                    raise ValueError(
                        f"sampling.focus.focus_bins[{i}] must be an integer, got {type(b).__name__}"
                    )
            window_bins = set()
            for b in focus_bins:
                for r in range(b - focus_radius_bins, b + focus_radius_bins + 1):
                    if 0 <= r < n_bins:
                        window_bins.add(r)
            window_bins_list = sorted(window_bins)
            n_per_bin_by_bin = [
                focus_n_per_bin if i in window_bins else base_n_per_bin
                for i in range(n_bins)
            ]
            selected_indices, per_bin_n = _select_examples_per_bin_adaptive(
                normalized_examples, bin_idxs, token_lens, n_bins, n_per_bin_by_bin, seed=dataset_seed
            )
            n_per_bin = max(per_bin_n) if per_bin_n else 0  # for manifest compatibility
            print(
                f"  sampling_focus: base={base_n_per_bin} focus={focus_n_per_bin} "
                f"window_bins={window_bins_list} total={len(selected_indices)}"
            )
        else:
            raise ValueError(
                f"Unsupported sampling.strategy: {strategy!r}. Use 'uniform' or 'focus_transition'."
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
    n_examples = len(entries)
    manifest = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "task": task,  # Preserve exact task string
        "tokenizer_name": tokenizer_name,
        "n_bins": n_bins,
        "n_per_bin": n_per_bin,
        "n_examples": n_examples,
        "bin_edges": bin_edges_list,
        "entries": entries,
        "examples": examples_dict  # Dict keyed by example_id
    }
    if per_bin_n is not None:
        manifest["per_bin_n"] = per_bin_n

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
