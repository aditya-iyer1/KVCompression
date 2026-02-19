"""Normalize raw LongBench examples into canonical schema.

Phase B: Converts raw dataset rows into stable, JSON-serializable example dicts.
"""

import hashlib
import json
from typing import Any, Dict, List


def _extract_field(raw: Dict[str, Any], candidates: List[str], field_name: str, required: bool = True) -> str:
    """Extract a field from raw dict using candidate keys.
    
    Args:
        raw: Raw example dict.
        candidates: List of candidate keys to try (in order).
        field_name: Human-readable name for error messages.
        required: If True, raise error if field not found.
    
    Returns:
        Extracted field value as string.
    
    Raises:
        ValueError: If required field is missing.
    """
    for key in candidates:
        if key in raw:
            value = raw[key]
            if value is not None:
                return str(value).strip()
    
    if required:
        available_keys = ", ".join(sorted(raw.keys()))
        raise ValueError(
            f"Missing required field '{field_name}'. "
            f"Tried keys: {', '.join(candidates)}. "
            f"Available keys: {available_keys}"
        )
    
    return ""


def _normalize_answers(raw: Dict[str, Any]) -> List[str]:
    """Extract and normalize answers field to list[str].
    
    Args:
        raw: Raw example dict.
    
    Returns:
        List of non-empty answer strings.
    
    Raises:
        ValueError: If no valid answers found.
    """
    # Try candidate keys
    candidates = ["answers", "answer", "target", "output", "ground_truth"]
    answer_value = None
    
    for key in candidates:
        if key in raw:
            answer_value = raw[key]
            break
    
    if answer_value is None:
        available_keys = ", ".join(sorted(raw.keys()))
        raise ValueError(
            f"Missing required field 'answers'. "
            f"Tried keys: {', '.join(candidates)}. "
            f"Available keys: {available_keys}"
        )
    
    # Convert to list[str]
    if isinstance(answer_value, str):
        # Single string -> wrap in list
        answers = [answer_value.strip()]
    elif isinstance(answer_value, (list, tuple)):
        # List-like -> convert each item to string
        answers = [str(item).strip() for item in answer_value if item is not None]
    else:
        # Other types -> convert to string and wrap
        answers = [str(answer_value).strip()]
    
    # Filter out empty strings
    answers = [a for a in answers if a]
    
    if not answers:
        raise ValueError(
            f"Answers field is empty after normalization. "
            f"Original value: {answer_value!r}"
        )
    
    return answers


def _compute_example_id(raw: Dict[str, Any], task: str, question: str, context: str, answers: List[str]) -> str:
    """Compute deterministic example_id.
    
    Prefers existing ID fields, otherwise derives from content hash.
    
    Args:
        raw: Raw example dict.
        task: Task name.
        question: Normalized question.
        context: Normalized context.
        answers: Normalized answers list.
    
    Returns:
        Stable example_id string.
    """
    # Prefer existing ID fields
    for id_key in ["id", "example_id", "_id", "idx", "index"]:
        if id_key in raw:
            value = raw[id_key]
            if value is not None:
                return str(value).strip()
    
    # Derive from content hash
    # Create stable string representation
    content_str = json.dumps(
        {
            "task": task,
            "question": question,
            "context": context,
            "answers": sorted(answers)  # Sort for stability
        },
        sort_keys=True,
        ensure_ascii=False
    )
    
    # Compute hash
    hash_obj = hashlib.sha256(content_str.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    # Use first 16 chars for readability (collision risk is acceptable for this use case)
    return f"{task}_{hash_hex[:16]}"


def normalize_one(raw: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Normalize a single raw example into canonical schema.
    
    Args:
        raw: Raw example dict from dataset.
        task: Task name (preserved in meta).
    
    Returns:
        Canonical example dict with fields:
        - example_id (str)
        - question (str)
        - context (str)
        - answers (list[str])
        - meta (dict)
    
    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict, got {type(raw).__name__}")
    
    # Extract question
    question = _extract_field(
        raw,
        candidates=["question", "query", "input", "prompt"],
        field_name="question",
        required=True
    )
    
    # Extract context (may be empty for some tasks)
    context = _extract_field(
        raw,
        candidates=["context", "passage", "article", "document", "text"],
        field_name="context",
        required=False  # Some tasks may not have context
    )
    
    # Extract and normalize answers
    answers = _normalize_answers(raw)
    
    # Compute example_id
    example_id = _compute_example_id(raw, task, question, context, answers)
    
    # Build meta dict (preserve task and any other useful metadata)
    meta = {
        "task": task,
        "source_keys": list(raw.keys())  # Helpful for debugging
    }
    
    # Preserve any existing meta fields if present
    if "meta" in raw and isinstance(raw["meta"], dict):
        meta.update(raw["meta"])
    
    # Build canonical example
    normalized = {
        "example_id": example_id,
        "question": question,
        "context": context,
        "answers": answers,
        "meta": meta
    }
    
    return normalized


def normalize_longbench_examples(raw_examples: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    """Normalize a list of raw LongBench examples.
    
    Args:
        raw_examples: List of raw example dicts from dataset.
        task: Task name (preserved in meta).
    
    Returns:
        List of normalized example dicts (same length as input).
        Each dict conforms to canonical schema.
    
    Raises:
        ValueError: If any example fails normalization.
    """
    if not isinstance(raw_examples, list):
        raise ValueError(f"Expected list, got {type(raw_examples).__name__}")
    
    normalized = []
    errors = []
    
    for i, raw in enumerate(raw_examples):
        try:
            norm_example = normalize_one(raw, task)
            normalized.append(norm_example)
        except ValueError as e:
            errors.append(f"Example {i}: {e}")
    
    if errors:
        error_msg = f"Failed to normalize {len(errors)} of {len(raw_examples)} examples:\n"
        error_msg += "\n".join(f"  - {err}" for err in errors[:10])  # Show first 10
        if len(errors) > 10:
            error_msg += f"\n  ... and {len(errors) - 10} more"
        raise ValueError(error_msg)
    
    # Verify output length matches input
    if len(normalized) != len(raw_examples):
        raise RuntimeError(
            f"Normalization length mismatch: input {len(raw_examples)}, output {len(normalized)}"
        )
    
    return normalized
