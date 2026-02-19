"""Core execution loop for running inference on manifest entries.

Executes manifest entries for a single KV setting and persists results to DB.
"""

import json
import sqlite3
from typing import Any, Dict

from ..db import dao
from ..engines.base import BaseEngine


def _compute_dataset_id(dataset_name: str, task: str) -> str:
    """Compute dataset_id from dataset name and task.
    
    Matches Phase B convention: {dataset.name}__{dataset.task}
    
    Args:
        dataset_name: Dataset name.
        task: Task identifier.
    
    Returns:
        Dataset ID string.
    """
    return f"{dataset_name}__{task}"


def _build_messages(context: str, question: str) -> list[Dict[str, str]]:
    """Build OpenAI-style messages for inference.
    
    Args:
        context: Context text.
        question: Question text.
    
    Returns:
        List of message dicts in OpenAI format.
    """
    return [
        {
            "role": "system",
            "content": "Answer the question using the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]


def run_one_setting(
    conn: sqlite3.Connection,
    settings: Dict[str, Any],
    engine: BaseEngine,
    model_name: str,
    run_id: str,
    kv_budget: float
) -> None:
    """Run inference for a single KV budget setting.
    
    Reads manifest entries, executes inference, and persists results to DB.
    
    Args:
        conn: SQLite connection.
        settings: Settings dict (from load_settings).
        engine: Engine instance (from endpoints.build_engine).
        model_name: Model name string.
        run_id: Run identifier.
        kv_budget: KV budget value for this run.
    """
    # Identify dataset_id
    dataset_name = settings["dataset"]["name"]
    task = settings["dataset"]["task"]
    dataset_id = _compute_dataset_id(dataset_name, task)
    
    # Read manifest entries from DB
    manifest_entries = dao.get_manifest_entries(conn, dataset_id)
    
    if not manifest_entries:
        print(f"  Warning: No manifest entries found for dataset_id={dataset_id}")
        return
    
    print(f"  Processing {len(manifest_entries)} manifest entries")
    
    # Get example IDs
    example_ids = [entry["example_id"] for entry in manifest_entries]
    
    # Fetch examples from DB
    examples = dao.get_examples_by_ids(conn, example_ids)
    
    if len(examples) != len(example_ids):
        missing = set(example_ids) - set(examples.keys())
        print(f"  Warning: {len(missing)} examples not found in DB")
    
    # Extract decoding parameters
    decoding = settings.get("decoding", {})
    temperature = decoding.get("temperature", 0.0)
    max_tokens = decoding.get("max_tokens", 256)
    
    # Extract timeout
    timeout_s = None
    engine_settings = settings.get("engine", {})
    request_settings = engine_settings.get("request", {})
    if request_settings:
        timeout_s = request_settings.get("timeout_s")
    
    # Process each manifest entry
    for idx, entry in enumerate(manifest_entries):
        example_id = entry["example_id"]
        
        if example_id not in examples:
            print(f"  Skipping entry {idx}: example_id {example_id} not found")
            continue
        
        example = examples[example_id]
        
        # Build messages
        context = example["context"]
        question = example["question"]
        messages = _build_messages(context, question)
        
        # Call engine
        try:
            result = engine.generate(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s
            )
            
            # Extract result data
            text = result.text
            finish_reason = result.finish_reason
            raw_json = json.dumps(result.raw) if result.raw else None
            
            # Extract usage
            usage = result.usage or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            
            # Extract timings
            timings = result.timings or {}
            latency_s = timings.get("latency_s")
            
            # Persist to DB (placeholder - will be implemented in dao.py)
            # TODO: Call dao.create_request(conn, run_id, entry_id, messages, prompt_tokens, max_tokens, temperature)
            # TODO: Call dao.create_response(conn, request_id, text, finish_reason, completion_tokens, raw_json)
            # TODO: Call dao.create_telemetry(conn, request_id, latency_s, ...)
            # These will be implemented when dao.py is extended with Phase C functions
            
            # For now, just log progress
            if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest_entries):
                print(f"    Processed {idx + 1}/{len(manifest_entries)} entries")
        
        except Exception as e:
            # Error handling: record failure and continue
            print(f"    Error processing entry {idx} (example_id={example_id}): {e}")
            # TODO: Call dao.create_failure(conn, request_id, failure_type, message)
            # This will be implemented when dao.py is extended
            continue
    
    print(f"  Completed processing for run_id={run_id}")
