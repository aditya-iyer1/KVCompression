"""Telemetry extraction and normalization for inference requests.

Extracts timing and usage information from engine results for DB persistence.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..engines.base import EngineResult


def now_iso() -> str:
    """Get current UTC time as ISO 8601 string.
    
    Returns:
        ISO 8601 formatted timestamp string (UTC).
    """
    return datetime.now(timezone.utc).isoformat()


def extract_telemetry(engine_result: EngineResult) -> Dict[str, Any]:
    """Extract telemetry data from engine result.
    
    Normalizes timing and usage information into a JSON-serializable dict
    suitable for DB persistence.
    
    Args:
        engine_result: EngineResult from engine.generate().
    
    Returns:
        Dict with telemetry fields:
        - latency_s (float | None)
        - ttfb_s (float | None)
        - prompt_tokens (int | None)
        - completion_tokens (int | None)
        - total_tokens (int | None)
        - finish_reason (str | None)
        - notes (dict | None)
    """
    telemetry: Dict[str, Any] = {}
    
    # Extract timings
    timings = engine_result.timings or {}
    telemetry["latency_s"] = timings.get("latency_s")
    telemetry["ttfb_s"] = timings.get("ttfb_s")  # May be None if not available
    
    # Extract usage
    usage = engine_result.usage or {}
    telemetry["prompt_tokens"] = usage.get("prompt_tokens")
    telemetry["completion_tokens"] = usage.get("completion_tokens")
    telemetry["total_tokens"] = usage.get("total_tokens")
    
    # Extract finish_reason
    telemetry["finish_reason"] = engine_result.finish_reason
    
    # Optional notes (can include additional metadata)
    notes = {}
    if engine_result.raw:
        # Include any additional fields from raw response that might be useful
        raw = engine_result.raw
        if "model" in raw:
            notes["model"] = raw["model"]
        if "id" in raw:
            notes["response_id"] = raw["id"]
    
    telemetry["notes"] = notes if notes else None
    
    return telemetry
