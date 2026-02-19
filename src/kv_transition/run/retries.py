"""Retry utility for transient HTTP/runtime failures.

Provides retry logic with exponential backoff for engine calls.
"""

import random
import time
from typing import Any, Callable


def is_retryable_error(exc: Exception) -> bool:
    """Determine if an exception is retryable.
    
    Treats common transient issues as retryable:
    - Timeouts
    - Connection errors
    - HTTP 429 (rate limit) and 5xx (server errors)
    
    Args:
        exc: Exception to check.
    
    Returns:
        True if the error is retryable, False otherwise.
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()
    
    # Timeout errors
    if "timeout" in exc_type.lower() or "timeout" in exc_msg or "timed out" in exc_msg:
        return True
    
    # Connection errors
    if "connection" in exc_type.lower() or "connection" in exc_msg:
        return True
    
    # Network errors
    if "network" in exc_msg or "urlerror" in exc_type.lower():
        return True
    
    # HTTP errors - check for 429 (rate limit) or 5xx (server errors)
    if "429" in exc_msg or "rate limit" in exc_msg:
        return True
    
    if "5" in exc_msg and any(code in exc_msg for code in ["500", "502", "503", "504"]):
        return True
    
    # Check for HTTP status codes in error messages
    if "http" in exc_msg:
        # Look for status codes
        if "429" in exc_msg:
            return True
        # 5xx server errors
        if any(f"5{xx}" in exc_msg for xx in ["00", "02", "03", "04"]):
            return True
    
    # Runtime errors that might be transient (e.g., from urllib)
    if "temporarily unavailable" in exc_msg or "service unavailable" in exc_msg:
        return True
    
    return False


def retry_call(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0
) -> Any:
    """Call a function with retry logic for transient failures.
    
    Uses exponential backoff with jitter for retries.
    
    Args:
        fn: Function to call (no arguments).
        max_attempts: Maximum number of attempts (default: 3).
        base_delay_s: Base delay in seconds for exponential backoff (default: 0.5).
        max_delay_s: Maximum delay in seconds (default: 8.0).
    
    Returns:
        Return value of fn().
    
    Raises:
        Exception: The last exception raised by fn() if all attempts fail.
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            
            # Don't retry if not retryable or if this was the last attempt
            if not is_retryable_error(e) or attempt == max_attempts - 1:
                raise
            
            # Calculate delay with exponential backoff
            delay = min(base_delay_s * (2 ** attempt), max_delay_s)
            
            # Add small jitter (0-20% of delay)
            jitter = random.uniform(0, delay * 0.2)
            total_delay = delay + jitter
            
            # Sleep before retry
            time.sleep(total_delay)
    
    # Should not reach here, but re-raise last exception if we do
    if last_exception:
        raise last_exception
    
    raise RuntimeError("retry_call: unexpected state")
