"""Failure taxonomy and classification for inference responses.

Provides stable labels for common failure modes in LLM responses.
"""

from typing import Optional


# Failure labels (stable string constants)
EMPTY_OUTPUT = "EMPTY_OUTPUT"
TRUNCATED = "TRUNCATED"
FORMAT_ERROR = "FORMAT_ERROR"
REFUSAL = "REFUSAL"
TIMEOUT = "TIMEOUT"
CONTEXT_LENGTH_EXCEEDED = "CONTEXT_LENGTH_EXCEEDED"
RATE_LIMITED = "RATE_LIMITED"
ENGINE_ERROR = "ENGINE_ERROR"


def classify_failure(
    text: Optional[str],
    finish_reason: Optional[str],
    error_message: Optional[str] = None
) -> Optional[str]:
    """Classify failure type from response data.
    
    Returns a failure label if the response indicates a failure,
    otherwise returns None.
    
    Args:
        text: Response text (may be None or empty).
        finish_reason: Finish reason from engine (e.g., "stop", "length").
        error_message: Optional error message from exception or engine.
    
    Returns:
        Failure label string or None if not a failure.
    """
    # Check error_message first for rate limit (so 429 is classified even with non-empty text)
    if error_message:
        error_lower = error_message.lower()
        rate_limit_patterns = [
            "status 429",
            "http 429",
            "rate limit",
            "rate limit reached",
            "tokens per min",
            " tpm",
            "tpm ",
        ]
        if any(p in error_lower for p in rate_limit_patterns):
            return RATE_LIMITED
        if "please try again" in error_lower and (
            "rate" in error_lower or "limit" in error_lower or "429" in error_lower
        ):
            return RATE_LIMITED

    # Check for empty output
    if text is None or (isinstance(text, str) and not text.strip()):
        return EMPTY_OUTPUT
    
    # Check for truncation (finish_reason takes precedence)
    if finish_reason == "length":
        return TRUNCATED
    
    # Check for timeout in error message (before truncation in error message)
    if error_message:
        error_lower = error_message.lower()
        if "timeout" in error_lower or "timed out" in error_lower:
            return TIMEOUT
    
    # Check for context length exceeded (before generic truncation/engine errors)
    if error_message:
        error_lower = error_message.lower()
        context_length_patterns = [
            "maximum context length",
            "context length is",
            "context length exceeded",
            "please reduce the length of the input messages",
            "input messages are too long",
            "(parameter=input_tokens",
            "max context length",
        ]
        for pattern in context_length_patterns:
            if pattern in error_lower:
                return CONTEXT_LENGTH_EXCEEDED
    
    # Check error message for truncation indicators
    if error_message:
        error_lower = error_message.lower()
        if "truncat" in error_lower or "max tokens" in error_lower:
            return TRUNCATED
    
    # Check for refusal patterns in text
    if text:
        text_lower = text.lower()
        # Minimal refusal patterns
        refusal_patterns = [
            "i can't help",
            "i cannot help",
            "i can't assist",
            "i cannot assist",
            "i can't comply",
            "i cannot comply",
            "i'm not able to",
            "i am not able to",
            "i'm unable to",
            "i am unable to",
            "i cannot provide",
            "i can't provide",
            "i'm sorry, but i cannot",
            "i'm sorry, but i can't",
        ]
        for pattern in refusal_patterns:
            if pattern in text_lower:
                return REFUSAL
    
    # Check for format error (obvious cases)
    if text and text.strip().startswith("ERROR:"):
        return FORMAT_ERROR
    
    # Engine error (fallback for any error message that doesn't match above)
    if error_message:
        return ENGINE_ERROR
    
    # No failure detected
    return None
