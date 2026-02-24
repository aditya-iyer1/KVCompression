"""Unit tests for failure taxonomy classification."""

import pytest

from kv_transition.eval.failure_taxonomy import (
    CONTEXT_LENGTH_EXCEEDED,
    classify_failure,
)


def test_context_length_exceeded_classified():
    """Context length exceeded error messages classify as CONTEXT_LENGTH_EXCEEDED."""
    error_message = (
        "HTTP 400: maximum context length exceeded. "
        "Please reduce the length of the input messages. "
        "Details: (parameter=input_tokens)"
    )
    # Non-empty text so we don't hit EMPTY_OUTPUT; error_message drives classification
    result = classify_failure(
        text="x",
        finish_reason=None,
        error_message=error_message,
    )
    assert result == CONTEXT_LENGTH_EXCEEDED
