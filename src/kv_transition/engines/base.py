"""Base engine interface for inference providers.

Defines the abstract interface and result container used by the runner.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EngineResult:
    """Container for engine generation results.
    
    Attributes:
        text: Generated text output.
        raw: Raw provider response payload (if available).
        usage: Token usage information (if available), e.g., {"prompt_tokens": 10, "completion_tokens": 20}.
        finish_reason: Reason for completion (e.g., "stop", "length", "error").
        timings: Timing information (if available), e.g., {"latency_s": 0.5, "ttfb_s": 0.1}.
    """
    text: str
    raw: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    timings: Optional[Dict[str, Any]] = None


class BaseEngine(ABC):
    """Abstract base class for inference engines.
    
    All engine implementations must provide a generate method that takes
    OpenAI-style chat messages and returns an EngineResult.
    """
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: Optional[float] = None,
        **kwargs: Any
    ) -> EngineResult:
        """Generate text from messages.
        
        Args:
            messages: OpenAI-style chat messages, e.g.,
                [{"role": "user", "content": "Hello"}]
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout_s: Optional timeout in seconds.
            **kwargs: Additional provider-specific parameters.
        
        Returns:
            EngineResult with generated text and metadata.
        
        Raises:
            TimeoutError: If request times out.
            RuntimeError: For other engine/provider errors.
        """
        pass
