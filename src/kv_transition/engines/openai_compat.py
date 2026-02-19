"""OpenAI-compatible HTTP engine implementation.

Works with OpenAI API, vLLM, SGLang, and other OpenAI-compatible servers.
"""

import json
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import BaseEngine, EngineResult


def _normalize_base_url(base_url: str) -> str:
    """Normalize base_url to ensure it ends with /v1 if needed.
    
    Args:
        base_url: Base URL (may or may not end with /v1).
    
    Returns:
        Normalized base URL ending with /v1.
    """
    base_url = base_url.rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"
    return base_url


def _make_request(
    url: str,
    data: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: Optional[float] = None
) -> Dict[str, Any]:
    """Make HTTP POST request and return JSON response.
    
    Args:
        url: Request URL.
        data: JSON body data.
        headers: Request headers.
        timeout_s: Optional timeout in seconds.
    
    Returns:
        Parsed JSON response.
    
    Raises:
        HTTPError: If HTTP status is not 2xx.
        URLError: For network/connection errors.
        RuntimeError: For other errors.
    """
    json_data = json.dumps(data).encode('utf-8')
    request = Request(url, data=json_data, headers=headers, method='POST')
    
    try:
        timeout = timeout_s if timeout_s is not None else 120.0
        with urlopen(request, timeout=timeout) as response:
            status_code = response.getcode()
            
            if not (200 <= status_code < 300):
                # Read error response
                error_body = response.read().decode('utf-8', errors='replace')
                error_snippet = error_body[:200] if len(error_body) > 200 else error_body
                raise HTTPError(
                    url, status_code,
                    f"HTTP {status_code}: {error_snippet}",
                    response.headers,
                    None
                )
            
            response_data = json.loads(response.read().decode('utf-8'))
            return response_data
    
    except HTTPError as e:
        # Re-raise HTTP errors with context
        error_body = ""
        try:
            if hasattr(e, 'read'):
                error_body = e.read().decode('utf-8', errors='replace')[:200]
        except Exception:
            pass
        
        raise RuntimeError(
            f"HTTP request failed with status {e.code}: {error_body or str(e)}"
        ) from e
    
    except URLError as e:
        raise RuntimeError(f"Network error: {e}") from e
    
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response: {e}") from e


class OpenAICompatEngine(BaseEngine):
    """OpenAI-compatible HTTP engine.
    
    Works with OpenAI API, vLLM, SGLang, and other OpenAI-compatible servers.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None
    ):
        """Initialize OpenAI-compatible engine.
        
        Args:
            base_url: Base URL for the API (e.g., "https://api.openai.com" or "http://localhost:8000").
            api_key: Optional API key for authentication.
            default_headers: Optional default headers to include in all requests.
        """
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key
        self.default_headers = default_headers or {}
    
    def generate(
        self,
        messages: list[Dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: Optional[float] = None,
        **kwargs: Any
    ) -> EngineResult:
        """Generate text using OpenAI-compatible API.
        
        Args:
            messages: OpenAI-style chat messages.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout_s: Optional request timeout in seconds.
            **kwargs: Additional parameters to pass to API.
        
        Returns:
            EngineResult with generated text and metadata.
        
        Raises:
            RuntimeError: For HTTP/network errors or invalid responses.
            TimeoutError: If request times out (wrapped in RuntimeError).
        """
        # Build request URL
        url = f"{self.base_url}/chat/completions"
        
        # Build request body
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Build headers
        headers = {
            "Content-Type": "application/json",
            **self.default_headers
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Measure latency
        start_time = time.time()
        
        try:
            # Make request
            response_data = _make_request(url, body, headers, timeout_s=timeout_s)
            
            # Calculate latency
            latency_s = time.time() - start_time
            
            # Parse response
            choices = response_data.get("choices", [])
            if not choices:
                text = ""
                finish_reason = None
            else:
                choice = choices[0]
                message = choice.get("message", {})
                text = message.get("content", "")
                finish_reason = choice.get("finish_reason")
            
            usage = response_data.get("usage")
            
            timings = {
                "latency_s": latency_s
            }
            
            return EngineResult(
                text=text,
                raw=response_data,
                usage=usage,
                finish_reason=finish_reason,
                timings=timings
            )
        
        except TimeoutError as e:
            raise RuntimeError(f"Request timed out after {timeout_s}s") from e
