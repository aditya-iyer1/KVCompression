"""Endpoint resolution for building engines from settings.

Resolves environment variables and constructs OpenAI-compatible engines.
"""

import os
from typing import Dict, Optional, Tuple

from .openai_compat import OpenAICompatEngine


def resolve_api_key(settings: Dict) -> Optional[str]:
    """Resolve API key from environment variable.
    
    Args:
        settings: Settings dict (from load_settings).
    
    Returns:
        API key string if found, None otherwise.
        Returns None if env var is not set (supports local servers without auth).
    """
    api_key_env = settings.get("engine", {}).get("api_key_env")
    if not api_key_env:
        return None
    
    return os.getenv(api_key_env)


def build_engine(settings: Dict) -> Tuple[OpenAICompatEngine, str]:
    """Build OpenAI-compatible engine from settings.
    
    Args:
        settings: Settings dict (from load_settings).
    
    Returns:
        Tuple of (engine, model_name):
        - engine: Configured OpenAICompatEngine instance.
        - model_name: Model name string from settings.
    
    Raises:
        ValueError: If required settings are missing.
    """
    # Extract engine settings
    engine_settings = settings.get("engine", {})
    base_url = engine_settings.get("base_url")
    
    if not base_url:
        raise ValueError("settings.engine.base_url is required")
    
    # Resolve API key
    api_key = resolve_api_key(settings)
    
    # Get optional default headers
    default_headers = engine_settings.get("headers")
    
    # Create engine
    engine = OpenAICompatEngine(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers
    )
    
    # Extract model name
    model_name = settings.get("model", {}).get("name")
    if not model_name:
        raise ValueError("settings.model.name is required")
    
    return (engine, model_name)
