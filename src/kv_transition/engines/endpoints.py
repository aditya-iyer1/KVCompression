"""Endpoint resolution for building engines from settings.

Resolves environment variables and constructs OpenAI-compatible engines.
"""

import os
from typing import Dict, Optional, Tuple

from .base import BaseEngine
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


def build_engine(settings: Dict) -> Tuple[BaseEngine, str]:
    """Build engine from settings.
    
    Args:
        settings: Settings dict (from load_settings).
    
    Returns:
        Tuple of (engine, model_name):
        - engine: Configured engine instance.
        - model_name: Model name string from settings.
    
    Raises:
        ValueError: If required settings are missing.
    """
    # Always require model name
    model_name = settings.get("model", {}).get("name")
    if not model_name:
        raise ValueError("settings.model.name is required")

    # Extract engine settings
    engine_settings = settings.get("engine", {})
    engine_type = engine_settings.get("type") or "openai_compat"

    if engine_type == "openai_compat":
        base_url = engine_settings.get("base_url")
        if not base_url:
            raise ValueError("settings.engine.base_url is required for engine.type=openai_compat")

        # Resolve API key
        api_key = resolve_api_key(settings)

        # Get optional default headers
        default_headers = engine_settings.get("headers")

        engine: BaseEngine = OpenAICompatEngine(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )
        return (engine, model_name)

    if engine_type == "kvfactory":
        from .kvfactory_engine import KVFactoryEngine

        repo_dir = engine_settings.get("kvfactory_repo_dir")
        if not repo_dir:
            raise ValueError("settings.engine.kvfactory_repo_dir is required for engine.type=kvfactory")

        method = engine_settings.get("method") or "snapkv"
        attn_implementation = engine_settings.get("attn_implementation") or "sdpa"
        max_capacity_prompts = engine_settings.get("max_capacity_prompts", -1)

        save_root_dir = engine_settings.get("save_root_dir")
        if not save_root_dir:
            exp_group_id = settings.get("output", {}).get("exp_group_id") or "default"
            save_root_dir = os.path.join("runs", str(exp_group_id), "kvfactory")

        engine = KVFactoryEngine(
            repo_dir,
            method=method,
            attn_implementation=attn_implementation,
            max_capacity_prompts=max_capacity_prompts,
            save_root_dir=save_root_dir,
        )
        return (engine, model_name)

    raise ValueError(
        f"Unsupported settings.engine.type={engine_type!r}; expected 'openai_compat' or 'kvfactory'."
    )
