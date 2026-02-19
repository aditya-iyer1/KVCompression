"""Configuration loading and validation for KV Transition evaluation harness.

Phase A: Load, merge, validate, and resolve environment variables in config files.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override dict into base dict (override takes precedence)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_env_vars(obj: Any, warnings: List[str]) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in strings.
    
    If env var is missing, leaves placeholder intact and adds warning.
    """
    if isinstance(obj, str):
        # Match ${VAR_NAME} patterns
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, obj)
        
        if not matches:
            return obj
        
        resolved = obj
        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is not None:
                resolved = resolved.replace(f"${{{var_name}}}", env_value)
            else:
                warnings.append(f"Environment variable '{var_name}' not set, leaving placeholder intact")
        
        return resolved
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v, warnings) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item, warnings) for item in obj]
    else:
        return obj


def _get_nested_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested value from config using dot-separated path."""
    keys = path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def _has_nested_key(config: Dict[str, Any], path: str) -> bool:
    """Check if nested key exists in config (even if value is None)."""
    keys = path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return False
    return True


def _validate_settings(config: Dict[str, Any]) -> List[str]:
    """Validate required keys and types. Returns list of error messages (empty if valid)."""
    errors: List[str] = []
    
    # Helper to check required key
    def require(path: str, expected_type: type, min_value: Optional[float] = None):
        value = _get_nested_value(config, path)
        if value is None:
            errors.append(f"Missing required key: {path}")
            return
        if not isinstance(value, expected_type):
            errors.append(f"Key {path} must be {expected_type.__name__}, got {type(value).__name__}")
            return
        if min_value is not None and value < min_value:
            errors.append(f"Key {path} must be >= {min_value}, got {value}")
    
    # Dataset validation
    require("dataset.name", str)
    require("dataset.task", str)
    require("dataset.n_per_bin", int, min_value=1)
    
    # Binning validation
    require("binning.n_bins", int, min_value=1)
    
    # Model validation
    model_name = _get_nested_value(config, "model.name")
    if model_name is None:
        errors.append("Missing required key: model.name")
    elif not isinstance(model_name, str):
        errors.append(f"Key model.name must be str, got {type(model_name).__name__}")
    elif not model_name.strip():
        # Warn if empty after env resolution, but don't fail (might be resolved later)
        pass
    
    # Engine validation
    require("engine.base_url", str)
    require("engine.api_key_env", str)
    
    # KV validation
    require("kv.policy", str)
    kv_budgets = _get_nested_value(config, "kv.budgets")
    if kv_budgets is None:
        errors.append("Missing required key: kv.budgets")
    elif not isinstance(kv_budgets, list):
        errors.append(f"Key kv.budgets must be a list, got {type(kv_budgets).__name__}")
    elif len(kv_budgets) == 0:
        errors.append("Key kv.budgets must be non-empty")
    else:
        for i, budget in enumerate(kv_budgets):
            if not isinstance(budget, (int, float)):
                errors.append(f"Key kv.budgets[{i}] must be a number, got {type(budget).__name__}")
    
    # Decoding validation
    decoding_temp = _get_nested_value(config, "decoding.temperature")
    if decoding_temp is None:
        errors.append("Missing required key: decoding.temperature")
    elif not isinstance(decoding_temp, (int, float)):
        errors.append(f"Key decoding.temperature must be a number, got {type(decoding_temp).__name__}")
    
    require("decoding.max_tokens", int, min_value=1)
    
    # Output validation
    require("output.exp_group_id", str)
    
    # DB validation (allow None/null or string, including sentinel values like "auto")
    if not _has_nested_key(config, "db.path"):
        errors.append("Missing required key: db.path")
    else:
        db_path = _get_nested_value(config, "db.path")
        if db_path is not None and not isinstance(db_path, str):
            errors.append(f"Key db.path must be str or null, got {type(db_path).__name__}")
    
    return errors


def load_settings(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load, merge, validate, and resolve environment variables in config.
    
    Args:
        config_path: Path to experiment YAML config file.
        overrides: Optional dict to override config values (highest priority).
    
    Returns:
        Merged and validated config dict with '_warnings' key containing non-fatal warnings.
    
    Raises:
        FileNotFoundError: If default.yaml or config_path doesn't exist.
        ValueError: If validation fails (missing required keys or wrong types).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load default.yaml as base
    default_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")
    
    with open(default_path, 'r') as f:
        base_config = yaml.safe_load(f) or {}
    
    # Load experiment config
    with open(config_path, 'r') as f:
        experiment_config = yaml.safe_load(f) or {}
    
    # Deep merge: base -> experiment -> overrides
    merged = _deep_merge(base_config, experiment_config)
    if overrides:
        merged = _deep_merge(merged, overrides)
    
    # Resolve environment variables
    warnings: List[str] = []
    resolved = _resolve_env_vars(merged, warnings)
    
    # Validate
    validation_errors = _validate_settings(resolved)
    if validation_errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
        raise ValueError(error_msg)
    
    # Attach warnings to config (non-fatal issues)
    resolved['_warnings'] = warnings
    
    return resolved
