"""CLI entrypoint for KV Transition evaluation harness.

Phase A: Minimal CLI for config loading/validation and path resolution.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths
from .settings import load_settings


def _parse_overrides(override_args: List[str]) -> Dict[str, Any]:
    """Parse --override KEY=VALUE arguments into nested dict.
    
    Supports dot-notation for nested keys (e.g., "output.exp_group_id=test").
    All values are treated as strings (no type coercion).
    
    Args:
        override_args: List of "KEY=VALUE" strings.
    
    Returns:
        Nested dict suitable for load_settings(overrides=...).
    
    Raises:
        ValueError: If override format is invalid.
    """
    overrides: Dict[str, Any] = {}
    
    for override_str in override_args:
        if "=" not in override_str:
            raise ValueError(f"Invalid override format: {override_str}. Expected KEY=VALUE")
        
        key_str, value = override_str.split("=", 1)
        keys = key_str.split(".")
        
        # Build nested dict
        current = overrides
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                raise ValueError(f"Cannot override nested key '{key_str}': '{key}' is not a dict")
            current = current[key]
        
        # Set final value
        current[keys[-1]] = value
    
    return overrides


def _print_config(config: Dict[str, Any]) -> None:
    """Print merged config as YAML to stdout."""
    try:
        import yaml
        # Remove internal keys before printing
        print_config = {k: v for k, v in config.items() if not k.startswith("_")}
        yaml.dump(print_config, sys.stdout, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback to pretty-print dict if yaml not available
        import json
        print_config = {k: v for k, v in config.items() if not k.startswith("_")}
        json.dump(print_config, sys.stdout, indent=2)


def cmd_resolve(config_path: Path, overrides: Optional[Dict[str, Any]] = None) -> int:
    """Resolve and print exp_group_id, run_dir, and db_path.
    
    Args:
        config_path: Path to YAML config file.
        overrides: Optional config overrides.
    
    Returns:
        Exit code (0 on success, 2 on error).
    """
    try:
        config = load_settings(config_path, overrides=overrides)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 2
    
    exp_group_id = config["output"]["exp_group_id"]
    db_path_cfg = config.get("db", {}).get("path")
    
    # Resolve paths
    run_d = paths.run_dir(exp_group_id)
    db_p = paths.db_path(exp_group_id, db_path_cfg)
    
    # Print machine-friendly output (one key=value per line)
    print(f"exp_group_id={exp_group_id}")
    print(f"run_dir={run_d}")
    print(f"db_path={db_p}")
    
    return 0


def cmd_validate(config_path: Path, overrides: Optional[Dict[str, Any]] = None, print_config_flag: bool = False) -> int:
    """Validate config and print warnings.
    
    Args:
        config_path: Path to YAML config file.
        overrides: Optional config overrides.
        print_config_flag: If True, print merged config to stdout.
    
    Returns:
        Exit code (0 on success, 2 on validation error).
    """
    try:
        config = load_settings(config_path, overrides=overrides)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 2
    
    # Print warnings if any
    warnings = config.get("_warnings", [])
    for warning in warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    
    # Print config if requested
    if print_config_flag:
        _print_config(config)
    
    print("OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
    
    Returns:
        Exit code (0 on success, 2 on error).
    """
    parser = argparse.ArgumentParser(
        description="KV Transition evaluation harness CLI",
        prog="python -m kv_transition"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Common options (added to each subcommand)
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Path to YAML config file (e.g., config/experiments/snapkv_longbench.yaml)"
    )
    common_parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override config value (supports dot-notation, e.g., 'output.exp_group_id=test'). Can be repeated."
    )
    
    # resolve command
    resolve_parser = subparsers.add_parser(
        "resolve",
        parents=[common_parser],
        help="Resolve and print exp_group_id, run_dir, and db_path"
    )
    
    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        parents=[common_parser],
        help="Validate config and print warnings"
    )
    validate_parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged config to stdout (for debugging)"
    )
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Parse overrides
    overrides = None
    if args.overrides:
        try:
            overrides = _parse_overrides(args.overrides)
        except ValueError as e:
            print(f"Error parsing overrides: {e}", file=sys.stderr)
            return 2
    
    # Route to command
    if args.command == "resolve":
        return cmd_resolve(args.config, overrides=overrides)
    elif args.command == "validate":
        return cmd_validate(args.config, overrides=overrides, print_config_flag=getattr(args, 'print_config', False))
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
