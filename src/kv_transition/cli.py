"""CLI entrypoint for KV Transition evaluation harness.

Phase A: Minimal CLI for config loading/validation and path resolution.
Phase D: Scoring completed runs.
Phase E: Aggregation and transition detection.
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


def _get_run_ids(conn, exp_group_id: str, run_id: Optional[str] = None) -> List[str]:
    """Get list of run_ids to process.
    
    Args:
        conn: SQLite connection.
        exp_group_id: Experiment group identifier.
        run_id: Optional specific run_id. If provided, returns [run_id].
                 If None, queries all runs for exp_group_id.
    
    Returns:
        List of run_ids, ordered by kv_budget DESC.
    
    Raises:
        ValueError: If run_id specified but not found, or no runs found.
    """
    if run_id:
        cursor = conn.execute("SELECT run_id FROM runs WHERE run_id = ? AND exp_group_id = ?", (run_id, exp_group_id))
        if cursor.fetchone() is None:
            raise ValueError(f"Run {run_id} not found for exp_group_id {exp_group_id}")
        return [run_id]
    
    # Query all runs for exp_group_id, ordered by kv_budget DESC
    cursor = conn.execute("""
        SELECT run_id FROM runs
        WHERE exp_group_id = ?
        ORDER BY kv_budget DESC
    """, (exp_group_id,))
    
    run_ids = [row[0] for row in cursor.fetchall()]
    if not run_ids:
        raise ValueError(f"No runs found for exp_group_id {exp_group_id}")
    
    return run_ids


def cmd_score(config_path: Path, run_id: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> int:
    """Score completed runs (Phase D).
    
    Computes EM/F1 scores and classifies failures for requests in the specified run(s).
    
    Args:
        config_path: Path to YAML config file.
        run_id: Optional specific run_id. If omitted, scores all runs for exp_group_id.
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
    db_p = paths.db_path(exp_group_id, db_path_cfg)
    
    # Open DB and init schema
    try:
        from .db import connect, schema
        conn = connect.connect(db_p)
        schema.init_schema(conn)
    except Exception as e:
        print(f"Error opening database: {e}", file=sys.stderr)
        return 2
    
    try:
        # Get run_ids to score
        run_ids = _get_run_ids(conn, exp_group_id, run_id)
        
        # Score each run
        from .eval.score import score_run
        
        for rid in run_ids:
            try:
                score_run(conn, rid)
                print(f"Scored run: {rid}")
            except Exception as e:
                print(f"Error scoring run {rid}: {e}", file=sys.stderr)
                return 2
        
        conn.close()
        return 0
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        conn.close()
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        conn.close()
        return 2


def cmd_analyze(config_path: Path, run_id: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> int:
    """Analyze runs and generate aggregates, transition detection, and plots (Phase E).
    
    Computes bin-level aggregates, bootstrap CIs, transition detection, and plots.
    
    Args:
        config_path: Path to YAML config file.
        run_id: Optional specific run_id. If omitted, analyzes all runs for exp_group_id.
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
    db_p = paths.db_path(exp_group_id, db_path_cfg)
    
    # Get analysis seed from settings
    seed = config.get("analysis", {}).get("seed", 1337)
    drop_threshold = config.get("analysis", {}).get("drop_threshold", 0.15)
    
    # Open DB and init schema
    try:
        from .db import connect, schema, dao
        conn = connect.connect(db_p)
        schema.init_schema(conn)
    except Exception as e:
        print(f"Error opening database: {e}", file=sys.stderr)
        return 2
    
    try:
        # Get run_ids to analyze
        run_ids = _get_run_ids(conn, exp_group_id, run_id)
        
        # Import analysis modules
        from .analysis import queries, aggregate, bootstrap, transition, plots
        
        # Process each run: compute aggregates and persist bin_stats
        runs_data = []  # For transition detection and plotting
        
        for rid in run_ids:
            # Get run metadata
            run_meta = queries.get_run_metadata(conn, rid)
            if not run_meta:
                print(f"Warning: Run {rid} metadata not found, skipping", file=sys.stderr)
                continue
            
            kv_budget = run_meta["kv_budget"]
            
            # Get dataset_id from first request (all requests in a run share dataset_id)
            cursor = conn.execute("SELECT dataset_id FROM requests WHERE run_id = ? LIMIT 1", (rid,))
            dataset_row = cursor.fetchone()
            if not dataset_row:
                print(f"Warning: No requests found for run {rid}, skipping", file=sys.stderr)
                continue
            
            dataset_id = dataset_row[0]
            
            # Get bin-level rows
            bin_rows = queries.get_bin_level_rows(conn, rid)
            if not bin_rows:
                print(f"Warning: No bin-level data for run {rid}, skipping", file=sys.stderr)
                continue
            
            # Compute aggregates
            bin_stats = aggregate.aggregate_run_bins(conn, rid)
            
            # Add bootstrap CIs
            bin_stats = bootstrap.add_bootstrap_cis(bin_stats, bin_rows, seed=seed)
            
            # Get bin structure for token_min/token_max
            bin_structure = queries.get_bin_structure(conn, dataset_id)
            bin_edges = {row["bin_idx"]: (row["token_min"], row["token_max"]) for row in bin_structure}
            
            # Add token_min/token_max to bin_stats
            for stat in bin_stats:
                bin_idx = stat["bin_idx"]
                if bin_idx in bin_edges:
                    stat["token_min"] = bin_edges[bin_idx][0]
                    stat["token_max"] = bin_edges[bin_idx][1]
            
            # Persist bin_stats
            dao.upsert_bin_stats(conn, rid, dataset_id, bin_stats)
            print(f"Analyzed run: {rid} (budget={kv_budget})")
            
            # Collect data for transition detection and plotting
            runs_data.append({
                "run_id": rid,
                "kv_budget": kv_budget,
                "kv_policy": run_meta.get("kv_policy", ""),
                "bins": bin_stats
            })
        
        if not runs_data:
            print("Error: No runs processed", file=sys.stderr)
            conn.close()
            return 2
        
        # Detect transition
        transition_result = transition.detect_transition(runs_data, drop_threshold=drop_threshold)
        
        # Get kv_policy from first run (all runs in a group share policy)
        kv_policy = runs_data[0].get("kv_policy", "")
        if not kv_policy:
            # Fallback: get from first run metadata
            first_run_meta = queries.get_run_metadata(conn, runs_data[0]["run_id"])
            kv_policy = first_run_meta.get("kv_policy", "") if first_run_meta else ""
        
        # Persist transition summary
        summary = {
            "exp_group_id": exp_group_id,
            "kv_policy": kv_policy,
            "method": transition_result.get("method", "overall_mean_drop"),
            "drop_threshold": drop_threshold,
            "pre_budget": transition_result.get("pre_budget"),
            "transition_budget": transition_result.get("transition_budget"),
            "acc_pre": transition_result.get("acc_pre"),
            "acc_post": transition_result.get("acc_post"),
            "drop": transition_result.get("drop"),
            "transition_bin_idx": transition_result.get("transition_bin_idx")
        }
        dao.upsert_transition_summary(conn, summary)
        
        # Generate plots
        try:
            plot_paths = plots.save_run_plots(config, runs_data)
            print(f"Plots saved:")
            for path in plot_paths:
                print(f"  {path}")
        except ImportError as e:
            print(f"Warning: Could not generate plots: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error generating plots: {e}", file=sys.stderr)
        
        # Print transition result
        if transition_result.get("transition_budget") is not None:
            print(f"Transition detected at budget {transition_result['transition_budget']:.2f} "
                  f"(drop: {transition_result.get('drop', 0):.3f})")
        else:
            print("No transition detected")
        
        conn.close()
        return 0
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        conn.close()
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        conn.close()
        return 2


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
    
    # score command
    score_parser = subparsers.add_parser(
        "score",
        parents=[common_parser],
        help="Score completed runs (Phase D)"
    )
    score_parser.add_argument(
        "--run-id",
        type=str,
        help="Specific run_id to score (default: all runs for exp_group_id)"
    )
    
    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        parents=[common_parser],
        help="Analyze runs and generate aggregates, transition detection, and plots (Phase E)"
    )
    analyze_parser.add_argument(
        "--run-id",
        type=str,
        help="Specific run_id to analyze (default: all runs for exp_group_id)"
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
    elif args.command == "score":
        return cmd_score(args.config, run_id=getattr(args, 'run_id', None), overrides=overrides)
    elif args.command == "analyze":
        return cmd_analyze(args.config, run_id=getattr(args, 'run_id', None), overrides=overrides)
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
