"""CLI entrypoint for KV Transition evaluation harness.

Phase A: Minimal CLI for config loading/validation and path resolution.
Phase D: Scoring completed runs.
Phase E: Aggregation and transition detection.
Phase F: Report generation.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import paths
from .settings import load_settings

from dotenv import load_dotenv


load_dotenv()


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


def _get_git_hash() -> Optional[str]:
    """Get current git commit hash if available.
    
    Returns:
        Git commit hash string, or None if git is not available or not in a git repo.
    """
    try:
        repo_root = paths.repo_root()
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, RuntimeError):
        pass
    return None


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


def cmd_prepare(
    config_path: Path,
    split: str = "test",
    cache_dir: Optional[str] = None,
    notes: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> int:
    """Prepare dataset: load, normalize, bin, and persist manifest (Phase B).
    
    Loads raw examples, builds manifest, and persists to database and manifest.json.
    
    Args:
        config_path: Path to YAML config file.
        split: Dataset split to load (default: "test").
        cache_dir: Optional cache directory for dataset loader.
        notes: Optional notes string to store in experiments.notes.
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
    
    # Ensure run_dir exists
    run_d.mkdir(parents=True, exist_ok=True)
    
    # Open DB and init schema
    try:
        from .db import connect, schema
        conn = connect.connect(db_p)
        schema.init_schema(conn)
    except Exception as e:
        print(f"Error opening database: {e}", file=sys.stderr)
        return 2
    
    try:
        # Dump config as YAML (remove internal keys)
        try:
            import yaml
            config_for_yaml = {k: v for k, v in config.items() if not k.startswith("_")}
            config_yaml = yaml.dump(config_for_yaml, default_flow_style=False, sort_keys=False)
        except ImportError:
            import json
            config_for_yaml = {k: v for k, v in config.items() if not k.startswith("_")}
            config_yaml = json.dumps(config_for_yaml, indent=2)
        
        # Get git hash
        git_hash = _get_git_hash()
        
        # Upsert experiments row
        created_at = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT OR REPLACE INTO experiments
            (exp_group_id, created_at, config_yaml, git_hash, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (exp_group_id, created_at, config_yaml, git_hash, notes))
        conn.commit()
        
        # Derive dataset_id from config (same convention as manifest._compute_dataset_id)
        dataset_name = config.get("dataset", {}).get("name")
        task = config.get("dataset", {}).get("task")
        
        if not dataset_name or not task:
            print("Error: config.dataset.name and config.dataset.task are required", file=sys.stderr)
            conn.close()
            return 2
        
        dataset_id = f"{dataset_name}__{task}"
        
        # Load raw examples
        from .data.longbench_loader import load_longbench
        
        cache_dir_path = Path(cache_dir) if cache_dir else None
        raw_examples = load_longbench(task=task, cache_dir=cache_dir_path, split=split)
        
        # Build manifest
        from .data.manifest import build_manifest, write_manifest
        
        manifest = build_manifest(config, raw_examples)
        
        # Write manifest.json
        manifest_path_obj = write_manifest(manifest, dataset_id)
        
        # Persist to DB using DAO functions
        from .db import dao
        
        # Get resolved model name from config (fully resolved after env var substitution)
        model_name = config.get("model", {}).get("name")
        if not model_name or not model_name.strip():
            print("Error: config.model.name is required and must be resolved", file=sys.stderr)
            conn.close()
            return 2
        
        # Use resolved model name as tokenizer_name (ensure it's not a template placeholder)
        tokenizer_name = model_name.strip()
        if tokenizer_name.startswith("${") and tokenizer_name.endswith("}"):
            print(f"Error: config.model.name appears unresolved: {tokenizer_name}", file=sys.stderr)
            conn.close()
            return 2
        
        # Check if dataset already exists with a different tokenizer_name
        cursor = conn.execute(
            "SELECT tokenizer_name FROM datasets WHERE dataset_id = ?",
            (dataset_id,)
        )
        existing_row = cursor.fetchone()
        if existing_row and existing_row[0] is not None:
            existing_tokenizer = existing_row[0]
            if existing_tokenizer != tokenizer_name:
                print(
                    f"Error: Dataset {dataset_id} already exists with tokenizer_name={existing_tokenizer}, "
                    f"but current config specifies tokenizer_name={tokenizer_name}. "
                    f"Tokenizer name mismatch detected.",
                    file=sys.stderr
                )
                conn.close()
                return 2
        
        # Upsert dataset with resolved tokenizer_name
        dao.upsert_dataset(conn, dataset_id, dataset_name, task, tokenizer_name)
        
        # Insert examples (from manifest.examples dict)
        # Build token_len lookup from entries
        token_len_lookup = {entry["example_id"]: entry["token_len"] for entry in manifest["entries"]}
        
        examples_list = []
        for example_id, ex_data in manifest["examples"].items():
            token_len = token_len_lookup.get(example_id, 0)
            
            examples_list.append({
                "example_id": example_id,
                "question": ex_data["question"],
                "context": ex_data["context"],
                "answers": ex_data["answers"],
                "meta": ex_data["meta"],
                "token_len": token_len
            })
        
        dao.insert_examples(conn, dataset_id, examples_list)
        
        # Insert bins
        bin_edges = manifest["bin_edges"]
        # Count examples per bin
        n_examples_per_bin: Dict[int, int] = {}
        for entry in manifest["entries"]:
            bin_idx = entry["bin_idx"]
            n_examples_per_bin[bin_idx] = n_examples_per_bin.get(bin_idx, 0) + 1
        
        dao.insert_bins(conn, dataset_id, bin_edges, n_examples_per_bin)
        
        # Insert manifest entries
        dao.insert_manifest_entries(conn, dataset_id, manifest["entries"])
        
        conn.commit()
        
        # Print success summary
        n_examples = len(examples_list)
        print(f"Prepared dataset:")
        print(f"  exp_group_id: {exp_group_id}")
        print(f"  db_path: {db_p}")
        print(f"  dataset_id: {dataset_id}")
        print(f"  n_examples: {n_examples}")
        print(f"  manifest: {manifest_path_obj}")
        
        conn.close()
        return 0
        
    except Exception as e:
        print(f"Error preparing dataset: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        conn.close()
        return 2


def cmd_run(config_path: Path, overrides: Optional[Dict[str, Any]] = None) -> int:
    """Run inference over a prepared manifest (Phase C).
    
    Loads settings, checks that Phase B artifacts exist, then orchestrates runs.
    
    Args:
        config_path: Path to YAML config file.
        overrides: Optional config overrides.
    
    Returns:
        Exit code (0 on success, 2 on error).
    """
    # 1) Load/validate config
    try:
        config = load_settings(config_path, overrides=overrides)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 2
    
    # 2) Resolve exp_group_id/run_dir/db_path
    exp_group_id = config["output"]["exp_group_id"]
    db_path_cfg = config.get("db", {}).get("path")
    
    run_d = paths.run_dir(exp_group_id)
    db_p = paths.db_path(exp_group_id, db_path_cfg)
    
    # Ensure run_dir exists (orchestrate will reuse it)
    run_d.mkdir(parents=True, exist_ok=True)
    
    # 3) Open DB and init schema
    try:
        from .db import connect, schema
        conn = connect.connect(db_p)
        schema.init_schema(conn)
    except Exception as e:
        print(f"Error opening database: {e}", file=sys.stderr)
        return 2
    
    try:
        # 4) Ensure Phase B preconditions
        dataset_name = config.get("dataset", {}).get("name")
        task = config.get("dataset", {}).get("task")
        
        if not dataset_name or not task:
            print("Error: config.dataset.name and config.dataset.task are required", file=sys.stderr)
            conn.close()
            return 2
        
        dataset_id = f"{dataset_name}__{task}"
        
        # Check that dataset exists
        cursor = conn.execute(
            "SELECT 1 FROM datasets WHERE dataset_id = ?",
            (dataset_id,)
        )
        if cursor.fetchone() is None:
            print(
                "No prepared manifest found; run `kv-transition prepare` first.",
                file=sys.stderr
            )
            conn.close()
            return 2
        
        # Check that manifest_entries exist for this dataset
        cursor = conn.execute(
            "SELECT 1 FROM manifest_entries WHERE dataset_id = ? LIMIT 1",
            (dataset_id,)
        )
        if cursor.fetchone() is None:
            print(
                "No prepared manifest found; run `kv-transition prepare` first.",
                file=sys.stderr
            )
            conn.close()
            return 2
        
        # Close this connection before orchestration; orchestrate() manages its own conn
        conn.close()
        
        # 5) Call orchestrate(settings) to perform Phase C runs
        from .run.orchestrate import orchestrate
        
        orchestrate(config)
        return 0
    
    except Exception as e:
        print(f"Error during run: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        conn.close()
        return 2


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
            run_meta = dict(run_meta)
            
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


def _validate_report_prerequisites(
    exp_group_id: str,
    db_path: Path,
    run_dir: Path,
    allow_partial: bool = False
) -> Tuple[bool, List[str]]:
    """Validate prerequisites for report generation.
    
    Args:
        exp_group_id: Experiment group ID.
        db_path: Path to database file.
        run_dir: Path to run directory.
        allow_partial: If True, return warnings instead of errors.
    
    Returns:
        Tuple of (is_valid, list_of_missing_items).
        If allow_partial is False and prerequisites are missing, is_valid is False.
        If allow_partial is True, is_valid is always True but missing_items contains warnings.
    """
    missing = []
    
    # Check database file exists
    if not db_path.exists():
        missing.append(f"Database file missing: {db_path}")
    else:
        # Check database contains bin_stats table (lightweight check)
        try:
            from .db import connect, schema
            conn = connect.connect(db_path)
            schema.init_schema(conn)
            
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='bin_stats'
            """)
            if cursor.fetchone() is None:
                missing.append(f"Database table 'bin_stats' missing (run 'analyze' command first)")
            
            # Check if bin_stats has any data for this exp_group_id
            cursor = conn.execute("""
                SELECT COUNT(*) FROM bin_stats bs
                JOIN runs r ON bs.run_id = r.run_id
                WHERE r.exp_group_id = ?
            """, (exp_group_id,))
            row = cursor.fetchone()
            count = row[0] if row else 0
            if count == 0:
                missing.append(f"No bin_stats data found for exp_group_id '{exp_group_id}' (run 'analyze' command first)")
            
            conn.close()
        except Exception as e:
            missing.append(f"Error checking database: {e}")
    
    # Check plots directory exists
    plots_dir = run_dir / "plots"
    if not plots_dir.exists():
        missing.append(f"Plots directory missing: {plots_dir}")
    else:
        # Check expected plot files
        expected_plots = ["acc_by_bin.png", "fail_by_bin.png", "latency_p50_by_bin.png"]
        for plot_file in expected_plots:
            plot_path = plots_dir / plot_file
            if not plot_path.exists():
                missing.append(f"Plot file missing: {plot_path}")
    
    if not allow_partial and missing:
        return (False, missing)
    
    return (True, missing)


def cmd_report(config_path: Path, allow_partial: bool = False, overrides: Optional[Dict[str, Any]] = None) -> int:
    """Generate markdown report from persisted DB outputs (Phase F).
    
    Reads experiment metadata, runs, transition summary, bin stats, and plot files,
    then renders a Jinja template to produce a markdown report.
    
    Args:
        config_path: Path to YAML config file.
        allow_partial: If True, allow report generation even if prerequisites are missing.
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
    run_d = paths.run_dir(exp_group_id)
    
    # Validate prerequisites
    is_valid, missing = _validate_report_prerequisites(exp_group_id, db_p, run_d, allow_partial=allow_partial)
    
    if not is_valid:
        print("Error: Missing required prerequisites for report generation:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print("", file=sys.stderr)
        print("To fix:", file=sys.stderr)
        if any("bin_stats" in item or "analyze" in item.lower() for item in missing):
            print("  Run: uv run python -m kv_transition analyze -c <config>", file=sys.stderr)
        if any("plot" in item.lower() for item in missing):
            print("  Run: uv run python -m kv_transition analyze -c <config> (generates plots)", file=sys.stderr)
        print("", file=sys.stderr)
        print("Or use --allow-partial to generate a partial report.", file=sys.stderr)
        return 2
    
    if missing and allow_partial:
        print("Warning: Generating partial report (--allow-partial enabled). Missing items:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print("", file=sys.stderr)
    
    # Open DB and init schema
    try:
        from .db import connect, schema
        conn = connect.connect(db_p)
        schema.init_schema(conn)
    except Exception as e:
        print(f"Error opening database: {e}", file=sys.stderr)
        return 2
    
    try:
        # Generate report
        from .report.build import build_report
        
        report_path = build_report(conn, config)
        print(f"Report generated: {report_path}")
        
        conn.close()
        return 0
        
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        conn.close()
        return 2
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        conn.close()
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        conn.close()
        return 2


def cmd_all(config_path: Path, run_id: Optional[str] = None, allow_partial: bool = False, overrides: Optional[Dict[str, Any]] = None) -> int:
    """Run complete pipeline: score → analyze → report (Phase D→E→F).
    
    Executes scoring, analysis, and report generation in sequence without re-running inference.
    
    Args:
        config_path: Path to YAML config file.
        run_id: Optional specific run_id. If omitted, processes all runs for exp_group_id.
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
    
    # Open DB and init schema once
    try:
        from .db import connect, schema, dao
        conn = connect.connect(db_p)
        schema.init_schema(conn)
    except Exception as e:
        print(f"Error opening database: {e}", file=sys.stderr)
        return 2
    
    try:
        # Get run_ids to process
        run_ids = _get_run_ids(conn, exp_group_id, run_id)
        
        # ===== Phase D: Score =====
        print("Phase D: Scoring runs...")
        from .eval.score import score_run
        
        for rid in run_ids:
            try:
                score_run(conn, rid)
                print(f"  Scored run: {rid}")
            except Exception as e:
                print(f"Error scoring run {rid}: {e}", file=sys.stderr)
                conn.close()
                return 2
        
        # ===== Phase E: Analyze =====
        print("Phase E: Analyzing runs...")
        from .analysis import queries, aggregate, bootstrap, transition, plots
        
        runs_data = []  # For transition detection and plotting
        
        for rid in run_ids:
            # Get run metadata
            run_meta = queries.get_run_metadata(conn, rid)
            if not run_meta:
                print(f"Warning: Run {rid} metadata not found, skipping", file=sys.stderr)
                continue
            
            kv_budget = run_meta["kv_budget"]
            
            # Get dataset_id from first request
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
            print(f"  Analyzed run: {rid} (budget={kv_budget})")
            
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
        
        # Get kv_policy from first run
        kv_policy = runs_data[0].get("kv_policy", "")
        if not kv_policy:
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
            print(f"  Plots saved:")
            for path in plot_paths:
                print(f"    {path}")
        except ImportError as e:
            print(f"Warning: Could not generate plots: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error generating plots: {e}", file=sys.stderr)
        
        # Print transition result
        if transition_result.get("transition_budget") is not None:
            print(f"  Transition detected at budget {transition_result['transition_budget']:.2f} "
                  f"(drop: {transition_result.get('drop', 0):.3f})")
        else:
            print("  No transition detected")
        
        # ===== Phase F: Report =====
        print("Phase F: Generating report...")
        conn.close()
        
        # Call cmd_report with allow_partial flag (it will handle validation and generation)
        report_exit_code = cmd_report(config_path, allow_partial=allow_partial, overrides=overrides)
        if report_exit_code != 0:
            return report_exit_code
        
        return 0
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        conn.close()
        return 2
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        conn.close()
        return 2
    except FileNotFoundError as e:
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
    
    # run command
    run_parser = subparsers.add_parser(
        "run",
        parents=[common_parser],
        help="Run inference over prepared manifest (Phase C)"
    )
    
    # prepare command
    prepare_parser = subparsers.add_parser(
        "prepare",
        parents=[common_parser],
        help="Prepare dataset: load, normalize, bin, and persist manifest (Phase B)"
    )
    prepare_parser.add_argument(
        "--split",
        type=str,
        default="test",
        help='Dataset split to load (default: "test")'
    )
    prepare_parser.add_argument(
        "--cache-dir",
        type=str,
        help="Optional cache directory for dataset loader"
    )
    prepare_parser.add_argument(
        "--notes",
        type=str,
        help="Optional notes string to store in experiments.notes"
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
    
    # report command
    report_parser = subparsers.add_parser(
        "report",
        parents=[common_parser],
        help="Generate markdown report from persisted DB outputs (Phase F)"
    )
    report_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow report generation even if some prerequisites are missing (prints warnings)"
    )
    
    # all command
    all_parser = subparsers.add_parser(
        "all",
        parents=[common_parser],
        help="Run complete pipeline: score → analyze → report (Phase D→E→F)"
    )
    all_parser.add_argument(
        "--run-id",
        type=str,
        help="Specific run_id to score/analyze (default: all runs for exp_group_id). Report is always group-level."
    )
    all_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow report generation even if some prerequisites are missing (prints warnings)"
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
    if args.command == "run":
        return cmd_run(args.config, overrides=overrides)
    elif args.command == "prepare":
        return cmd_prepare(
            args.config,
            split=getattr(args, 'split', 'test'),
            cache_dir=getattr(args, 'cache_dir', None),
            notes=getattr(args, 'notes', None),
            overrides=overrides
        )
    elif args.command == "resolve":
        return cmd_resolve(args.config, overrides=overrides)
    elif args.command == "validate":
        return cmd_validate(args.config, overrides=overrides, print_config_flag=getattr(args, 'print_config', False))
    elif args.command == "score":
        return cmd_score(args.config, run_id=getattr(args, 'run_id', None), overrides=overrides)
    elif args.command == "analyze":
        return cmd_analyze(args.config, run_id=getattr(args, 'run_id', None), overrides=overrides)
    elif args.command == "report":
        return cmd_report(args.config, allow_partial=getattr(args, 'allow_partial', False), overrides=overrides)
    elif args.command == "all":
        return cmd_all(args.config, run_id=getattr(args, 'run_id', None), allow_partial=getattr(args, 'allow_partial', False), overrides=overrides)
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
