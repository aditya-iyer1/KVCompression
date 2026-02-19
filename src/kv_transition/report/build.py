"""Report generation from persisted DB outputs and plot files.

Generates markdown reports by rendering Jinja templates with data from the database.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


def _get_value(row, key):
    """Safely get value from Row, returning None if key doesn't exist."""
    try:
        return row[key] if key in row.keys() else None
    except (KeyError, IndexError):
        return None


def build_report(conn: sqlite3.Connection, settings: Dict) -> Path:
    """Generate markdown report from persisted DB outputs and plot files.
    
    Reads experiment metadata, runs, transition summary, bin stats, and plot files,
    then renders a Jinja template to produce a markdown report.
    
    Args:
        conn: SQLite connection.
        settings: Settings dict with exp_group_id.
    
    Returns:
        Path to generated report.md file.
    
    Raises:
        ImportError: If jinja2 is not installed.
        FileNotFoundError: If template file not found.
    """
    if not HAS_JINJA2:
        raise ImportError(
            "jinja2 is required for report generation. Install with: pip install jinja2"
        )
    
    # Determine paths
    exp_group_id = settings["output"]["exp_group_id"]
    
    from .. import paths
    
    run_dir = paths.run_dir(exp_group_id)
    report_path = run_dir / "report.md"
    plots_dir = run_dir / "plots"
    
    # Template path relative to repo root
    repo_root = paths.repo_root()
    template_dir = repo_root / "src" / "kv_transition" / "report" / "templates"
    template_path = template_dir / "report.md.jinja"
    
    if not template_path.exists():
        # Fallback: try report.py.jinja if report.md.jinja doesn't exist
        fallback_template = template_dir / "report.py.jinja"
        if fallback_template.exists():
            template_path = fallback_template
        else:
            raise FileNotFoundError(
                f"Template not found: {template_path} or {fallback_template}"
            )
    
    # Gather experiment metadata
    cursor = conn.execute("""
        SELECT * FROM experiments
        WHERE exp_group_id = ?
    """, (exp_group_id,))
    exp_row = cursor.fetchone()
    
    experiment = None
    if exp_row:
        experiment = {
            "exp_group_id": exp_row["exp_group_id"],
            "created_at": exp_row["created_at"] if "created_at" in exp_row.keys() else None,
            "config_yaml": exp_row["config_yaml"] if "config_yaml" in exp_row.keys() else None,
            "git_hash": exp_row["git_hash"] if "git_hash" in exp_row.keys() else None,
            "notes": exp_row["notes"] if "notes" in exp_row.keys() else None
        }
    
    # Gather runs for exp_group_id
    cursor = conn.execute("""
        SELECT run_id, kv_policy, kv_budget, engine_name, base_url, model_name, created_at
        FROM runs
        WHERE exp_group_id = ?
        ORDER BY kv_budget DESC
    """, (exp_group_id,))
    run_rows = cursor.fetchall()
    
    # Import DAO for bin_stats
    from ..db import dao
    
    runs = []
    for run_row in run_rows:
        run_id = run_row["run_id"]
        
        # Get bin_stats for this run
        bin_stats_rows = dao.get_bin_stats_for_run(conn, run_id)
        bin_stats = []
        for stat_row in bin_stats_rows:
            # Convert Row to dict
            bin_stat = {
                "bin_idx": stat_row["bin_idx"],
                "token_min": _get_value(stat_row, "token_min"),
                "token_max": _get_value(stat_row, "token_max"),
                "n": _get_value(stat_row, "n"),
                "acc_mean": _get_value(stat_row, "acc_mean"),
                "acc_std": _get_value(stat_row, "acc_std"),
                "acc_ci_low": _get_value(stat_row, "acc_ci_low"),
                "acc_ci_high": _get_value(stat_row, "acc_ci_high"),
                "em_mean": _get_value(stat_row, "em_mean"),
                "fail_rate": _get_value(stat_row, "fail_rate"),
                "lat_p50": _get_value(stat_row, "lat_p50"),
                "lat_p95": _get_value(stat_row, "lat_p95"),
                "tok_p50": _get_value(stat_row, "tok_p50"),
                "tok_p95": _get_value(stat_row, "tok_p95")
            }
            bin_stats.append(bin_stat)
        
        # Convert Row to dict
        run_dict = {
            "run_id": run_id,
            "kv_policy": _get_value(run_row, "kv_policy"),
            "kv_budget": _get_value(run_row, "kv_budget"),
            "engine_name": _get_value(run_row, "engine_name"),
            "base_url": _get_value(run_row, "base_url"),
            "model_name": _get_value(run_row, "model_name"),
            "created_at": _get_value(run_row, "created_at"),
            "bin_stats": bin_stats
        }
        runs.append(run_dict)
    
    # Gather transition summary
    transition_row = dao.get_transition_summary(conn, exp_group_id)
    transition = None
    if transition_row:
        # Handle "drop" column (quoted in SQL)
        cursor = conn.execute('SELECT "drop" FROM transition_summary WHERE exp_group_id = ?', (exp_group_id,))
        drop_row = cursor.fetchone()
        drop = drop_row[0] if drop_row else None
        
        transition = {
            "kv_policy": _get_value(transition_row, "kv_policy"),
            "method": _get_value(transition_row, "method"),
            "drop_threshold": _get_value(transition_row, "drop_threshold"),
            "pre_budget": _get_value(transition_row, "pre_budget"),
            "transition_budget": _get_value(transition_row, "transition_budget"),
            "acc_pre": _get_value(transition_row, "acc_pre"),
            "acc_post": _get_value(transition_row, "acc_post"),
            "drop": drop,
            "transition_bin_idx": _get_value(transition_row, "transition_bin_idx"),
            "created_at": _get_value(transition_row, "created_at")
        }
    
    # Gather plot artifacts
    plots = {}
    expected_plots = ["acc_by_bin.png", "fail_by_bin.png", "latency_p50_by_bin.png"]
    
    if plots_dir.exists():
        for plot_name in expected_plots:
            plot_path = plots_dir / plot_name
            if plot_path.exists():
                # Store relative path from report location
                plots[plot_name] = f"plots/{plot_name}"
    
    # Build context
    context = {
        "exp_group_id": exp_group_id,
        "experiment": experiment,
        "runs": runs,
        "transition": transition,
        "plots": plots,
        "generated_at": datetime.utcnow().isoformat()
    }
    
    # Render template
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template(template_path.name)
    rendered = template.render(**context)
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")
    
    return report_path
