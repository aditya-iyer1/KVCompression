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


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
    """, (table_name,))
    return cursor.fetchone() is not None


def _compute_integrity_summary(conn: sqlite3.Connection, exp_group_id: str) -> str:
    """Compute integrity summary metrics and return as markdown section.
    
    Computes failure/quality counters from SQLite DB:
    - % empty outputs (response text missing/empty)
    - % refusals (from failures table if present)
    - % engine errors/timeouts (from failures table if present)
    - % format errors (from failures table if present)
    - % missing telemetry (if telemetry table exists)
    
    Args:
        conn: SQLite connection.
        exp_group_id: Experiment group ID.
    
    Returns:
        Markdown string for Integrity Summary section.
    """
    # Get all request_ids for this exp_group_id
    try:
        cursor = conn.execute("""
            SELECT r.request_id
            FROM requests r
            JOIN runs ru ON r.run_id = ru.run_id
            WHERE ru.exp_group_id = ?
        """, (exp_group_id,))
        all_requests = cursor.fetchall()
        total_requests = len(all_requests)
    except sqlite3.OperationalError:
        # requests table might not exist
        return "## Integrity Summary\n\n**Status:** Requests table not available → N/A\n"
    
    if total_requests == 0:
        return "## Integrity Summary\n\n**Status:** No requests found for this experiment.\n"
    
    # Initialize counters
    empty_outputs = 0
    refusals = 0
    engine_errors = 0
    timeouts = 0
    format_errors = 0
    truncated = 0
    missing_telemetry = 0
    failures_available = False
    telemetry_available = False
    
    # Check if failures table exists
    if _table_exists(conn, "failures"):
        failures_available = True
        try:
            cursor = conn.execute("""
                SELECT f.error_type, COUNT(*) as count
                FROM failures f
                JOIN requests r ON f.request_id = r.request_id
                JOIN runs ru ON r.run_id = ru.run_id
                WHERE ru.exp_group_id = ?
                GROUP BY f.error_type
            """, (exp_group_id,))
            failure_rows = cursor.fetchall()
            
            for row in failure_rows:
                try:
                    error_type = row["error_type"]
                    count = row["count"]
                except (KeyError, TypeError):
                    # Fallback for tuple-style rows
                    error_type = row[0] if len(row) > 0 else None
                    count = row[1] if len(row) > 1 else 0
                
                if error_type:
                    error_type_upper = error_type.upper()
                    if "REFUSAL" in error_type_upper:
                        refusals += count
                    elif "TIMEOUT" in error_type_upper:
                        timeouts += count
                    elif "ENGINE_ERROR" in error_type_upper or "ENGINE" in error_type_upper:
                        engine_errors += count
                    elif "FORMAT_ERROR" in error_type_upper or "FORMAT" in error_type_upper:
                        format_errors += count
                    elif "TRUNCATED" in error_type_upper:
                        truncated += count
                    elif "EMPTY_OUTPUT" in error_type_upper:
                        empty_outputs += count
        except sqlite3.OperationalError:
            failures_available = False
    
    # Count empty outputs from responses table
    if _table_exists(conn, "responses"):
        try:
            cursor = conn.execute("""
                SELECT COUNT(*) as count
                FROM responses res
                JOIN requests r ON res.request_id = r.request_id
                JOIN runs ru ON r.run_id = ru.run_id
                WHERE ru.exp_group_id = ?
                AND (res.text IS NULL OR res.text = '' OR TRIM(res.text) = '')
            """, (exp_group_id,))
            row = cursor.fetchone()
            if row:
                try:
                    empty_from_responses = row["count"]
                except (KeyError, TypeError):
                    empty_from_responses = row[0] if len(row) > 0 else 0
                # Use max of failures table count or responses table count
                empty_outputs = max(empty_outputs, empty_from_responses)
        except sqlite3.OperationalError:
            pass
    
    # Count missing telemetry
    if _table_exists(conn, "telemetry"):
        telemetry_available = True
        try:
            cursor = conn.execute("""
                SELECT COUNT(*) as count
                FROM requests r
                JOIN runs ru ON r.run_id = ru.run_id
                LEFT JOIN telemetry t ON r.request_id = t.request_id
                WHERE ru.exp_group_id = ?
                AND t.request_id IS NULL
            """, (exp_group_id,))
            row = cursor.fetchone()
            if row:
                try:
                    missing_telemetry = row["count"]
                except (KeyError, TypeError):
                    missing_telemetry = row[0] if len(row) > 0 else 0
        except sqlite3.OperationalError:
            telemetry_available = False
    
    # Compute percentages
    def pct(count, total):
        if total == 0:
            return 0.0
        return (count / total) * 100.0
    
    # Build markdown section
    lines = ["## Integrity Summary", ""]
    
    # Table header
    lines.append("| Metric | Count | Percentage |")
    lines.append("|--------|-------|------------|")
    
    # Empty outputs
    lines.append(f"| Empty outputs | {empty_outputs} | {pct(empty_outputs, total_requests):.1f}% |")
    
    # Failures (if available)
    if failures_available:
        if refusals > 0:
            lines.append(f"| Refusals | {refusals} | {pct(refusals, total_requests):.1f}% |")
        if timeouts > 0:
            lines.append(f"| Timeouts | {timeouts} | {pct(timeouts, total_requests):.1f}% |")
        if engine_errors > 0:
            lines.append(f"| Engine errors | {engine_errors} | {pct(engine_errors, total_requests):.1f}% |")
        if format_errors > 0:
            lines.append(f"| Format errors | {format_errors} | {pct(format_errors, total_requests):.1f}% |")
        if truncated > 0:
            lines.append(f"| Truncated | {truncated} | {pct(truncated, total_requests):.1f}% |")
    else:
        lines.append("| Failures (by type) | N/A | N/A | *Failures table missing* |")
    
    # Missing telemetry
    if telemetry_available:
        lines.append(f"| Missing telemetry | {missing_telemetry} | {pct(missing_telemetry, total_requests):.1f}% |")
    else:
        lines.append("| Missing telemetry | N/A | N/A | *Telemetry table missing* |")
    
    lines.append("")
    lines.append("**Total requests:** " + str(total_requests))
    lines.append("")
    
    # Interpretation notes
    lines.append("### Interpretation")
    lines.append("")
    
    total_failures = empty_outputs + refusals + timeouts + engine_errors + format_errors + truncated
    failure_rate = pct(total_failures, total_requests)
    
    if failure_rate > 10.0:
        lines.append("- **High failure rate detected** (" + f"{failure_rate:.1f}%): Failure rate may be driving accuracy collapse rather than quality degradation.")
    elif failure_rate > 5.0:
        lines.append("- **Moderate failure rate** (" + f"{failure_rate:.1f}%): Monitor failure patterns across bins to distinguish system failures from quality issues.")
    else:
        lines.append("- **Low failure rate** (" + f"{failure_rate:.1f}%): Failures are unlikely to be the primary driver of accuracy changes.")
    
    if empty_outputs > 0:
        empty_pct = pct(empty_outputs, total_requests)
        lines.append(f"- **Empty outputs** ({empty_pct:.1f}%): May indicate KV compression truncation or model refusal; check correlation with failure rate spikes.")
    
    if refusals > 0 or timeouts > 0 or engine_errors > 0:
        lines.append("- **Infrastructure failures** (refusals/timeouts/engine errors): Distinguish from compression-induced failures; these should be retried or excluded from analysis.")
    
    lines.append("")
    
    return "\n".join(lines)


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
    
    # Compute and append Integrity Summary
    integrity_section = _compute_integrity_summary(conn, exp_group_id)
    rendered_with_integrity = rendered + "\n\n" + integrity_section
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered_with_integrity, encoding="utf-8")
    
    return report_path
