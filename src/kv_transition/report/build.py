"""Report generation from persisted DB outputs and plot files.

Generates markdown reports by rendering Jinja templates with data from the database.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def get_scored_count(conn: sqlite3.Connection, exp_group_id: str) -> int:
    """Return number of scored requests (scores joined to runs) for this exp_group_id."""
    if not _table_exists(conn, "scores") or not _table_exists(conn, "runs"):
        return 0
    try:
        cursor = conn.execute("""
            SELECT COUNT(*) FROM scores s
            JOIN requests r ON r.request_id = s.request_id
            JOIN runs ru ON r.run_id = ru.run_id
            WHERE ru.exp_group_id = ?
        """, (exp_group_id,))
        row = cursor.fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def validate_report_prerequisites(
    db_path: Path,
    run_dir: Path,
    exp_group_id: str,
    allow_partial: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate prerequisites for report generation.
    
    When the run group has 0 scored rows, acc_by_bin and latency_p50_by_bin are not required;
    report can succeed and will show a no-scorable-responses note.
    When scored_count > 0, all expected plots are required unless allow_partial is True.
    
    Returns:
        (is_valid, list_of_missing_items).
    """
    missing: List[str] = []
    if not db_path.exists():
        missing.append(f"Database file missing: {db_path}")
        return (allow_partial, missing) if allow_partial else (False, missing)
    scored_count = 0
    conn = None
    try:
        from ..db import connect, schema
        conn = connect.connect(db_path)
        schema.init_schema(conn)
        scored_count = get_scored_count(conn, exp_group_id)
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='bin_stats'
        """)
        if cursor.fetchone() is None:
            missing.append("Database table 'bin_stats' missing (run 'analyze' command first)")
        else:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM bin_stats bs
                JOIN runs r ON bs.run_id = r.run_id
                WHERE r.exp_group_id = ?
            """, (exp_group_id,))
            row = cursor.fetchone()
            count = row[0] if row else 0
            if count == 0:
                missing.append(f"No bin_stats data found for exp_group_id '{exp_group_id}' (run 'analyze' command first)")
    except Exception as e:
        missing.append(f"Error checking database: {e}")
    finally:
        if conn is not None:
            conn.close()
    plots_dir = run_dir / "plots"
    if not plots_dir.exists():
        missing.append(f"Plots directory missing: {plots_dir}")
    else:
        if scored_count == 0:
            # Edge case: no scorable responses; do not require any plot files
            required_plots = []
        else:
            required_plots = ["acc_by_bin.png", "fail_by_bin.png", "latency_p50_by_bin.png"]
        for plot_file in required_plots:
            plot_path = plots_dir / plot_file
            if not plot_path.exists():
                missing.append(f"Plot file missing: {plot_path}")
    if not allow_partial and missing:
        return (False, missing)
    return (True, missing)


def _compute_integrity_summary(conn: sqlite3.Connection, exp_group_id: str) -> str:
    """Compute integrity summary metrics and return as markdown section.
    
    Uses failures.error_type and responses.text. Empty outputs = responses with
    blank/NULL text only (no double-count with failures). Failure count from
    failures table; breakdown by error_type.
    """
    def pct(count: int, total: int) -> float:
        if total == 0:
            return 0.0
        return (count / total) * 100.0

    def row_val(r, key: str, fallback_idx: int = 0):
        try:
            return r[key] if key in r.keys() else (r[fallback_idx] if len(r) > fallback_idx else 0)
        except (KeyError, TypeError, IndexError):
            return 0

    # total_requests from requests
    try:
        cursor = conn.execute("""
            SELECT COUNT(*) AS c FROM requests r
            JOIN runs ru ON r.run_id = ru.run_id
            WHERE ru.exp_group_id = ?
        """, (exp_group_id,))
        row = cursor.fetchone()
        total_requests = row_val(row, "c", 0)
    except sqlite3.OperationalError:
        return "## Integrity Summary\n\n**Status:** Requests table not available → N/A\n"

    if total_requests == 0:
        return "## Integrity Summary\n\n**Status:** No requests found for this experiment.\n"

    n_failures = 0
    error_type_breakdown: List[tuple] = []  # (error_type, count)
    if _table_exists(conn, "failures"):
        try:
            cursor = conn.execute("""
                SELECT COUNT(*) AS c FROM failures f
                JOIN requests r ON f.request_id = r.request_id
                JOIN runs ru ON r.run_id = ru.run_id
                WHERE ru.exp_group_id = ?
            """, (exp_group_id,))
            n_failures = row_val(cursor.fetchone(), "c", 0)
            cursor = conn.execute("""
                SELECT f.error_type, COUNT(*) AS count
                FROM failures f
                JOIN requests r ON f.request_id = r.request_id
                JOIN runs ru ON r.run_id = ru.run_id
                WHERE ru.exp_group_id = ?
                GROUP BY f.error_type
            """, (exp_group_id,))
            for r in cursor.fetchall():
                et = row_val(r, "error_type", 0)
                cnt = row_val(r, "count", 1)
                if et is None or et == "":
                    et = "(empty)"
                error_type_breakdown.append((et, cnt))
        except sqlite3.OperationalError:
            pass

    # Empty outputs: only from responses.text (blank/NULL) — do not double-count with failures
    empty_outputs = 0
    if _table_exists(conn, "responses"):
        try:
            cursor = conn.execute("""
                SELECT COUNT(*) AS c FROM responses res
                JOIN requests r ON res.request_id = r.request_id
                JOIN runs ru ON r.run_id = ru.run_id
                WHERE ru.exp_group_id = ?
                AND (res.text IS NULL OR res.text = '' OR TRIM(res.text) = '')
            """, (exp_group_id,))
            empty_outputs = row_val(cursor.fetchone(), "c", 0)
        except sqlite3.OperationalError:
            pass

    missing_telemetry = 0
    telemetry_available = _table_exists(conn, "telemetry")
    if telemetry_available:
        try:
            cursor = conn.execute("""
                SELECT COUNT(*) AS c FROM requests r
                JOIN runs ru ON r.run_id = ru.run_id
                LEFT JOIN telemetry t ON r.request_id = t.request_id
                WHERE ru.exp_group_id = ? AND t.request_id IS NULL
            """, (exp_group_id,))
            missing_telemetry = row_val(cursor.fetchone(), "c", 0)
        except sqlite3.OperationalError:
            telemetry_available = False

    # Build markdown
    lines = ["## Integrity Summary", ""]
    lines.append("| Metric | Count | Percentage |")
    lines.append("|--------|-------|------------|")

    lines.append(f"| Failures | {n_failures} | {pct(n_failures, total_requests):.1f}% |")
    for et, cnt in error_type_breakdown:
        lines.append(f"| — {et} | {cnt} | {pct(cnt, total_requests):.1f}% |")
    lines.append(f"| Empty outputs (responses.text blank/NULL) | {empty_outputs} | {pct(empty_outputs, total_requests):.1f}% |")
    if telemetry_available:
        lines.append(f"| Missing telemetry | {missing_telemetry} | {pct(missing_telemetry, total_requests):.1f}% |")
    else:
        lines.append("| Missing telemetry | N/A | N/A | *Telemetry table missing* |")

    lines.append("")
    lines.append("**Total requests:** " + str(total_requests))
    lines.append("")
    lines.append("### Interpretation")
    lines.append("")

    failure_rate = pct(n_failures, total_requests)
    if failure_rate > 10.0:
        lines.append("- **High failure rate** (" + f"{failure_rate:.1f}%): May be driving accuracy collapse rather than quality degradation.")
    elif failure_rate > 5.0:
        lines.append("- **Moderate failure rate** (" + f"{failure_rate:.1f}%): Monitor failure types across bins.")
    else:
        lines.append("- **Low failure rate** (" + f"{failure_rate:.1f}%): Unlikely to be the main driver of accuracy changes.")
    if empty_outputs > 0:
        lines.append(f"- **Empty outputs** ({pct(empty_outputs, total_requests):.1f}%): Check KV/truncation or refusal; correlate with failure rate.")
    if n_failures > 0 and error_type_breakdown:
        lines.append("- **Failure breakdown** above; retry or exclude infrastructure errors (timeouts, context length) from analysis.")
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
    
    scored_count = get_scored_count(conn, exp_group_id)
    no_scorable_responses = scored_count == 0
    context = {
        "exp_group_id": exp_group_id,
        "experiment": experiment,
        "runs": runs,
        "transition": transition,
        "plots": plots,
        "generated_at": datetime.utcnow().isoformat(),
        "no_scorable_responses": no_scorable_responses,
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
