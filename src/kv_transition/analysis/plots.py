"""Matplotlib plotting utilities for bin-level visualization.

Generates accuracy, failure rate, and latency curves per KV budget.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _resolve_output_dir(settings: Dict, out_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve output directory for plots.
    
    Args:
        settings: Settings dict with exp_group_id.
        out_dir: Optional explicit output directory.
    
    Returns:
        Path to plots directory.
    """
    if out_dir:
        return Path(out_dir)
    
    from .. import paths
    
    exp_group_id = settings.get("output", {}).get("exp_group_id")
    if not exp_group_id:
        raise ValueError("exp_group_id not found in settings")
    
    run_dir = paths.run_dir(exp_group_id)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return plots_dir


def _extract_bin_data(runs: List[Dict], metric_key: str) -> Dict[float, Dict[int, float]]:
    """Extract bin-level data for a metric, grouped by kv_budget.
    
    Args:
        runs: List of run dicts with kv_budget and bins.
        metric_key: Key to extract from bin dicts (e.g., 'acc_mean', 'fail_rate').
    
    Returns:
        Dict mapping kv_budget -> dict mapping bin_idx -> metric value.
    """
    data_by_budget = {}
    
    for run in runs:
        kv_budget = run.get('kv_budget')
        if kv_budget is None:
            continue
        
        bins = run.get('bins', [])
        data_by_budget[kv_budget] = {}
        
        for bin_stat in bins:
            bin_idx = bin_stat.get('bin_idx')
            metric_value = bin_stat.get(metric_key)
            
            if bin_idx is not None and metric_value is not None:
                data_by_budget[kv_budget][bin_idx] = float(metric_value)
    
    return data_by_budget


def _plot_metric_by_bin(
    runs: List[Dict],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path
) -> Optional[Path]:
    """Plot a metric vs bin_idx with one line per kv_budget.
    
    Args:
        runs: List of run dicts.
        metric_key: Key to extract from bin dicts.
        ylabel: Y-axis label.
        title: Plot title.
        output_path: Path to save plot.
    
    Returns:
        Path to saved plot, or None if matplotlib not available or no data.
    """
    if plt is None:
        return None
    
    # Extract data by budget
    data_by_budget = _extract_bin_data(runs, metric_key)
    
    if not data_by_budget:
        return None
    
    # Sort budgets (ascending for consistent ordering)
    sorted_budgets = sorted(data_by_budget.keys())
    
    # Get all bin indices
    all_bin_indices = set()
    for budget_data in data_by_budget.values():
        all_bin_indices.update(budget_data.keys())
    
    if not all_bin_indices:
        return None
    
    sorted_bin_indices = sorted(all_bin_indices)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot one line per budget
    for kv_budget in sorted_budgets:
        budget_data = data_by_budget[kv_budget]
        
        # Extract values for this budget, in bin_idx order
        x_values = []
        y_values = []
        
        for bin_idx in sorted_bin_indices:
            if bin_idx in budget_data:
                x_values.append(bin_idx)
                y_values.append(budget_data[bin_idx])
        
        if x_values and y_values:
            ax.plot(x_values, y_values, marker='o', label=f'KV budget {kv_budget}')
    
    ax.set_xlabel('Bin Index')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def save_run_plots(
    settings: Dict,
    runs: List[Dict],
    *,
    out_dir: Optional[Union[str, Path]] = None
) -> List[Path]:
    """Save bin-level plots for runs.
    
    Generates accuracy, failure rate, and latency plots (if data available).
    
    Args:
        settings: Settings dict with exp_group_id.
        runs: List of run dicts, each with:
            - kv_budget
            - bins: list of bin dicts with bin_idx, acc_mean, fail_rate, lat_p50
        out_dir: Optional explicit output directory.
    
    Returns:
        List of paths to saved plot files.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Resolve output directory
    plots_dir = _resolve_output_dir(settings, out_dir)
    
    saved_paths = []
    
    # Plot 1: Accuracy vs bin_idx
    acc_path = plots_dir / "acc_by_bin.png"
    result = _plot_metric_by_bin(
        runs,
        metric_key='acc_mean',
        ylabel='Mean Accuracy (F1)',
        title='Accuracy by Bin Index',
        output_path=acc_path
    )
    if result:
        saved_paths.append(result)
    
    # Plot 2: Failure rate vs bin_idx
    fail_path = plots_dir / "fail_by_bin.png"
    result = _plot_metric_by_bin(
        runs,
        metric_key='fail_rate',
        ylabel='Failure Rate',
        title='Failure Rate by Bin Index',
        output_path=fail_path
    )
    if result:
        saved_paths.append(result)
    
    # Plot 3: Latency p50 vs bin_idx (if data available)
    # Check if any run has lat_p50 data
    has_latency_data = False
    for run in runs:
        bins = run.get('bins', [])
        for bin_stat in bins:
            if bin_stat.get('lat_p50') is not None:
                has_latency_data = True
                break
        if has_latency_data:
            break
    
    if has_latency_data:
        lat_path = plots_dir / "latency_p50_by_bin.png"
        result = _plot_metric_by_bin(
            runs,
            metric_key='lat_p50',
            ylabel='Latency P50 (seconds)',
            title='Latency P50 by Bin Index',
            output_path=lat_path
        )
        if result:
            saved_paths.append(result)
    
    return saved_paths
