"""Bin-level aggregation for Phase E statistics computation.

Computes summary statistics per bin for a run (accuracy, std, failure rate, latency percentiles).
"""

import math
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional


def _percentile(values: List[float], p: float) -> Optional[float]:
    """Compute percentile of a list of values.
    
    Uses simple sorted-list approach. Returns None if values is empty.
    
    Args:
        values: List of numeric values (must be non-empty).
        p: Percentile (0.0 to 1.0, e.g., 0.5 for median, 0.95 for 95th).
    
    Returns:
        Percentile value or None if values is empty.
    """
    if not values:
        return None
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    # Linear interpolation method
    index = (n - 1) * p
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    
    if lower == upper:
        return sorted_vals[lower]
    
    weight = index - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def _mean(values: List[float]) -> Optional[float]:
    """Compute mean of values.
    
    Args:
        values: List of numeric values.
    
    Returns:
        Mean value or None if values is empty.
    """
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: List[float]) -> Optional[float]:
    """Compute population standard deviation of values.
    
    Args:
        values: List of numeric values.
    
    Returns:
        Standard deviation or None if values is empty or has only one value.
    """
    if len(values) < 2:
        return None
    
    mean_val = _mean(values)
    if mean_val is None:
        return None
    
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def aggregate_run_bins(conn: sqlite3.Connection, run_id: str) -> List[Dict]:
    """Compute bin-level aggregates for a run.
    
    Groups requests by bin_idx and computes summary statistics per bin.
    
    Args:
        conn: SQLite connection.
        run_id: Run identifier.
    
    Returns:
        List of dicts, one per bin, with keys:
        - bin_idx
        - n (count of requests)
        - acc_mean (mean F1 score)
        - acc_std (std of F1 scores)
        - em_mean (mean EM score, optional)
        - fail_rate (proportion of failures)
        - lat_p50 (50th percentile latency)
        - lat_p95 (95th percentile latency)
        - tok_p50 (50th percentile total tokens)
        - tok_p95 (95th percentile total tokens)
        Sorted by bin_idx.
    """
    from .queries import get_bin_level_rows
    
    # Fetch bin-level rows
    rows = get_bin_level_rows(conn, run_id)
    
    if not rows:
        return []
    
    # Group by bin_idx
    bins_data: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {
        'f1': [],
        'em': [],
        'failure': [],
        'latency_s': [],
        'total_tokens': []
    })
    
    for row in rows:
        bin_idx = row['bin_idx']
        if bin_idx is None:
            continue
        
        # Collect F1 scores
        if row['f1'] is not None:
            bins_data[bin_idx]['f1'].append(row['f1'])
        
        # Collect EM scores
        if row['em'] is not None:
            bins_data[bin_idx]['em'].append(row['em'])
        
        # Collect failure indicators
        if row['failure'] is not None:
            bins_data[bin_idx]['failure'].append(float(row['failure']))
        
        # Collect latency values (ignore None)
        if row['latency_s'] is not None:
            bins_data[bin_idx]['latency_s'].append(row['latency_s'])
        
        # Collect token counts (ignore None)
        if row['total_tokens'] is not None:
            bins_data[bin_idx]['total_tokens'].append(float(row['total_tokens']))
    
    # Compute aggregates per bin
    results = []
    for bin_idx in sorted(bins_data.keys()):
        data = bins_data[bin_idx]
        
        n = len(data['f1']) if data['f1'] else 0
        
        # Accuracy metrics (F1 as primary)
        acc_mean = _mean(data['f1']) if data['f1'] else None
        acc_std = _std(data['f1']) if data['f1'] else None
        em_mean = _mean(data['em']) if data['em'] else None
        
        # Failure rate
        fail_rate = _mean(data['failure']) if data['failure'] else None
        
        # Latency percentiles
        lat_p50 = _percentile(data['latency_s'], 0.5) if data['latency_s'] else None
        lat_p95 = _percentile(data['latency_s'], 0.95) if data['latency_s'] else None
        
        # Token percentiles
        tok_p50 = _percentile(data['total_tokens'], 0.5) if data['total_tokens'] else None
        tok_p95 = _percentile(data['total_tokens'], 0.95) if data['total_tokens'] else None
        
        result = {
            'bin_idx': bin_idx,
            'n': n,
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'em_mean': em_mean,
            'fail_rate': fail_rate,
            'lat_p50': lat_p50,
            'lat_p95': lat_p95,
            'tok_p50': tok_p50,
            'tok_p95': tok_p95
        }
        
        results.append(result)
    
    return results
