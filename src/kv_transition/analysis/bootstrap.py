"""Bootstrap confidence intervals for bin-level statistics.

Provides percentile bootstrap CIs for mean accuracy (F1 mean).
"""

import random
import sqlite3
from typing import Dict, List, Optional, Tuple, Union


def bootstrap_mean_ci(
    values: List[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 1337
) -> Tuple[Optional[float], Optional[float]]:
    """Compute percentile bootstrap confidence interval for the mean.
    
    Resamples with replacement from values, computes mean for each resample,
    then takes percentiles of the bootstrap distribution.
    
    Args:
        values: List of numeric values.
        n_boot: Number of bootstrap resamples (default: 1000).
        alpha: Significance level (default: 0.05 for 95% CI).
        seed: Random seed for deterministic behavior (default: 1337).
    
    Returns:
        Tuple of (ci_low, ci_high) or (None, None) if values is empty.
    """
    if not values:
        return (None, None)
    
    # Set seed for deterministic behavior
    rng = random.Random(seed)
    
    # Bootstrap resampling
    bootstrap_means = []
    n = len(values)
    
    for _ in range(n_boot):
        # Resample with replacement
        resample = [rng.choice(values) for _ in range(n)]
        # Compute mean of resample
        bootstrap_mean = sum(resample) / n
        bootstrap_means.append(bootstrap_mean)
    
    # Sort bootstrap means
    bootstrap_means.sort()
    
    # Compute percentile bounds
    lower_percentile = alpha / 2
    upper_percentile = 1 - alpha / 2
    
    # Get indices for percentiles
    lower_idx = int(lower_percentile * n_boot)
    upper_idx = int(upper_percentile * n_boot)
    
    # Clamp indices to valid range
    lower_idx = max(0, min(lower_idx, n_boot - 1))
    upper_idx = max(0, min(upper_idx, n_boot - 1))
    
    ci_low = bootstrap_means[lower_idx]
    ci_high = bootstrap_means[upper_idx]
    
    return (ci_low, ci_high)


def add_bootstrap_cis(
    bin_stats: List[Dict],
    per_request_rows: Union[List[Dict], List[sqlite3.Row]],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 1337
) -> List[Dict]:
    """Add bootstrap confidence intervals to bin_stats.
    
    For each bin in bin_stats, extracts F1 values from per_request_rows
    and computes CI for mean accuracy.
    
    Args:
        bin_stats: List of bin stat dicts (from aggregate_run_bins).
        per_request_rows: List of per-request rows (from get_bin_level_rows).
        n_boot: Number of bootstrap resamples (default: 1000).
        alpha: Significance level (default: 0.05 for 95% CI).
        seed: Random seed for deterministic behavior (default: 1337).
    
    Returns:
        Updated bin_stats list with acc_ci_low and acc_ci_high keys added.
    """
    # Extract F1 values by bin_idx from per_request_rows
    f1_by_bin: Dict[int, List[float]] = {}
    
    for row in per_request_rows:
        bin_idx = row['bin_idx'] if isinstance(row, dict) else row['bin_idx']
        f1 = row['f1'] if isinstance(row, dict) else row['f1']
        
        if bin_idx is not None and f1 is not None:
            if bin_idx not in f1_by_bin:
                f1_by_bin[bin_idx] = []
            f1_by_bin[bin_idx].append(float(f1))
    
    # Add CIs to each bin_stat
    result = []
    for bin_stat in bin_stats:
        bin_idx = bin_stat['bin_idx']
        f1_values = f1_by_bin.get(bin_idx, [])
        
        # Compute CI
        ci_low, ci_high = bootstrap_mean_ci(f1_values, n_boot=n_boot, alpha=alpha, seed=seed)
        
        # Create updated dict
        updated_stat = bin_stat.copy()
        updated_stat['acc_ci_low'] = ci_low
        updated_stat['acc_ci_high'] = ci_high
        
        result.append(updated_stat)
    
    return result
