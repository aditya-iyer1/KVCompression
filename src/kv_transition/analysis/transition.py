"""Transition zone detection for KV budget stability analysis.

Identifies the "instability transition zone" from bin-level accuracy curves.
"""

from typing import Dict, List, Optional


def _compute_overall_acc(bins: List[Dict]) -> Optional[float]:
    """Compute overall accuracy as mean of bin acc_mean values.
    
    Ignores bins with missing acc_mean.
    
    Args:
        bins: List of bin dicts with acc_mean.
    
    Returns:
        Overall accuracy (mean of acc_mean values) or None if no valid bins.
    """
    acc_values = []
    for bin_stat in bins:
        acc_mean = bin_stat.get('acc_mean')
        if acc_mean is not None:
            acc_values.append(float(acc_mean))
    
    if not acc_values:
        return None
    
    return sum(acc_values) / len(acc_values)


def _find_largest_bin_drop(pre_bins: List[Dict], post_bins: List[Dict]) -> Optional[int]:
    """Find the bin_idx where the largest per-bin drop occurs.
    
    Args:
        pre_bins: List of bin dicts for pre-transition budget.
        post_bins: List of bin dicts for post-transition budget.
    
    Returns:
        bin_idx with largest drop, or None if no valid comparison.
    """
    # Create lookup by bin_idx
    pre_by_bin = {bin_stat['bin_idx']: bin_stat.get('acc_mean') for bin_stat in pre_bins if 'bin_idx' in bin_stat}
    post_by_bin = {bin_stat['bin_idx']: bin_stat.get('acc_mean') for bin_stat in post_bins if 'bin_idx' in bin_stat}
    
    # Find common bins
    common_bins = set(pre_by_bin.keys()) & set(post_by_bin.keys())
    
    if not common_bins:
        return None
    
    # Find largest drop
    largest_drop = 0.0
    transition_bin_idx = None
    
    for bin_idx in common_bins:
        pre_acc = pre_by_bin[bin_idx]
        post_acc = post_by_bin[bin_idx]
        
        if pre_acc is not None and post_acc is not None:
            drop = pre_acc - post_acc
            if drop > largest_drop:
                largest_drop = drop
                transition_bin_idx = bin_idx
    
    return transition_bin_idx


def detect_transition(
    runs: List[Dict],
    *,
    drop_threshold: float = 0.15
) -> Dict:
    """Detect transition zone from bin-level accuracy curves.
    
    Identifies the first budget where overall accuracy drops by >= drop_threshold
    relative to the previous higher budget.
    
    Args:
        runs: List of run dicts, each with:
            - run_id
            - kv_budget
            - bins: list of bin dicts with bin_idx, acc_mean
        drop_threshold: Minimum drop in overall accuracy to trigger transition (default: 0.15).
    
    Returns:
        Dict with:
        - transition_budget: Budget where drop occurs (or None)
        - pre_budget: Previous budget (or None)
        - acc_pre: Overall accuracy before transition (or None)
        - acc_post: Overall accuracy after transition (or None)
        - drop: Absolute drop in accuracy (or None)
        - transition_bin_idx: Bin with largest per-bin drop (or None)
        - method: String describing the detection method
    """
    if not runs:
        return {
            'transition_budget': None,
            'pre_budget': None,
            'acc_pre': None,
            'acc_post': None,
            'drop': None,
            'transition_bin_idx': None,
            'method': 'overall_mean_drop'
        }
    
    # Sort budgets descending (1.0 -> smaller)
    sorted_runs = sorted(runs, key=lambda r: r.get('kv_budget', 0.0), reverse=True)
    
    # Compute overall accuracy for each budget
    budget_accs = []
    for run in sorted_runs:
        kv_budget = run.get('kv_budget')
        bins = run.get('bins', [])
        overall_acc = _compute_overall_acc(bins)
        
        if overall_acc is not None:
            budget_accs.append({
                'budget': kv_budget,
                'acc': overall_acc,
                'bins': bins,
                'run_id': run.get('run_id')
            })
    
    if len(budget_accs) < 2:
        # Need at least 2 budgets to detect transition
        return {
            'transition_budget': None,
            'pre_budget': None,
            'acc_pre': None,
            'acc_post': None,
            'drop': None,
            'transition_bin_idx': None,
            'method': 'overall_mean_drop'
        }
    
    # Find first budget where drop >= threshold
    for i in range(1, len(budget_accs)):
        pre = budget_accs[i - 1]
        post = budget_accs[i]
        
        acc_pre = pre['acc']
        acc_post = post['acc']
        drop = acc_pre - acc_post
        
        if drop >= drop_threshold:
            # Found transition
            transition_bin_idx = _find_largest_bin_drop(pre['bins'], post['bins'])
            
            return {
                'transition_budget': post['budget'],
                'pre_budget': pre['budget'],
                'acc_pre': acc_pre,
                'acc_post': acc_post,
                'drop': drop,
                'transition_bin_idx': transition_bin_idx,
                'method': 'overall_mean_drop'
            }
    
    # No transition detected
    return {
        'transition_budget': None,
        'pre_budget': None,
        'acc_pre': None,
        'acc_post': None,
        'drop': None,
        'transition_bin_idx': None,
        'method': 'overall_mean_drop'
    }
