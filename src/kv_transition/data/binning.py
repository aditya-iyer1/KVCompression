"""Binning utilities for token length-based example assignment.

Phase B: Assigns examples to bins based on context length for balanced evaluation.
"""

from typing import List, Tuple


def compute_bin_edges(lengths: List[int], n_bins: int) -> List[Tuple[int, int]]:
    """Compute bin edges using quantile-style splits.
    
    Partitions the sorted lengths into roughly balanced bins and returns
    inclusive ranges (min_len, max_len) for each bin.
    
    Args:
        lengths: List of token lengths (non-empty).
        n_bins: Number of bins to create.
    
    Returns:
        List of (min_len, max_len) tuples, one per bin.
        Ranges are inclusive on both ends.
    
    Raises:
        ValueError: If lengths is empty or n_bins < 1.
    """
    if not lengths:
        raise ValueError("lengths must be non-empty")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    
    # Sort lengths for quantile-style splitting
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    
    # Handle edge case: more bins than examples
    if n_bins >= n:
        # Each example gets its own bin (or some bins are empty)
        edges = []
        for i in range(n_bins):
            if i < n:
                # Single value bin
                edges.append((sorted_lengths[i], sorted_lengths[i]))
            else:
                # Empty bin - use last value as placeholder
                edges.append((sorted_lengths[-1], sorted_lengths[-1]))
        return edges
    
    # Compute quantile-style splits
    # Divide sorted lengths into roughly equal-sized groups
    bin_size = n / n_bins
    edges = []
    
    for i in range(n_bins):
        # Compute start and end indices for this bin
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)
        
        # Ensure last bin includes all remaining items
        if i == n_bins - 1:
            end_idx = n
        
        # Get min and max for this bin
        bin_min = sorted_lengths[start_idx]
        bin_max = sorted_lengths[end_idx - 1]
        
        edges.append((bin_min, bin_max))
    
    return edges


def assign_bin(length: int, edges: List[Tuple[int, int]]) -> int:
    """Assign a length to a bin index.
    
    Args:
        length: Token length to assign.
        edges: List of (min_len, max_len) bin edges from compute_bin_edges.
    
    Returns:
        Bin index in [0, len(edges)-1].
        Returns the first bin where length falls within [min, max] (inclusive).
        If length is between bins, assigns to the bin with the closest boundary.
        If length is less than all bins, returns 0.
        If length is greater than all bins, returns last bin index.
    """
    if not edges:
        raise ValueError("edges must be non-empty")
    
    # Check each bin (in order) for exact match
    for bin_idx, (min_len, max_len) in enumerate(edges):
        if min_len <= length <= max_len:
            return bin_idx
    
    # Handle length between bins: find closest bin boundary
    first_min, _ = edges[0]
    _, last_max = edges[-1]
    
    if length < first_min:
        return 0
    elif length > last_max:
        return len(edges) - 1
    
    # Length is between bins - find closest bin
    # Use the bin whose min is closest (ties go to lower bin)
    best_bin = 0
    min_distance = abs(length - edges[0][0])
    
    for bin_idx, (min_len, max_len) in enumerate(edges):
        # Check distance to both boundaries
        dist_to_min = abs(length - min_len)
        dist_to_max = abs(length - max_len)
        min_dist = min(dist_to_min, dist_to_max)
        
        if min_dist < min_distance:
            min_distance = min_dist
            best_bin = bin_idx
    
    return best_bin


def bin_examples(token_lens: List[int], n_bins: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Bin examples by token length.
    
    Args:
        token_lens: List of token lengths (one per example, in order).
        n_bins: Number of bins to create.
    
    Returns:
        Tuple of (bin_idxs, edges):
        - bin_idxs: List of bin indices, one per input length (aligned to input order).
        - edges: List of (min_len, max_len) tuples for each bin.
    
    Raises:
        ValueError: If token_lens is empty or n_bins < 1.
    """
    if not token_lens:
        raise ValueError("token_lens must be non-empty")
    
    # Compute bin edges
    edges = compute_bin_edges(token_lens, n_bins)
    
    # Assign each length to a bin
    bin_idxs = [assign_bin(length, edges) for length in token_lens]
    
    return (bin_idxs, edges)
