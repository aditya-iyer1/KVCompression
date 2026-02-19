"""Evaluation metrics for comparing model outputs to gold answers.

Provides Exact Match (EM) and F1 score computation with text normalization.
"""

import re
from typing import List


def normalize_text(s: str) -> str:
    """Normalize text for comparison.
    
    Applies lowercase, trims whitespace, collapses multiple whitespace to single spaces,
    and removes minimal punctuation for robust matching.
    
    Args:
        s: Input text string.
    
    Returns:
        Normalized text string.
    """
    if not s:
        return ""
    
    # Lowercase
    normalized = s.lower()
    
    # Remove leading/trailing whitespace
    normalized = normalized.strip()
    
    # Collapse multiple whitespace (spaces, tabs, newlines) to single spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove common punctuation that might cause mismatches
    # Keep alphanumeric and spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Final trim
    normalized = normalized.strip()
    
    return normalized


def exact_match(pred: str, gold: str) -> float:
    """Compute exact match score.
    
    Returns 1.0 if normalized prediction exactly matches normalized gold answer,
    otherwise 0.0.
    
    Args:
        pred: Prediction string.
        gold: Gold answer string.
    
    Returns:
        1.0 for exact match, 0.0 otherwise.
    """
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    
    # Both empty is a match
    if not pred_norm and not gold_norm:
        return 1.0
    
    # One empty, one not, is not a match
    if not pred_norm or not gold_norm:
        return 0.0
    
    return 1.0 if pred_norm == gold_norm else 0.0


def f1_score(pred: str, gold: str) -> float:
    """Compute F1 score based on token overlap.
    
    Uses whitespace tokenization and computes precision, recall, and F1
    after normalization.
    
    Args:
        pred: Prediction string.
        gold: Gold answer string.
    
    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    
    # Both empty: F1 = 1.0
    if not pred_norm and not gold_norm:
        return 1.0
    
    # One empty, one not: F1 = 0.0
    if not pred_norm or not gold_norm:
        return 0.0
    
    # Tokenize (whitespace split)
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # Count overlaps
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    
    # Common tokens
    common = pred_set & gold_set
    num_common = len(common)
    
    if num_common == 0:
        return 0.0
    
    # Precision: common / pred_tokens
    precision = num_common / len(pred_tokens)
    
    # Recall: common / gold_tokens
    recall = num_common / len(gold_tokens)
    
    # F1: harmonic mean
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def best_exact_match(pred: str, golds: List[str]) -> float:
    """Compute best exact match score over multiple gold answers.
    
    Returns the maximum exact match score when comparing prediction
    against all gold answers.
    
    Args:
        pred: Prediction string.
        golds: List of gold answer strings.
    
    Returns:
        Maximum exact match score (0.0 to 1.0).
    """
    if not golds:
        return 0.0
    
    scores = [exact_match(pred, gold) for gold in golds]
    return max(scores)


def best_f1(pred: str, golds: List[str]) -> float:
    """Compute best F1 score over multiple gold answers.
    
    Returns the maximum F1 score when comparing prediction
    against all gold answers.
    
    Args:
        pred: Prediction string.
        golds: List of gold answer strings.
    
    Returns:
        Maximum F1 score (0.0 to 1.0).
    """
    if not golds:
        return 0.0
    
    scores = [f1_score(pred, gold) for gold in golds]
    return max(scores)
