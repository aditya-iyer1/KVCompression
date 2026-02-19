"""Tokenizer utility for token counting.

Phase B: Provides token counting for binning consistency.
"""

from typing import Dict, Optional


def get_tokenizer_name(settings: Dict) -> str:
    """Get tokenizer name from settings, falling back to model name.
    
    Args:
        settings: Settings dict (from load_settings).
    
    Returns:
        Tokenizer name string (from tokenizer.name or model.name).
    """
    # Try tokenizer.name first
    tokenizer_name = settings.get("tokenizer", {}).get("name")
    if tokenizer_name and tokenizer_name.strip():
        return tokenizer_name.strip()
    
    # Fall back to model.name
    model_name = settings.get("model", {}).get("name")
    if model_name and model_name.strip():
        return model_name.strip()
    
    # Last resort: return empty string (caller should handle)
    return ""


def count_tokens(text: str, tokenizer_name: Optional[str] = None) -> int:
    """Count tokens in text using tokenizer or deterministic fallback.
    
    Args:
        text: Text to count tokens for.
        tokenizer_name: Optional tokenizer name (e.g., "gpt-4", "cl100k_base").
                       If None or tiktoken unavailable, uses fallback.
    
    Returns:
        Token count (integer, stable across runs).
    
    Note:
        Uses tiktoken if available and tokenizer_name is provided.
        Otherwise uses deterministic whitespace-based approximation.
    """
    if not text:
        return 0
    
    # Try tiktoken if available and tokenizer_name provided
    if tokenizer_name:
        try:
            import tiktoken
            
            # Map common model names to encodings
            encoding_map = {
                "gpt-4": "cl100k_base",
                "gpt-3.5-turbo": "cl100k_base",
                "gpt-4o": "o200k_base",
                "gpt-4-turbo": "cl100k_base",
                "gpt-3.5-turbo-16k": "cl100k_base",
            }
            
            # Try to get encoding
            encoding_name = encoding_map.get(tokenizer_name.lower(), tokenizer_name)
            
            try:
                encoding = tiktoken.get_encoding(encoding_name)
                return len(encoding.encode(text))
            except (KeyError, ValueError):
                # Encoding not found, fall through to fallback
                pass
        except ImportError:
            # tiktoken not available, fall through to fallback
            pass
    
    # Deterministic fallback: whitespace-based approximation
    # This is stable and reproducible, though less accurate than real tokenization
    # Formula: split on whitespace + estimate punctuation/characters
    words = text.split()
    base_count = len(words)
    
    # Add rough estimate for punctuation and special characters
    # Count non-whitespace, non-alphanumeric characters
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    # Rough approximation: ~3-4 chars per token for English
    estimated_tokens = base_count + (special_chars // 3)
    
    return max(1, estimated_tokens)  # At least 1 token for non-empty text
