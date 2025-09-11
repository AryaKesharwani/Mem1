"""
Shared utilities for the Memory Agent system.

Contains common functions used across multiple memory modules to avoid code duplication.
"""

import hashlib
import struct
import math
import logging
from typing import Optional, List, Dict
from .config import MemoryConfig


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Args:
        text: Input text to estimate tokens for
        
    Returns:
        Estimated number of tokens (rough approximation: 1 token ≈ 4 characters)
    """
    if not text:
        return 0
    
    # Rough approximation: 1 token ≈ 4 characters for English text
    return max(1, len(text) // 4)


_EMBEDDING_MODEL_CACHE: Dict[str, object] = {}


async def generate_embedding(text: str, config: MemoryConfig) -> Optional[List[float]]:
    """
    Generate embedding for text using the configured embedding model.
    
    Args:
        text: Text to generate embedding for
        config: Memory configuration containing embedding settings
        
    Returns:
        List of floats representing the embedding, or None if generation fails
    """
    if not text or not text.strip():
        return None
    
    try:
        # Try to use sentence-transformers for real embeddings
        try:
            from sentence_transformers import SentenceTransformer

            # Cache model instance per embedding_model for performance
            model = _EMBEDDING_MODEL_CACHE.get(config.embedding_model)
            if model is None:
                model = SentenceTransformer(config.embedding_model)
                _EMBEDDING_MODEL_CACHE[config.embedding_model] = model

            # Generate embedding
            embedding = model.encode(text, convert_to_tensor=False)
            
            # Convert to list and ensure correct dimension
            embedding_list = embedding.tolist()
            
            # Pad or truncate to desired dimension if needed
            target_dim = config.vector_dimension
            if len(embedding_list) != target_dim:
                if len(embedding_list) > target_dim:
                    embedding_list = embedding_list[:target_dim]
                else:
                    # Pad with zeros
                    embedding_list.extend([0.0] * (target_dim - len(embedding_list)))
            
            return embedding_list
            
        except ImportError:
            logger = logging.getLogger(__name__)
            logger.warning("sentence-transformers not available, using fallback embedding")
            
            # Fallback to deterministic dummy embedding based on text hash
            text_hash = hashlib.md5(text.encode()).digest()
            
            # Convert hash to list of floats
            embedding = []
            for i in range(0, len(text_hash), 4):
                chunk = text_hash[i:i+4]
                if len(chunk) == 4:
                    value = struct.unpack('f', chunk)[0]
                    embedding.append(float(value))
            
            # Pad or truncate to desired dimension
            target_dim = config.vector_dimension
            while len(embedding) < target_dim:
                embedding.extend(embedding[:min(len(embedding), target_dim - len(embedding))])
            
            embedding = embedding[:target_dim]
            
            # Normalize the embedding
            norm = math.sqrt(sum(x*x for x in embedding))
            if norm > 0:
                embedding = [x/norm for x in embedding]
            
            return embedding
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Embedding generation failed: {str(e)}")
        return None


def generate_record_id(content: str, prefix: str = "") -> str:
    """
    Generate a unique ID for a record based on its content.
    
    Args:
        content: Content to generate ID from
        prefix: Optional prefix for the ID
        
    Returns:
        Unique string ID
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{prefix}_{content_hash}" if prefix else content_hash


def format_time_ago(days_ago: int) -> str:
    """
    Format a time difference in human-readable format.
    
    Args:
        days_ago: Number of days ago
        
    Returns:
        Human-readable time string
    """
    if days_ago == 0:
        return "today"
    elif days_ago == 1:
        return "yesterday"
    elif days_ago < 7:
        return f"{days_ago} days ago"
    elif days_ago < 30:
        weeks_ago = days_ago // 7
        return f"{weeks_ago} week{'s' if weeks_ago > 1 else ''} ago"
    elif days_ago < 365:
        months_ago = days_ago // 30
        return f"{months_ago} month{'s' if months_ago > 1 else ''} ago"
    else:
        years_ago = days_ago // 365
        return f"{years_ago} year{'s' if years_ago > 1 else ''} ago"


def truncate_text(text: str, max_tokens: int, preserve_sentences: bool = True) -> str:
    """
    Truncate text to fit within a token budget.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens allowed
        preserve_sentences: Whether to try to preserve sentence boundaries
        
    Returns:
        Truncated text
    """
    if not text:
        return text
    
    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text
    
    # Calculate approximate character limit
    char_limit = max_tokens * 4  # Rough approximation
    
    if not preserve_sentences or len(text) <= char_limit:
        return text[:char_limit] + "..."
    
    # Try to find a good sentence boundary
    truncated = text[:char_limit]
    
    # Look for sentence endings near the cut point
    for boundary in ['. ', '! ', '? ']:
        last_boundary = truncated.rfind(boundary)
        if last_boundary > char_limit * 0.8:  # Don't cut too much
            return truncated[:last_boundary + 1] + "..."
    
    # No good boundary found, use character limit
    return truncated + "..."
