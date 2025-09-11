"""
Single reranker service used by router and RAG.

Provides context reranking and merging functionality with token budget management.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from .config import MemoryConfig
from .models import MemoryType, MemoryRecord
from .utils import estimate_tokens


class ContextReranker:
    """
    Single reranker service for context blocks from multiple memory systems.
    
    Handles token budget allocation, relevance scoring, and context formatting
    to provide the most useful information within constraints.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize the context reranker."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def merge_context(
        self,
        context_blocks: List[Dict[str, Any]],
        token_budget: int,
        query: str = ""
    ) -> str:
        """
        Merge and rank context blocks within token budget.
        
        Args:
            context_blocks: List of context blocks from memory systems
            token_budget: Maximum tokens allowed for merged context
            query: Original query for relevance scoring
            
        Returns:
            Merged context string within token budget
        """
        if not context_blocks:
            return ""
        
        try:
            # Score and sort blocks by relevance
            scored_blocks = []
            for block in context_blocks:
                score = self._calculate_block_score(block, query)
                scored_blocks.append((block, score))
            
            # Sort by score (descending)
            scored_blocks.sort(key=lambda x: x[1], reverse=True)
            
            # Select blocks within token budget
            selected_blocks = self._select_within_budget(
                [block for block, _ in scored_blocks], 
                token_budget
            )
            
            # Format final context
            return self._format_merged_context(selected_blocks)
            
        except Exception as e:
            self.logger.error(f"Context merging failed: {e}")
            # Fallback to simple merge
            return self._simple_merge(context_blocks, token_budget)
    
    def _calculate_block_score(self, block: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for a context block."""
        base_score = 0.5
        
        # Memory type weights
        memory_type = block.get("memory_type", "").lower()
        type_weights = {
            "stm": 1.0,      # Recent context is very important
            "semantic": 0.8,  # Facts are generally useful
            "episodic": 0.7,  # Past events are somewhat important
            "rag": 0.9       # Documents are very relevant when matched
        }
        
        base_score *= type_weights.get(memory_type, 0.5)
        
        # Confidence and importance boosts
        confidence = block.get("confidence", 0.5)
        importance = block.get("importance", 0.5)
        relevance = block.get("relevance_score", 0.5)
        
        # Combine scores
        final_score = base_score * (0.4 * confidence + 0.3 * importance + 0.3 * relevance)
        
        # Lexical overlap bonus if query provided
        if query and block.get("content"):
            overlap_score = self._calculate_lexical_overlap(query, block["content"])
            final_score += 0.1 * overlap_score
        
        return min(1.0, final_score)
    
    def _calculate_lexical_overlap(self, query: str, content: str) -> float:
        """Calculate lexical overlap between query and content."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _select_within_budget(
        self, 
        blocks: List[Dict[str, Any]], 
        token_budget: int
    ) -> List[Dict[str, Any]]:
        """Select context blocks that fit within token budget."""
        selected_blocks = []
        used_tokens = 0
        
        for block in blocks:
            block_tokens = block.get("token_count", estimate_tokens(block.get("content", "")))
            formatted_tokens = block_tokens + 20  # Buffer for formatting
            
            if used_tokens + formatted_tokens <= token_budget:
                selected_blocks.append(block)
                used_tokens += formatted_tokens
            else:
                # Try truncated version if there's space
                remaining_tokens = token_budget - used_tokens - 20
                if remaining_tokens > 50:
                    truncated_block = self._truncate_block(block, remaining_tokens)
                    if truncated_block:
                        selected_blocks.append(truncated_block)
                break
        
        return selected_blocks
    
    def _truncate_block(self, block: Dict[str, Any], max_tokens: int) -> Optional[Dict[str, Any]]:
        """Truncate a context block to fit within token limit."""
        content = block.get("content", "")
        if not content:
            return None
        
        max_chars = max_tokens * 4  # Rough approximation
        if len(content) <= max_chars:
            return block
        
        # Try to truncate at sentence boundary
        truncated = content[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        if last_period > max_chars * 0.7:
            truncated = truncated[:last_period + 1]
        elif last_newline > max_chars * 0.7:
            truncated = truncated[:last_newline]
        
        truncated += "... [truncated]"
        
        # Create truncated block
        truncated_block = block.copy()
        truncated_block["content"] = truncated
        truncated_block["token_count"] = estimate_tokens(truncated)
        truncated_block["truncated"] = True
        
        return truncated_block
    
    def _format_merged_context(self, blocks: List[Dict[str, Any]]) -> str:
        """Format selected context blocks into final context string."""
        sections = []
        
        for block in blocks:
            memory_type = block.get("memory_type", "UNKNOWN").upper()
            content = block.get("content", "").strip()
            
            if content:
                sections.append(f"[{memory_type}]\n{content}")
        
        return "\n\n".join(sections)
    
    def _simple_merge(self, context_blocks: List[Dict[str, Any]], token_budget: int) -> str:
        """Simple fallback merge when scoring fails."""
        sections = []
        used_tokens = 0
        
        for block in context_blocks:
            memory_type = block.get("memory_type", "UNKNOWN").upper()
            content = block.get("content", "").strip()
            
            if not content:
                continue
            
            # Estimate tokens for this section
            section_tokens = estimate_tokens(f"[{memory_type}]\n{content}") + 5
            
            if used_tokens + section_tokens <= token_budget:
                sections.append(f"[{memory_type}]\n{content}")
                used_tokens += section_tokens
            else:
                # Try to fit truncated version
                remaining_tokens = token_budget - used_tokens - 20
                if remaining_tokens > 50:
                    max_chars = remaining_tokens * 4
                    truncated_content = content[:max_chars] + "... [truncated]"
                    sections.append(f"[{memory_type}]\n{truncated_content}")
                break
        
        return "\n\n".join(sections)
