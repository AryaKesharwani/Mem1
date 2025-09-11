"""
Memory Agent - Multi-system memory orchestrator for AI agents.

This package provides a configurable memory system that coordinates between:
- Short-term Memory (STM): Conversation context and working memory
- Semantic Memory: Facts, preferences, and stable knowledge
- Episodic Memory: Events, experiences, and temporal context
- RAG Memory: Document-based knowledge retrieval

Main entry points:
- handle(): Query memory systems for context
- post_write(): Write conversation data to memory systems
"""

from .agent import MemoryAgent
from .config import MemoryConfig, get_config
from .models import (
    MemoryType,
    MemoryRecord,
    ContextBlock,
    RoutingDecision,
    QueryContext,
    WriteContext,
    MemoryResponse,
    SemanticFact,
    EpisodicEvent,
    DocumentChunk
)
from .api import handle, post_write, handle_sync, post_write_sync, health_check, get_system_info

__version__ = "1.0.0"
__author__ = "Memory Agent Team"

# Public API exports
__all__ = [
    # Main API functions
    "handle",
    "post_write", 
    "handle_sync",
    "post_write_sync",
    "health_check",
    "get_system_info",
    
    # Core classes
    "MemoryAgent",
    "MemoryConfig",
    "get_config",
    
    # Data models
    "MemoryType",
    "MemoryRecord",
    "ContextBlock", 
    "RoutingDecision",
    "QueryContext",
    "WriteContext",
    "MemoryResponse",
    "SemanticFact",
    "EpisodicEvent",
    "DocumentChunk",
    
    # Package metadata
    "__version__",
    "__author__"
]