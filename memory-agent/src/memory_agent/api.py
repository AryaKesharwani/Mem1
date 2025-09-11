"""
Memory Agent API - Main entry points for the memory system.

This module provides the two core API functions:
1. handle() - Query memory systems and return context
2. post_write() - Write conversation data to appropriate memory systems
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from .agent import MemoryAgent
from .config import get_config


# Global memory agent instance (initialized lazily)
_memory_agent: Optional[MemoryAgent] = None


def _get_memory_agent() -> MemoryAgent:
    """Get or create the global memory agent instance."""
    global _memory_agent
    if _memory_agent is None:
        _memory_agent = MemoryAgent()
    return _memory_agent


async def handle(
    query: str,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    topic_hint: Optional[str] = None,
    user_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Handle a query by routing to appropriate memory systems and returning merged context.
    
    This is the main entry point for querying the memory agent. It:
    1. Analyzes the query to determine which memory systems to use
    2. Queries relevant memory systems in parallel (STM, Semantic, Episodic, RAG)
    3. Merges and ranks results within token budget
    4. Returns context + routing metadata
    
    Args:
        query: The user's query text to find relevant context for
        tenant_id: Organization identifier for multi-tenant data isolation
        user_id: User identifier within the organization
        agent_id: Agent identifier for this conversation session
        conversation_id: Unique identifier for this conversation thread
        topic_hint: Optional hint about query topic for better routing
        user_metadata: Optional user context (preferences, current state, etc.)
        
    Returns:
        Dict containing:
        - merged_context: String with all relevant context merged and formatted
        - context_blocks: List of individual context blocks from each memory system
        - routing_decision: Details about which systems were queried and why
        - query_metadata: Performance metrics and system information
        
    Example:
        >>> result = await handle(
        ...     query="Plan a Paris trip for next weekend",
        ...     tenant_id="org1",
        ...     user_id="user123",
        ...     agent_id="travel_agent",
        ...     conversation_id="conv_456"
        ... )
        >>> print(result["merged_context"])
        [STM] Previous conversation about travel preferences...
        [SEMANTIC] User prefers boutique hotels and local cuisine...
        [RAG] Paris travel guide: Top attractions include...
    """
    # Input validation
    if not all([query, tenant_id, user_id, agent_id, conversation_id]):
        raise ValueError("Missing required parameters: query, tenant_id, user_id, agent_id, conversation_id")
    
    # Get memory agent instance
    memory_agent = _get_memory_agent()
    
    # Handle the query
    try:
        result = await memory_agent.handle_query(
            query=query.strip(),
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            topic_hint=topic_hint,
            user_metadata=user_metadata or {}
        )
        
        # Log successful query
        logger = logging.getLogger(__name__)
        logger.info(f"Query handled for user {user_id} in conversation {conversation_id}")
        
        return result
        
    except Exception as e:
        # Log error but don't expose internal details
        logger = logging.getLogger(__name__)
        logger.error(f"Error in handle(): {str(e)}")
        
        # Return minimal safe response
        return {
            "merged_context": "",
            "context_blocks": [],
            "routing_decision": {
                "use_stm": True,
                "confidence": 0.0,
                "reasoning": "Error occurred during query processing"
            },
            "query_metadata": {
                "error": "Internal error occurred",
                "timestamp": None
            }
        }


async def post_write(
    user_text: str,
    assistant_text: str,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    topic_hint: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Write user and assistant messages to appropriate memory systems.
    
    This function analyzes the conversation turn and intelligently writes to:
    - STM: Always updated with conversation context
    - Semantic: If stable facts/preferences are detected (confidence > threshold)
    - Episodic: If significant events/experiences are mentioned
    - RAG: If new documents or structured information is provided
    
    Args:
        user_text: The user's message text
        assistant_text: The assistant's response text
        tenant_id: Organization identifier for data isolation
        user_id: User identifier within the organization
        agent_id: Agent identifier for this conversation
        conversation_id: Unique conversation identifier
        topic_hint: Optional topic hint for better categorization
        metadata: Optional metadata to store with the memory (tags, importance, etc.)
        
    Returns:
        Dict containing:
        - write_results: Results from each memory system that was written to
        - write_decision: Details about which systems were written to and why
        - write_metadata: Performance metrics and system information
        
    Example:
        >>> result = await post_write(
        ...     user_text="I prefer riverside hotels over beach-side ones",
        ...     assistant_text="Noted! I'll update your hotel preferences...",
        ...     tenant_id="org1",
        ...     user_id="user123",
        ...     agent_id="travel_agent",
        ...     conversation_id="conv_456",
        ...     topic_hint="hotel_preferences"
        ... )
        >>> print(result["write_decision"]["write_semantic"])  # True - preference detected
    """
    # Input validation
    if not all([user_text, assistant_text, tenant_id, user_id, agent_id, conversation_id]):
        raise ValueError("Missing required parameters: user_text, assistant_text, tenant_id, user_id, agent_id, conversation_id")
    
    # Get memory agent instance
    memory_agent = _get_memory_agent()
    
    # Handle the write
    try:
        result = await memory_agent.handle_write(
            user_text=user_text.strip(),
            assistant_text=assistant_text.strip(),
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            topic_hint=topic_hint,
            metadata=metadata or {}
        )
        
        # Log successful write
        logger = logging.getLogger(__name__)
        logger.info(f"Write completed for user {user_id} in conversation {conversation_id}")
        
        return result
        
    except Exception as e:
        # Log error but don't expose internal details
        logger = logging.getLogger(__name__)
        logger.error(f"Error in post_write(): {str(e)}")
        
        # Return minimal safe response
        return {
            "write_results": {},
            "write_decision": {
                "write_stm": True,  # At minimum, try to write to STM
                "error": "Error occurred during write processing"
            },
            "write_metadata": {
                "error": "Internal error occurred",
                "timestamp": None
            }
        }


# Synchronous wrapper functions for backwards compatibility
def handle_sync(
    query: str,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    topic_hint: Optional[str] = None,
    user_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for handle() function.
    
    Use this when you can't use async/await in your calling code.
    For better performance, prefer the async handle() function.
    """
    return asyncio.run(handle(
        query, tenant_id, user_id, agent_id, conversation_id,
        topic_hint, user_metadata
    ))


def post_write_sync(
    user_text: str,
    assistant_text: str,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    topic_hint: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for post_write() function.
    
    Use this when you can't use async/await in your calling code.
    For better performance, prefer the async post_write() function.
    """
    return asyncio.run(post_write(
        user_text, assistant_text, tenant_id, user_id, agent_id,
        conversation_id, topic_hint, metadata
    ))


# Health check and utility functions
async def health_check() -> Dict[str, Any]:
    """
    Check the health status of the memory agent system.
    
    Returns:
        Dict with system status and configuration info
    """
    try:
        config = get_config()
        memory_agent = _get_memory_agent()
        
        return {
            "status": "healthy",
            "config": {
                "systems_enabled": {
                    "stm": config.enable_stm,
                    "semantic": config.enable_semantic,
                    "episodic": config.enable_episodic,
                    "rag": config.enable_rag
                },
                "token_budget": config.default_token_budget,
                "llm_arbiter_enabled": config.enable_llm_arbiter
            },
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }


def get_system_info() -> Dict[str, Any]:
    """
    Get information about the memory agent system configuration.
    
    Returns:
        Dict with system configuration and capabilities
    """
    config = get_config()
    
    return {
        "version": "1.0.0",
        "memory_systems": {
            "stm": {
                "enabled": config.enable_stm,
                "window_size": config.stm_window_size,
                "token_allocation": config.stm_token_allocation
            },
            "semantic": {
                "enabled": config.enable_semantic,
                "confidence_threshold": config.semantic_confidence_threshold,
                "token_allocation": config.semantic_token_allocation
            },
            "episodic": {
                "enabled": config.enable_episodic,
                "confidence_threshold": config.episodic_confidence_threshold,
                "token_allocation": config.episodic_token_allocation
            },
            "rag": {
                "enabled": config.enable_rag,
                "relevance_threshold": config.rag_relevance_threshold,
                "token_allocation": config.rag_token_allocation,
                "chunk_size": config.chunk_size
            }
        },
        "routing": {
            "llm_arbiter_enabled": config.enable_llm_arbiter,
            "arbiter_threshold": config.arbiter_confidence_threshold,
            "llm_model": config.llm_model
        },
        "data_paths": {
            "data_root": config.data_root
        }
    }