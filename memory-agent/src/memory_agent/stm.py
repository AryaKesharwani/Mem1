"""
Short-term Memory (STM) - Working memory and conversation context.

Provides temporary storage for active conversations, working memory,
and recent context that needs to be quickly accessible using Redis.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import deque

from .config import MemoryConfig
from .models import MemoryRecord, MemoryType, ContextBlock
from .utils import estimate_tokens

# Try to import Redis, fall back to in-memory if not available
try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class STMBuffer:
    """
    Ring buffer for short-term memory storage.
    
    Maintains a sliding window of recent conversation turns,
    user context, and working memory within a fixed size limit.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize the STM buffer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ring buffer for conversation turns
        self.conversation_buffer = deque(maxlen=config.stm_window_size)
        
        # Working memory for temporary state
        self.working_memory = {}
        
        # Metadata tracking
        self.last_updated = datetime.now(timezone.utc)
        self.total_turns = 0
    
    def add_conversation_turn(
        self,
        user_text: str,
        assistant_text: str,
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation turn to the buffer.
        
        Args:
            user_text: User's message
            assistant_text: Assistant's response
            conversation_id: Conversation identifier
            metadata: Optional metadata for the turn
        """
        turn_data = {
            "turn_id": self.total_turns + 1,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc),
            "metadata": metadata or {}
        }
        
        self.conversation_buffer.append(turn_data)
        self.total_turns += 1
        self.last_updated = datetime.now(timezone.utc)
        
        self.logger.debug(f"Added conversation turn {self.total_turns} to STM buffer")
    
    def get_recent_context(
        self,
        max_turns: Optional[int] = None,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation context.
        
        Args:
            max_turns: Maximum number of turns to return
            conversation_id: Filter by specific conversation
            
        Returns:
            List of recent conversation turns
        """
        # Filter by conversation if specified
        if conversation_id:
            filtered_turns = [
                turn for turn in self.conversation_buffer
                if turn.get("conversation_id") == conversation_id
            ]
        else:
            filtered_turns = list(self.conversation_buffer)
        
        # Limit number of turns
        if max_turns:
            filtered_turns = filtered_turns[-max_turns:]
        
        return filtered_turns
    
    def update_working_memory(self, key: str, value: Any) -> None:
        """Update working memory with key-value pair."""
        self.working_memory[key] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc)
        }
        self.last_updated = datetime.now(timezone.utc)
        
        self.logger.debug(f"Updated working memory: {key}")
    
    def get_working_memory(self, key: Optional[str] = None) -> Any:
        """Get value from working memory."""
        if key:
            entry = self.working_memory.get(key)
            return entry["value"] if entry else None
        else:
            return {k: v["value"] for k, v in self.working_memory.items()}
    
    def clear_working_memory(self, key: Optional[str] = None) -> None:
        """Clear working memory (specific key or all)."""
        if key:
            self.working_memory.pop(key, None)
        else:
            self.working_memory.clear()
        
        self.last_updated = datetime.now(timezone.utc)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffer state."""
        total_tokens = 0
        for entry in self.conversation_buffer:
            total_tokens += estimate_tokens(entry.get("user_text", "")) + estimate_tokens(entry.get("assistant_text", ""))
        return {
            "buffer_size": len(self.conversation_buffer),
            "max_size": self.conversation_buffer.maxlen,
            "total_tokens": total_tokens,
            "working_memory_keys": len(self.working_memory),
            "last_updated": self.last_updated.isoformat()
        }


class STMManager:
    """
    Manager for short-term memory operations.
    
    Coordinates multiple STM buffers for different conversations
    and provides query/write interface for the memory agent.
    Uses Redis backend if available, falls back to in-memory.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize STM manager with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Always initialize memory backend first
        self._init_memory_backend()
        
        # Use Redis STM if available and enabled, otherwise in-memory
        if REDIS_AVAILABLE and getattr(config, 'use_redis_stm', True):
            try:
                # Initialize Redis client from config
                self._backend: Optional[Redis] = Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    password=config.redis_password,
                    ssl=config.redis_ssl,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                # Test connection
                self._backend.ping()
                self._use_redis = True
                self.logger.info("STMManager initialized with Redis backend")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis STM, falling back to in-memory: {e}")
                self._backend = None
                self._use_redis = False
        else:
            self._backend = None
            self._use_redis = False

    def _conv_key(self, tenant_id: str, user_id: str, conversation_id: str) -> str:
        prefix = getattr(self.config, 'redis_prefix', 'memory_agent')
        return f"{prefix}:stm:conv:{tenant_id}:{user_id}:{conversation_id}"

    def _wm_key(self, tenant_id: str, user_id: str, conversation_id: str) -> str:
        prefix = getattr(self.config, 'redis_prefix', 'memory_agent')
        return f"{prefix}:stm:wm:{tenant_id}:{user_id}:{conversation_id}"
    
    def _init_memory_backend(self):
        """Initialize in-memory backend."""
        # Dictionary of conversation buffers
        self.buffers: Dict[str, STMBuffer] = {}
        
        # Track last cleanup time
        self.last_cleanup = datetime.now(timezone.utc)
        
        self.logger.info("STMManager initialized with in-memory backend")
    
    async def query_stm(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        max_turns: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Query short-term memory for conversation context."""
        if self._use_redis:
            return await self._query_redis_backend(query, tenant_id, user_id, conversation_id, max_turns)
        else:
            return await self._query_memory_backend(query, tenant_id, user_id, conversation_id, max_turns)
    
    async def query_conversation_history(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query conversation history from STM."""
        if self._use_redis:
            return await self._query_redis_history(conversation_id, tenant_id, user_id, limit)
        else:
            return await self._query_memory_history(conversation_id, tenant_id, user_id, limit)
    
    async def write_conversation_history(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
        user_text: str,
        assistant_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Write conversation turn to STM."""
        if self._use_redis:
            return await self._write_redis_backend(conversation_id, tenant_id, user_id, user_text, assistant_text, metadata)
        else:
            return await self._write_memory_backend(conversation_id, tenant_id, user_id, user_text, assistant_text, metadata)
    
    async def _query_memory_backend(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        max_turns: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query short-term memory for relevant context using in-memory backend.
        
        Args:
            query: The query to find relevant context for
            tenant_id: Tenant identifier
            user_id: User identifier
            conversation_id: Conversation identifier
            max_turns: Maximum conversation turns to include
            
        Returns:
            Context from STM or None if no relevant context found
        """
        buffer_key = f"{tenant_id}:{user_id}:{conversation_id}"
        
        # Debug logging
        self.logger.info(f"STM Query - Looking for buffer key: {buffer_key}")
        self.logger.info(f"STM Query - Available buffers: {list(self.buffers.keys())}")
        
        # Get or create buffer for this conversation
        if buffer_key not in self.buffers:
            self.logger.warning(f"STM Query - Buffer not found for key: {buffer_key}")
            return None
        
        buffer = self.buffers[buffer_key]
        max_turns = max_turns or self.config.stm_max_turns
        
        # Get recent conversation turns
        recent_turns = list(buffer.conversation_buffer)[-max_turns:]
        
        self.logger.info(f"STM Query - Found {len(recent_turns)} turns in buffer")
        
        if not recent_turns:
            return None
        
        # Format turns for context
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn['user_text']}")
            context_parts.append(f"Assistant: {turn['assistant_text']}")
        
        return {
            "system": "stm",
            "memory_type": "stm",
            "content": "\n".join(context_parts),
            "metadata": {"source": "stm", "turn_count": len(recent_turns)},
            "token_count": sum(len(turn['user_text']) + len(turn['assistant_text']) for turn in recent_turns) // 4,
            # Provide defaults used by reranker scoring
            "confidence": 1.0,
            "importance": 0.8,
            "relevance_score": 0.9
        }
    
    async def _query_memory_history(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query conversation history using in-memory backend."""
        buffer_key = f"{tenant_id}:{user_id}:{conversation_id}"
        
        if buffer_key not in self.buffers:
            return []
        
        buffer = self.buffers[buffer_key]
        limit = limit or self.config.stm_max_turns
        
        # Return recent turns
        recent_turns = list(buffer.conversation_buffer)[-limit:]
        return recent_turns
    
    async def _write_memory_backend(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
        user_text: str,
        assistant_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Write to in-memory backend."""
        buffer_key = f"{tenant_id}:{user_id}:{conversation_id}"
        
        # Get or create buffer for this conversation
        if buffer_key not in self.buffers:
            self.buffers[buffer_key] = STMBuffer(self.config)
        
        buffer = self.buffers[buffer_key]
        
        # Add conversation turn
        buffer.add_conversation_turn(
            user_text=user_text,
            assistant_text=assistant_text,
            conversation_id=conversation_id,
            metadata=metadata or {}
        )
        
        # Periodic cleanup
        await self._cleanup_expired_buffers()
        
        return True
    
    async def _cleanup_expired_buffers(self):
        """Clean up expired STM buffers to prevent memory leaks."""
        if not hasattr(self, 'buffers'):
            return
            
        now = datetime.now(timezone.utc)
        
        # Skip cleanup if not enough time has passed
        if (now - self.last_cleanup).total_seconds() < 300:  # 5 minutes
            return
        
        # Remove buffers that haven't been updated in cache TTL
        expired_keys = []
        for key, buffer in self.buffers.items():
            if (now - buffer.last_updated).total_seconds() > self.config.stm_cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.buffers[key]
            self.logger.debug(f"Cleaned up expired STM buffer: {key}")
        
        self.last_cleanup = now
        
        if expired_keys:
            self.logger.info(f"STM cleanup: removed {len(expired_keys)} expired buffers")
    
    async def write_stm(
        self,
        user_text: str,
        assistant_text: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Write conversation turn to short-term memory.
        
        Args:
            user_text: User's message
            assistant_text: Assistant's response
            tenant_id: Tenant identifier
            user_id: User identifier
            conversation_id: Conversation identifier
            metadata: Optional metadata for the turn
            
        Returns:
            Write result with status and metadata
        """
        try:
            # Get or create STM buffer
            buffer_key = f"{tenant_id}:{user_id}:{conversation_id}"
            
            # Get or create buffer for this conversation
            if not hasattr(self, 'buffers'):
                self.buffers = {}
                
            if buffer_key not in self.buffers:
                self.buffers[buffer_key] = STMBuffer(self.config)
            
            buffer = self.buffers[buffer_key]
            
            # Add conversation turn
            buffer.add_conversation_turn(
                user_text=user_text,
                assistant_text=assistant_text,
                conversation_id=conversation_id,
                metadata=metadata or {}
            )
            
            # Update working memory if metadata contains updates
            if metadata and "working_memory_updates" in metadata:
                for key, value in metadata["working_memory_updates"].items():
                    buffer.update_working_memory(key, value)
            
            # Periodic cleanup
            await self._cleanup_expired_buffers()
            
            return {
                "status": "success",
                "system": "stm",
                "turns_stored": buffer.total_turns,
                "buffer_size": len(buffer.conversation_buffer)
            }
            
        except Exception as e:
            self.logger.error(f"STM write failed: {str(e)}")
            return {
                "status": "error",
                "system": "stm",
                "error": str(e)
            }
    
    def _get_buffer(self, tenant_id: str, user_id: str, conversation_id: str) -> STMBuffer:
        """Get or create STM buffer for the given identifiers."""
        buffer_key = f"{tenant_id}:{user_id}:{conversation_id}"
        
        if not hasattr(self, 'buffers'):
            self.buffers = {}
            
        if buffer_key not in self.buffers:
            self.buffers[buffer_key] = STMBuffer(self.config)
            self.logger.debug(f"Created new STM buffer for {buffer_key}")
        
        return self.buffers[buffer_key]
    
    def _format_conversation_context(self, turns: List[Dict[str, Any]]) -> str:
        """Format conversation turns into readable context."""
        if not turns:
            return ""
        
        formatted_turns = []
        
        for turn in turns:
            turn_text = f"User: {turn['user_text']}\nAssistant: {turn['assistant_text']}"
            
            # Add timestamp if available
            if "timestamp" in turn:
                timestamp = turn["timestamp"]
                if isinstance(timestamp, datetime):
                    time_str = timestamp.strftime("%H:%M")
                    turn_text = f"[{time_str}] {turn_text}"
            
            formatted_turns.append(turn_text)
        
        return "\n\n".join(formatted_turns)
    
    def _format_working_memory(self, working_memory: Dict[str, Any]) -> str:
        """Format working memory into readable text."""
        if not working_memory:
            return ""
        
        formatted_items = []
        for key, value in working_memory.items():
            formatted_items.append(f"- {key}: {value}")
        
        return "\n".join(formatted_items)
    
    def _create_stm_records(
        self,
        turns: List[Dict[str, Any]],
        tenant_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Create memory records from conversation turns."""
        records = []
        
        for turn in turns:
            # Create record content from the turn
            content = f"User: {turn['user_text']}\nAssistant: {turn['assistant_text']}"
            
            # Create record for the turn
            record = {
                "id": f"stm_{turn.get('turn_id', 0)}",
                "content": content,
                "memory_type": MemoryType.STM.value,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "conversation_id": turn.get("conversation_id"),
                "confidence": 1.0,  # STM is always high confidence
                "importance": 0.8,  # Recent context is important
                "token_count": estimate_tokens(content),
                "relevance_score": 0.9,  # Assume high relevance for recent context
                "created_at": turn.get("timestamp", datetime.now(timezone.utc)),
                "metadata": turn.get("metadata", {})
            }
            
            records.append(record)
        
        return records
    
    def _calculate_recency_score(self, timestamp: Optional[datetime]) -> float:
        """Calculate recency score based on timestamp."""
        if not timestamp:
            return 1.0
        
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # Calculate hours since timestamp
        hours_ago = (now - timestamp).total_seconds() / 3600
        
        # Exponential decay: score = e^(-hours/24)
        # Recent messages (< 1 hour) get score close to 1.0
        # Messages from 24 hours ago get score around 0.37
        import math
        recency_score = math.exp(-hours_ago / 24)
        
        return max(0.1, min(1.0, recency_score))
    
    async def _write_redis_backend(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
        user_text: str,
        assistant_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Write a conversation turn and working memory to Redis backend."""
        if not self._backend:
            return await self._write_memory_backend(conversation_id, tenant_id, user_id, user_text, assistant_text, metadata)
        try:
            turn = {
                "user_text": user_text,
                "assistant_text": assistant_text,
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}
            }
            list_key = self._conv_key(tenant_id, user_id, conversation_id)
            ttl = int(getattr(self.config, 'stm_cache_ttl', 3600))
            window = int(getattr(self.config, 'stm_window_size', 10))

            pipe = self._backend.pipeline()
            pipe.rpush(list_key, json.dumps(turn))
            # Keep only the last N items (window)
            pipe.ltrim(list_key, -window, -1)
            pipe.expire(list_key, ttl)

            # Working memory updates (optional)
            if metadata and isinstance(metadata.get("working_memory_updates"), dict):
                wm_key = self._wm_key(tenant_id, user_id, conversation_id)
                for k, v in metadata["working_memory_updates"].items():
                    pipe.hset(wm_key, k, json.dumps({"value": v, "timestamp": datetime.now(timezone.utc).isoformat()}))
                pipe.expire(wm_key, ttl)

            pipe.execute()
            return True
        except Exception as e:
            self.logger.warning(f"Redis STM write failed, falling back to memory: {e}")
            return await self._write_memory_backend(conversation_id, tenant_id, user_id, user_text, assistant_text, metadata)
    
    async def _query_redis_backend(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        max_turns: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Query recent conversation context from Redis backend."""
        if not self._backend:
            return await self._query_memory_backend(query, tenant_id, user_id, conversation_id, max_turns)
        try:
            list_key = self._conv_key(tenant_id, user_id, conversation_id)
            max_turns = max_turns or self.config.stm_max_turns
            # Get last N turns
            raw_items = self._backend.lrange(list_key, -max_turns, -1)
            if not raw_items:
                return None
            turns = []
            for item in raw_items:
                try:
                    turns.append(json.loads(item))
                except Exception:
                    continue

            if not turns:
                return None

            # Format context
            context_parts = []
            for t in turns:
                context_parts.append(f"User: {t.get('user_text','')}")
                context_parts.append(f"Assistant: {t.get('assistant_text','')}")

            content = "\n".join(context_parts)
            token_count = sum(len(t.get('user_text','')) + len(t.get('assistant_text','')) for t in turns) // 4
            return {
                "system": "stm",
                "memory_type": "stm",
                "content": content,
                "metadata": {"source": "stm", "turn_count": len(turns)},
                "token_count": token_count,
                "confidence": 1.0,
                "importance": 0.8,
                "relevance_score": 0.9
            }
        except Exception as e:
            self.logger.warning(f"Redis STM query failed, falling back to memory: {e}")
            return await self._query_memory_backend(query, tenant_id, user_id, conversation_id, max_turns)
    
    async def _query_redis_history(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query full conversation history from Redis backend (bounded by limit)."""
        if not self._backend:
            return await self._query_memory_history(conversation_id, tenant_id, user_id, limit)
        try:
            list_key = self._conv_key(tenant_id, user_id, conversation_id)
            limit = limit or self.config.stm_max_turns
            raw_items = self._backend.lrange(list_key, -limit, -1)
            turns: List[Dict[str, Any]] = []
            for item in raw_items:
                try:
                    t = json.loads(item)
                    turns.append(t)
                except Exception:
                    continue
            return turns
        except Exception as e:
            self.logger.warning(f"Redis STM history failed, falling back to memory: {e}")
            return await self._query_memory_history(conversation_id, tenant_id, user_id, limit)
    


# Convenience functions for direct STM operations
async def query_short_term_memory(
    query: str,
    tenant_id: str,
    user_id: str,
    conversation_id: str,
    config: Optional[MemoryConfig] = None,
    max_turns: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Convenience function to query STM directly."""
    from .config import get_config
    
    config = config or get_config()
    stm_manager = STMManager(config)
    
    return await stm_manager.query_stm(
        query, tenant_id, user_id, conversation_id, max_turns
    )


async def write_short_term_memory(
    user_text: str,
    assistant_text: str,
    tenant_id: str,
    user_id: str,
    conversation_id: str,
    config: Optional[MemoryConfig] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to write to STM directly."""
    from .config import get_config
    
    config = config or get_config()
    stm_manager = STMManager(config)
    
    return await stm_manager.write_stm(
        user_text, assistant_text, tenant_id, user_id, conversation_id, metadata
    )
