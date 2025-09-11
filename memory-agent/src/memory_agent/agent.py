import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone

from .config import get_config, MemoryConfig


class MemoryAgent:
    """
    Main orchestrator for the memory agent system.
    Coordinates between different memory systems and handles routing decisions.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory agent with configuration."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Memory system instances (will be initialized when needed)
        self._stm_store = None          # Short-term memory cache
        self._semantic_store = None     # Semantic facts and preferences
        self._episodic_store = None     # Episodic events and experiences
        self._rag_store = None          # RAG document store
        
        # Core components (will be initialized when needed)
        self._router = None             # Routing decision engine
        self._arbiter = None            # LLM-based routing fallback
        self._reranker = None           # Cross-system result reranker
        
        self.logger.info("MemoryAgent initialized with config")
    
    async def handle_query(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        topic_hint: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle a query by routing to appropriate memory systems and returning context.
        
        Args:
            query: The user's query text
            tenant_id: Organization identifier for multi-tenancy
            user_id: User identifier within the organization
            agent_id: Agent identifier for this conversation
            conversation_id: Unique conversation identifier
            topic_hint: Optional hint about the query topic
            user_metadata: Optional user context metadata
            
        Returns:
            Dict containing merged context, routing decisions, and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Step 1: Create query context
            query_context = self._create_query_context(
                query, tenant_id, user_id, agent_id, conversation_id,
                topic_hint, user_metadata or {}
            )
            
            # Step 2: Make routing decision (which memory systems to query)
            routing_decision = await self._make_routing_decision(query_context)
            
            # Step 3: Query selected memory systems in parallel
            context_blocks = await self._query_memory_systems(
                query_context, routing_decision
            )
            
            # Step 4: Rerank and merge results within token budget
            merged_context = await self._merge_and_rank_context(
                context_blocks, routing_decision
            )
            
            # Step 5: Prepare response
            query_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            response = {
                "merged_context": merged_context,
                "context_blocks": context_blocks,
                "routing_decision": routing_decision,
                "query_metadata": {
                    "query_time_ms": query_time_ms,
                    "systems_queried": routing_decision.get("systems_queried", []),
                    "total_tokens": sum(block.get("token_count", 0) for block in context_blocks),
                    "timestamp": start_time.isoformat()
                }
            }
            
            self.logger.info(f"Query handled successfully in {query_time_ms:.2f}ms")
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling query: {str(e)}")
            # Return minimal context on error
            return {
                "merged_context": "",
                "context_blocks": [],
                "routing_decision": {
                    "use_stm": True,
                    "use_semantic": False,
                    "use_episodic": False,
                    "use_rag": False,
                    "confidence": 0.0,
                    "reasoning": "Error occurred during query processing"
                },
                "query_metadata": {
                    "error": str(e),
                    "timestamp": start_time.isoformat()
                }
            }
    
    async def handle_write(
        self,
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
        Handle writing user and assistant text to appropriate memory systems.
        
        Args:
            user_text: The user's message text
            assistant_text: The assistant's response text
            tenant_id: Organization identifier
            user_id: User identifier
            agent_id: Agent identifier
            conversation_id: Conversation identifier
            topic_hint: Optional topic hint for better categorization
            metadata: Optional metadata to store with the memory
            
        Returns:
            Dict containing write results and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Step 1: Create write context
            write_context = self._create_write_context(
                user_text, assistant_text, tenant_id, user_id, agent_id,
                conversation_id, topic_hint, metadata or {}
            )
            
            # Step 2: Determine what to write to which memory systems
            write_decision = await self._make_write_decision(write_context)
            
            # Step 3: Write to selected memory systems in parallel
            write_results = await self._write_to_memory_systems(
                write_context, write_decision
            )
            
            # Step 4: Prepare response
            write_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            response = {
                "write_results": write_results,
                "write_decision": write_decision,
                "write_metadata": {
                    "write_time_ms": write_time_ms,
                    "systems_written": write_decision.get("systems_written", []),
                    "timestamp": start_time.isoformat()
                }
            }
            
            self.logger.info(f"Write handled successfully in {write_time_ms:.2f}ms")
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling write: {str(e)}")
            return {
                "write_results": {},
                "write_decision": {"error": str(e)},
                "write_metadata": {"error": str(e)}
            }
    
    def _create_query_context(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        topic_hint: Optional[str],
        user_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create standardized query context object."""
        return {
            "query": query,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "topic_hint": topic_hint,
            "user_metadata": user_metadata,
            "timestamp": datetime.now(timezone.utc),
            "token_budget": self.config.default_token_budget
        }
    
    def _create_write_context(
        self,
        user_text: str,
        assistant_text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        topic_hint: Optional[str],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create standardized write context object."""
        return {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "topic_hint": topic_hint,
            "metadata": metadata,
            "timestamp": datetime.now(timezone.utc)
        }
    
    async def _make_routing_decision(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make routing decision about which memory systems to query.
        Uses deterministic rules first, falls back to LLM arbiter if needed.
        """
        # Initialize router if needed
        if self._router is None:
            from .router import MemoryRouter
            self._router = MemoryRouter(self.config)
        
        # Get routing decision from router
        routing_decision = await self._router.route_query(query_context)
        
        # If confidence is low and arbiter is enabled, use LLM fallback
        if (routing_decision.get("confidence", 1.0) < self.config.arbiter_confidence_threshold 
            and self.config.enable_llm_arbiter):
            
            if self._arbiter is None:
                from .arbiter import MemoryArbiter
                self._arbiter = MemoryArbiter(self.config)
            
            # Override with arbiter decision
            arbiter_decision = await self._arbiter.make_decision(query_context, routing_decision)
            routing_decision.update(arbiter_decision)
        
        return routing_decision
    
    async def _make_write_decision(self, write_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine which memory systems to write to based on content analysis.
        """
        # Initialize router if needed
        if self._router is None:
            from .router import MemoryRouter
            self._router = MemoryRouter(self.config)
        
        return await self._router.route_write(write_context)
    
    async def _query_memory_systems(
        self,
        query_context: Dict[str, Any],
        routing_decision: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Query selected memory systems in parallel and return context blocks.
        """
        tasks = []
        
        # Query STM if enabled
        if routing_decision.get("use_stm", False) and self.config.enable_stm:
            tasks.append(self._query_stm(query_context))
        
        # Query Semantic if enabled
        if routing_decision.get("use_semantic", False) and self.config.enable_semantic:
            tasks.append(self._query_semantic(query_context))
        
        # Query Episodic if enabled
        if routing_decision.get("use_episodic", False) and self.config.enable_episodic:
            tasks.append(self._query_episodic(query_context))
        
        # Query RAG if enabled
        if routing_decision.get("use_rag", False) and self.config.enable_rag:
            tasks.append(self._query_rag(query_context))
        
        # Execute queries in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions and None results
            context_blocks = [r for r in results if r is not None and not isinstance(r, Exception)]
        else:
            context_blocks = []
        
        return context_blocks
    
    async def _write_to_memory_systems(
        self,
        write_context: Dict[str, Any],
        write_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Write to selected memory systems in parallel.
        """
        tasks = []
        results = {}
        
        # Write to STM if enabled
        if write_decision.get("write_stm", False) and self.config.enable_stm:
            tasks.append(("stm", self._write_stm(write_context)))
        
        # Write to Semantic if enabled
        if write_decision.get("write_semantic", False) and self.config.enable_semantic:
            tasks.append(("semantic", self._write_semantic(write_context)))
        
        # Write to Episodic if enabled
        if write_decision.get("write_episodic", False) and self.config.enable_episodic:
            tasks.append(("episodic", self._write_episodic(write_context)))
        
        # Write to RAG if enabled
        if write_decision.get("write_rag", False) and self.config.enable_rag:
            tasks.append(("rag", self._write_rag(write_context)))
        
        # Execute writes in parallel
        if tasks:
            task_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            for i, (system_name, _) in enumerate(tasks):
                result = task_results[i]
                if not isinstance(result, Exception):
                    results[system_name] = result
                else:
                    results[system_name] = {"error": str(result)}
        
        return results
    
    async def _merge_and_rank_context(
        self,
        context_blocks: List[Dict[str, Any]],
        routing_decision: Dict[str, Any]
    ) -> str:
        """
        Merge and rank context blocks within token budget.
        """
        if not context_blocks:
            return ""
        
        # Use router's context merging functionality
        if self._router is None:
            from .router import MemoryRouter
            self._router = MemoryRouter(self.config)
        
        merged_context = self._router.merge_context_blocks(
            context_blocks,
            routing_decision.get("token_budget", self.config.default_token_budget),
            routing_decision.get("query", "")
        )
        
        return merged_context
    
    # Placeholder methods for memory system interactions
    # These will be implemented when the respective stores are created
    
    async def _query_stm(self, query_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query short-term memory store."""
        try:
            if self._stm_store is None:
                from .stm import STMManager
                self._stm_store = STMManager(self.config)
            
            # Use the proper query_stm method instead of query_conversation_history
            result = await self._stm_store.query_stm(
                query_context["query"],
                query_context["tenant_id"],
                query_context["user_id"],
                query_context["conversation_id"],
                max_turns=self.config.stm_max_turns
            )
            
            if result:
                # Return the result as-is since query_stm already formats it properly
                return result
            else:
                # Fallback: try to get conversation history and format it
                results = await self._stm_store.query_conversation_history(
                    query_context["conversation_id"],
                    query_context["tenant_id"],
                    query_context["user_id"],
                    limit=self.config.stm_max_turns
                )
                
                if results:
                    content_parts = []
                    for r in results:
                        content_parts.append(f"User: {r['user_text']}")
                        content_parts.append(f"Assistant: {r['assistant_text']}")
                    
                    return {
                        "system": "stm",
                        "content": "[STM] Recent conversation:\n" + "\n".join(content_parts),
                        "metadata": {"source": "stm", "turn_count": len(results)},
                        "token_count": sum(len(r['user_text']) + len(r['assistant_text']) for r in results) // 4
                    }
        except Exception as e:
            self.logger.warning(f"STM query failed: {str(e)}")
        return None
    
    async def _query_rag(self, query_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query RAG document store."""
        try:
            if self._rag_store is None:
                from .rag import RAGMemoryManager
                self._rag_store = RAGMemoryManager(self.config)
            
            results = await self._rag_store.query_rag(
                query_context["query"],
                query_context["tenant_id"],
                query_context["user_id"],
                limit=self.config.rag_max_chunks
            )
            
            if results:
                return results
        except Exception as e:
            self.logger.warning(f"RAG query failed: {str(e)}")
        return None
    
    async def _write_stm(self, write_context: Dict[str, Any]) -> Dict[str, Any]:
        """Write to short-term memory store."""
        try:
            if self._stm_store is None:
                from .stm import STMManager
                self._stm_store = STMManager(self.config)
            
            result = await self._stm_store.write_conversation_history(
                write_context["conversation_id"],
                write_context["tenant_id"],
                write_context["user_id"],
                write_context["user_text"],
                write_context["assistant_text"]
            )
            
            return {"status": "success", "system": "stm"}
        except Exception as e:
            self.logger.warning(f"STM write failed: {str(e)}")
            return {"status": "failure", "system": "stm", "error": str(e)}
    
    async def _write_rag(self, write_context: Dict[str, Any]) -> Dict[str, Any]:
        """Write to RAG document store."""
        try:
            if self._rag_store is None:
                from .rag import RAGMemoryManager
                self._rag_store = RAGMemoryManager(self.config)
            
            # For conversation turns, create a document from the conversation
            conversation_content = f"User: {write_context['user_text']}\nAssistant: {write_context['assistant_text']}"
            document_id = f"conv_{write_context['conversation_id']}_{write_context['timestamp'].isoformat()}"
            
            result = await self._rag_store.ingest_document(
                content=conversation_content,
                document_id=document_id,
                tenant_id=write_context["tenant_id"],
                user_id=write_context["user_id"],
                content_type="text/plain",
                metadata={
                    "conversation_id": write_context["conversation_id"],
                    "agent_id": write_context["agent_id"],
                    "topic_hint": write_context.get("topic_hint"),
                    **write_context.get("metadata", {})
                }
            )
            
            return {"status": "success", "system": "rag", "chunks_created": result.get("chunks_stored", 0)}
        except Exception as e:
            self.logger.error(f"RAG write failed: {str(e)}")
            return {"status": "failure", "system": "rag", "error": str(e)}
    
    async def _query_semantic(self, query_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query semantic memory store."""
        try:
            if self._semantic_store is None:
                from .memory_store import UnifiedMemoryManager
                self._semantic_store = UnifiedMemoryManager(self.config)
            
            results = await self._semantic_store.query_memories(
                query_context["query"],
                query_context["tenant_id"],
                query_context["user_id"],
                memory_types=["semantic"],
                limit=self.config.semantic_max_results
            )
            
            semantic_results = results.get("semantic", [])
            if semantic_results:
                content_parts = []
                for record, score in semantic_results:
                    if score >= self.config.semantic_confidence_threshold:
                        content_parts.append(f"- {record.content} (confidence: {score:.2f})")
                
                if content_parts:
                    return {
                        "system": "semantic",
                        "memory_type": "semantic",
                        "content": "\n".join(content_parts),
                        "metadata": {"source": "semantic", "fact_count": len(content_parts)},
                        "token_count": sum(len(part) for part in content_parts) // 4,
                        "confidence": max(score for _, score in semantic_results),
                        "relevance_score": max(score for _, score in semantic_results)
                    }
        except Exception as e:
            self.logger.warning(f"Semantic query failed: {str(e)}")
        return None
    
    async def _query_episodic(self, query_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query episodic memory store."""
        try:
            if self._episodic_store is None:
                from .memory_store import UnifiedMemoryManager
                self._episodic_store = UnifiedMemoryManager(self.config)
            
            results = await self._episodic_store.query_memories(
                query_context["query"],
                query_context["tenant_id"],
                query_context["user_id"],
                memory_types=["episodic"],
                limit=self.config.episodic_max_results
            )
            
            episodic_results = results.get("episodic", [])
            if episodic_results:
                content_parts = []
                for record, score in episodic_results:
                    if score >= self.config.episodic_confidence_threshold:
                        # Format with relative time if available
                        time_info = ""
                        if hasattr(record, 'created_at') and record.created_at:
                            from .utils import format_time_ago
                            from datetime import datetime, timezone
                            days_ago = (datetime.now(timezone.utc) - record.created_at).days
                            time_info = f" ({format_time_ago(days_ago)})"
                        content_parts.append(f"- {record.content}{time_info} (relevance: {score:.2f})")
                
                if content_parts:
                    return {
                        "system": "episodic",
                        "memory_type": "episodic", 
                        "content": "\n".join(content_parts),
                        "metadata": {"source": "episodic", "event_count": len(content_parts)},
                        "token_count": sum(len(part) for part in content_parts) // 4,
                        "confidence": max(score for _, score in episodic_results),
                        "relevance_score": max(score for _, score in episodic_results)
                    }
        except Exception as e:
            self.logger.warning(f"Episodic query failed: {str(e)}")
        return None

    async def _write_semantic(self, write_context: Dict[str, Any]) -> Dict[str, Any]:
        """Write to semantic memory store."""
        try:
            if self._semantic_store is None:
                from .memory_store import UnifiedMemoryManager
                self._semantic_store = UnifiedMemoryManager(self.config)
            
            result = await self._semantic_store.write_memories(
                write_context["user_text"],
                write_context["assistant_text"],
                write_context["tenant_id"],
                write_context["user_id"],
                write_context["agent_id"],
                write_context["conversation_id"],
                write_context.get("topic_hint")
            )
            
            semantic_ids = result.get("semantic", [])
            return {
                "status": "success" if semantic_ids else "no_facts_extracted",
                "system": "semantic",
                "facts_stored": len(semantic_ids),
                "fact_ids": semantic_ids
            }
        except Exception as e:
            self.logger.warning(f"Semantic write failed: {str(e)}")
            return {"status": "failure", "system": "semantic", "error": str(e)}
    
    async def _write_episodic(self, write_context: Dict[str, Any]) -> Dict[str, Any]:
        """Write to episodic memory store."""
        try:
            if self._episodic_store is None:
                from .memory_store import UnifiedMemoryManager
                self._episodic_store = UnifiedMemoryManager(self.config)
            
            result = await self._episodic_store.write_memories(
                write_context["user_text"],
                write_context["assistant_text"],
                write_context["tenant_id"],
                write_context["user_id"],
                write_context["agent_id"],
                write_context["conversation_id"],
                write_context.get("topic_hint")
            )
            
            episodic_ids = result.get("episodic", [])
            return {
                "status": "success" if episodic_ids else "no_events_extracted",
                "system": "episodic",
                "events_stored": len(episodic_ids),
                "event_ids": episodic_ids
            }
        except Exception as e:
            self.logger.warning(f"Episodic write failed: {str(e)}")
            return {"status": "failure", "system": "episodic", "error": str(e)}
    
    # Main API methods
    async def handle(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        topic_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main API method to handle queries and return merged context.
        
        This is the primary entry point for querying the memory system.
        """
        return await self.handle_query(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            topic_hint=topic_hint
        )
    
    async def post_write(
        self,
        user_text: str,
        assistant_text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        topic_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main API method to write conversation data to memory systems.
        
        This is the primary entry point for persisting conversation data.
        """
        return await self.write_conversation(
            user_text=user_text,
            assistant_text=assistant_text,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            topic_hint=topic_hint
        )
    
    async def write_conversation(
        self,
        user_text: str,
        assistant_text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        topic_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Write conversation data to memory systems.
        
        This method handles writing conversation data to appropriate memory systems.
        """
        return await self.handle_write(
            user_text=user_text,
            assistant_text=assistant_text,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            topic_hint=topic_hint
        )