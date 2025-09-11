import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone

from .config import MemoryConfig
from .models import MemoryType, RoutingDecision
from .utils import estimate_tokens
from .services import ContextReranker


class MemoryRouter:
    """
    Deterministic router that decides which memory systems to query/write based on query patterns.
    Uses fast rule-based routing with confidence scoring.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize router with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reranker = ContextReranker(self.config)
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for query analysis."""
        # Temporal patterns for episodic memory
        self.temporal_patterns = [
            re.compile(r'\b(yesterday|today|tomorrow|last\s+\w+|next\s+\w+)\b', re.IGNORECASE),
            re.compile(r'\b(when|what\s+time|at\s+\d+|on\s+\w+day)\b', re.IGNORECASE),
            re.compile(r'\b(\d{1,2}[:/]\d{1,2}|\d{4}-\d{2}-\d{2})\b'),  # dates/times
            re.compile(r'\b(ago|before|after|during|while)\b', re.IGNORECASE),
        ]
        
        # Identity/preference patterns for semantic memory
        self.semantic_patterns = [
            re.compile(r'\b(I\s+prefer|I\s+like|I\s+hate|I\s+love|my\s+favorite)\b', re.IGNORECASE),
            re.compile(r"\b(about\s+me|my\s+style|my\s+preferences|I\s+am|I\'m)\b", re.IGNORECASE),
            re.compile(r'\b(always|never|usually|typically|generally)\b', re.IGNORECASE),
            re.compile(r'\b(remember\s+that\s+I|note\s+that\s+I|keep\s+in\s+mind)\b', re.IGNORECASE),
        ]
        
        # Document/knowledge patterns for RAG
        self.rag_patterns = [
            re.compile(r'\b(manual|documentation|guide|policy|procedure)\b', re.IGNORECASE),
            re.compile(r'\b(according\s+to|based\s+on|as\s+per|reference)\b', re.IGNORECASE),
            re.compile(r'\b(product|feature|specification|requirement)\b', re.IGNORECASE),
            re.compile(r'\b(how\s+to|step\s+by\s+step|instructions)\b', re.IGNORECASE),
        ]
        
        # Question patterns that need context
        self.context_patterns = [
            re.compile(r'\b(what|how|why|when|where|who)\b', re.IGNORECASE),
            re.compile(r'\b(explain|describe|tell\s+me|show\s+me)\b', re.IGNORECASE),
            re.compile(r'\b(help|assist|support)\b', re.IGNORECASE),
        ]
    
    async def route_query(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a query to appropriate memory systems using deterministic rules.
        
        Args:
            query_context: Query context with query text and metadata
            
        Returns:
            Dict with routing decisions and confidence scores
        """
        query = query_context.get("query", "")
        topic_hint = query_context.get("topic_hint")
        
        # Initialize routing decision
        routing = RoutingDecision()
        routing.token_budget = query_context.get("token_budget", self.config.default_token_budget)
        
        # Always include STM for conversation context
        routing.use_stm = self.config.enable_stm
        confidence_scores = []
        reasoning_parts = []
        
        # Rule 1: Temporal cues → Episodic memory
        if self._has_temporal_cues(query):
            if self.config.enable_episodic:
                routing.use_episodic = True
                confidence_scores.append(0.9)
                reasoning_parts.append("temporal cues detected")
        
        # Rule 2: Identity/preferences → Semantic memory
        if self._has_semantic_cues(query):
            if self.config.enable_semantic:
                routing.use_semantic = True
                confidence_scores.append(0.85)
                reasoning_parts.append("preference/identity patterns detected")
        
        # Rule 3: Document/knowledge intent → RAG
        if self._has_rag_cues(query) or self._is_knowledge_query(query):
            if self.config.enable_rag:
                routing.use_rag = True
                confidence_scores.append(0.8)
                reasoning_parts.append("document/knowledge query detected")
        
        # Rule 4: Topic hint override
        if topic_hint:
            self._apply_topic_hint(routing, topic_hint, confidence_scores, reasoning_parts)
        
        # Rule 5: Fallback for ambiguous queries
        if not any([routing.use_semantic, routing.use_episodic, routing.use_rag]):
            # Default to semantic + episodic for general queries
            if self.config.enable_semantic:
                routing.use_semantic = True
            if self.config.enable_episodic:
                routing.use_episodic = True
            confidence_scores.append(0.6)
            reasoning_parts.append("fallback for general query")
        
        # Calculate overall confidence
        routing.confidence = max(confidence_scores) if confidence_scores else 0.5
        routing.reasoning = "; ".join(reasoning_parts)
        
        # Allocate token budget
        self._allocate_token_budget(routing)
        
        self.logger.debug(f"Query routing: {routing.get_query_systems()}, confidence: {routing.confidence}")
        
        return {
            "use_stm": routing.use_stm,
            "use_semantic": routing.use_semantic,
            "use_episodic": routing.use_episodic,
            "use_rag": routing.use_rag,
            "confidence": routing.confidence,
            "reasoning": routing.reasoning,
            "token_budget": routing.token_budget,
            "token_allocation": routing.token_allocation,
            "systems_queried": [t.value for t in routing.get_query_systems()],
            "query": query,
        }
    
    def _has_temporal_cues(self, query: str) -> bool:
        """Check if query contains temporal references."""
        return any(pattern.search(query) for pattern in self.temporal_patterns)
    
    def _has_semantic_cues(self, query: str) -> bool:
        """Check if query contains identity/preference patterns."""
        return any(pattern.search(query) for pattern in self.semantic_patterns)
    
    def _has_rag_cues(self, query: str) -> bool:
        """Check if query contains document/knowledge patterns."""
        return any(pattern.search(query) for pattern in self.rag_patterns)
    
    def _is_knowledge_query(self, query: str) -> bool:
        """Check if query is asking for factual knowledge."""
        # Look for question patterns + specific domains
        has_question = any(pattern.search(query) for pattern in self.context_patterns)
        
        # Check for specific knowledge domains
        knowledge_terms = [
            "definition", "meaning", "explain", "what is", "how does",
            "tutorial", "example", "best practice", "comparison"
        ]
        has_knowledge_intent = any(term in query.lower() for term in knowledge_terms)
        
        return has_question and has_knowledge_intent
    
    def _apply_topic_hint(self, routing: RoutingDecision, topic_hint: str, 
                         confidence_scores: List[float], reasoning_parts: List[str]):
        """Apply topic hint to routing decision."""
        hint_lower = topic_hint.lower()
        
        if any(term in hint_lower for term in ["preference", "profile", "user", "personal"]):
            if self.config.enable_semantic:
                routing.use_semantic = True
                confidence_scores.append(0.95)
                reasoning_parts.append(f"topic hint: {topic_hint}")
        
        elif any(term in hint_lower for term in ["event", "history", "timeline", "past"]):
            if self.config.enable_episodic:
                routing.use_episodic = True
                confidence_scores.append(0.95)
                reasoning_parts.append(f"topic hint: {topic_hint}")
        
        elif any(term in hint_lower for term in ["document", "manual", "guide", "knowledge"]):
            if self.config.enable_rag:
                routing.use_rag = True
                confidence_scores.append(0.95)
                reasoning_parts.append(f"topic hint: {topic_hint}")
    
    def _contains_facts_or_preferences(self, text: str) -> bool:
        """Check if text contains facts or preferences worth storing."""
        # Look for preference statements
        preference_indicators = [
            "i prefer", "i like", "i don't like", "i hate", "i love",
            "my favorite", "i always", "i never", "i usually",
            "remember that i", "note that i", "keep in mind"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in preference_indicators)
    
    def _contains_events_or_experiences(self, text: str) -> bool:
        """Check if text contains events or experiences worth storing."""
        # Look for event indicators
        event_indicators = [
            "yesterday", "today", "last week", "last month", "when i",
            "i went to", "i visited", "i met", "i did", "i tried",
            "happened", "occurred", "experience", "remember when"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in event_indicators)
    
    def _contains_document_content(self, user_text: str, assistant_text: str) -> bool:
        """Check if content contains structured information worth storing in RAG."""
        combined_text = f"{user_text} {assistant_text}".lower()
        
        # Look for structured content indicators
        document_indicators = [
            "here's the document", "attached file", "pdf", "manual",
            "step 1", "step 2", "procedure", "instructions",
            "specification", "requirements", "guidelines"
        ]
        
        return any(indicator in combined_text for indicator in document_indicators)
    
    def _apply_write_topic_hint(self, routing: RoutingDecision, topic_hint: str,
                               confidence_scores: List[float], reasoning_parts: List[str]):
        """Apply topic hint to write routing decision."""
        hint_lower = topic_hint.lower()
        
        if "preference" in hint_lower or "profile" in hint_lower:
            if self.config.enable_semantic:
                routing.write_semantic = True
                confidence_scores.append(0.9)
                reasoning_parts.append(f"write topic hint: {topic_hint}")
        
        elif "event" in hint_lower or "experience" in hint_lower:
            if self.config.enable_episodic:
                routing.write_episodic = True
                confidence_scores.append(0.9)
                reasoning_parts.append(f"write topic hint: {topic_hint}")
    
    def _allocate_token_budget(self, routing: RoutingDecision):
        """Allocate token budget across selected memory systems."""
        active_systems = routing.get_query_systems()
        
        if not active_systems:
            return
        
        # Base allocations from config
        base_allocations = {
            MemoryType.STM: self.config.stm_token_allocation,
            MemoryType.SEMANTIC: self.config.semantic_token_allocation,
            MemoryType.EPISODIC: self.config.episodic_token_allocation,
            MemoryType.RAG: self.config.rag_token_allocation
        }
        
        # Calculate total requested
        total_requested = sum(base_allocations[system] for system in active_systems)
        
        # Scale down if over budget
        if total_requested > routing.token_budget:
            scale_factor = routing.token_budget / total_requested
            routing.token_allocation = {
                (system.value if isinstance(system, MemoryType) else str(system)): int(base_allocations[system] * scale_factor)
                for system in active_systems
            }
        else:
            routing.token_allocation = {
                (system.value if isinstance(system, MemoryType) else str(system)): base_allocations[system]
                for system in active_systems
            }
    
    async def route_write(self, write_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a write operation to appropriate memory systems.
        
        Args:
            write_context: Write context with user/assistant text and metadata
            
        Returns:
            Dict with write routing decisions and confidence scores
        """
        user_text = write_context.get("user_text", "")
        assistant_text = write_context.get("assistant_text", "")
        topic_hint = write_context.get("topic_hint")
        
        # Initialize routing decision
        routing = RoutingDecision()
        confidence_scores = []
        reasoning_parts = []
        
        # Always write to STM for conversation history
        routing.write_stm = self.config.enable_stm
        
        # Rule 1: Check for facts or preferences → Semantic memory
        if self._contains_facts_or_preferences(f"{user_text} {assistant_text}"):
            if self.config.enable_semantic:
                routing.write_semantic = True
                confidence_scores.append(0.8)
                reasoning_parts.append("facts/preferences detected")
        
        # Rule 2: Check for events or experiences → Episodic memory
        if self._contains_events_or_experiences(f"{user_text} {assistant_text}"):
            if self.config.enable_episodic:
                routing.write_episodic = True
                confidence_scores.append(0.8)
                reasoning_parts.append("events/experiences detected")
        
        # Rule 3: Check for document content → RAG memory
        if self._contains_document_content(user_text, assistant_text):
            if self.config.enable_rag:
                routing.write_rag = True
                confidence_scores.append(0.9)
                reasoning_parts.append("document content detected")
        
        # Rule 4: Topic hint override
        if topic_hint:
            self._apply_write_topic_hint(routing, topic_hint, confidence_scores, reasoning_parts)
        
        # Calculate overall confidence
        routing.confidence = max(confidence_scores) if confidence_scores else 0.7
        routing.reasoning = "; ".join(reasoning_parts) if reasoning_parts else "conversation turn storage"
        
        self.logger.debug(f"Write routing: {routing.get_write_systems()}, confidence: {routing.confidence}")
        
        return {
            "write_stm": routing.write_stm,
            "write_semantic": routing.write_semantic,
            "write_episodic": routing.write_episodic,
            "write_rag": routing.write_rag,
            "confidence": routing.confidence,
            "reasoning": routing.reasoning,
            "systems_written": [t.value for t in routing.get_write_systems()]
        }
    
    def merge_context_blocks(
        self,
        context_blocks: List[Dict[str, Any]],
        token_budget: int,
        query: str = ""
    ) -> str:
        """
        Merge context blocks using the reranker service.
        
        Args:
            context_blocks: List of context blocks from memory systems
            token_budget: Maximum tokens allowed for merged context
            query: Original query for relevance scoring
            
        Returns:
            Merged context string within token budget
        """
        return self.reranker.merge_context(context_blocks, token_budget, query)
