"""
Unified memory store for both Semantic and Episodic memory.

Consolidates semantic fact extraction and episodic event extraction
into a single module with vector-backed storage.
"""

import asyncio
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .config import MemoryConfig
from .models import MemoryRecord, MemoryType
from .stores import VectorStore, create_vector_store
from .utils import generate_embedding, estimate_tokens, truncate_text, generate_record_id


# Try to import ChromaDB, fall back to in-memory if not available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class SemanticFactExtractor:
    """Extracts semantic facts and preferences from text."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for fact extraction."""
        self.preference_patterns = [
            # Original patterns
            re.compile(r'\b(I\s+(?:like|love|enjoy|prefer|hate|dislike))\s+(.+)', re.IGNORECASE),
            re.compile(r'\b(my\s+favorite)\s+(.+?)\s+(?:is|are)\s+(.+)', re.IGNORECASE),
            # Additional patterns for comparative preferences
            re.compile(r'\b(I\s+prefer)\s+(.+?)\s+(?:over|to|rather\s+than)\s+(.+)', re.IGNORECASE),
            re.compile(r'\b(I\s+(?:always|usually|often))\s+(?:choose|pick|order|get)\s+(.+)', re.IGNORECASE),
            re.compile(r'\b(I\s+(?:never|rarely|don\'t))\s+(?:eat|drink|like|choose)\s+(.+)', re.IGNORECASE),
        ]
        
        self.personal_patterns = [
            re.compile(r'\b(my\s+name\s+is|I\s+am|I\'m)\s+(.+)', re.IGNORECASE),
            re.compile(r'\b(I\s+live\s+in|I\'m\s+from)\s+(.+)', re.IGNORECASE),
        ]
    
    async def extract_facts(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract semantic facts from text."""
        facts = []
        
        # Extract preferences
        for i, pattern in enumerate(self.preference_patterns):
            matches = pattern.finditer(text)
            for match in matches:
                groups = match.groups()
                
                # Handle comparative preferences (I prefer X over Y) - pattern index 2
                if i == 2 and len(groups) >= 3:  # This is the comparative pattern
                    predicate = groups[0].strip().lower()
                    preferred = groups[1].strip()
                    not_preferred = groups[2].strip()
                    
                    if len(preferred) > 2 and len(not_preferred) > 2:
                        fact = {
                            "fact_id": generate_record_id(f"{predicate}_{preferred}_over_{not_preferred}", "pref"),
                            "fact_type": "preference",
                            "subject": "user",
                            "predicate": f"{predicate} {preferred} over {not_preferred}",
                            "object": preferred,
                            "confidence": 0.9,
                            "source_text": match.group(0),
                            "importance": 0.8,
                            "metadata": {**(context or {}), "preference_type": "comparative", "not_preferred": not_preferred}
                        }
                        facts.append(fact)
                        self.logger.debug(f"Extracted comparative preference: {fact}")
                
                # Handle "my favorite X is Y" pattern - pattern index 1
                elif i == 1 and len(groups) >= 3:
                    predicate = groups[0].strip().lower()
                    category = groups[1].strip()
                    obj = groups[2].strip()
                    
                    if len(obj) > 2:
                        fact = {
                            "fact_id": generate_record_id(f"{predicate}_{category}_{obj}", "pref"),
                            "fact_type": "preference",
                            "subject": "user",
                            "predicate": f"{predicate} {category}",
                            "object": obj,
                            "confidence": 0.9,
                            "source_text": match.group(0),
                            "importance": 0.8,
                            "metadata": {**(context or {}), "category": category}
                        }
                        facts.append(fact)
                        self.logger.debug(f"Extracted favorite preference: {fact}")
                
                # Handle regular preferences (I like/love/prefer X) - patterns 0, 3, 4
                elif len(groups) >= 2:
                    predicate = groups[0].strip().lower()
                    obj = groups[1].strip()
                    
                    if len(obj) > 3:
                        fact = {
                            "fact_id": generate_record_id(f"{predicate}_{obj}", "pref"),
                            "fact_type": "preference",
                            "subject": "user",
                            "predicate": predicate,
                            "object": obj,
                            "confidence": 0.8,
                            "source_text": match.group(0),
                            "importance": 0.7,
                            "metadata": context or {}
                        }
                        facts.append(fact)
                        self.logger.debug(f"Extracted regular preference: {fact}")
        
        # Extract personal info
        for pattern in self.personal_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                predicate = match.group(1).strip().lower()
                obj = match.group(2).strip()
                
                if len(obj) > 2:
                    fact = {
                        "fact_id": generate_record_id(f"{predicate}_{obj}", "info"),
                        "fact_type": "personal_info",
                        "subject": "user",
                        "predicate": predicate,
                        "object": obj,
                        "confidence": 0.9,
                        "source_text": match.group(0),
                        "importance": 0.8,
                        "metadata": context or {}
                    }
                    facts.append(fact)
                    self.logger.debug(f"Extracted personal info: {fact}")
        
        self.logger.info(f"Extracted {len(facts)} facts from text: {text[:100]}...")
        return facts


class EpisodicEventExtractor:
    """Extracts episodic events and experiences from text."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for event extraction."""
        self.action_patterns = [
            re.compile(r'\b(I\s+(?:went|visited|saw|met|did|made|bought))\s+(.+)', re.IGNORECASE),
            re.compile(r'\b(yesterday|today|last\s+\w+)\s+(.+)', re.IGNORECASE),
        ]
    
    async def extract_events(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract episodic events from text."""
        events = []
        
        for pattern in self.action_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                action = match.group(1).strip()
                description = match.group(2).strip()
                
                if len(description) > 3:
                    event = {
                        "event_id": generate_record_id(f"{action}_{description}", "event"),
                        "event_type": "action",
                        "description": f"{action} {description}",
                        "participants": ["user"],
                        "location": None,
                        "timestamp": datetime.now(timezone.utc),
                        "confidence": 0.8,
                        "importance": 0.7,
                        "emotional_context": None,
                        "metadata": context or {}
                    }
                    events.append(event)
        
        return events


class UnifiedMemoryManager:
    """
    Unified manager for both semantic and episodic memory.
    Handles extraction, storage, and retrieval of facts and events.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize extractors
        self.fact_extractor = SemanticFactExtractor(config)
        self.event_extractor = EpisodicEventExtractor(config)
        
        # Memory stores (lazy initialization)
        self.stores: Dict[str, VectorStore] = {}
    
    async def write_memories(
        self,
        user_text: str,
        assistant_text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        conversation_id: str,
        topic_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Write both semantic facts and episodic events from conversation."""
        results = {"semantic": [], "episodic": []}
        
        # Extract and store semantic facts
        if self.config.enable_semantic:
            facts = await self.fact_extractor.extract_facts(
                f"{user_text} {assistant_text}",
                {"conversation_id": conversation_id, "topic_hint": topic_hint}
            )
            
            if facts:
                semantic_records = []
                for fact in facts:
                    record = MemoryRecord(
                        id=fact["fact_id"],
                        content=f"{fact['predicate']}: {fact['object']}",
                        memory_type=MemoryType.SEMANTIC,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        conversation_id=conversation_id,
                        confidence=fact["confidence"],
                        importance=fact["importance"],
                        embedding=await generate_embedding(f"{fact['predicate']}: {fact['object']}", self.config),
                        metadata=fact["metadata"]
                    )
                    semantic_records.append(record)
                
                store = await self._get_store(tenant_id, user_id, "semantic")
                stored_ids = await store.add_records(semantic_records)
                results["semantic"] = stored_ids
        
        # Extract and store episodic events
        if self.config.enable_episodic:
            events = await self.event_extractor.extract_events(
                f"{user_text} {assistant_text}",
                {"conversation_id": conversation_id, "topic_hint": topic_hint}
            )
            
            if events:
                episodic_records = []
                for event in events:
                    record = MemoryRecord(
                        id=event["event_id"],
                        content=event["description"],
                        memory_type=MemoryType.EPISODIC,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        conversation_id=conversation_id,
                        confidence=event["confidence"],
                        importance=event["importance"],
                        embedding=await generate_embedding(event["description"], self.config),
                        metadata=event["metadata"]
                    )
                    episodic_records.append(record)
                
                store = await self._get_store(tenant_id, user_id, "episodic")
                stored_ids = await store.add_records(episodic_records)
                results["episodic"] = stored_ids
        
        return results
    
    async def query_memories(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Query both semantic and episodic memories."""
        results = {}
        query_embedding = await generate_embedding(query, self.config)
        
        if not query_embedding:
            return results
        
        memory_types = memory_types or ["semantic", "episodic"]
        
        # Query semantic memory
        if "semantic" in memory_types and self.config.enable_semantic:
            store = await self._get_store(tenant_id, user_id, "semantic")
            semantic_results = await store.search_records(
                query_embedding,
                limit=limit,
                filters={"tenant_id": tenant_id, "user_id": user_id, "memory_type": "semantic"}
            )
            results["semantic"] = semantic_results
        
        # Query episodic memory
        if "episodic" in memory_types and self.config.enable_episodic:
            store = await self._get_store(tenant_id, user_id, "episodic")
            episodic_results = await store.search_records(
                query_embedding,
                limit=limit,
                filters={"tenant_id": tenant_id, "user_id": user_id, "memory_type": "episodic"}
            )
            results["episodic"] = episodic_results
        
        return results
    
    async def _get_store(self, tenant_id: str, user_id: str, memory_type: str):
        """Get or create memory store for tenant/user/type."""
        store_key = f"{tenant_id}_{user_id}_{memory_type}"
        
        if store_key not in self.stores:
            collection_name = f"{tenant_id}_{user_id}_{memory_type}_collection"
            
            self.stores[store_key] = create_vector_store(
                config=self.config,
                store_type="memory",  # Use in-memory for now
                collection_name=collection_name
            )
        
        return self.stores[store_key]
