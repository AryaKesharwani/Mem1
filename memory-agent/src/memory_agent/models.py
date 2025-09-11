from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum


class MemoryType(Enum):
    """Enumeration of memory system types."""
    SEMANTIC = "semantic"
    EPISODIC = "episodic" 
    RAG = "rag"
    STM = "stm"


@dataclass
class MemoryRecord:
    """Base memory record with metadata and content."""
    id: str
    content: str
    memory_type: MemoryType
    tenant_id: str
    user_id: str
    agent_id: str
    conversation_id: Optional[str] = None
    
    # Scoring and relevance
    confidence: float = 1.0
    importance: float = 1.0
    recency_score: float = 1.0
    relevance_score: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Vector embeddings
    embedding: Optional[List[float]] = None
    
    def composite_score(self) -> float:
        """Calculate composite score: relevance × importance × recency."""
        return self.relevance_score * self.importance * self.recency_score


@dataclass
class ContextBlock:
    """A block of context from a specific memory system."""
    memory_type: MemoryType
    content: str
    records: List[MemoryRecord]
    token_count: int
    composite_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate composite score if not provided."""
        if not self.composite_score and self.records:
            # Calculate average composite score from records
            self.composite_score = sum(r.composite_score() for r in self.records) / len(self.records)


@dataclass
class RoutingDecision:
    """Decision about which memory systems to query and write to."""
    # Query decisions
    use_semantic: bool = False
    use_episodic: bool = False
    use_rag: bool = False
    use_stm: bool = True  # Always include STM by default
    
    # Write decisions
    write_semantic: bool = False
    write_episodic: bool = False
    write_rag: bool = False
    write_stm: bool = True  # Always update STM
    
    # Confidence and reasoning
    confidence: float = 1.0
    reasoning: str = ""
    
    # Token budget allocation
    token_budget: int = 4000
    token_allocation: Dict[str, int] = field(default_factory=dict)
    
    def get_query_systems(self) -> List[MemoryType]:
        """Get list of memory systems to query."""
        systems = []
        if self.use_stm:
            systems.append(MemoryType.STM)
        if self.use_semantic:
            systems.append(MemoryType.SEMANTIC)
        if self.use_episodic:
            systems.append(MemoryType.EPISODIC)
        if self.use_rag:
            systems.append(MemoryType.RAG)
        return systems
    
    def get_write_systems(self) -> List[MemoryType]:
        """Get list of memory systems to write to."""
        systems = []
        if self.write_stm:
            systems.append(MemoryType.STM)
        if self.write_semantic:
            systems.append(MemoryType.SEMANTIC)
        if self.write_episodic:
            systems.append(MemoryType.EPISODIC)
        if self.write_rag:
            systems.append(MemoryType.RAG)
        return systems


@dataclass
class QueryContext:
    """Context for a query including user info and conversation state."""
    query: str
    tenant_id: str
    user_id: str
    agent_id: str
    conversation_id: str
    
    # Optional context
    topic_hint: Optional[str] = None
    user_metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal context
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WriteContext:
    """Context for writing to memory systems."""
    user_text: str
    assistant_text: str
    tenant_id: str
    user_id: str
    agent_id: str
    conversation_id: str
    
    # Optional context
    topic_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal context
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MemoryResponse:
    """Response from memory agent containing context and routing info."""
    context_blocks: List[ContextBlock]
    routing_decision: RoutingDecision
    total_tokens: int
    query_context: QueryContext
    
    # Performance metrics
    query_time_ms: float = 0.0
    systems_queried: List[MemoryType] = field(default_factory=list)
    
    def get_merged_context(self) -> str:
        """Get merged context from all blocks."""
        return "\n\n".join([
            f"[{block.memory_type.value.upper()}]\n{block.content}"
            for block in sorted(self.context_blocks, key=lambda x: x.composite_score, reverse=True)
        ])


@dataclass
class SemanticFact:
    """A semantic fact with confidence and importance scoring."""
    fact_id: str
    fact_type: str
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str = ""
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodicEvent:
    """An episodic event with temporal and contextual information."""
    event_id: str
    description: str
    timestamp: datetime
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0


@dataclass
class DocumentChunk:
    """A chunk of a document for RAG."""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None