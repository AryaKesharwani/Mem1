"""
Minimal vector store adapters for Memory Agent.

Provides simple abstractions for vector storage with InMemory (default) 
and ChromaDB (optional) implementations.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import math
# Optional numpy import with lightweight fallback
try:
    import numpy as np  # type: ignore
    _NUMPY_AVAILABLE = True
except ImportError:  # Fallback shim to avoid hard dependency in tests
    _NUMPY_AVAILABLE = False

    class _LinalgShim:
        @staticmethod
        def norm(v):
            try:
                return math.sqrt(sum(float(x) * float(x) for x in v))
            except Exception:
                return 0.0

    class _NPShim:
        linalg = _LinalgShim()

        @staticmethod
        def array(seq, dtype=None):
            try:
                return [float(x) for x in seq]
            except Exception:
                return list(seq)

        @staticmethod
        def dot(a, b):
            try:
                return float(sum(float(x) * float(y) for x, y in zip(a, b)))
            except Exception:
                return 0.0

        # Provide float32 constant for compatibility
        float32 = float

    np = _NPShim()  # type: ignore

from .config import MemoryConfig
from .models import MemoryRecord, MemoryType

# Try to import Pinecone, fall back to ChromaDB or in-memory if not available
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Try to import ChromaDB, fall back to in-memory if not available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""
    
    def __init__(self, config: MemoryConfig, collection_name: str = "default"):
        self.config = config
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def add_records(self, records: List[MemoryRecord]) -> List[str]:
        """Add memory records to the store."""
        pass
    
    @abstractmethod
    async def search_records(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search for similar records using vector similarity."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        pass


class InMemoryStore(VectorStore):
    """In-memory vector store using dictionaries and numpy."""
    
    def __init__(self, config: MemoryConfig, collection_name: str = "default"):
        super().__init__(config, collection_name)
        self.records: Dict[str, MemoryRecord] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.logger.info(f"InMemoryStore initialized: {collection_name}")
    
    async def add_records(self, records: List[MemoryRecord]) -> List[str]:
        """Add records to in-memory store."""
        record_ids = []
        for record in records:
            record_id = record.id or str(uuid.uuid4())
            record.id = record_id
            self.records[record_id] = record
            if record.embedding:
                self.embeddings[record_id] = np.array(record.embedding, dtype=np.float32)
            record_ids.append(record_id)
        return record_ids
    
    async def search_records(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search using cosine similarity."""
        if not self.embeddings:
            return []
        
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        similarities = []
        for record_id, embedding in self.embeddings.items():
            if record_id not in self.records:
                continue
            
            record = self.records[record_id]
            
            # Apply filters
            if filters:
                if filters.get("tenant_id") and record.tenant_id != filters["tenant_id"]:
                    continue
                if filters.get("user_id") and record.user_id != filters["user_id"]:
                    continue
                if filters.get("memory_type") and record.memory_type.value != filters["memory_type"]:
                    continue
            
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm == 0:
                continue
            
            similarity = np.dot(query_vec, embedding) / (query_norm * embedding_norm)
            similarities.append((record, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_records": len(self.records),
            "records_with_embeddings": len(self.embeddings),
            "collection_name": self.collection_name,
            "store_type": "in_memory"
        }


class PineconeStore(VectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, config: MemoryConfig, collection_name: str = "default"):
        super().__init__(config, collection_name)
        
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=config.pinecone_api_key,
            environment=config.pinecone_environment
        )
        
        # Get or create index
        self.index_name = collection_name
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=config.vector_dimension,
                metric="cosine"
            )
        
        self.index = pinecone.Index(self.index_name)
        self.logger.info(f"PineconeStore initialized: {collection_name}")
    
    async def add_records(self, records: List[MemoryRecord]) -> List[str]:
        """Add records to Pinecone."""
        if not records:
            return []
        
        vectors_to_upsert = []
        record_ids = []
        
        for record in records:
            record_id = record.id or str(uuid.uuid4())
            record.id = record_id
            
            vectors_to_upsert.append({
                "id": record_id,
                "values": record.embedding or [0.0] * self.config.vector_dimension,
                "metadata": self._prepare_metadata(record)
            })
            record_ids.append(record_id)
        
        # Batch upsert to Pinecone
        self.index.upsert(vectors=vectors_to_upsert)
        
        return record_ids
    
    async def search_records(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search using Pinecone."""
        query_response = self.index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True,
            filter=self._prepare_filter(filters) if filters else None
        )
        
        records_with_scores = []
        for match in query_response.matches:
            record_id = match.id
            similarity = float(match.score)
            metadata = match.metadata
            
            record = self._metadata_to_record(record_id, metadata)
            records_with_scores.append((record, similarity))
        
        return records_with_scores
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_records": stats.total_vector_count,
                "collection_name": self.collection_name,
                "store_type": "pinecone",
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _prepare_metadata(self, record: MemoryRecord) -> Dict[str, Any]:
        """Prepare metadata for Pinecone storage."""
        return {
            "content": record.content,
            "memory_type": record.memory_type.value if isinstance(record.memory_type, MemoryType) else record.memory_type,
            "tenant_id": record.tenant_id,
            "user_id": record.user_id,
            "agent_id": record.agent_id,
            "conversation_id": record.conversation_id or "",
            "confidence": record.confidence,
            "importance": record.importance,
            "created_at": record.created_at.isoformat(),
        }
    
    def _prepare_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Prepare filter for Pinecone filtering."""
        if not filters:
            return None
        
        pinecone_filter = {}
        for key, value in filters.items():
            if key in ["tenant_id", "user_id", "memory_type", "conversation_id", "agent_id"]:
                pinecone_filter[key] = {"$eq": value}
        
        return pinecone_filter if pinecone_filter else None
    
    def _metadata_to_record(self, record_id: str, metadata: Dict[str, Any]) -> MemoryRecord:
        """Convert Pinecone metadata back to MemoryRecord."""
        return MemoryRecord(
            id=record_id,
            content=metadata["content"],
            memory_type=MemoryType(metadata["memory_type"]),
            tenant_id=metadata["tenant_id"],
            user_id=metadata["user_id"],
            agent_id=metadata.get("agent_id"),
            conversation_id=metadata.get("conversation_id"),
            confidence=metadata["confidence"],
            importance=metadata["importance"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
        )


class ChromaStore(VectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, config: MemoryConfig, collection_name: str = "default"):
        super().__init__(config, collection_name)
        
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=f"{config.data_root}/chroma"
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Memory Agent collection: {collection_name}"}
        )
        
        self.logger.info(f"ChromaStore initialized: {collection_name}")
    
    async def add_records(self, records: List[MemoryRecord]) -> List[str]:
        """Add records to ChromaDB."""
        if not records:
            return []
        
        record_ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for record in records:
            record_id = record.id or str(uuid.uuid4())
            record.id = record_id
            
            record_ids.append(record_id)
            embeddings.append(record.embedding or [0.0] * self.config.vector_dimension)
            documents.append(record.content)
            metadatas.append(self._prepare_metadata(record))
        
        self.collection.add(
            ids=record_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return record_ids
    
    async def search_records(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search using ChromaDB."""
        where_clause = self._prepare_where_clause(filters)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause if where_clause else None
        )
        
        records_with_scores = []
        if results["ids"] and results["ids"][0]:
            for i, record_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = max(0.0, 1.0 - distance)
                
                metadata = results["metadatas"][0][i]
                document = results["documents"][0][i]
                
                record = self._metadata_to_record(record_id, document, metadata)
                records_with_scores.append((record, similarity))
        
        return records_with_scores
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        try:
            count = self.collection.count()
            return {
                "total_records": count,
                "collection_name": self.collection_name,
                "store_type": "chromadb"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _prepare_metadata(self, record: MemoryRecord) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage."""
        return {
            "memory_type": record.memory_type.value if isinstance(record.memory_type, MemoryType) else record.memory_type,
            "tenant_id": record.tenant_id,
            "user_id": record.user_id,
            "conversation_id": record.conversation_id or "",
            "confidence": record.confidence,
            "importance": record.importance,
            "created_at": record.created_at.isoformat(),
        }
    
    def _prepare_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Prepare where clause for ChromaDB filtering."""
        if not filters:
            return None
        
        where_clause = {}
        for key, value in filters.items():
            if key in ["tenant_id", "user_id", "memory_type", "conversation_id"]:
                where_clause[key] = value
        
        return where_clause if where_clause else None
    
    def _metadata_to_record(self, record_id: str, document: str, metadata: Dict[str, Any]) -> MemoryRecord:
        """Convert ChromaDB metadata back to MemoryRecord."""
        return MemoryRecord(
            id=record_id,
            content=document,
            memory_type=MemoryType(metadata["memory_type"]),
            tenant_id=metadata["tenant_id"],
            user_id=metadata["user_id"],
            conversation_id=metadata.get("conversation_id"),
            confidence=metadata["confidence"],
            importance=metadata["importance"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
        )


def create_vector_store(
    config: MemoryConfig,
    store_type: str = "pinecone",
    collection_name: str = "default"
) -> VectorStore:
    """Factory function to create vector store instances."""
    if store_type.lower() == "pinecone" and PINECONE_AVAILABLE:
        return PineconeStore(config, collection_name)
    elif store_type.lower() == "chroma" and CHROMA_AVAILABLE:
        return ChromaStore(config, collection_name)
    else:
        # Fallback to in-memory store
        if store_type.lower() == "pinecone" and not PINECONE_AVAILABLE:
            logging.warning("Pinecone not available, falling back to in-memory store")
        elif store_type.lower() == "chroma" and not CHROMA_AVAILABLE:
            logging.warning("ChromaDB not available, falling back to in-memory store")
        return InMemoryStore(config, collection_name)
