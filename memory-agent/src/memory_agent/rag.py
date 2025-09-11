"""
RAG Memory - Document ingestion, chunking, and retrieval.

Provides document processing capabilities for PDF, text, HTML, and other formats
with chunking, embedding, and hybrid retrieval (semantic + keyword + metadata).
"""

import logging
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
import json

from .config import MemoryConfig
from .models import MemoryRecord, MemoryType, DocumentChunk
from .stores import VectorStore, create_vector_store
from .utils import generate_embedding, estimate_tokens


class DocumentProcessor:
    """
    Document processor for various file formats.
    
    Handles ingestion, parsing, and chunking of documents
    for RAG memory storage.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize the document processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def process_document(
        self,
        content: Union[str, bytes],
        document_id: str,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process a document into chunks.
        
        Args:
            content: Document content (text or bytes)
            document_id: Unique document identifier
            content_type: MIME type of the document
            metadata: Optional document metadata
            
        Returns:
            List of document chunks
        """
        try:
            # Extract text based on content type
            if isinstance(content, bytes):
                text = await self._extract_text_from_bytes(content, content_type)
            else:
                text = content
            
            if not text or not text.strip():
                self.logger.warning(f"No text extracted from document {document_id}")
                return []
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Create chunks
            chunks = self._create_chunks(
                text=cleaned_text,
                document_id=document_id,
                metadata=metadata or {}
            )
            
            self.logger.info(f"Processed document {document_id} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Document processing failed for {document_id}: {str(e)}")
            return []
    
    async def _extract_text_from_bytes(self, content: bytes, content_type: str) -> str:
        """Extract text from binary content based on content type."""
        try:
            if content_type == "application/pdf":
                return await self._extract_pdf_text(content)
            elif content_type in ["text/html", "application/xhtml+xml"]:
                return await self._extract_html_text(content)
            elif content_type.startswith("text/"):
                return content.decode("utf-8", errors="ignore")
            else:
                # Try to decode as text
                return content.decode("utf-8", errors="ignore")
                
        except Exception as e:
            self.logger.warning(f"Text extraction failed for {content_type}: {str(e)}")
            return ""
    
    async def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            from pdfminer.high_level import extract_text
            from io import BytesIO
            
            pdf_file = BytesIO(content)
            text = extract_text(pdf_file)
            return text
            
        except ImportError:
            self.logger.warning("pdfminer.six not available for PDF processing")
            return ""
        except Exception as e:
            self.logger.warning(f"PDF text extraction failed: {str(e)}")
            return ""
    
    async def _extract_html_text(self, content: bytes) -> str:
        """Extract text from HTML content."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean up whitespace
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            self.logger.warning("beautifulsoup4 not available for HTML processing")
            return content.decode("utf-8", errors="ignore")
        except Exception as e:
            self.logger.warning(f"HTML text extraction failed: {str(e)}")
            return content.decode("utf-8", errors="ignore")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize curly quotes to ASCII
        try:
            text = (text
                    .replace('\u201c', '"')  # left double quotation mark
                    .replace('\u201d', '"')  # right double quotation mark
                    .replace('\u2018', "'")  # left single quotation mark
                    .replace('\u2019', "'")  # right single quotation mark)
                   )
        except Exception:
            # If any unexpected encoding issues occur, continue with original text
            pass
        
        return text.strip()
    
    def _create_chunks(
        self,
        text: str,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create document chunks from text."""
        if not text:
            return []
        
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        # Simple character-based chunking with overlap
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 20% of the chunk
                search_start = max(start + int(chunk_size * 0.8), start + 1)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                
                if sentence_end > start:
                    end = sentence_end
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                # Estimate token count
                token_count = estimate_tokens(chunk_content)
                
                # Create chunk
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    metadata=metadata.copy()
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(end - overlap, start + 1)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find a good sentence boundary for chunking."""
        # Look for sentence endings (., !, ?) followed by space or newline
        sentence_pattern = r'[.!?]\s+'
        
        # Search backwards from end position
        search_text = text[start:end]
        matches = list(re.finditer(sentence_pattern, search_text))
        
        if matches:
            # Use the last sentence boundary found
            last_match = matches[-1]
            return start + last_match.end()
        
        # If no sentence boundary found, look for paragraph breaks
        paragraph_pattern = r'\n\s*\n'
        matches = list(re.finditer(paragraph_pattern, search_text))
        
        if matches:
            last_match = matches[-1]
            return start + last_match.start()
        
        # No good boundary found, use the end position
        return end


class RAGMemoryManager:
    """
    Manager for RAG memory operations.
    
    Handles document ingestion, storage, and retrieval with
    hybrid search capabilities (semantic + keyword + metadata).
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize the RAG memory manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Document processor
        self.processor = DocumentProcessor(config)
        
        # Vector stores for different tenants/users (initialized lazily)
        self.stores: Dict[str, VectorStore] = {}
        
        # Embedding function (placeholder - would use actual embedding model)
        self._embedding_function = None
    
    async def ingest_document(
        self,
        content: Union[str, bytes],
        document_id: str,
        tenant_id: str,
        user_id: str,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into RAG memory.
        
        Args:
            content: Document content
            document_id: Unique document identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            content_type: MIME type of the document
            metadata: Optional document metadata
            
        Returns:
            Ingestion result with status and metadata
        """
        try:
            # Process document into chunks
            chunks = await self.processor.process_document(
                content=content,
                document_id=document_id,
                content_type=content_type,
                metadata=metadata
            )
            
            if not chunks:
                return {
                    "status": "error",
                    "system": "rag",
                    "error": "No chunks created from document"
                }
            
            # Convert chunks to memory records
            records = []
            for chunk in chunks:
                # Generate embedding for chunk
                embedding = await generate_embedding(chunk.content, self.config)
                
                # Create memory record
                record = MemoryRecord(
                    id=chunk.chunk_id,
                    content=chunk.content,
                    memory_type=MemoryType.RAG,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    agent_id="rag_system",
                    conversation_id=None,
                    confidence=1.0,  # Documents are high confidence
                    importance=0.7,  # Default importance for documents
                    recency_score=1.0,  # New documents are recent
                    relevance_score=0.0,  # Will be calculated during search
                    embedding=embedding,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "content_type": content_type,
                        **chunk.metadata
                    }
                )
                
                records.append(record)
            
            # Store records in vector store
            store = await self._get_store(tenant_id, user_id)
            stored_ids = await store.add_records(records)
            
            self.logger.info(f"Ingested document {document_id}: {len(stored_ids)} chunks stored")
            
            return {
                "status": "success",
                "system": "rag",
                "document_id": document_id,
                "chunks_stored": len(stored_ids),
                "chunk_ids": stored_ids
            }
            
        except Exception as e:
            self.logger.error(f"Document ingestion failed for {document_id}: {str(e)}")
            return {
                "status": "error",
                "system": "rag",
                "error": str(e)
            }
    
    async def query_rag(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query RAG memory for relevant document chunks.
        
        Args:
            query: The search query
            tenant_id: Tenant identifier
            user_id: User identifier
            limit: Maximum number of results
            filters: Optional filters for search
            
        Returns:
            Context block with RAG content or None
        """
        try:
            if not query.strip():
                return None
            
            # Generate query embedding
            query_embedding = await generate_embedding(query, self.config)
            if not query_embedding:
                return None
            
            # Get vector store
            store = await self._get_store(tenant_id, user_id)
            
            # Prepare search filters
            search_filters = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "memory_type": MemoryType.RAG.value
            }
            if filters:
                search_filters.update(filters)
            
            # Search for similar chunks
            results = await store.search_records(
                query_embedding=query_embedding,
                limit=limit,
                filters=search_filters
            )
            
            if not results:
                return None
            
            # Filter by relevance threshold
            filtered_results = [
                (record, score) for record, score in results
                if score >= self.config.rag_relevance_threshold
            ]
            
            if not filtered_results:
                return None
            
            # Format context content
            context_content = self._format_rag_context(filtered_results, query)
            
            # Create context block
            context_block = {
                "memory_type": "rag",
                "content": context_content,
                "token_count": estimate_tokens(context_content),
                "records": [
                    {
                        "id": record.id,
                        "content": record.content,
                        "memory_type": record.memory_type.value,
                        "tenant_id": record.tenant_id,
                        "user_id": record.user_id,
                        "confidence": record.confidence,
                        "importance": record.importance,
                        "relevance_score": score,
                        "metadata": record.metadata
                    }
                    for record, score in filtered_results
                ],
                "metadata": {
                    "query": query,
                    "results_count": len(filtered_results),
                    "max_relevance": max(score for _, score in filtered_results),
                    "documents": list(set(
                        record.metadata.get("document_id", "unknown")
                        for record, _ in filtered_results
                    ))
                }
            }
            
            self.logger.debug(f"RAG query returned {len(filtered_results)} relevant chunks")
            
            return context_block
            
        except Exception as e:
            self.logger.error(f"RAG query failed: {str(e)}")
            return None
    
    async def _get_store(self, tenant_id: str, user_id: str) -> VectorStore:
        """Get or create vector store for tenant/user."""
        store_key = f"{tenant_id}:{user_id}"
        
        if store_key not in self.stores:
            collection_name = f"{tenant_id}_{user_id}_rag_collection"
            
            self.stores[store_key] = create_vector_store(
                config=self.config,
                store_type="memory",  # Use in-memory for now
                collection_name=collection_name
            )
            
            self.logger.debug(f"Created RAG vector store for {store_key}")
        
        return self.stores[store_key]
    
    def _format_rag_context(
        self,
        results: List[Tuple[MemoryRecord, float]],
        query: str
    ) -> str:
        """Format RAG search results into context string."""
        if not results:
            return ""
        
        formatted_chunks = []
        
        for record, score in results:
            document_id = record.metadata.get("document_id", "unknown")
            chunk_index = record.metadata.get("chunk_index", 0)
            
            # Format chunk with metadata
            chunk_header = f"[Document: {document_id}, Chunk {chunk_index}]"
            chunk_content = record.content
            
            formatted_chunks.append(f"{chunk_header}\n{chunk_content}")
        
        return "\n\n".join(formatted_chunks)


# Convenience functions for direct RAG operations
async def ingest_document(
    content: Union[str, bytes],
    document_id: str,
    tenant_id: str,
    user_id: str,
    content_type: str = "text/plain",
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[MemoryConfig] = None
) -> Dict[str, Any]:
    """Convenience function to ingest a document directly."""
    from .config import get_config
    
    config = config or get_config()
    rag_manager = RAGMemoryManager(config)
    
    return await rag_manager.ingest_document(
        content, document_id, tenant_id, user_id, content_type, metadata
    )


async def query_documents(
    query: str,
    tenant_id: str,
    user_id: str,
    limit: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    config: Optional[MemoryConfig] = None
) -> Optional[Dict[str, Any]]:
    """Convenience function to query RAG memory directly."""
    from .config import get_config
    
    config = config or get_config()
    rag_manager = RAGMemoryManager(config)
    
    return await rag_manager.query_rag(
        query, tenant_id, user_id, limit, filters
    )
