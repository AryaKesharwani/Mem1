"""
Memory Agent Web API - FastAPI-based REST API server.

This module provides HTTP endpoints for the memory agent system:
- POST /query - Query memory systems for context
- POST /write - Write conversation data to memory systems
- GET /health - Health check endpoint
- GET /info - System information endpoint
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union

from .api import handle, post_write, health_check, get_system_info
from .rag import ingest_document, query_documents, RAGMemoryManager
from .s3_storage import create_s3_storage
from .config import get_config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Memory Agent API",
    description="Multi-system memory orchestrator for AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for querying memory systems."""
    query: str = Field(..., description="The user's query text to find relevant context for")
    tenant_id: str = Field(..., description="Organization identifier for multi-tenant data isolation")
    user_id: str = Field(..., description="User identifier within the organization")
    agent_id: str = Field(..., description="Agent identifier for this conversation session")
    conversation_id: str = Field(..., description="Unique identifier for this conversation thread")
    topic_hint: Optional[str] = Field(None, description="Optional hint about query topic for better routing")
    user_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional user context (preferences, current state, etc.)")


class WriteRequest(BaseModel):
    """Request model for writing conversation data."""
    user_text: str = Field(..., description="The user's message text")
    assistant_text: str = Field(..., description="The assistant's response text")
    tenant_id: str = Field(..., description="Organization identifier for data isolation")
    user_id: str = Field(..., description="User identifier within the organization")
    agent_id: str = Field(..., description="Agent identifier for this conversation")
    conversation_id: str = Field(..., description="Unique conversation identifier")
    topic_hint: Optional[str] = Field(None, description="Optional topic hint for better categorization")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata to store with the memory")


class ContextBlock(BaseModel):
    """Model for individual context blocks from memory systems."""
    system: str = Field(..., description="Memory system name (stm, semantic, episodic, rag)")
    memory_type: str = Field(..., description="Type of memory")
    content: str = Field(..., description="The actual context content")
    metadata: Dict[str, Any] = Field(..., description="System-specific metadata")
    token_count: Optional[int] = Field(None, description="Estimated token count")
    confidence: Optional[float] = Field(None, description="Confidence score")
    relevance_score: Optional[float] = Field(None, description="Relevance score")


class RoutingDecision(BaseModel):
    """Model for routing decision information."""
    use_stm: bool = Field(..., description="Whether STM was queried")
    use_semantic: bool = Field(..., description="Whether semantic memory was queried")
    use_episodic: bool = Field(..., description="Whether episodic memory was queried")
    use_rag: bool = Field(..., description="Whether RAG was queried")
    confidence: float = Field(..., description="Routing confidence score")
    reasoning: str = Field(..., description="Explanation of routing decision")
    systems_queried: List[str] = Field(..., description="List of systems that were queried")


class QueryMetadata(BaseModel):
    """Model for query metadata."""
    query_time_ms: float = Field(..., description="Query processing time in milliseconds")
    systems_queried: List[str] = Field(..., description="Systems that were queried")
    total_tokens: int = Field(..., description="Total tokens in response")
    timestamp: str = Field(..., description="Query timestamp")


class QueryResponse(BaseModel):
    """Response model for query requests."""
    merged_context: str = Field(..., description="All relevant context merged and formatted")
    context_blocks: List[ContextBlock] = Field(..., description="Individual context blocks from each memory system")
    routing_decision: RoutingDecision = Field(..., description="Details about which systems were queried and why")
    query_metadata: QueryMetadata = Field(..., description="Performance metrics and system information")


class WriteResult(BaseModel):
    """Model for individual write results."""
    status: str = Field(..., description="Write status (success, failure, no_facts_extracted, etc.)")
    system: str = Field(..., description="Memory system name")
    facts_stored: Optional[int] = Field(None, description="Number of facts stored (semantic)")
    events_stored: Optional[int] = Field(None, description="Number of events stored (episodic)")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created (RAG)")
    error: Optional[str] = Field(None, description="Error message if failed")


class WriteDecision(BaseModel):
    """Model for write decision information."""
    write_stm: bool = Field(..., description="Whether STM was written to")
    write_semantic: bool = Field(..., description="Whether semantic memory was written to")
    write_episodic: bool = Field(..., description="Whether episodic memory was written to")
    write_rag: bool = Field(..., description="Whether RAG was written to")
    systems_written: List[str] = Field(..., description="List of systems that were written to")


class WriteMetadata(BaseModel):
    """Model for write metadata."""
    write_time_ms: float = Field(..., description="Write processing time in milliseconds")
    systems_written: List[str] = Field(..., description="Systems that were written to")
    timestamp: str = Field(..., description="Write timestamp")


class WriteResponse(BaseModel):
    """Response model for write requests."""
    write_results: Dict[str, WriteResult] = Field(..., description="Results from each memory system that was written to")
    write_decision: WriteDecision = Field(..., description="Details about which systems were written to and why")
    write_metadata: WriteMetadata = Field(..., description="Performance metrics and system information")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="System health status")
    config: Optional[Dict[str, Any]] = Field(None, description="System configuration")
    timestamp: float = Field(..., description="Health check timestamp")


class SystemInfoResponse(BaseModel):
    """Response model for system information."""
    version: str = Field(..., description="System version")
    memory_systems: Dict[str, Any] = Field(..., description="Memory system configurations")
    routing: Dict[str, Any] = Field(..., description="Routing configuration")
    data_paths: Dict[str, Any] = Field(..., description="Data storage paths")


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    tenant_id: str = Field(..., description="Organization identifier")
    user_id: str = Field(..., description="User identifier")
    document_id: Optional[str] = Field(None, description="Custom document ID (auto-generated if not provided)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional document metadata")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    status: str = Field(..., description="Upload status")
    document_id: str = Field(..., description="Document identifier")
    chunks_stored: int = Field(..., description="Number of chunks created")
    chunk_ids: List[str] = Field(..., description="List of chunk IDs")
    content_type: str = Field(..., description="Detected content type")
    file_size: int = Field(..., description="File size in bytes")


class RAGQueryRequest(BaseModel):
    """Request model for direct RAG queries."""
    query: str = Field(..., description="Search query")
    tenant_id: str = Field(..., description="Organization identifier")
    user_id: str = Field(..., description="User identifier")
    limit: int = Field(5, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional search filters")


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str = Field(..., description="Original query")
    results_count: int = Field(..., description="Number of results found")
    max_relevance: float = Field(..., description="Highest relevance score")
    documents: List[str] = Field(..., description="List of document IDs in results")
    chunks: List[Dict[str, Any]] = Field(..., description="Document chunks with metadata")
    context_content: str = Field(..., description="Formatted context content")


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Memory Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
        "rag": "/rag/upload, /rag/query, /rag/documents",
        "s3": "S3 document storage with Pinecone vector search"
    }


@app.post("/query", response_model=QueryResponse)
async def query_memory(request: QueryRequest):
    """
    Query memory systems for relevant context.
    
    This endpoint analyzes the query and returns merged context from appropriate memory systems.
    """
    try:
        logger.info(f"Query request from user {request.user_id} in conversation {request.conversation_id}")
        
        # Call the memory agent API
        result = await handle(
            query=request.query,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            agent_id=request.agent_id,
            conversation_id=request.conversation_id,
            topic_hint=request.topic_hint,
            user_metadata=request.user_metadata
        )
        
        # Convert context blocks to response models
        context_blocks = []
        for block in result.get("context_blocks", []):
            context_blocks.append(ContextBlock(
                system=block.get("system", ""),
                memory_type=block.get("memory_type", ""),
                content=block.get("content", ""),
                metadata=block.get("metadata", {}),
                token_count=block.get("token_count"),
                confidence=block.get("confidence"),
                relevance_score=block.get("relevance_score")
            ))
        
        # Convert routing decision
        routing_data = result.get("routing_decision", {})
        routing_decision = RoutingDecision(
            use_stm=routing_data.get("use_stm", False),
            use_semantic=routing_data.get("use_semantic", False),
            use_episodic=routing_data.get("use_episodic", False),
            use_rag=routing_data.get("use_rag", False),
            confidence=routing_data.get("confidence", 0.0),
            reasoning=routing_data.get("reasoning", ""),
            systems_queried=routing_data.get("systems_queried", [])
        )
        
        # Convert query metadata
        metadata = result.get("query_metadata", {})
        query_metadata = QueryMetadata(
            query_time_ms=metadata.get("query_time_ms", 0.0),
            systems_queried=metadata.get("systems_queried", []),
            total_tokens=metadata.get("total_tokens", 0),
            timestamp=metadata.get("timestamp", datetime.now().isoformat())
        )
        
        response = QueryResponse(
            merged_context=result.get("merged_context", ""),
            context_blocks=context_blocks,
            routing_decision=routing_decision,
            query_metadata=query_metadata
        )
        
        logger.info(f"Query completed successfully for user {request.user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/write", response_model=WriteResponse)
async def write_memory(request: WriteRequest):
    """
    Write conversation data to appropriate memory systems.
    
    This endpoint analyzes the conversation turn and writes to relevant memory systems.
    """
    try:
        logger.info(f"Write request from user {request.user_id} in conversation {request.conversation_id}")
        
        # Call the memory agent API
        result = await post_write(
            user_text=request.user_text,
            assistant_text=request.assistant_text,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            agent_id=request.agent_id,
            conversation_id=request.conversation_id,
            topic_hint=request.topic_hint,
            metadata=request.metadata
        )
        
        # Convert write results
        write_results = {}
        for system_name, system_result in result.get("write_results", {}).items():
            write_results[system_name] = WriteResult(
                status=system_result.get("status", "unknown"),
                system=system_result.get("system", system_name),
                facts_stored=system_result.get("facts_stored"),
                events_stored=system_result.get("events_stored"),
                chunks_created=system_result.get("chunks_created"),
                error=system_result.get("error")
            )
        
        # Convert write decision
        write_decision_data = result.get("write_decision", {})
        write_decision = WriteDecision(
            write_stm=write_decision_data.get("write_stm", False),
            write_semantic=write_decision_data.get("write_semantic", False),
            write_episodic=write_decision_data.get("write_episodic", False),
            write_rag=write_decision_data.get("write_rag", False),
            systems_written=write_decision_data.get("systems_written", [])
        )
        
        # Convert write metadata
        metadata = result.get("write_metadata", {})
        write_metadata = WriteMetadata(
            write_time_ms=metadata.get("write_time_ms", 0.0),
            systems_written=metadata.get("systems_written", []),
            timestamp=metadata.get("timestamp", datetime.now().isoformat())
        )
        
        response = WriteResponse(
            write_results=write_results,
            write_decision=write_decision,
            write_metadata=write_metadata
        )
        
        logger.info(f"Write completed successfully for user {request.user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in write endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Check the health status of the memory agent system.
    
    Returns system status and configuration information.
    """
    try:
        result = await health_check()
        return HealthResponse(
            status=result.get("status", "unknown"),
            config=result.get("config"),
            timestamp=result.get("timestamp", 0.0)
        )
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            config=None,
            timestamp=0.0
        )


@app.get("/info", response_model=SystemInfoResponse)
async def system_info():
    """
    Get information about the memory agent system configuration.
    
    Returns system configuration and capabilities.
    """
    try:
        result = get_system_info()
        return SystemInfoResponse(
            version=result.get("version", "1.0.0"),
            memory_systems=result.get("memory_systems", {}),
            routing=result.get("routing", {}),
            data_paths=result.get("data_paths", {})
        )
    except Exception as e:
        logger.error(f"Error in system info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/rag/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    user_id: str = Form(...),
    document_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """
    Upload a document for RAG processing.
    
    Supports various file formats including PDF, HTML, and plain text.
    The document will be processed into chunks and stored in the vector database.
    """
    try:
        logger.info(f"Document upload request: {file.filename} for user {user_id}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file"
            )
        
        # Generate document ID if not provided
        if not document_id:
            import hashlib
            import time
            timestamp = int(time.time())
            file_hash = hashlib.md5(content).hexdigest()[:8]
            document_id = f"doc_{timestamp}_{file_hash}"
        
        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            try:
                import json
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON metadata: {metadata}")
        
        # Add file metadata
        parsed_metadata.update({
            "filename": file.filename,
            "upload_timestamp": datetime.now().isoformat(),
            "file_size": len(content)
        })
        
        # Detect content type
        content_type = file.content_type or "application/octet-stream"
        
        # Ingest document
        result = await ingest_document(
            content=content,
            document_id=document_id,
            tenant_id=tenant_id,
            user_id=user_id,
            content_type=content_type,
            metadata=parsed_metadata
        )
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {result.get('error', 'Unknown error')}"
            )
        
        response = DocumentUploadResponse(
            status=result["status"],
            document_id=result["document_id"],
            chunks_stored=result["chunks_stored"],
            chunk_ids=result["chunk_ids"],
            content_type=content_type,
            file_size=len(content)
        )
        
        logger.info(f"Document uploaded successfully: {document_id} ({result['chunks_stored']} chunks)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag_documents(request: RAGQueryRequest):
    """
    Query RAG documents directly.
    
    This endpoint searches through uploaded documents and returns relevant chunks
    without going through the full memory routing system.
    """
    try:
        logger.info(f"RAG query request: '{request.query}' for user {request.user_id}")
        
        # Query RAG documents
        result = await query_documents(
            query=request.query,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            limit=request.limit,
            filters=request.filters
        )
        
        if not result:
            # Return empty response if no results found
            return RAGQueryResponse(
                query=request.query,
                results_count=0,
                max_relevance=0.0,
                documents=[],
                chunks=[],
                context_content=""
            )
        
        # Extract chunks from result
        chunks = result.get("records", [])
        
        response = RAGQueryResponse(
            query=request.query,
            results_count=result.get("metadata", {}).get("results_count", 0),
            max_relevance=result.get("metadata", {}).get("max_relevance", 0.0),
            documents=result.get("metadata", {}).get("documents", []),
            chunks=chunks,
            context_content=result.get("content", "")
        )
        
        logger.info(f"RAG query completed: {response.results_count} results found")
        return response
        
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/rag/documents/{tenant_id}/{user_id}", response_model=List[Dict[str, Any]])
async def list_documents(tenant_id: str, user_id: str, limit: int = 100):
    """
    List documents stored in S3 for a tenant/user.
    
    Returns metadata about stored documents including S3 keys and upload timestamps.
    """
    try:
        config = get_config()
        s3_storage = create_s3_storage(config)
        
        if not s3_storage:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="S3 storage not configured"
            )
        
        documents = await s3_storage.list_documents(tenant_id, user_id, limit)
        return documents
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/rag/documents/{tenant_id}/{user_id}/{document_id}")
async def get_document(tenant_id: str, user_id: str, document_id: str):
    """
    Retrieve a document from S3.
    
    Returns the document content and metadata.
    """
    try:
        config = get_config()
        s3_storage = create_s3_storage(config)
        
        if not s3_storage:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="S3 storage not configured"
            )
        
        document = await s3_storage.retrieve_document(document_id, tenant_id, user_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/rag/documents/{tenant_id}/{user_id}/{document_id}/url")
async def get_document_url(
    tenant_id: str, 
    user_id: str, 
    document_id: str, 
    expiration: int = 3600
):
    """
    Generate a presigned URL for document access.
    
    Returns a temporary URL that can be used to access the document directly from S3.
    """
    try:
        config = get_config()
        s3_storage = create_s3_storage(config)
        
        if not s3_storage:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="S3 storage not configured"
            )
        
        url = await s3_storage.get_document_url(document_id, tenant_id, user_id, expiration)
        
        if not url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {"url": url, "expires_in": expiration}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating document URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.delete("/rag/documents/{tenant_id}/{user_id}/{document_id}")
async def delete_document(tenant_id: str, user_id: str, document_id: str):
    """
    Delete a document from S3.
    
    Removes the document from S3 storage. Note: This does not remove vector embeddings from Pinecone.
    """
    try:
        config = get_config()
        s3_storage = create_s3_storage(config)
        
        if not s3_storage:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="S3 storage not configured"
            )
        
        success = await s3_storage.delete_document(document_id, tenant_id, user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {"status": "success", "message": "Document deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the memory agent on startup."""
    logger.info("Memory Agent API starting up...")
    try:
        # Initialize the memory agent by calling health check
        await health_check()
        logger.info("Memory Agent API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Memory Agent API: {str(e)}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Memory Agent API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
