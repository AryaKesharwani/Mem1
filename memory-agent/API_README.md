# Memory Agent API - REST API Documentation

## Overview

The Memory Agent API provides HTTP endpoints for interacting with the multi-system memory orchestrator. This API allows you to query memory systems for context and write conversation data to appropriate memory stores.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file with the following variables:

```bash
# Required for LLM arbiter
OPENAI_API_KEY=your_openai_api_key_here

# Required for vector store
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=aws-us-west-2

# Optional Redis configuration
MEMORY_REDIS_HOST=127.0.0.1
MEMORY_REDIS_PORT=6379
MEMORY_REDIS_DB=0

# Memory system configuration
MEMORY_ENABLE_SEMANTIC=true
MEMORY_ENABLE_EPISODIC=true
MEMORY_ENABLE_RAG=true
MEMORY_ENABLE_STM=true
MEMORY_TOKEN_BUDGET=4000

# S3 Document Storage Configuration
S3_BUCKET_NAME=your-memory-agent-documents
S3_REGION=us-east-1
S3_PREFIX=memory-agent-documents
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

### 3. Start the API Server

```bash
python run_api.py
```

The server will start on `http://localhost:8000` by default.

### 4. Configure S3 and Pinecone

#### S3 Setup (Document Storage)

1. **Create S3 Bucket**: Create a bucket in AWS S3 for document storage
2. **Configure AWS Credentials**: Set up AWS access keys with S3 permissions
3. **Set Environment Variables**: Add S3 configuration to your `.env` file

```bash
# S3 Configuration
S3_BUCKET_NAME=your-memory-agent-documents
S3_REGION=us-east-1
S3_PREFIX=memory-agent-documents
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

#### Pinecone Setup (Vector Search)

1. **Create Pinecone Account**: Sign up at [pinecone.io](https://pinecone.io)
2. **Get API Key**: Retrieve your API key from the Pinecone console
3. **Choose Environment**: Select your preferred environment
4. **Set Environment Variables**: Add Pinecone configuration to your `.env` file

```bash
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=aws-us-west-2
```

### 5. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### POST /query

Query memory systems for relevant context.

**Request Body:**
```json
{
  "query": "What are my food preferences?",
  "tenant_id": "org1",
  "user_id": "user123",
  "agent_id": "assistant",
  "conversation_id": "conv_456",
  "topic_hint": "preferences",
  "user_metadata": {
    "current_context": "meal_planning"
  }
}
```

**Response:**
```json
{
  "merged_context": "[STM] Recent conversation about food...\n[SEMANTIC] User prefers Italian food...",
  "context_blocks": [
    {
      "system": "stm",
      "memory_type": "stm",
      "content": "[STM] Recent conversation:\nUser: I prefer Italian food\nAssistant: Got it!",
      "metadata": {"source": "stm", "turn_count": 3},
      "token_count": 45,
      "confidence": 0.95,
      "relevance_score": 0.92
    }
  ],
  "routing_decision": {
    "use_stm": true,
    "use_semantic": true,
    "use_episodic": false,
    "use_rag": false,
    "confidence": 0.85,
    "reasoning": "Query about preferences matches semantic memory patterns",
    "systems_queried": ["stm", "semantic"]
  },
  "query_metadata": {
    "query_time_ms": 125.5,
    "systems_queried": ["stm", "semantic"],
    "total_tokens": 156,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### POST /write

Write conversation data to appropriate memory systems.

**Request Body:**
```json
{
  "user_text": "I prefer Italian food over Chinese food",
  "assistant_text": "Got it — I'll remember that preference for future recommendations",
  "tenant_id": "org1",
  "user_id": "user123",
  "agent_id": "assistant",
  "conversation_id": "conv_456",
  "topic_hint": "food_preferences",
  "metadata": {
    "importance": "high",
    "category": "preference"
  }
}
```

**Response:**
```json
{
  "write_results": {
    "stm": {
      "status": "success",
      "system": "stm"
    },
    "semantic": {
      "status": "success",
      "system": "semantic",
      "facts_stored": 1,
      "fact_ids": ["fact_123"]
    }
  },
  "write_decision": {
    "write_stm": true,
    "write_semantic": true,
    "write_episodic": false,
    "write_rag": false,
    "systems_written": ["stm", "semantic"]
  },
  "write_metadata": {
    "write_time_ms": 89.2,
    "systems_written": ["stm", "semantic"],
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### GET /health

Check the health status of the memory agent system.

**Response:**
```json
{
  "status": "healthy",
  "config": {
    "systems_enabled": {
      "stm": true,
      "semantic": true,
      "episodic": true,
      "rag": true
    },
    "token_budget": 4000,
    "llm_arbiter_enabled": true
  },
  "timestamp": 1705312200.123
}
```

### GET /info

Get detailed system configuration and capabilities.

**Response:**
```json
{
  "version": "1.0.0",
  "memory_systems": {
    "stm": {
      "enabled": true,
      "window_size": 10,
      "token_allocation": 1000
    },
    "semantic": {
      "enabled": true,
      "confidence_threshold": 0.7,
      "token_allocation": 1500
    },
    "episodic": {
      "enabled": true,
      "confidence_threshold": 0.6,
      "token_allocation": 1000
    },
    "rag": {
      "enabled": true,
      "relevance_threshold": 0.5,
      "token_allocation": 500,
      "chunk_size": 512
    }
  },
  "routing": {
    "llm_arbiter_enabled": true,
    "arbiter_threshold": 0.5,
    "llm_model": "gpt-3.5-turbo"
  },
  "data_paths": {
    "data_root": "data"
  }
}
```

## RAG (Retrieval-Augmented Generation) Endpoints

### POST /rag/upload

Upload a document for RAG processing. Supports PDF, HTML, and text files.

**Request:** `multipart/form-data`
- `file`: The document file to upload
- `tenant_id`: Organization identifier
- `user_id`: User identifier
- `document_id`: Optional custom document ID
- `metadata`: Optional JSON metadata

**Response:**
```json
{
  "status": "success",
  "document_id": "doc_1705312200_a1b2c3d4",
  "chunks_stored": 15,
  "chunk_ids": ["doc_1705312200_a1b2c3d4_chunk_0", "..."],
  "content_type": "application/pdf",
  "file_size": 245760,
  "s3_stored": true,
  "s3_key": "memory-agent-documents/org1/user123/2024/01/doc_1705312200_a1b2c3d4.pdf",
  "s3_url": "s3://your-bucket/memory-agent-documents/org1/user123/2024/01/doc_1705312200_a1b2c3d4.pdf",
  "content_hash": "sha256_hash_of_content"
}
```

### POST /rag/query

Query RAG documents directly for relevant content.

**Request Body:**
```json
{
  "query": "What are the key characteristics of Italian cuisine?",
  "tenant_id": "org1",
  "user_id": "user123",
  "limit": 5,
  "filters": {
    "category": "food_guide"
  }
}
```

**Response:**
```json
{
  "query": "What are the key characteristics of Italian cuisine?",
  "results_count": 3,
  "max_relevance": 0.89,
  "documents": ["italian_cuisine_guide"],
  "chunks": [
    {
      "id": "italian_cuisine_guide_chunk_2",
      "content": "Key characteristics of Italian cuisine: Use of fresh, seasonal ingredients...",
      "memory_type": "rag",
      "confidence": 1.0,
      "relevance_score": 0.89,
      "metadata": {
        "document_id": "italian_cuisine_guide",
        "chunk_index": 2,
        "category": "food_guide"
      }
    }
  ],
  "context_content": "[Document: italian_cuisine_guide, Chunk 2]\nKey characteristics of Italian cuisine: Use of fresh, seasonal ingredients..."
}
```

## S3 Document Management Endpoints

### GET /rag/documents/{tenant_id}/{user_id}

List documents stored in S3 for a tenant/user.

**Response:**
```json
[
  {
    "document_id": "italian_cuisine_guide",
    "s3_key": "memory-agent-documents/org1/user123/2024/01/italian_cuisine_guide.txt",
    "content_type": "text/plain",
    "content_length": 1024,
    "upload_timestamp": "2024-01-15T10:30:00Z",
    "last_modified": "2024-01-15T10:30:00Z",
    "size": 1024
  }
]
```

### GET /rag/documents/{tenant_id}/{user_id}/{document_id}

Retrieve a document from S3.

**Response:**
```json
{
  "document_id": "italian_cuisine_guide",
  "content": "Italian Cuisine Guide\n\nItalian food is renowned worldwide...",
  "content_bytes": "base64_encoded_content",
  "content_type": "text/plain",
  "metadata": {
    "document_id": "italian_cuisine_guide",
    "tenant_id": "org1",
    "user_id": "user123",
    "upload_timestamp": "2024-01-15T10:30:00Z"
  },
  "s3_key": "memory-agent-documents/org1/user123/2024/01/italian_cuisine_guide.txt",
  "content_length": 1024
}
```

### GET /rag/documents/{tenant_id}/{user_id}/{document_id}/url

Generate a presigned URL for document access.

**Query Parameters:**
- `expiration`: URL expiration time in seconds (default: 3600)

**Response:**
```json
{
  "url": "https://your-bucket.s3.amazonaws.com/path/to/document?AWSAccessKeyId=...",
  "expires_in": 3600
}
```

### DELETE /rag/documents/{tenant_id}/{user_id}/{document_id}

Delete a document from S3.

**Response:**
```json
{
  "status": "success",
  "message": "Document deleted"
}
```

## Configuration Options

### Server Configuration

The API server can be configured using command-line arguments:

```bash
python run_api.py --host 0.0.0.0 --port 8000 --reload --log-level INFO
```

**Available options:**
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--workers`: Number of worker processes (default: 1)

### Environment Variables

All memory agent configuration options are available via environment variables. See the main README.md for the complete list.

## Error Handling

The API returns appropriate HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request data
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: Server-side errors

Error responses include detailed error messages:

```json
{
  "detail": "Internal server error: Connection failed to Redis"
}
```

## Development

### Running in Development Mode

```bash
python run_api.py --reload --log-level DEBUG
```

### Testing the API

You can test the API using curl:

```bash
# Health check
curl http://localhost:8000/health

# Query memory
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are my preferences?",
    "tenant_id": "org1",
    "user_id": "user123",
    "agent_id": "assistant",
    "conversation_id": "conv_456"
  }'

# Write to memory
curl -X POST http://localhost:8000/write \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "I like Italian food",
    "assistant_text": "Noted!",
    "tenant_id": "org1",
    "user_id": "user123",
    "agent_id": "assistant",
    "conversation_id": "conv_456"
  }'

# Upload document for RAG
curl -X POST http://localhost:8000/rag/upload \
  -F "file=@document.pdf" \
  -F "tenant_id=org1" \
  -F "user_id=user123" \
  -F "document_id=my_document" \
  -F 'metadata={"category": "guide", "language": "english"}'

# Query RAG documents
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are Italian food characteristics?",
    "tenant_id": "org1",
    "user_id": "user123",
    "limit": 5
  }'
```

### Python RAG Usage Example

```python
import aiohttp
import asyncio

async def rag_example():
    async with aiohttp.ClientSession() as session:
        # Upload a document
        with open('document.pdf', 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='document.pdf')
            data.add_field('tenant_id', 'org1')
            data.add_field('user_id', 'user123')
            data.add_field('document_id', 'my_guide')
            data.add_field('metadata', '{"category": "guide", "language": "english"}')
            
            async with session.post('http://localhost:8000/rag/upload', data=data) as resp:
                upload_result = await resp.json()
                print(f"Document uploaded: {upload_result['document_id']}")
        
        # Query RAG documents
        async with session.post('http://localhost:8000/rag/query', json={
            "query": "What are the characteristics of Italian cuisine?",
            "tenant_id": "org1",
            "user_id": "user123",
            "limit": 3
        }) as resp:
            rag_result = await resp.json()
            print(f"Found {rag_result['results_count']} relevant chunks")
            print(f"Context: {rag_result['context_content'][:100]}...")
        
        # Query with RAG integration (full memory system)
        async with session.post('http://localhost:8000/query', json={
            "query": "Tell me about Italian food",
            "tenant_id": "org1",
            "user_id": "user123",
            "agent_id": "assistant",
            "conversation_id": "conv_456"
        }) as resp:
            full_result = await resp.json()
            print(f"Full context: {full_result['merged_context'][:200]}...")

asyncio.run(rag_example())
```

## Complete Demo Workflow

### 1. Upload Documents to S3 + Pinecone

```bash
# Upload a PDF document
curl -X POST http://localhost:8000/rag/upload \
  -F "file=@italian_cuisine_guide.pdf" \
  -F "tenant_id=demo_org" \
  -F "user_id=demo_user" \
  -F "document_id=italian_cuisine_guide" \
  -F 'metadata={"category": "food_guide", "language": "english"}'

# Upload another document
curl -X POST http://localhost:8000/rag/upload \
  -F "file=@wine_guide.pdf" \
  -F "tenant_id=demo_org" \
  -F "user_id=demo_user" \
  -F "document_id=wine_guide" \
  -F 'metadata={"category": "wine_guide", "language": "english"}'
```

### 2. Query Documents with Pinecone Vector Search

```bash
# Direct RAG query
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key characteristics of Italian cuisine?",
    "tenant_id": "demo_org",
    "user_id": "demo_user",
    "limit": 3
  }'

# Full memory system query (includes RAG + other memory types)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about Italian food and wine",
    "tenant_id": "demo_org",
    "user_id": "demo_user",
    "agent_id": "travel_agent",
    "conversation_id": "italy_trip_001"
  }'
```

### 3. Manage Documents in S3

```bash
# List all documents
curl http://localhost:8000/rag/documents/demo_org/demo_user

# Get document content
curl http://localhost:8000/rag/documents/demo_org/demo_user/italian_cuisine_guide

# Get presigned URL for direct S3 access
curl http://localhost:8000/rag/documents/demo_org/demo_user/italian_cuisine_guide/url?expiration=7200

# Delete document from S3
curl -X DELETE http://localhost:8000/rag/documents/demo_org/demo_user/italian_cuisine_guide
```

### 4. Complete Python Demo

```python
import asyncio
import aiohttp
import json

async def complete_demo():
    async with aiohttp.ClientSession() as session:
        # 1. Upload documents
        print("📄 Uploading documents...")
        print("   Note: Use /rag/upload endpoint to upload actual files (PDF, HTML, text)")
        print("   Example:")
        print("   with open('document.pdf', 'rb') as f:")
        print("       data = aiohttp.FormData()")
        print("       data.add_field('file', f, filename='document.pdf')")
        print("       data.add_field('tenant_id', 'demo_org')")
        print("       data.add_field('user_id', 'demo_user')")
        print("       data.add_field('document_id', 'my_document')")
        print("       async with session.post('http://localhost:8000/rag/upload', data=data) as resp:")
        print("           result = await resp.json()")
        print("           print(f'Document uploaded: {result[\"document_id\"]}')")
        
        # 2. Query with Pinecone vector search
        print("\n🔍 Querying with Pinecone...")
        async with session.post('http://localhost:8000/rag/query', json={
            "query": "What are the key characteristics of Italian cuisine?",
            "tenant_id": "demo_org",
            "user_id": "demo_user",
            "limit": 3
        }) as resp:
            rag_result = await resp.json()
            print(f"✅ Found {rag_result['results_count']} relevant chunks")
            print(f"   Max relevance: {rag_result['max_relevance']:.2f}")
            print(f"   Documents: {rag_result['documents']}")
        
        # 3. List documents in S3
        print("\n📋 Listing documents in S3...")
        async with session.get('http://localhost:8000/rag/documents/demo_org/demo_user') as resp:
            documents = await resp.json()
            print(f"✅ Found {len(documents)} documents in S3")
            for doc in documents:
                print(f"   - {doc['document_id']}: {doc['content_type']}")
        
        # 4. Get presigned URL
        print("\n🔗 Getting presigned URL...")
        async with session.get('http://localhost:8000/rag/documents/demo_org/demo_user/italian_cuisine_guide/url') as resp:
            url_result = await resp.json()
            print(f"✅ Presigned URL: {url_result['url'][:50]}...")
            print(f"   Expires in: {url_result['expires_in']} seconds")
        
        # 5. Full memory system integration
        print("\n🧠 Testing full memory system...")
        
        # Write conversation data
        async with session.post('http://localhost:8000/write', json={
            "user_text": "I'm planning a trip to Italy and want to learn about the food",
            "assistant_text": "Great! Let me help you learn about Italian cuisine.",
            "tenant_id": "demo_org",
            "user_id": "demo_user",
            "agent_id": "travel_agent",
            "conversation_id": "italy_trip_001"
        }) as resp:
            write_result = await resp.json()
            print(f"✅ Conversation written to: {write_result['write_decision']['systems_written']}")
        
        # Query full memory system
        async with session.post('http://localhost:8000/query', json={
            "query": "Tell me about Italian cuisine for my trip",
            "tenant_id": "demo_org",
            "user_id": "demo_user",
            "agent_id": "travel_agent",
            "conversation_id": "italy_trip_001"
        }) as resp:
            full_result = await resp.json()
            print(f"✅ Full memory query completed")
            print(f"   Systems queried: {full_result['routing_decision']['systems_queried']}")
            print(f"   Context blocks: {len(full_result['context_blocks'])}")
            
            # Show which systems provided context
            for block in full_result['context_blocks']:
                system = block['system']
                if system == 'rag':
                    print(f"   - RAG: Found {len(block.get('records', []))} document chunks")
                else:
                    print(f"   - {system.upper()}: Additional context")
        
        print("\n🎉 Demo completed successfully!")
        print("\nArchitecture Summary:")
        print("✅ Documents stored in S3 with organized folder structure")
        print("✅ Vector embeddings stored in Pinecone for fast similarity search")
        print("✅ API provides unified interface for document management")
        print("✅ Full memory system integrates RAG with other memory types")

asyncio.run(complete_demo())
```

## Production Deployment

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY run_api.py .

EXPOSE 8000

CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn

```bash
pip install gunicorn
gunicorn memory_agent.web_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Setup

For production, ensure all required environment variables are set:

```bash
export OPENAI_API_KEY="your_key"
export PINECONE_API_KEY="your_key"
export PINECONE_ENVIRONMENT="your_environment"
export MEMORY_REDIS_HOST="your_redis_host"
export MEMORY_REDIS_PORT="6379"
```

## Monitoring and Logging

The API includes comprehensive logging and monitoring:

- Request/response logging
- Performance metrics (query/write times)
- Error tracking
- Health check endpoint for monitoring systems

## Security Considerations

- Configure CORS appropriately for production
- Use HTTPS in production
- Validate and sanitize all input data
- Implement rate limiting if needed
- Secure API keys and credentials
