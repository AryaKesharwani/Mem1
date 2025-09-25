# Memory Agent - Pinecone Configuration

## Overview

The Memory Agent has been configured to use **Pinecone** as the primary vector database for storing and retrieving semantic and episodic memories.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the `pinecone-client>=3.0.0` package.

### 2. Environment Configuration

Create a `.env` file in the memory-agent directory with the following variables:

```bash
# Pinecone Configuration (Primary Vector Database)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=aws-us-west-2

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Redis Configuration (for Short-term Memory)
MEMORY_REDIS_HOST=127.0.0.1
MEMORY_REDIS_PORT=6379
MEMORY_REDIS_DB=0
MEMORY_REDIS_PASSWORD=
MEMORY_REDIS_SSL=false
MEMORY_REDIS_PREFIX=memory_agent

# Memory System Configuration
MEMORY_ENABLE_SEMANTIC=true
MEMORY_ENABLE_EPISODIC=true
MEMORY_ENABLE_RAG=true
MEMORY_ENABLE_STM=true

# Token Budget Configuration
MEMORY_TOKEN_BUDGET=4000
MEMORY_STM_TOKENS=1000

# Model Configuration
MEMORY_LLM_MODEL=gpt-3.5-turbo
MEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Data Storage
MEMORY_DATA_ROOT=data
```

### 3. Pinecone Setup

1. **Create a Pinecone Account**: Sign up at [pinecone.io](https://pinecone.io)
2. **Get API Key**: Retrieve your API key from the Pinecone console
3. **Choose Environment**: Select your preferred environment (e.g., `us-west1-gcp`, `us-east1-gcp`)
4. **Set Environment Variables**: Add your API key and environment to the `.env` file

### 4. Fallback Behavior

The system is designed with graceful fallbacks:

1. **Primary**: Pinecone (if available and configured)
2. **Secondary**: ChromaDB (if Pinecone unavailable)
3. **Fallback**: In-memory store (if neither Pinecone nor ChromaDB available)

## Vector Store Architecture

### PineconeStore Features

- **Modern API**: Uses the latest Pinecone client API (Pinecone class)
- **Automatic Index Creation**: Creates Pinecone indexes automatically with ServerlessSpec
- **Metadata Filtering**: Supports filtering by tenant_id, user_id, memory_type, etc.
- **Batch Operations**: Efficient batch upsert operations
- **Cosine Similarity**: Uses cosine similarity for vector search
- **Statistics**: Provides index statistics and health metrics
- **Cloud Agnostic**: Supports both AWS and GCP environments

### Collection Naming

Collections are automatically named using the pattern:
```
{tenant_id}-{user_id}-{memory_type}-collection
```

Examples:
- `acme-user123-semantic-collection`
- `acme-user123-episodic-collection`

**Note**: Pinecone requires collection names to use only lowercase alphanumeric characters and hyphens. The system automatically sanitizes collection names to meet these requirements.

## Configuration Options

### MemoryConfig Settings

```python
# Pinecone settings
pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

# Vector store settings
embedding_model: str = "all-MiniLM-L6-v2"
vector_dimension: int = 384
similarity_threshold: float = 0.7
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Your Pinecone API key | Required |
| `PINECONE_ENVIRONMENT` | Pinecone environment (format: `aws-us-west-2`, `gcp-us-west1`) | `aws-us-west-2` |
| `MEMORY_EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `MEMORY_DATA_ROOT` | Data storage directory | `data` |

## Usage Example

```python
from memory_agent.config import MemoryConfig
from memory_agent.stores import create_vector_store

# Load configuration (automatically reads from .env)
config = MemoryConfig()

# Create Pinecone store
store = create_vector_store(
    config=config,
    store_type="pinecone",  # This is now the default
    collection_name="my_collection"
)

# The store will automatically fallback to ChromaDB or in-memory
# if Pinecone is not available or configured
```

## Migration from ChromaDB

If you were previously using ChromaDB, the migration is seamless:

1. Install Pinecone: `pip install pinecone-client>=3.0.0`
2. Set up your Pinecone API key and environment
3. The system will automatically use Pinecone as the primary store
4. ChromaDB remains available as a fallback option

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `pinecone-client>=3.0.0` is installed
2. **API Key Error**: Verify your Pinecone API key is correct
3. **Environment Error**: Check that your Pinecone environment exists
4. **Index Creation**: Ensure you have permissions to create indexes in your Pinecone project

### Debug Mode

Enable debug logging to see fallback behavior:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show warnings when falling back to alternative vector stores.
