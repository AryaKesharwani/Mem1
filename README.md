# Memory Agent - Multi-System Memory Orchestrator

A sophisticated memory system for AI agents that coordinates between multiple memory types to provide intelligent context retrieval and storage.

## 🧠 Memory Systems

- **STM (Short-term Memory)**: Conversation context and working memory
- **Semantic Memory**: Facts, preferences, and stable knowledge about users
- **Episodic Memory**: Events, experiences, and temporal context
- **RAG Memory**: Document-based knowledge retrieval

## 🚀 Features

- **Intelligent Routing**: Deterministic rules with LLM arbiter fallback
- **Multi-tenant Support**: Organization and user-level data isolation
- **Token Budget Management**: Efficient context merging within limits
- **Confidence Scoring**: Quality-based memory persistence
- **Fallback Mechanisms**: Graceful degradation when systems fail
- **Comprehensive Testing**: 10+ test scenarios with edge case handling

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Yashkalwar/Mem1.git
cd Mem1
```

2. Install dependencies:
```bash
cd memory-agent
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## 🔧 Configuration

The system uses environment variables for configuration:

```bash
OPENAI_API_KEY=your_openai_api_key_here
MEMORY_ENABLE_SEMANTIC=true
MEMORY_ENABLE_EPISODIC=true
MEMORY_ENABLE_RAG=true
MEMORY_ENABLE_STM=true
MEMORY_TOKEN_BUDGET=4000
```

## 🎯 Usage

### Basic Query Example

```python
from memory_agent import handle, post_write

# Write conversation data
await post_write(
    user_text="I prefer Italian food over Chinese food",
    assistant_text="I'll remember your preference for Italian cuisine",
    tenant_id="org1",
    user_id="user123",
    agent_id="assistant",
    conversation_id="conv_456"
)

# Query for context
result = await handle(
    query="What are my food preferences?",
    tenant_id="org1",
    user_id="user123",
    agent_id="assistant",
    conversation_id="conv_456"
)

print(result["merged_context"])
```

### Response Format

```json
{
    "merged_context": "[STM] Recent conversation...\n[SEMANTIC] User prefers Italian food...",
    "routing_decision": {
        "use_stm": true,
        "use_semantic": true,
        "confidence": 0.85,
        "reasoning": "preference patterns detected"
    },
    "query_metadata": {
        "query_time_ms": 45.2,
        "systems_queried": ["stm", "semantic"],
        "total_tokens": 150
    }
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_memory_agent.py
```

Or use the test runner:

```bash
python run_tests.py
```

### Test Coverage

- System health and configuration validation
- All memory systems (STM, Semantic, Episodic, RAG)
- Routing decisions and confidence scoring
- Edge cases (empty queries, special characters)
- Fallback mechanisms and error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer     │───▶│  Memory Agent    │───▶│  Memory Router  │
│  (handle/write) │    │  (Orchestrator)  │    │ (Route Queries) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   LLM Arbiter    │    │ Memory Systems  │
                       │ (Fallback Logic) │    │ STM│SEM│EPI│RAG │
                       └──────────────────┘    └─────────────────┘
```

## 🔍 Key Improvements

### STM Memory Persistence
- Fixed instance isolation with global buffer sharing
- Improved conversation context retention
- Enhanced fallback mechanisms

### Semantic Memory Enhancement
- Better preference extraction patterns
- Comparative preference support ("I prefer X over Y")
- Confidence-based fact persistence

### Router Integration
- Deterministic routing with LLM fallback
- Confidence threshold management
- Comprehensive error handling

## 📊 Performance

- **Query Response Time**: ~45ms average
- **Token Budget**: Configurable (default 4000 tokens)
- **Memory Systems**: Parallel querying for optimal performance
- **Fallback Latency**: <100ms for degraded operations

## 🛠️ Development

### Project Structure

```
memory-agent/
├── src/memory_agent/
│   ├── agent.py          # Main orchestrator
│   ├── router.py         # Routing logic
│   ├── arbiter.py        # LLM fallback
│   ├── stm.py           # Short-term memory
│   ├── memory_store.py   # Semantic/Episodic
│   ├── rag.py           # Document retrieval
│   └── config.py        # Configuration
├── requirements.txt
└── README.md
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Support

For issues and questions:
- Create an issue on GitHub
- Check the test suite for usage examples
- Review the configuration options in `config.py`
