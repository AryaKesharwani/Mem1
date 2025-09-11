import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

@dataclass
class MemoryConfig:
    """Configuration for memory systems with feature flags and thresholds."""
    
    # Feature flags - enable/disable memory systems
    enable_semantic: bool = True
    enable_episodic: bool = True
    enable_rag: bool = True
    enable_stm: bool = True
    
    # Confidence thresholds for persistence
    semantic_confidence_threshold: float = 0.7  # Only persist facts above this confidence
    episodic_confidence_threshold: float = 0.6  # Lower threshold for events
    rag_relevance_threshold: float = 0.5        # Minimum relevance for RAG results
    
    # Token budget management
    default_token_budget: int = 4000            # Total tokens available for context
    stm_token_allocation: int = 1000            # Reserved tokens for STM
    semantic_token_allocation: int = 800        # Max tokens for semantic context
    episodic_token_allocation: int = 1200       # Max tokens for episodic context
    rag_token_allocation: int = 1000            # Max tokens for RAG context
    
    # Memory system weights for scoring
    recency_weight: float = 0.3                 # How much to weight recent items
    importance_weight: float = 0.4              # How much to weight important items
    relevance_weight: float = 0.3               # How much to weight relevance
    
    # STM (Short-term memory) settings
    stm_window_size: int = 10                   # Number of conversation turns to keep
    stm_cache_ttl: int = 3600                   # Cache TTL in seconds (1 hour)
    stm_max_turns: int = 10                     # Maximum turns to return in queries
    # Redis settings for STM
    use_redis_stm: bool = True                  # Use Redis backend for STM
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_prefix: str = "memory_agent"         # Namespace prefix for Redis keys
    
    # Semantic memory settings
    semantic_max_results: int = 5               # Maximum semantic facts to return
    # Include low-confidence semantic items as fallback (top-k)
    semantic_include_low_conf: bool = True
    semantic_low_conf_fallback_k: int = 2
    
    # Episodic memory settings
    episodic_max_results: int = 5               # Maximum episodic events to return
    
    # RAG memory settings
    rag_max_chunks: int = 3                     # Maximum RAG chunks to return
    
    # Vector store settings
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model
    vector_dimension: int = 384                 # Embedding dimension
    similarity_threshold: float = 0.7           # Minimum similarity for matches
    
    # RAG settings
    chunk_size: int = 512                       # Document chunk size in tokens
    chunk_overlap: int = 50                     # Overlap between chunks
    max_chunks_per_doc: int = 150               # Maximum chunks per document
    
    # LLM settings for arbiter
    enable_llm_arbiter: bool = True             # Enable LLM fallback for routing
    arbiter_confidence_threshold: float = 0.7   # Use arbiter when below this confidence
    llm_model: str = "gpt-3.5-turbo"           # Model for arbiter decisions
    llm_temperature: float = 0.1                # Low temperature for consistent routing
    
    # OpenAI API key
    openai_api_key: str = os.getenv("OPENAI_API_KEY")  # Load from environment variable
    
    # Data paths
    data_root: str = "data"                     # Root directory for data storage
    
    def __post_init__(self):
        """Validate configuration and apply environment overrides."""
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate token allocations don't exceed budget
        total_allocation = (
            self.stm_token_allocation + 
            self.semantic_token_allocation + 
            self.episodic_token_allocation + 
            self.rag_token_allocation
        )
        if total_allocation > self.default_token_budget:
            raise ValueError(f"Token allocations ({total_allocation}) exceed budget ({self.default_token_budget})")
        
        # Validate weights sum to 1.0
        weight_sum = self.recency_weight + self.importance_weight + self.relevance_weight
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {weight_sum}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Feature flags
        self.enable_semantic = self._get_bool_env("MEMORY_ENABLE_SEMANTIC", self.enable_semantic)
        self.enable_episodic = self._get_bool_env("MEMORY_ENABLE_EPISODIC", self.enable_episodic)
        self.enable_rag = self._get_bool_env("MEMORY_ENABLE_RAG", self.enable_rag)
        self.enable_stm = self._get_bool_env("MEMORY_ENABLE_STM", self.enable_stm)
        
        # Thresholds
        self.semantic_confidence_threshold = self._get_float_env("MEMORY_SEMANTIC_THRESHOLD", self.semantic_confidence_threshold)
        self.episodic_confidence_threshold = self._get_float_env("MEMORY_EPISODIC_THRESHOLD", self.episodic_confidence_threshold)
        self.rag_relevance_threshold = self._get_float_env("MEMORY_RAG_THRESHOLD", self.rag_relevance_threshold)
        
        # Token budgets
        self.default_token_budget = self._get_int_env("MEMORY_TOKEN_BUDGET", self.default_token_budget)
        self.stm_token_allocation = self._get_int_env("MEMORY_STM_TOKENS", self.stm_token_allocation)
        self.semantic_low_conf_fallback_k = self._get_int_env("MEMORY_SEMANTIC_LOWCONF_K", self.semantic_low_conf_fallback_k)
        self.semantic_include_low_conf = self._get_bool_env("MEMORY_SEMANTIC_INCLUDE_LOWCONF", self.semantic_include_low_conf)
        
        # LLM settings
        self.llm_model = os.getenv("MEMORY_LLM_MODEL", self.llm_model)
        self.embedding_model = os.getenv("MEMORY_EMBEDDING_MODEL", self.embedding_model)
        
        # Data paths
        self.data_root = os.getenv("MEMORY_DATA_ROOT", self.data_root)

        # Redis connection overrides
        self.use_redis_stm = self._get_bool_env("MEMORY_USE_REDIS_STM", self.use_redis_stm)
        self.redis_host = os.getenv("MEMORY_REDIS_HOST", self.redis_host)
        self.redis_port = self._get_int_env("MEMORY_REDIS_PORT", self.redis_port)
        self.redis_db = self._get_int_env("MEMORY_REDIS_DB", self.redis_db)
        self.redis_password = os.getenv("MEMORY_REDIS_PASSWORD", self.redis_password or "") or None
        self.redis_ssl = self._get_bool_env("MEMORY_REDIS_SSL", self.redis_ssl)
        self.redis_prefix = os.getenv("MEMORY_REDIS_PREFIX", self.redis_prefix)
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")
    
    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
    
    def _get_float_env(self, key: str, default: float) -> float:
        """Get float environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    
    def get_collection_path(self, tenant_id: str, user_id: str, collection_type: str) -> str:
        """Get path for user-specific collection."""
        return os.path.join(self.data_root, tenant_id, "Users", user_id, f"{collection_type}_collection")
    
    def get_shared_collection_path(self, tenant_id: str, collection_type: str) -> str:
        """Get path for shared organization collection."""
        return os.path.join(self.data_root, tenant_id, "Shared", f"org_{collection_type}_collection")
    
    def is_memory_system_enabled(self, memory_type: str) -> bool:
        """Check if a memory system is enabled."""
        return getattr(self, f"enable_{memory_type}", False)
    
    def get_token_allocation(self, memory_type: str) -> int:
        """Get token allocation for a memory system."""
        return getattr(self, f"{memory_type}_token_allocation", 0)


def load_config(config_path: Optional[str] = None) -> MemoryConfig:
    """Load configuration from file or environment variables."""
    if config_path and os.path.exists(config_path):
        # TODO: Add YAML config file loading if needed
        pass
    
    return MemoryConfig()


def get_config() -> MemoryConfig:
    """Get the global configuration instance."""
    return load_config()
