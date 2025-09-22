#!/usr/bin/env python3

import os
import sys
import asyncio
from pathlib import Path

# Add the memory agent to the path
sys.path.insert(0, str(Path(__file__).parent / "memory-agent" / "src"))

from memory_agent.config import MemoryConfig
from memory_agent.stores import create_vector_store

async def test_pinecone_integration():
    """Test Pinecone integration with the new API."""
    print("🧪 Testing Pinecone Integration")
    print("=" * 50)
    
    # Create config
    config = MemoryConfig()
    
    # Check if Pinecone is available
    try:
        from pinecone import Pinecone, ServerlessSpec
        print("✅ Pinecone client imported successfully")
    except ImportError as e:
        print(f"❌ Pinecone client not available: {e}")
        print("💡 Install with: pip install pinecone-client")
        return False
    
    # Check API key
    if not config.pinecone_api_key:
        print("⚠️  PINECONE_API_KEY not set - using fallback store")
        print("💡 Set PINECONE_API_KEY environment variable to test Pinecone")
        
        # Test fallback behavior
        try:
            store = create_vector_store(config, "pinecone", "test_collection")
            print(f"✅ Fallback store created: {type(store).__name__}")
            return True
        except Exception as e:
            print(f"❌ Fallback store creation failed: {e}")
            return False
    
    print(f"✅ Pinecone API key configured")
    print(f"✅ Environment: {config.pinecone_environment}")
    
    # Test store creation
    try:
        store = create_vector_store(config, "pinecone", "test-collection")
        print(f"✅ Pinecone store created: {type(store).__name__}")
        
        # Test stats
        stats = await store.get_stats()
        print(f"✅ Store stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pinecone store creation failed: {e}")
        print("💡 Check your API key and environment settings")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_pinecone_integration())
    if success:
        print("\n🎉 Pinecone integration test completed successfully!")
    else:
        print("\n❌ Pinecone integration test failed!")
        sys.exit(1)
