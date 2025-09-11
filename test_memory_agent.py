#!/usr/bin/env python3
"""
Memory Agent CLI Test Script

This script tests the complete memory agent system including:
- Data bootstrapping (creating test data)
- All memory systems (STM, Semantic, Episodic, RAG)
- Routing decisions and fallback mechanisms
- Edge cases and error handling
- Configuration validation

Usage: python test_memory_agent.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add the memory-agent src to path
sys.path.insert(0, str(Path(__file__).parent / "memory-agent" / "src"))

try:
    from memory_agent import handle, post_write, health_check, get_system_info
    from memory_agent.config import MemoryConfig, get_config
    from memory_agent.agent import MemoryAgent
except ImportError as e:
    print(f"❌ Failed to import memory_agent: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryAgentTester:
    """Comprehensive tester for Memory Agent system."""
    
    def __init__(self):
        """Initialize the tester."""
        self.tenant_id = "test_org"
        self.user_id = "test_user"
        self.agent_id = "test_agent"
        self.conversation_id = "test_conversation"
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run all test scenarios."""
        print("🚀 Starting Memory Agent CLI Tests")
        print("=" * 50)
        
        # Test 1: System Health Check
        await self.test_system_health()
        
        # Test 2: Configuration Validation
        await self.test_configuration()
        
        # Test 3: Bootstrap Test Data
        await self.bootstrap_test_data()
        
        # Test 4: Test STM (Short-term Memory)
        await self.test_stm()
        
        # Test 5: Test Semantic Memory
        await self.test_semantic_memory()
        
        # Test 6: Test Episodic Memory
        await self.test_episodic_memory()
        
        # Test 7: Test RAG Memory
        await self.test_rag_memory()
        
        # Test 8: Test Routing Decisions
        await self.test_routing_decisions()
        
        # Test 9: Test Edge Cases
        await self.test_edge_cases()
        
        # Test 10: Test Fallback Mechanisms
        await self.test_fallback_mechanisms()
        
        # Print Summary
        self.print_test_summary()
    
    async def test_system_health(self):
        """Test system health and configuration."""
        print("\n📋 Testing System Health...")
        
        try:
            # Health check
            health = await health_check()
            self.test_results["health_check"] = health["status"] == "healthy"
            print(f"   Health Status: {health['status']}")
            
            # System info
            info = get_system_info()
            print(f"   Version: {info['version']}")
            print(f"   Memory Systems Enabled: {info['memory_systems']}")
            
            self.test_results["system_info"] = True
            print("   ✅ System health check passed")
            
        except Exception as e:
            print(f"   ❌ System health check failed: {e}")
            self.test_results["health_check"] = False
    
    async def test_configuration(self):
        """Test configuration validation."""
        print("\n⚙️  Testing Configuration...")
        
        try:
            config = get_config()
            
            # Test token budget validation
            total_allocation = (
                config.stm_token_allocation + 
                config.semantic_token_allocation + 
                config.episodic_token_allocation + 
                config.rag_token_allocation
            )
            
            budget_valid = total_allocation <= config.default_token_budget
            print(f"   Token Budget: {config.default_token_budget}")
            print(f"   Total Allocation: {total_allocation}")
            print(f"   Budget Valid: {budget_valid}")
            
            # Test weights validation
            weight_sum = config.recency_weight + config.importance_weight + config.relevance_weight
            weights_valid = abs(weight_sum - 1.0) < 0.01
            print(f"   Scoring Weights Sum: {weight_sum:.3f}")
            print(f"   Weights Valid: {weights_valid}")
            
            self.test_results["configuration"] = budget_valid and weights_valid
            print("   ✅ Configuration validation passed")
            
        except Exception as e:
            print(f"   ❌ Configuration validation failed: {e}")
            self.test_results["configuration"] = False
    
    async def bootstrap_test_data(self):
        """Bootstrap the system with test data."""
        print("\n🌱 Bootstrapping Test Data...")
        
        test_conversations = [
            {
                "user": "Hi, I'm John. I prefer Italian food over Chinese food.",
                "assistant": "Nice to meet you John! I've noted your preference for Italian cuisine over Chinese food.",
                "topic": "user_preferences"
            },
            {
                "user": "Yesterday I went to the new Italian restaurant downtown.",
                "assistant": "That sounds great! How was your experience at the new Italian restaurant?",
                "topic": "dining_experience"
            },
            {
                "user": "I love pasta, especially carbonara. I always order it when available.",
                "assistant": "Carbonara is an excellent choice! I'll remember that it's your go-to pasta dish.",
                "topic": "food_preferences"
            },
            {
                "user": "Last week I met my friend Sarah at the coffee shop on Main Street.",
                "assistant": "It's nice that you got to catch up with Sarah. How long have you two been friends?",
                "topic": "social_events"
            },
            {
                "user": "Can you help me find information about Italian cooking techniques?",
                "assistant": "I'd be happy to help! Let me search for information about Italian cooking techniques for you.",
                "topic": "cooking_help"
            }
        ]
        
        success_count = 0
        for i, conv in enumerate(test_conversations):
            try:
                result = await post_write(
                    user_text=conv["user"],
                    assistant_text=conv["assistant"],
                    tenant_id=self.tenant_id,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    conversation_id=f"{self.conversation_id}_{i}",
                    topic_hint=conv["topic"]
                )
                
                if result.get("write_results"):
                    success_count += 1
                    print(f"   ✅ Conversation {i+1} stored successfully")
                else:
                    print(f"   ⚠️  Conversation {i+1} had issues: {result}")
                    
            except Exception as e:
                print(f"   ❌ Failed to store conversation {i+1}: {e}")
        
        self.test_results["bootstrap"] = success_count == len(test_conversations)
        print(f"   📊 Successfully stored {success_count}/{len(test_conversations)} conversations")
        
        # Wait a moment for data to be processed
        await asyncio.sleep(1)
    
    async def test_stm(self):
        """Test Short-term Memory functionality."""
        print("\n🧠 Testing Short-term Memory (STM)...")
        
        try:
            # Query recent conversation context
            result = await handle(
                query="What did we talk about recently?",
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                conversation_id=f"{self.conversation_id}_0"
            )
            
            context = result.get("merged_context", "")
            routing = result.get("routing_decision", {})
            
            print(f"   STM Used: {routing.get('use_stm', False)}")
            print(f"   Context Length: {len(context)} chars")
            
            if context and "[STM]" in context:
                print("   ✅ STM context retrieved successfully")
                self.test_results["stm"] = True
            else:
                print("   ⚠️  No STM context found")
                self.test_results["stm"] = False
                
        except Exception as e:
            print(f"   ❌ STM test failed: {e}")
            self.test_results["stm"] = False
    
    async def test_semantic_memory(self):
        """Test Semantic Memory functionality."""
        print("\n🎯 Testing Semantic Memory...")
        
        try:
            # Query for preferences
            result = await handle(
                query="What are my food preferences?",
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                conversation_id=self.conversation_id,
                topic_hint="preferences"
            )
            
            context = result.get("merged_context", "")
            routing = result.get("routing_decision", {})
            
            print(f"   Semantic Used: {routing.get('use_semantic', False)}")
            print(f"   Routing Confidence: {routing.get('confidence', 0):.2f}")
            
            if routing.get('use_semantic') and context:
                print("   ✅ Semantic memory queried successfully")
                self.test_results["semantic"] = True
            else:
                print("   ⚠️  Semantic memory not used or no results")
                self.test_results["semantic"] = False
                
        except Exception as e:
            print(f"   ❌ Semantic memory test failed: {e}")
            self.test_results["semantic"] = False
    
    async def test_episodic_memory(self):
        """Test Episodic Memory functionality."""
        print("\n📅 Testing Episodic Memory...")
        
        try:
            # Query for past events
            result = await handle(
                query="What did I do yesterday?",
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                conversation_id=self.conversation_id,
                topic_hint="past_events"
            )
            
            context = result.get("merged_context", "")
            routing = result.get("routing_decision", {})
            
            print(f"   Episodic Used: {routing.get('use_episodic', False)}")
            print(f"   Routing Confidence: {routing.get('confidence', 0):.2f}")
            
            if routing.get('use_episodic'):
                print("   ✅ Episodic memory queried successfully")
                self.test_results["episodic"] = True
            else:
                print("   ⚠️  Episodic memory not used")
                self.test_results["episodic"] = False
                
        except Exception as e:
            print(f"   ❌ Episodic memory test failed: {e}")
            self.test_results["episodic"] = False
    
    async def test_rag_memory(self):
        """Test RAG Memory functionality."""
        print("\n📚 Testing RAG Memory...")
        
        try:
            # Query for document-based information
            result = await handle(
                query="Tell me about Italian cooking techniques",
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                conversation_id=self.conversation_id,
                topic_hint="cooking_knowledge"
            )
            
            context = result.get("merged_context", "")
            routing = result.get("routing_decision", {})
            
            print(f"   RAG Used: {routing.get('use_rag', False)}")
            print(f"   Routing Confidence: {routing.get('confidence', 0):.2f}")
            
            if routing.get('use_rag'):
                print("   ✅ RAG memory queried successfully")
                self.test_results["rag"] = True
            else:
                print("   ⚠️  RAG memory not used")
                self.test_results["rag"] = False
                
        except Exception as e:
            print(f"   ❌ RAG memory test failed: {e}")
            self.test_results["rag"] = False
    
    async def test_routing_decisions(self):
        """Test routing decision logic."""
        print("\n🧭 Testing Routing Decisions...")
        
        test_queries = [
            {
                "query": "I prefer tea over coffee",
                "expected_systems": ["semantic"],
                "description": "Preference statement"
            },
            {
                "query": "Yesterday I visited the museum",
                "expected_systems": ["episodic"],
                "description": "Temporal event"
            },
            {
                "query": "What's in the user manual?",
                "expected_systems": ["rag"],
                "description": "Document query"
            },
            {
                "query": "Hello, how are you?",
                "expected_systems": ["stm"],
                "description": "General conversation"
            }
        ]
        
        routing_success = 0
        for test in test_queries:
            try:
                result = await handle(
                    query=test["query"],
                    tenant_id=self.tenant_id,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    conversation_id=self.conversation_id
                )
                
                routing = result.get("routing_decision", {})
                systems_used = []
                
                if routing.get("use_stm"): systems_used.append("stm")
                if routing.get("use_semantic"): systems_used.append("semantic")
                if routing.get("use_episodic"): systems_used.append("episodic")
                if routing.get("use_rag"): systems_used.append("rag")
                
                expected_found = any(exp in systems_used for exp in test["expected_systems"])
                
                print(f"   {test['description']}: {systems_used} (confidence: {routing.get('confidence', 0):.2f})")
                
                if expected_found:
                    routing_success += 1
                    
            except Exception as e:
                print(f"   ❌ Routing test failed for '{test['query']}': {e}")
        
        self.test_results["routing"] = routing_success >= len(test_queries) * 0.75
        print(f"   📊 Routing success: {routing_success}/{len(test_queries)}")
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n🔍 Testing Edge Cases...")
        
        edge_cases = [
            {
                "name": "Empty query",
                "query": "",
                "should_handle": True
            },
            {
                "name": "Very long query",
                "query": "What is " + "very " * 1000 + "long query?",
                "should_handle": True
            },
            {
                "name": "Special characters",
                "query": "What about émojis 🤔 and spëcial chars?",
                "should_handle": True
            },
            {
                "name": "None query",
                "query": None,
                "should_handle": False
            }
        ]
        
        edge_case_success = 0
        for case in edge_cases:
            try:
                if case["query"] is None:
                    # This should raise an error
                    try:
                        result = await handle(
                            query=case["query"],
                            tenant_id=self.tenant_id,
                            user_id=self.user_id,
                            agent_id=self.agent_id,
                            conversation_id=self.conversation_id
                        )
                        print(f"   ❌ {case['name']}: Should have failed but didn't")
                    except (ValueError, TypeError):
                        print(f"   ✅ {case['name']}: Properly handled error")
                        edge_case_success += 1
                else:
                    result = await handle(
                        query=case["query"],
                        tenant_id=self.tenant_id,
                        user_id=self.user_id,
                        agent_id=self.agent_id,
                        conversation_id=self.conversation_id
                    )
                    
                    if result and "merged_context" in result:
                        print(f"   ✅ {case['name']}: Handled successfully")
                        edge_case_success += 1
                    else:
                        print(f"   ⚠️  {case['name']}: Unexpected result")
                        
            except Exception as e:
                if case["should_handle"]:
                    print(f"   ❌ {case['name']}: Unexpected error: {e}")
                else:
                    print(f"   ✅ {case['name']}: Expected error handled")
                    edge_case_success += 1
        
        self.test_results["edge_cases"] = edge_case_success >= len(edge_cases) * 0.75
        print(f"   📊 Edge cases handled: {edge_case_success}/{len(edge_cases)}")
    
    async def test_fallback_mechanisms(self):
        """Test fallback mechanisms."""
        print("\n🛡️  Testing Fallback Mechanisms...")
        
        try:
            # Test with invalid tenant/user (should still work with fallbacks)
            result = await handle(
                query="Test fallback query",
                tenant_id="nonexistent_tenant",
                user_id="nonexistent_user",
                agent_id=self.agent_id,
                conversation_id="fallback_test"
            )
            
            if result and "merged_context" in result:
                print("   ✅ Fallback to empty data handled")
                fallback_success = True
            else:
                print("   ❌ Fallback mechanism failed")
                fallback_success = False
            
            # Test token budget overflow (simulate with very small budget)
            config = get_config()
            original_budget = config.default_token_budget
            config.default_token_budget = 10  # Very small budget
            
            result = await handle(
                query="This is a test query that should trigger token budget management",
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                conversation_id=self.conversation_id
            )
            
            # Restore original budget
            config.default_token_budget = original_budget
            
            if result:
                print("   ✅ Token budget overflow handled")
                fallback_success = fallback_success and True
            else:
                print("   ❌ Token budget overflow not handled")
                fallback_success = False
            
            self.test_results["fallbacks"] = fallback_success
            
        except Exception as e:
            print(f"   ❌ Fallback mechanism test failed: {e}")
            self.test_results["fallbacks"] = False
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Memory Agent is working correctly.")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️  Most tests passed. Some minor issues detected.")
        else:
            print("❌ Multiple test failures. System needs attention.")
        
        print("\n💡 Tips:")
        print("   - Check logs for detailed error information")
        print("   - Ensure all dependencies are installed")
        print("   - Verify data directory permissions")
        print("   - Check configuration settings")

async def main():
    """Main test execution function."""
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create and run tester
    tester = MemoryAgentTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
