#!/usr/bin/env python3
"""
Memory Agent API Client - Example usage of the REST API.

This script demonstrates how to interact with the Memory Agent API.
"""

import asyncio
import json
import aiohttp
import argparse
from typing import Dict, Any


class MemoryAgentClient:
    """Client for interacting with the Memory Agent API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query(self, query: str, tenant_id: str, user_id: str, 
                   agent_id: str, conversation_id: str, 
                   topic_hint: str = None, user_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query memory systems for context."""
        url = f"{self.base_url}/query"
        data = {
            "query": query,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "conversation_id": conversation_id
        }
        
        if topic_hint:
            data["topic_hint"] = topic_hint
        if user_metadata:
            data["user_metadata"] = user_metadata
        
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def write(self, user_text: str, assistant_text: str, tenant_id: str,
                   user_id: str, agent_id: str, conversation_id: str,
                   topic_hint: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Write conversation data to memory systems."""
        url = f"{self.base_url}/write"
        data = {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "conversation_id": conversation_id
        }
        
        if topic_hint:
            data["topic_hint"] = topic_hint
        if metadata:
            data["metadata"] = metadata
        
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def health(self) -> Dict[str, Any]:
        """Check system health."""
        url = f"{self.base_url}/health"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    
    async def info(self) -> Dict[str, Any]:
        """Get system information."""
        url = f"{self.base_url}/info"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    
    async def upload_document(self, file_path: str, tenant_id: str, user_id: str,
                            document_id: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload a document for RAG processing."""
        url = f"{self.base_url}/rag/upload"
        
        # Prepare form data
        data = aiohttp.FormData()
        data.add_field('tenant_id', tenant_id)
        data.add_field('user_id', user_id)
        if document_id:
            data.add_field('document_id', document_id)
        if metadata:
            import json
            data.add_field('metadata', json.dumps(metadata))
        
        # Add file
        with open(file_path, 'rb') as f:
            data.add_field('file', f, filename=file_path.split('/')[-1])
        
        async with self.session.post(url, data=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def query_rag(self, query: str, tenant_id: str, user_id: str,
                       limit: int = 5, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query RAG documents directly."""
        url = f"{self.base_url}/rag/query"
        data = {
            "query": query,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "limit": limit
        }
        if filters:
            data["filters"] = filters
        
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()


async def interactive_mode():
    """Interactive mode for testing the API."""
    print("Memory Agent API Interactive Mode")
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 50)
    
    async with MemoryAgentClient() as client:
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'help':
                    print("Commands:")
                    print("  health - Check system health")
                    print("  info - Get system information")
                    print("  write - Write conversation data")
                    print("  query - Query for context")
                    print("  upload - Upload document for RAG")
                    print("  rag-query - Query RAG documents directly")
                    print("  quit - Exit")
                elif command == 'health':
                    result = await client.health()
                    print(json.dumps(result, indent=2))
                elif command == 'info':
                    result = await client.info()
                    print(json.dumps(result, indent=2))
                elif command == 'write':
                    print("Enter conversation data:")
                    user_text = input("User text: ")
                    assistant_text = input("Assistant text: ")
                    tenant_id = input("Tenant ID (default: demo): ") or "demo"
                    user_id = input("User ID (default: user1): ") or "user1"
                    agent_id = input("Agent ID (default: agent1): ") or "agent1"
                    conversation_id = input("Conversation ID (default: conv1): ") or "conv1"
                    
                    result = await client.write(
                        user_text=user_text,
                        assistant_text=assistant_text,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        conversation_id=conversation_id
                    )
                    print(json.dumps(result, indent=2))
                elif command == 'query':
                    query = input("Query: ")
                    tenant_id = input("Tenant ID (default: demo): ") or "demo"
                    user_id = input("User ID (default: user1): ") or "user1"
                    agent_id = input("Agent ID (default: agent1): ") or "agent1"
                    conversation_id = input("Conversation ID (default: conv1): ") or "conv1"
                    
                    result = await client.query(
                        query=query,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        conversation_id=conversation_id
                    )
                    print(json.dumps(result, indent=2))
                elif command == 'upload':
                    file_path = input("File path: ")
                    tenant_id = input("Tenant ID (default: demo): ") or "demo"
                    user_id = input("User ID (default: user1): ") or "user1"
                    document_id = input("Document ID (optional): ") or None
                    
                    result = await client.upload_document(
                        file_path=file_path,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        document_id=document_id
                    )
                    print(json.dumps(result, indent=2))
                elif command == 'rag-query':
                    query = input("RAG query: ")
                    tenant_id = input("Tenant ID (default: demo): ") or "demo"
                    user_id = input("User ID (default: user1): ") or "user1"
                    limit = input("Limit (default: 5): ") or "5"
                    
                    result = await client.query_rag(
                        query=query,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        limit=int(limit)
                    )
                    print(json.dumps(result, indent=2))
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Memory Agent API Client")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive"],
        default="demo",
        help="Run mode (default: demo)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
