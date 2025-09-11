"""
Memory Arbiter - LLM-based routing decision fallback.

When deterministic routing rules have low confidence, the arbiter uses
an LLM to make more nuanced routing decisions about which memory systems
to query or write to.
"""

import logging
from typing import Dict, List, Optional, Any
import json

from .config import MemoryConfig
from .models import MemoryType


class MemoryArbiter:
    """
    LLM-based arbiter for memory routing decisions.
    
    Used as a fallback when deterministic routing rules have low confidence.
    Makes intelligent decisions about which memory systems to use based on
    query context and conversation state.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize the memory arbiter."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # LLM client will be initialized when needed
        self._llm_client = None
    
    async def make_decision(
        self,
        query_context: Dict[str, Any],
        current_routing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make routing decision using LLM when deterministic rules have low confidence.
        
        Args:
            query_context: Context about the query and user
            current_routing: Current routing decision from deterministic rules
            
        Returns:
            Dict with updated routing decisions and reasoning
        """
        try:
            # Prepare prompt for LLM
            prompt = self._build_routing_prompt(query_context, current_routing)
            
            # Get LLM decision
            llm_response = await self._query_llm(prompt)
            
            # Parse and validate response
            decision = self._parse_llm_response(llm_response)
            
            # Update routing with LLM decision
            updated_routing = self._merge_decisions(current_routing, decision)
            
            self.logger.info(f"Arbiter decision: {decision.get('reasoning', 'No reasoning provided')}")
            
            return updated_routing
            
        except Exception as e:
            self.logger.error(f"Arbiter decision failed: {str(e)}")
            # Return original routing on error
            return {
                "confidence": max(current_routing.get("confidence", 0.5), 0.5),
                "reasoning": f"Arbiter failed, using deterministic routing: {str(e)}"
            }
    
    def _build_routing_prompt(
        self,
        query_context: Dict[str, Any],
        current_routing: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM routing decision."""
        query = query_context.get("query", "")
        topic_hint = query_context.get("topic_hint", "")
        user_metadata = query_context.get("user_metadata", {})
        
        prompt = f"""You are a memory routing expert. Analyze this query and decide which memory systems to use.

QUERY: "{query}"
TOPIC HINT: "{topic_hint}"
USER CONTEXT: {json.dumps(user_metadata, indent=2)}

CURRENT ROUTING (low confidence):
- Use STM: {current_routing.get('use_stm', False)}
- Use Semantic: {current_routing.get('use_semantic', False)}
- Use Episodic: {current_routing.get('use_episodic', False)}
- Use RAG: {current_routing.get('use_rag', False)}
- Confidence: {current_routing.get('confidence', 0.0)}
- Reasoning: {current_routing.get('reasoning', 'No reasoning')}

MEMORY SYSTEMS:
- STM (Short-term): Recent conversation context, working memory
- Semantic: Facts, preferences, stable knowledge about the user
- Episodic: Past events, experiences, temporal context
- RAG: Document-based knowledge, external information

INSTRUCTIONS:
1. Analyze the query intent and context
2. Decide which memory systems would be most helpful
3. Provide confidence score (0.0-1.0)
4. Explain your reasoning

Respond in JSON format:
{{
    "use_stm": true/false,
    "use_semantic": true/false,
    "use_episodic": true/false,
    "use_rag": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of decision"
}}"""
        
        return prompt
    
    async def _query_llm(self, prompt: str) -> str:
        """Query LLM for routing decision."""
        # Initialize LLM client if needed
        if self._llm_client is None:
            self._llm_client = self._init_llm_client()
        
        try:
            # Use OpenAI client (can be extended for other providers)
            if self.config.llm_model.startswith("gpt"):
                return await self._query_openai(prompt)
            else:
                # Fallback to simple heuristic
                return self._fallback_decision(prompt)
                
        except Exception as e:
            self.logger.warning(f"LLM query failed: {str(e)}")
            return self._fallback_decision(prompt)
    
    async def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API for routing decision."""
        try:
            from openai import AsyncOpenAI
            # Prefer cached client if available
            client = self._llm_client
            if client is None:
                client = self._init_llm_client()
                self._llm_client = client
            
            response = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a memory routing expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.llm_temperature,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except ImportError:
            self.logger.warning("OpenAI library not available, using fallback")
            return self._fallback_decision(prompt)
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return self._fallback_decision(prompt)
    
    def _fallback_decision(self, prompt: str) -> str:
        """Provide fallback decision when LLM is unavailable."""
        # Simple heuristic-based decision
        query_lower = prompt.lower()
        
        use_semantic = any(word in query_lower for word in [
            "prefer", "like", "favorite", "always", "never", "usually",
            "personal", "about me", "my", "i am", "i'm"
        ])
        
        use_episodic = any(word in query_lower for word in [
            "remember", "last time", "before", "previously", "when",
            "yesterday", "ago", "happened", "did", "was"
        ])
        
        use_rag = any(word in query_lower for word in [
            "document", "file", "article", "guide", "manual",
            "information about", "tell me about", "what is"
        ])
        
        return json.dumps({
            "use_stm": True,  # Always include conversation context
            "use_semantic": use_semantic,
            "use_episodic": use_episodic,
            "use_rag": use_rag,
            "confidence": 0.6,
            "reasoning": "Fallback heuristic decision (LLM unavailable)"
        })
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()
            
            decision = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["use_stm", "use_semantic", "use_episodic", "use_rag", "confidence"]
            for field in required_fields:
                if field not in decision:
                    decision[field] = False if field.startswith("use_") else 0.5
            
            # Ensure confidence is in valid range
            decision["confidence"] = max(0.0, min(1.0, float(decision["confidence"])))
            
            return decision
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse LLM response: {str(e)}")
            # Return safe default
            return {
                "use_stm": True,
                "use_semantic": True,
                "use_episodic": False,
                "use_rag": False,
                "confidence": 0.5,
                "reasoning": "Failed to parse LLM response, using safe defaults"
            }
    
    def _merge_decisions(
        self,
        current_routing: Dict[str, Any],
        llm_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge LLM decision with current routing."""
        # LLM decision takes precedence for routing flags
        merged = current_routing.copy()
        
        # Update routing decisions
        merged.update({
            "use_stm": llm_decision.get("use_stm", current_routing.get("use_stm", True)),
            "use_semantic": llm_decision.get("use_semantic", current_routing.get("use_semantic", False)),
            "use_episodic": llm_decision.get("use_episodic", current_routing.get("use_episodic", False)),
            "use_rag": llm_decision.get("use_rag", current_routing.get("use_rag", False)),
            "confidence": llm_decision.get("confidence", 0.5),
            "reasoning": f"Arbiter: {llm_decision.get('reasoning', 'LLM routing decision')}"
        })
        
        return merged
    
    def _init_llm_client(self):
        """Initialize LLM client based on configuration."""
        try:
            from openai import AsyncOpenAI
            if not self.config.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            return AsyncOpenAI(api_key=self.config.openai_api_key)
        except Exception as e:
            # Leave client as None; caller will fallback
            self.logger.debug(f"LLM client init skipped: {e}")
            return None
