import os
import sys
import asyncio
from smart_sdk.agents import SMARTLLMAgent
from smart_sdk import Console
from src.Agents.model_config import model

system_message = """
You are a smart router for a software compatibility chatbot. Your job is simple:

1. **Analyze the user's query**
2. **Determine if it's:**
   - A casual/general question (greetings, "what can you do", etc.)
   - A technical software compatibility question

3. **Return a simple JSON response:**
   {
     "type": "casual" | "technical",
     "response": "direct response for casual questions",
     "intent": "extracted intent for technical questions"
   }

**Examples:**
- "Hello" → {"type": "casual", "response": "Hello! I'm here to help with software compatibility questions. How can I assist you today?", "intent": null}
- "What can you do?" → {"type": "casual", "response": "I'm a software compatibility assistant. I can help you check if software versions are compatible, analyze upgrade paths, and provide technical recommendations!", "intent": null}
- "Can I upgrade Apache?" → {"type": "technical", "response": null, "intent": "compatibility_check for Apache upgrade"}
- "Is MySQL compatible with Ubuntu?" → {"type": "technical", "response": null, "intent": "compatibility_check for MySQL on Ubuntu"}

**Rules:**
- If it's casual (hello, thanks, general questions), give a friendly response
- If it's technical (software, compatibility, upgrades), extract the intent
- Be decisive and accurate
"""

# Create the agent
router_agent = SMARTLLMAgent(
    name="SimpleRouter",
    model_client=model(),
    system_message=system_message,
    description="Simple router for query classification"
)

async def route_query(query: str):
    """Route a query to either casual response or RAG processing."""
    try:
        # Use the agent's run method instead of run_stream for simpler handling
        result = await router_agent.run(task=query)
        
        # Convert result to string if it's not already
        if hasattr(result, 'content'):
            result_str = str(result.content)
        elif hasattr(result, 'strip'):
            result_str = str(result)
        else:
            result_str = str(result)
        
        # Try to parse JSON response
        import json
        import re
        
        json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        else:
            # If LLM didn't return valid JSON, default to technical (safer for software compatibility bot)
            print(f"LLM response not in expected JSON format: {result_str}")
            return {
                "type": "technical",
                "response": None,
                "intent": "compatibility_check"
            }
    except Exception as e:
        print(f"Error in route_query: {e}")
        # If LLM fails, default to technical (safer for software compatibility bot)
        return {
            "type": "technical",
            "response": None,
            "intent": "compatibility_check"
        }

async def main():
    """Test the simple router."""
    test_queries = [
        "Hello, how are you?",
        "What can you help me with?",
        "Can I upgrade Apache from 2.4.41 to 2.4.50?",
        "Is MySQL compatible with Ubuntu 20.04?",
        "Thanks for your help!"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = await route_query(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error testing query '{query}': {e}")

if __name__ == "__main__":
    asyncio.run(main()) 