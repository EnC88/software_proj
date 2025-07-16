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
    result = ""
    async for chunk in router_agent.run_stream(task=query):
        if hasattr(chunk, "content"):
            result += str(chunk.content)
        else:
            result += str(chunk)
    
    # Try to parse JSON response
    import json
    import re
    
    try:
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        else:
            # Fallback: simple keyword-based routing
            query_lower = query.lower()
            casual_keywords = ['hello', 'hi', 'hey', 'how are you', 'what can you do', 'thanks', 'thank you']
            if any(keyword in query_lower for keyword in casual_keywords):
                return {
                    "type": "casual",
                    "response": "Hello! I'm here to help with software compatibility questions. How can I assist you today?",
                    "intent": None
                }
            else:
                return {
                    "type": "technical",
                    "response": None,
                    "intent": "compatibility_check"
                }
    except Exception as e:
        print(f"Error parsing router response: {e}")
        # Fallback to technical
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
        result = await route_query(query)
        print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 