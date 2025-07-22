import os
import sys
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smart_sdk.agents import SMARTLLMAgent
from smart_sdk import Console
from smart_sdk.model import AzureOpenAIChatCompletionClient
from src.Agents.model_config import model

class ChatTitleAgent:
    def __init__(self):
        self.agent = SMARTLLMAgent(
            name="ChatTitleAgent",
            model_client=model(),
            system_message="""
You are a helpful assistant. Summarize the following user message as a short, descriptive chat title (3-5 words, no punctuation, no quotes, no extra explanation). Only return the title.
""",
            description="An agent for generating short chat titles from user messages."
        )

    async def generate_title(self, message: str) -> str:
        prompt = f"Summarize this message as a chat title in 3-5 words: {message}"
        result = await self.agent.run(task=prompt)
        
        # Handle TaskResult object properly
        if hasattr(result, 'content'):
            return str(result.content).strip()
        elif hasattr(result, 'strip'):
            return str(result).strip()
        else:
            return str(result).strip()