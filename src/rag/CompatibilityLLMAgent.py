import os
import sys
import asyncio
from smart_sdk.agents import SMARTLLMAgent
from smart_sdk import Console
from src.rag.model_config import model

system_message = """
You are ContextQueryAgent, an expert assistant for analyzing user queries about software compatibility and IT systems.

Your job is to:
- Analyze the user's question
- Determine the user's intent (e.g., compatibility_check, upgrade_advice, general_info, etc.)
- Extract relevant entities such as software name, current version, target version, operating system, and any other key details
- Return a structured output with the intent and extracted entities

Instructions:
- Do NOT answer the compatibility question directly
- Only analyze and extract intent and entities
- If information is missing, leave the field blank or as null
- Use a structured JSON format for your output, e.g.:
  {
    "intent": "compatibility_check",
    "software": "Apache",
    "from_version": "2.4.41",
    "to_version": "2.4.50",
    "os": "Ubuntu 20.04"
  }
- If the intent is unclear, set intent to "unknown"
- Be concise and accurate
"""

# Define the agent at the module level for import
agent = SMARTLLMAgent(
    name="ContextQueryAgent",
    model_client=model(),
    system_message=system_message,
    description="An agent for extracting intent and entities from user queries about software compatibility."
)

async def main():
    user_question = "Can I upgrade Apache from 2.4.41 to 2.4.50 on Ubuntu 20.04?"
    await Console(agent.run_stream(task=user_question))

if __name__ == "__main__":
    asyncio.run(main())
