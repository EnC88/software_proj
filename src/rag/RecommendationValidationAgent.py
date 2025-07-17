import os
import sys
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smart_sdk.agents import SMARTLLMAgent
from src.Agents.model_config import model

class RecommendationValidationAgent:
    def __init__(self):
        self.agent = SMARTLLMAgent(
            name="RecommendationValidationAgent",
            model_client=model(),
            system_message="""
You are an expert software compatibility assistant. Given a user's software change request, a list of initial recommendations, and context (such as affected software and environments), use your external knowledge to:
- Filter out recommendations that are not relevant, are direct competitors, or do not make sense as upgrades, dependencies, or compatible software.
- Only keep recommendations that are contextually appropriate and helpful for the user's request.
- For each recommendation you keep, provide a brief explanation (1-2 sentences) of why it is relevant.
- Return the improved recommendations as a numbered list, each with its explanation. If none are relevant, return an empty list.
""",
            description="An agent for validating and filtering software recommendations using external knowledge."
        )

    async def validate_recommendations(self, change_request: str, recommendations: list, context: str = "") -> list:
        prompt = f"""
User Change Request:
{change_request}

Initial Recommendations:
{recommendations}

Context:
{context}

Please return a numbered list of only the relevant recommendations, each with a brief explanation. If none are relevant, return an empty list.
"""
        result = await self.agent.run(task=prompt)
        # Return the LLM's output as a string (or parse to list if needed)
        if hasattr(result, 'content'):
            return str(result.content).strip()
        elif hasattr(result, 'strip'):
            return str(result).strip()
        else:
            return str(result).strip() 