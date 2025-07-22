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
You are an expert software compatibility assistant. Your role is to ENHANCE and IMPROVE software recommendations, not just filter them out. Given a user's software change request and a list of recommendations:

PRIMARY GOAL: Help the user by providing useful, actionable advice. Your job is to make recommendations BETTER, not remove them.

Guidelines:
- KEEP recommendations that could be helpful, even if not directly related to the specific upgrade
- IMPROVE recommendations by adding context, explanations, or suggesting related actions
- Only REMOVE recommendations that are completely irrelevant or harmful
- For each recommendation, explain WHY it might be useful to the user
- Consider related infrastructure, dependencies, or best practices
- If a recommendation seems unrelated, think about how it might help with the broader system context

Examples of helpful additions:
- "Consider also upgrading related components for better compatibility"
- "This might be useful for your overall infrastructure planning"
- "While not directly related to the upgrade, this could improve your system"

Return a numbered list with improved recommendations and explanations. Only return an empty list if ALL recommendations are truly harmful or completely irrelevant.
""",
            description="An agent for enhancing and improving software recommendations using external knowledge."
        )

    async def validate_recommendations(self, change_request: str, recommendations: list, context: str = "") -> str:
        prompt = f"""
User Change Request:
{change_request}

Recommendations to enhance:
{recommendations}

Context:
{context}

Please enhance these recommendations by:
1. Keeping useful ones and explaining why they're helpful
2. Improving recommendations with additional context or related advice
3. Only removing recommendations that are completely irrelevant or harmful
4. Adding explanations for why each recommendation might be valuable

Return a numbered list with enhanced recommendations and explanations. Focus on being helpful rather than overly strict.
"""
        print("LLM Validation Prompt:\n", prompt)  # Debug print
        result = await self.agent.run(task=prompt)
        
        print(f"Raw LLM result type: {type(result)}")
        print(f"Raw LLM result: {result}")
        
        # Extract content using the same pattern as other agents
        if hasattr(result, 'content'):
            response_text = str(result.content).strip()
            print(f"Using result.content: {response_text}")
        elif hasattr(result, 'messages') and result.messages:
            print(f"Found messages array with {len(result.messages)} messages")
            # Try to find the TextMessage from our agent
            for i, message in enumerate(result.messages):
                print(f"Message {i}: type={getattr(message, 'type', 'unknown')}, source={getattr(message, 'source', 'unknown')}")
                if hasattr(message, 'type') and message.type == 'TextMessage' and hasattr(message, 'source') and message.source == 'RecommendationValidationAgent':
                    response_text = str(message.content).strip()
                    print(f"Found our agent's message: {response_text}")
                    break
            else:
                # Fallback to any TextMessage
                for i, message in enumerate(result.messages):
                    if hasattr(message, 'type') and message.type == 'TextMessage' and hasattr(message, 'content'):
                        response_text = str(message.content).strip()
                        print(f"Using fallback TextMessage {i}: {response_text}")
                        break
                else:
                    response_text = str(result).strip()
                    print(f"No TextMessage found, using str(result): {response_text}")
        elif hasattr(result, 'strip'):
            response_text = str(result).strip()
            print(f"Using result.strip(): {response_text}")
        else:
            response_text = str(result).strip()
            print(f"Using str(result): {response_text}")
        
        # Fix literal '\n' strings to actual newlines
        response_text = response_text.replace('\\n', '\n')
        
        # Clean up markdown formatting - remove ** or convert to plain text
        response_text = response_text.replace('**', '')  # Remove bold markers
        
        print(f"Final response_text: {response_text}")
        
        return response_text 