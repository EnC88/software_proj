from smart_sdk.model import AzureOpenAIEmbeddingClient

def embedding_model():
    return AzureOpenAIEmbeddingClient(
        api_key="YOUR_AZURE_API_KEY",
        endpoint="YOUR_AZURE_ENDPOINT",
        deployment_name="YOUR_EMBEDDING_DEPLOYMENT"
        # Add any other required parameters here
    ) 