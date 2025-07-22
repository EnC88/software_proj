from smart_sdk.model.model_client import AzureOpenAIChatCompletionClient
from azure.identity import CertificateCredential, get_bearer_token_provider
import os

def model() -> AzureOpenAIChatCompletionClient:
    client_id= 'EE939F2D-404D-440F-904F-000000000000'
    tenant_id= 'EE939F2D-404D-440F-904F-000000000000'
    api_version= '2024-02-21'
    deployment_name= 'gpt-4-turbo-2024-04-09'

    scope = "https://cognitiveservices.azure.com/.default"
    endpoint= 'https://eastus.api.cognitive.microsoft.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2023-05-15'
    certificate_path = "C:/Users/jason/Desktop/sys-whisper-ai/src/IntentAgent/cert.pem"
    IDA_CLIENT_ID = 'EE939F2D-404D-440F-904F-000000000000'
    IDA_COGBOT_RESOURCE='JPMC:URI:RS-112026-149613-cogbot-UAT'
    BASE_URL = 'https://localhost:5000'
    IDA_TOKEN_URL = 'https://idauatg2.jpmc.com/oauth2/token'
    api_key = 'EE939F2D-404D-440F-904F-000000000000'
    cred = CertificateCredential(IDA_CLIENT_ID, IDA_TOKEN_URL, certificate_path)
    azure_ad_token_provider = get_bearer_token_provider(cred, scope)
    access_token = azure_ad_token_provider.get_token(scope)
    model_config = {'api_key': api_key, 
                    'temperature': 0.0,
                    'model': deployment_name,
                    'max_tokens': 4000,
                    'azure_endpoint': endpoint,
                    'default_headers': {"Authorization": f"Bearer {access_token}", "user_sid": "simple-smart-agent"}
    }
    return AzureOpenAIChatCompletionClient(**model_config)