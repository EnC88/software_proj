#!/usr/bin/env python3
"""
LangChain-Enhanced Compatibility Analyzer
Replaces and improves upon the existing compatibility analyzer with LangChain's NLP capabilities
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from collections import Counter, defaultdict

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Local imports
from src.rag.langchain_engine import LangChainQueryEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define repo root for robust file access
REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class ChangeRequest:
    """Represents a user's software change request."""
    software_name: str
    version: Optional[str] = None
    action: str = "upgrade"  # upgrade, install, remove, downgrade
    environment: Optional[str] = None
    target_servers: Optional[List[str]] = None
    raw_text: str = ""
    confidence: float = 0.0

@dataclass
class CompatibilityResult:
    """Result of compatibility analysis."""
    is_compatible: bool
    confidence: float
    affected_servers: List[Dict[str, Any]]
    conflicts: List[str]
    recommendations: List[str]
    warnings: List[str]
    alternative_versions: List[str]
    reasoning: str = ""

class ChangeRequestParser(BaseModel):
    """Pydantic model for parsing change requests."""
    software_name: str = Field(description="Name of the software")
    version: Optional[str] = Field(description="Version of the software")
    action: str = Field(description="Action to perform: upgrade, install, remove, downgrade")
    environment: Optional[str] = Field(description="Target environment: DEV, UAT, PROD")
    target_servers: Optional[List[str]] = Field(description="Specific target servers")
    confidence: float = Field(description="Confidence in the parsing (0-1)")

class CompatibilityAnalysis(BaseModel):
    """Pydantic model for compatibility analysis results."""
    is_compatible: bool = Field(description="Whether the change is compatible")
    confidence: float = Field(description="Confidence in the analysis (0-1)")
    affected_servers: List[str] = Field(description="List of affected server IDs")
    conflicts: List[str] = Field(description="List of compatibility conflicts")
    recommendations: List[str] = Field(description="List of recommendations")
    warnings: List[str] = Field(description="List of warnings")
    alternative_versions: List[str] = Field(description="List of alternative versions")
    reasoning: str = Field(description="Detailed reasoning for the analysis")

class LangChainCompatibilityAnalyzer:
    """Enhanced compatibility analyzer using LangChain for better NLP and reasoning."""
    
    def __init__(self, 
                 use_openai: bool = False,
                 openai_api_key: str = None,
                 data_dir: str = None):
        """Initialize the LangChain compatibility analyzer.
        
        Args:
            use_openai: Whether to use OpenAI for LLM responses
            openai_api_key: OpenAI API key if using OpenAI
            data_dir: Directory containing source data
        """
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        self.data_dir = Path(data_dir) if data_dir else REPO_ROOT / 'data'
        
        # Initialize components
        self.llm = None
        self.query_engine = None
        self.parser = None
        self.analysis_chain = None
        self.memory = None
        
        # Load data
        self.analysis_data = {}
        self.server_data = {}
        self.compatibility_rules = {}
        
        self._initialize_components()
        self._load_data()
    
    def _initialize_components(self):
        """Initialize LangChain components."""
        try:
            # Initialize query engine
            self.query_engine = LangChainQueryEngine(
                use_openai=self.use_openai,
                openai_api_key=self.openai_api_key
            )
            
            # Initialize LLM if OpenAI is enabled
            if self.use_openai and self.openai_api_key:
                import os
                os.environ["OPENAI_API_KEY"] = self.openai_api_key
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=2000
                )
                
                # Initialize parser
                self.parser = PydanticOutputParser(pydantic_object=ChangeRequestParser)
                
                # Initialize memory
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                
                # Initialize analysis chain
                self._initialize_analysis_chain()
                
                logger.info("LangChain components initialized successfully")
            else:
                logger.info("Using local processing only (no OpenAI LLM)")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_analysis_chain(self):
        """Initialize the compatibility analysis chain."""
        try:
            # Create analysis prompt
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a software compatibility expert. Analyze the compatibility of software changes based on the provided context.

Your task is to:
1. Analyze the compatibility of the requested change
2. Identify affected servers and potential conflicts
3. Provide recommendations and warnings
4. Suggest alternative versions if needed

Use the following context to make your analysis:
{context}

Change Request: {change_request}

Provide a detailed analysis in the specified format."""),
                ("human", "Analyze the compatibility of this change request.")
            ])
            
            # Create the chain
            self.analysis_chain = LLMChain(
                llm=self.llm,
                prompt=analysis_prompt,
                memory=self.memory
            )
            
        except Exception as e:
            logger.error(f"Error initializing analysis chain: {e}")
    
    def _load_data(self):
        """Load compatibility data."""
        try:
            # Load compatibility analysis
            analysis_path = self.data_dir / 'processed' / 'compatibility_analysis.json'
            if analysis_path.exists():
                with open(analysis_path, 'r') as f:
                    self.analysis_data = json.load(f)
                
                # Create server lookup
                self.server_data = {
                    server['id']: server 
                    for server in self.analysis_data.get('servers', [])
                }
                
                # Load compatibility rules
                self.compatibility_rules = self.analysis_data.get('compatibility_rules', {})
                
                logger.info(f"Loaded data for {len(self.server_data)} servers")
            else:
                logger.warning("Compatibility analysis file not found")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def parse_change_request(self, text: str) -> ChangeRequest:
        """Parse a natural language change request using LangChain.
        
        Args:
            text: Natural language request
            
        Returns:
            Parsed ChangeRequest object
        """
        if self.llm and self.parser:
            return self._parse_with_llm(text)
        else:
            return self._parse_with_rules(text)
    
    def _parse_with_llm(self, text: str) -> ChangeRequest:
        """Parse change request using LLM."""
        try:
            # Create parsing prompt
            parse_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at parsing software change requests. Extract the following information from the user's request:

- software_name: The name of the software
- version: The version number (if specified)
- action: The action to perform (upgrade, install, remove, downgrade)
- environment: Target environment (DEV, UAT, PROD)
- target_servers: Specific servers mentioned
- confidence: Your confidence in the parsing (0-1)

Respond in the specified JSON format."""),
                ("human", "Parse this change request: {text}")
            ])
            
            # Get response
            messages = parse_prompt.format_messages(text=text)
            response = self.llm(messages)
            
            # Parse response
            parsed = self.parser.parse(response.content)
            
            return ChangeRequest(
                software_name=parsed.software_name,
                version=parsed.version,
                action=parsed.action,
                environment=parsed.environment,
                target_servers=parsed.target_servers,
                raw_text=text,
                confidence=parsed.confidence
            )
            
        except Exception as e:
            logger.error(f"Error parsing with LLM: {e}")
            return self._parse_with_rules(text)
    
    def _parse_with_rules(self, text: str) -> ChangeRequest:
        """Parse change request using rule-based approach."""
        text = text.lower().strip()
        
        # Extract software name and version
        software_name = None
        version = None
        action = "upgrade"
        environment = None
        
        # Common software patterns
        software_patterns = [
            r'(apache|httpd|tomcat)',
            r'(nginx)',
            r'(websphere|ibm)',
            r'(mysql|postgresql|oracle|db2)',
            r'(python|java|node\.js|php)',
            r'(windows|linux|rhel|ubuntu)'
        ]
        
        # Version patterns
        version_patterns = [
            r'(\d+\.\d+\.\d+)',  # 2.4.50
            r'(\d+\.\d+)',       # 2.4
            r'version (\d+\.\d+\.\d+)',
            r'v(\d+\.\d+\.\d+)'
        ]
        
        # Action patterns
        if any(word in text for word in ['install', 'add', 'new']):
            action = "install"
        elif any(word in text for word in ['remove', 'uninstall', 'delete']):
            action = "remove"
        elif any(word in text for word in ['downgrade', 'rollback']):
            action = "downgrade"
        
        # Environment patterns
        env_patterns = {
            'dev': ['dev', 'development'],
            'uat': ['uat', 'staging', 'test'],
            'prod': ['prod', 'production', 'live']
        }
        
        for env, patterns in env_patterns.items():
            if any(pattern in text for pattern in patterns):
                environment = env.upper()
                break
        
        # Extract software name
        for pattern in software_patterns:
            match = re.search(pattern, text)
            if match:
                software_name = match.group(1).upper()
                break
        
        # Extract version
        for pattern in version_patterns:
            match = re.search(pattern, text)
            if match:
                version = match.group(1)
                break
        
        # Normalize software names
        if software_name:
            software_mapping = {
                'apache': 'APACHE HTTPD',
                'httpd': 'APACHE HTTPD', 
                'tomcat': 'APACHE TOMCAT',
                'websphere': 'WEBSPHERE',
                'ibm': 'WEBSPHERE',
                'mysql': 'MySQL',
                'postgresql': 'PostgreSQL',
                'oracle': 'Oracle',
                'db2': 'DB2',
                'python': 'Python',
                'java': 'Java',
                'node.js': 'Node.js',
                'php': 'PHP'
            }
            software_name = software_mapping.get(software_name, software_name)
        
        return ChangeRequest(
            software_name=software_name or "UNKNOWN",
            version=version,
            action=action,
            environment=environment,
            raw_text=text,
            confidence=0.7  # Default confidence for rule-based parsing
        )
    
    def analyze_compatibility(self, change_request: ChangeRequest, target_os: Optional[str] = None) -> CompatibilityResult:
        """Analyze compatibility using LangChain-enhanced approach.
        
        Args:
            change_request: Parsed change request
            target_os: Optional target OS filter
            
        Returns:
            CompatibilityResult object
        """
        if self.llm:
            return self._analyze_with_llm(change_request, target_os)
        else:
            return self._analyze_with_rules(change_request, target_os)
    
    def _analyze_with_llm(self, change_request: ChangeRequest, target_os: Optional[str] = None) -> CompatibilityResult:
        """Analyze compatibility using LLM."""
        try:
            # Query for relevant context
            query = f"compatibility analysis for {change_request.software_name} {change_request.version or ''} {change_request.action}"
            context_results = self.query_engine.query(query, top_k=5)
            
            # Prepare context
            context = ""
            if context_results.get('results'):
                for result in context_results['results']:
                    context += f"\n{result['content']}\n"
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze the compatibility of this software change:
            
            Software: {change_request.software_name}
            Version: {change_request.version or 'Not specified'}
            Action: {change_request.action}
            Environment: {change_request.environment or 'Not specified'}
            
            Context from knowledge base:
            {context}
            
            Provide a detailed analysis including:
            1. Compatibility assessment
            2. Affected servers
            3. Potential conflicts
            4. Recommendations
            5. Warnings
            6. Alternative versions
            7. Reasoning
            """
            
            # Get LLM response
            messages = [
                SystemMessage(content="You are a software compatibility expert. Provide detailed analysis in JSON format."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.llm(messages)
            
            # Parse response (simplified - in production you'd use a more robust parser)
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    analysis_json = json.loads(json_match.group())
                    
                    return CompatibilityResult(
                        is_compatible=analysis_json.get('is_compatible', False),
                        confidence=analysis_json.get('confidence', 0.5),
                        affected_servers=analysis_json.get('affected_servers', []),
                        conflicts=analysis_json.get('conflicts', []),
                        recommendations=analysis_json.get('recommendations', []),
                        warnings=analysis_json.get('warnings', []),
                        alternative_versions=analysis_json.get('alternative_versions', []),
                        reasoning=analysis_json.get('reasoning', '')
                    )
            except:
                pass
            
            # Fallback to rule-based analysis
            return self._analyze_with_rules(change_request, target_os)
            
        except Exception as e:
            logger.error(f"Error analyzing with LLM: {e}")
            return self._analyze_with_rules(change_request, target_os)
    
    def _analyze_with_rules(self, change_request: ChangeRequest, target_os: Optional[str] = None) -> CompatibilityResult:
        """Analyze compatibility using rule-based approach."""
        # This is a simplified version - you can enhance it with your existing logic
        affected_servers = []
        conflicts = []
        recommendations = []
        warnings = []
        alternative_versions = []
        
        # Find affected servers
        for server_id, server in self.server_data.items():
            if self._server_matches_criteria(server, change_request, target_os):
                affected_servers.append(server)
        
        # Basic compatibility check
        is_compatible = len(conflicts) == 0
        confidence = 0.8 if affected_servers else 0.5
        
        return CompatibilityResult(
            is_compatible=is_compatible,
            confidence=confidence,
            affected_servers=affected_servers,
            conflicts=conflicts,
            recommendations=recommendations,
            warnings=warnings,
            alternative_versions=alternative_versions,
            reasoning="Rule-based analysis completed"
        )
    
    def _server_matches_criteria(self, server: Dict[str, Any], change_request: ChangeRequest, target_os: Optional[str] = None) -> bool:
        """Check if a server matches the change request criteria."""
        # Implement your server matching logic here
        return True  # Simplified for now
    
    def parse_multiple_change_requests(self, text: str) -> List[ChangeRequest]:
        """Parse multiple change requests from text."""
        if self.llm:
            return self._parse_multiple_with_llm(text)
        else:
            return self._parse_multiple_with_rules(text)
    
    def _parse_multiple_with_llm(self, text: str) -> List[ChangeRequest]:
        """Parse multiple change requests using LLM."""
        try:
            # Create prompt for multiple requests
            prompt = f"""
            Parse the following text and extract multiple software change requests:
            
            {text}
            
            Identify each separate change request and extract:
            - software_name
            - version
            - action
            - environment
            - target_servers
            
            Return as a list of JSON objects.
            """
            
            messages = [
                SystemMessage(content="You are an expert at parsing multiple software change requests."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            
            # Parse response (simplified)
            try:
                # Extract JSON array from response
                json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
                if json_match:
                    requests_json = json.loads(json_match.group())
                    
                    change_requests = []
                    for req in requests_json:
                        change_requests.append(ChangeRequest(
                            software_name=req.get('software_name', 'UNKNOWN'),
                            version=req.get('version'),
                            action=req.get('action', 'upgrade'),
                            environment=req.get('environment'),
                            target_servers=req.get('target_servers'),
                            raw_text=text,
                            confidence=req.get('confidence', 0.7)
                        ))
                    
                    return change_requests
            except:
                pass
            
            # Fallback to single request parsing
            return [self.parse_change_request(text)]
            
        except Exception as e:
            logger.error(f"Error parsing multiple requests with LLM: {e}")
            return self._parse_multiple_with_rules(text)
    
    def _parse_multiple_with_rules(self, text: str) -> List[ChangeRequest]:
        """Parse multiple change requests using rule-based approach."""
        # Split text by common separators
        separators = [';', '\n', ' and ', ' also ', ' additionally ']
        
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                if len(parts) > 1:
                    return [self.parse_change_request(part.strip()) for part in parts if part.strip()]
        
        # If no separators found, treat as single request
        return [self.parse_change_request(text)]
    
    def analyze_multiple_compatibility(self, change_requests: List[ChangeRequest], target_os: Optional[str] = None) -> List[Tuple[ChangeRequest, CompatibilityResult]]:
        """Analyze compatibility for multiple change requests."""
        results = []
        for change_request in change_requests:
            result = self.analyze_compatibility(change_request, target_os)
            results.append((change_request, result))
        return results
    
    def format_analysis_result(self, result: CompatibilityResult, change_request: ChangeRequest) -> str:
        """Format analysis result for display."""
        status = "COMPATIBLE" if result.is_compatible else "INCOMPATIBLE"
        status_color = "#22c55e" if result.is_compatible else "#ef4444"
        
        formatted_result = f"""
        <div style="margin: 1rem 0; padding: 1rem; border-radius: 8px; background: #f8fafc;">
            <h3 style="color: {status_color};">Status: {status}</h3>
            <p><strong>Confidence:</strong> {result.confidence:.1%}</p>
            
            <h4>Affected Servers ({len(result.affected_servers)})</h4>
            <ul>
                {''.join([f'<li>{server.get("id", "Unknown")} - {server.get("server_info", {}).get("model", "Unknown")}</li>' for server in result.affected_servers[:5]])}
                {f'<li>... and {len(result.affected_servers) - 5} more</li>' if len(result.affected_servers) > 5 else ''}
            </ul>
            
            <h4>Conflicts ({len(result.conflicts)})</h4>
            <ul>
                {''.join([f'<li>{conflict}</li>' for conflict in result.conflicts])}
            </ul>
            
            <h4>Recommendations ({len(result.recommendations)})</h4>
            <ul>
                {''.join([f'<li>{rec}</li>' for rec in result.recommendations])}
            </ul>
            
            <h4>Warnings ({len(result.warnings)})</h4>
            <ul>
                {''.join([f'<li>{warning}</li>' for warning in result.warnings])}
            </ul>
            
            <h4>Alternative Versions ({len(result.alternative_versions)})</h4>
            <ul>
                {''.join([f'<li>{version}</li>' for version in result.alternative_versions])}
            </ul>
            
            <h4>Reasoning</h4>
            <p>{result.reasoning}</p>
        </div>
        """
        
        return formatted_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the analyzer."""
        return {
            'use_openai': self.use_openai,
            'llm_available': self.llm is not None,
            'query_engine_available': self.query_engine is not None,
            'total_servers': len(self.server_data),
            'total_rules': len(self.compatibility_rules),
            'query_engine_stats': self.query_engine.get_stats() if self.query_engine else {}
        }

def main():
    """Test the LangChain compatibility analyzer."""
    # Initialize the analyzer
    analyzer = LangChainCompatibilityAnalyzer()
    
    # Test parsing
    test_text = "I want to upgrade Apache to version 2.4.50 in production"
    change_request = analyzer.parse_change_request(test_text)
    
    print(f"Parsed change request: {change_request}")
    
    # Test analysis
    result = analyzer.analyze_compatibility(change_request)
    
    print(f"Analysis result: {result}")
    
    # Test multiple requests
    multi_text = "Upgrade Apache to 2.4.50 and install MySQL 8.0"
    multiple_requests = analyzer.parse_multiple_change_requests(multi_text)
    
    print(f"Multiple requests: {multiple_requests}")

if __name__ == "__main__":
    main() 