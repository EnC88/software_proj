#!/usr/bin/env python3
"""
Compatibility Analyzer for RAG Pipeline
Analyzes software change requests against existing infrastructure
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from src.models.query_parser import QueryParser
from src.Agents.ContextQueryAgent import agent as context_query_agent
import asyncio
from pathlib import Path
from src.rag.query_engine import QueryEngine
import os

# Load co-upgrade patterns for compatibility recommendations
CO_UPGRADE_PATH = os.path.join('data', 'processed', 'co_upgrade_patterns.json')
if os.path.exists(CO_UPGRADE_PATH):
    with open(CO_UPGRADE_PATH, 'r') as f:
        co_upgrade_patterns = json.load(f)
else:
    co_upgrade_patterns = {}

def get_co_upgrades(catalogid, top_n=5):
    entry = co_upgrade_patterns.get(catalogid)
    if not entry or not entry['co_upgrades']:
        return []
    sorted_co = sorted(entry['co_upgrades'].items(), key=lambda x: x[1]['count'], reverse=True)
    return [
        {'catalogid': co_id, 'model': co_info['model'], 'count': co_info['count']}
        for co_id, co_info in sorted_co[:top_n]
    ]

SOFTWARE_FAMILY_GROUPS = {
    "APACHE": ["APACHE HTTPD", "APACHE TOMCAT", "HTTPD", "TOMCAT"],
    "IBM": ["WEBSPHERE", "IBM HTTP SERVER", "IBM"],
    "NGINX": ["NGINX"],
    "MYSQL": ["MYSQL"],
    "POSTGRES": ["POSTGRES", "POSTGRESQL"],
    "ORACLE": ["ORACLE"],
    "DB2": ["DB2"],
    "PYTHON": ["PYTHON"],
    "JAVA": ["JAVA"],
    # Add more as needed
}

def get_software_family(software_name: str) -> str:
    """
    Given a software name, return its major family.
    If not found, return the uppercased software name itself.
    """
    name_upper = software_name.upper()
    for family, variants in SOFTWARE_FAMILY_GROUPS.items():
        if name_upper in [v.upper() for v in variants]:
            return family
    return name_upper

# Debug: See what context_query_agent actually i
# Debug: Check if the agent has the expected attributes
if hasattr(context_query_agent, 'run'):
    print(f"DEBUG: context_query_agent.run type: {type(context_query_agent.run)}")
    print(f"DEBUG: context_query_agent.run is callable: {callable(context_query_agent.run)}")
else:
    print("DEBUG: context_query_agent has no 'run' method!")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChangeRequest:
    """Represents a user's software change request."""
    software_name: str
    version: Optional[str] = None
    action: str = "upgrade"  # upgrade, install, remove, downgrade
    environment: Optional[str] = None
    target_servers: Optional[List[str]] = None
    raw_text: str = ""
    catalogid: Optional[str] = None  # NEW

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
    filtering_steps: List[Dict[str, Any]] = None

class CheckCompatibility:
    """Analyzes software change requests for compatibility."""
    
    def __init__(self, 
                 rules_path: str = 'data/processed/compatibility_rules.json',
                 analysis_path: str = 'data/processed/compatibility_analysis.json'):
        """Initialize the compatibility analyzer.
        
        Args:
            rules_path: Path to compatibility rules JSON
            analysis_path: Path to compatibility analysis JSON
        """
        self.rules_path = Path(rules_path)
        self.analysis_path = Path(analysis_path)
        self.rules = {}
        self.analysis = {}
        self.server_data = {}
        self.query_engine = QueryEngine()
        self.known_software_families = ["APACHE", "WEBSPHERE", "IBM HTTP SERVER", "NGINX", "TOMCAT", "MYSQL", "POSTGRESQL", "PYTHON", "JAVA"] # For parsing RAG results
        
        self._load_data()
    
    def _load_data(self):
        """Load compatibility rules and analysis data."""
        try:
            # Load compatibility rules
            if self.rules_path.exists():
                with open(self.rules_path, 'r') as f:
                    self.rules = json.load(f)
                logger.info(f"Loaded compatibility rules from {self.rules_path}")
            
            # Load analysis data
            if self.analysis_path.exists():
                with open(self.analysis_path, 'r') as f:
                    self.analysis = json.load(f)
                logger.info(f"Loaded compatibility analysis from {self.analysis_path}")
                
                # Create server lookup
                self.server_data = {server['id']: server for server in self.analysis.get('servers', [])}
                
                # --- Dynamically build the list of known software from the data ---
                known_families = set()
                import re
                for server in self.analysis.get('servers', []):
                    model_full = server.get('server_info', {}).get('model', 'Unknown')
                    if model_full != 'Unknown':
                        version_match = re.search(r'(\d+(?:\.\d+)*)', model_full)
                        product_family = model_full[:version_match.start()].strip().upper() if version_match else model_full.upper()
                        if product_family:
                            known_families.add(product_family)
                self.known_software_families = sorted(list(known_families), key=len, reverse=True) # Longer names first for better matching
                logger.info(f"Dynamically identified {len(self.known_software_families)} software families from data.")

            # Build catalog index from multiple sources
            self.catalog_index = self._build_catalog_index()

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _build_catalog_index(self):
        """Build a comprehensive catalog index from multiple data sources."""
        catalog_index = {}
        
        try:
            # Source 1: PCat data
            pcat_path = 'data/raw/PCat.csv'
            if os.path.exists(pcat_path):
                df = pd.read_csv(pcat_path)
                for _, row in df.iterrows():
                    if pd.notna(row.get('CATALOGID')) and pd.notna(row.get('MODEL')):
                        catalogid = str(row['CATALOGID']).strip()
                        model = str(row['MODEL']).strip()
                        catalog_index[model.upper()] = catalogid
                        # Also index without version for fuzzy matching
                        model_no_version = re.sub(r'\s+\d+\.\d+.*$', '', model.upper())
                        if model_no_version != model.upper():
                            catalog_index[model_no_version] = catalogid
            
            # Source 2: SOR history data (already loaded globally)
            if sor_hist_data is not None:
                # Old values
                old_data = sor_hist_data[sor_hist_data['OLDPRODUCTTYPE'].isin(['OPERATING SYSTEM', 'DATABASE', 'WEB SERVER'])][['OLDVALUE', 'OLD_MAPPED']].dropna()
                for _, row in old_data.iterrows():
                    if pd.notna(row['OLDVALUE']) and pd.notna(row['OLD_MAPPED']):
                        catalogid = str(row['OLDVALUE']).strip()
                        model = str(row['OLD_MAPPED']).strip()
                        catalog_index[model.upper()] = catalogid
                        # Also index without version
                        model_no_version = re.sub(r'\s+\d+\.\d+.*$', '', model.upper())
                        if model_no_version != model.upper():
                            catalog_index[model_no_version] = catalogid
                
                # New values
                new_data = sor_hist_data[sor_hist_data['NEWPRODUCTTYPE'].isin(['OPERATING SYSTEM', 'DATABASE', 'WEB SERVER'])][['NEWVALUE', 'NEW_MAPPED']].dropna()
                for _, row in new_data.iterrows():
                    if pd.notna(row['NEWVALUE']) and pd.notna(row['NEW_MAPPED']):
                        catalogid = str(row['NEWVALUE']).strip()
                        model = str(row['NEW_MAPPED']).strip()
                        catalog_index[model.upper()] = catalogid
                        # Also index without version
                        model_no_version = re.sub(r'\s+\d+\.\d+.*$', '', model.upper())
                        if model_no_version != model.upper():
                            catalog_index[model_no_version] = catalogid
            
            logger.info(f"Built catalog index with {len(catalog_index)} entries from multiple sources")
            
        except Exception as e:
            logger.error(f"Error building catalog index: {e}")
        
        return catalog_index

    def _resolve_catalogid(self, software_name: str, version: Optional[str], environment: Optional[str]) -> Optional[str]:
        """Find catalogid for a given software/version/environment using multiple strategies."""
        # Strategy 1: Check if catalogid is already in the software_name (from dropdown selection)
        if software_name and ' - ' in software_name:
            # Extract catalogid from "catalogid - model_name" format
            catalogid = software_name.split(' - ')[0].strip()
            if catalogid and catalogid != 'None':
                logger.info(f"Extracted catalogid from software_name: {catalogid}")
                return catalogid
        
        # Strategy 2: Use comprehensive catalog index
        if hasattr(self, 'catalog_index') and self.catalog_index:
            # Try exact match first
            if software_name.upper() in self.catalog_index:
                catalogid = self.catalog_index[software_name.upper()]
                logger.info(f"Found catalogid from catalog index (exact match): {catalogid}")
                return catalogid
            
            # Try partial matches
            for model, catalogid in self.catalog_index.items():
                if software_name.upper() in model or model in software_name.upper():
                    logger.info(f"Found catalogid from catalog index (partial match): {catalogid} for {model}")
                    return catalogid
        
        # Strategy 3: Search through server data (fallback)
        for server in self.analysis.get('servers', []):
            server_info = server.get('server_info', {})
            model = server_info.get('model', '').upper()
            env = server.get('environment', '').upper() if server.get('environment') else None
            if (
                software_name.upper() in model
                and (not version or version in model)
                and (not environment or (env and environment.upper() == env))
            ):
                catalogid = server.get('id')
                logger.info(f"Found catalogid from server search: {catalogid}")
                return catalogid
        
        logger.info(f"No catalogid found for software: {software_name}")
        return None
    
    async def parse_change_request_async(self, text: str) -> ChangeRequest:
        """Async version of parse_change_request that uses LLM classification."""
        text = text.lower().strip()
        
        print(f"DEBUG: parse_change_request_async called with: {text}")
        
        # Use LLM to classify the query intent AND extract entities
        print(f"DEBUG: About to call _classify_query_intent")
        action = await self._classify_query_intent(text)
        print(f"DEBUG: _classify_query_intent returned: {action}")
        
        # Use ContextQueryAgent to extract software entities in JSON format
        software_name = None
        version = None
        environment = None
        
        try:
            if context_query_agent and hasattr(context_query_agent, 'run'):
                result = await context_query_agent.run(task=text)
                
                # Extract JSON response from ContextQueryAgent
                if hasattr(result, 'messages') and result.messages:
                    for message in result.messages:
                        if hasattr(message, 'source') and message.source == 'ContextQueryAgent':
                            content = message.content
                            break
                    else:
                        content = result.messages[-1].content
                elif hasattr(result, 'content'):
                    content = str(result.content)
                else:
                    content = str(result)
                
                # Parse JSON response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    software_name = data.get('software')
                    version = data.get('to_version') or data.get('from_version')
                    environment = data.get('os')
                    
                    # Normalize software names if extracted
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
                        software_name = software_mapping.get(software_name.lower(), software_name.upper())
                        
                    logger.info(f"ContextQueryAgent extracted: software={software_name}, version={version}, os={environment}")
        except Exception as e:
            logger.warning(f"ContextQueryAgent entity extraction failed: {e}")
        
        # Fallback to regex patterns if LLM extraction failed
        if not software_name:
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
        
        # Resolve catalogid using the extracted information
        catalogid = self._resolve_catalogid(software_name, version, environment)
        
        return ChangeRequest(
            software_name=software_name or "UNKNOWN",
            version=version,
            action=action,
            environment=environment,
            raw_text=text,
            catalogid=catalogid
        )
    
    def parse_change_request(self, text: str) -> ChangeRequest:
        """Parse a natural language change request into structured data.
        
        Args:
            text: Natural language request (e.g., "I want to upgrade Apache to 2.4.50")
            
        Returns:
            Parsed ChangeRequest object
        """
        text = text.lower().strip()
        
        # Use fallback classification for now (synchronous)
        action = self._fallback_classify_intent(text)
        
        # Extract software name and version
        software_name = None
        version = None
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
        # Resolve catalogid
        catalogid = self._resolve_catalogid(software_name, version, environment)
        return ChangeRequest(
            software_name=software_name or "UNKNOWN",
            version=version,
            action=action,
            environment=environment,
            raw_text=text,
            catalogid=catalogid
        )
    
    async def _classify_query_intent(self, text: str) -> str:
        """Use ContextQueryAgent to classify the query intent and determine the appropriate action."""
        try:
            # Check if context_query_agent is properly initialized
            if not context_query_agent or not hasattr(context_query_agent, 'run') or not callable(getattr(context_query_agent, 'run', None)):
                logger.warning("ContextQueryAgent not properly initialized, using fallback classification")
                return self._fallback_classify_intent(text)

            # Test the run method without await first
            print(f"DEBUG: About to call context_query_agent.run with task: {text}")
            try:
                # Direct await without intermediate variable
                result = await context_query_agent.run(task=text)
            except Exception as e:
                print(f"DEBUG: Error calling run method: {e}")
                print(f"DEBUG: Error type: {type(e)}")
                raise
         
            # Extract the intent from the result
            if hasattr(result, 'messages') and result.messages:
                # Find the agent's response (not the user's message)
                for message in result.messages:
                    if hasattr(message, 'source') and message.source == 'ContextQueryAgent':
                        content = message.content
                        break
                else:
                    # If no agent message found, use the last message
                    content = result.messages[-1].content
            elif hasattr(result, 'content'):
                content = str(result.content)
            else:
                content = str(result)
            
            print(f"DEBUG: Extracted content: {content}")
            
            # Try to parse JSON response
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    intent = data.get('intent', 'unknown').lower()
                else:
                    # If no JSON found, try to extract intent from text
                    intent = content.lower()
            except json.JSONDecodeError:
                # Fallback to text extraction
                intent = content.lower()
            
            # Map ContextQueryAgent intents to our action types
            intent_mapping = {
                'general_info': 'info',
                'info': 'info',
                'compatibility_check': 'upgrade',
                'upgrade_advice': 'upgrade',
                'upgrade': 'upgrade',
                'install': 'install',
                'remove': 'remove',
                'downgrade': 'downgrade',
                'rollback': 'downgrade'
            }
            
            # Get the mapped action, default to 'upgrade'
            action = intent_mapping.get(intent, 'upgrade')
            
            logger.info(f"ContextQueryAgent classified '{text}' as '{intent}' -> '{action}'")
            return action
                
        except Exception as e:
            logger.error(f"Error classifying query intent with ContextQueryAgent: {e}")
            # Fallback to rule-based classification
            return self._fallback_classify_intent(text)
    
    def _fallback_classify_intent(self, text: str) -> str:
        """Fallback rule-based classification if LLM fails."""
        # Check if this is a general information question vs. personalized request
        general_question_indicators = [
            'what', 'which', 'how', 'when', 'where', 'why',
            'most compatible', 'best compatible', 'compatible with',
            'supported', 'supports', 'works with', 'runs on'
        ]
        
        is_general_question = any(indicator in text for indicator in general_question_indicators)
        
        if is_general_question:
            return "info"
        
        # Action patterns
        if any(word in text for word in ['install', 'add', 'new']):
            return "install"
        elif any(word in text for word in ['remove', 'uninstall', 'delete']):
            return "remove"
        elif any(word in text for word in ['downgrade', 'rollback']):
            return "downgrade"
        else:
            return "upgrade"  # Default
    
    def analyze_compatibility(self, change_request: ChangeRequest, target_os: Optional[str] = None, 
                            user_database: Optional[str] = None, user_web_servers: Optional[List[str]] = None) -> CompatibilityResult:
        """Analyze compatibility of a change request, optionally filtering by user configuration.
        Only process technical/compatibility queries. Info/general_info queries are not handled here.
        """
        affected_servers = []
        conflicts = []
        recommendations = []
        warnings = []
        alternative_versions = []
        filtering_steps = []
        total_servers = len(self.analysis.get('servers', []))
        filtering_steps.append({
            "stage": "initial",
            "count": total_servers,
            "description": f"All systems in database ({total_servers} total)"
        })

        # --- Dynamically find dependencies using the RAG Query Engine ---
        software_to_show_families = [change_request.software_name.upper()]
        rag_results = None
        try:
            dependency_query = f"What are the dependencies and compatible software for {change_request.software_name}?"
            rag_results = self.query_engine.query(dependency_query, top_k=3)
            if rag_results:
                context = self.query_engine.format_results_for_llm(rag_results)
                found_deps = re.findall(r'|'.join(re.escape(f) for f in self.known_software_families), context.upper())
                software_to_show_families.extend(found_deps)
                software_to_show_families = list(set(software_to_show_families))
        except Exception as e:
            pass

        # Only process technical/compatibility queries
        # (Skip info/general_info logic, e.g., os_counter, Counter, etc.)

        # --- Track filtering steps for transparency ---
        # Find affected servers
        if change_request.environment:
            affected_servers = [s for s in self.analysis.get('servers', []) if s['environment'] == change_request.environment]
            env_filtered_count = len(affected_servers)
            filtering_steps.append({
                "stage": "environment",
                "count": env_filtered_count,
                "description": f"Filtered by environment: {change_request.environment} ({env_filtered_count} servers)"
            })
        else:
            affected_servers = self.analysis.get('servers', [])
            filtering_steps.append({
                "stage": "environment",
                "count": len(affected_servers),
                "description": f"No environment filter applied ({len(affected_servers)} servers)"
            })
        
        logger.info(f"Total servers in analysis: {len(self.analysis.get('servers', []))}")
        logger.info(f"Affected servers after environment filter: {len(affected_servers)}")
        
        # --- Filter by User Configuration if provided ---
        original_server_count = len(affected_servers)
        filtered_servers = []
        
        for server in affected_servers:
            include_server = True
            
            # Filter by OS if provided - but be more lenient
            if target_os:
                server_os = server.get('os') or server.get('operating_system')
                # Only filter out if we have OS info AND it doesn't match
                # If no OS info is found, include the server anyway
                if server_os and target_os.lower() not in server_os.lower():
                    include_server = False
                    logger.info(f"Filtered out server {server.get('name', 'Unknown')} - OS mismatch: {server_os} vs {target_os}")
            
            if include_server:
                filtered_servers.append(server)
        
        affected_servers = filtered_servers
        user_config_filtered_count = len(affected_servers)
        
        # Create user config description
        user_config_desc = []
        if target_os:
            user_config_desc.append(f"OS: {target_os}")
        if user_database:
            user_config_desc.append(f"DB: {user_database}")
        if user_web_servers:
            user_config_desc.append(f"Web: {', '.join(user_web_servers)}")
        
        user_config_text = ", ".join(user_config_desc) if user_config_desc else "No user config"
        filtering_steps.append({
            "stage": "user_config",
            "count": user_config_filtered_count,
            "description": f"Filtered by user config ({user_config_text}): {user_config_filtered_count} servers"
        })
        
        logger.info(f"Servers after OS filtering: {len(affected_servers)} (original: {original_server_count})")
        
        # Check compatibility rules
        if change_request.software_name in self.rules.get('web_server_os', {}):
            software_rules = self.rules['web_server_os'][change_request.software_name]
            
            if change_request.version:
                version_rules = software_rules.get(change_request.version, {})
                
                # Check OS compatibility
                compatible_os = version_rules.get('compatible_os', [])
                if compatible_os:
                    # Check if any affected servers have incompatible OS
                    for server in affected_servers:
                        server_os = server.get('os') or server.get('operating_system')
                        if server_os and server_os not in compatible_os:
                            conflicts.append(f"Server {server['name']} has incompatible OS: {server_os}")
                
                # Check version constraints
                min_version = version_rules.get('min_os_version')
                max_version = version_rules.get('max_os_version')
                if min_version or max_version:
                    for server in affected_servers:
                        server_os = server.get('os') or server.get('operating_system')
                        if server_os and isinstance(server_os, str):
                            os_version = self._extract_version_from_os(server_os)
                            if os_version:
                                if min_version and os_version < min_version:
                                    conflicts.append(f"Server {server['name']} OS version {os_version} below minimum {min_version}")
                                if max_version and os_version > max_version:
                                    conflicts.append(f"Server {server['name']} OS version {os_version} above maximum {max_version}")
                
                # Get recommended OS
                recommended_os = version_rules.get('recommended_os')
                if recommended_os:
                    recommendations.append(f"Recommended OS: {recommended_os}")
            
            # Suggest alternative versions
            for version, rules in software_rules.items():
                if version != change_request.version:
                    alternative_versions.append(version)
        
        # Check environment rules
        if change_request.environment:
            env_rules = self.rules.get('environment_rules', {}).get(change_request.environment, {})
            allowed_versions = env_rules.get('allowed_versions')
            upgrade_frequency = env_rules.get('upgrade_frequency')
            
            if allowed_versions == 'stable' and change_request.action == 'upgrade':
                warnings.append(f"Environment {change_request.environment} prefers stable versions")
            
            if upgrade_frequency:
                recommendations.append(f"Upgrade frequency for {change_request.environment}: {upgrade_frequency}")
        
        # --- Find existing installations of target software and its dependencies ---
        # Use the filtered software families for recommendations (don't overwrite with all families)
        existing_installations = []
        logger.info(f"Looking for software families: {software_to_show_families}")
        
        for server in affected_servers:
            model_full = server.get('server_info', {}).get('model', 'Unknown')
            if model_full == 'Unknown':
                continue
            
            env = server.get('environment', 'Unknown')
            
            # Extract product family (full name) and version
            import re
            version_match = re.search(r'(\d+(?:\.\d+)*)', model_full)
            version = version_match.group(1) if version_match else "0.0"
            product_family = model_full[:version_match.start()].strip().upper() if version_match else model_full.upper()
            
            # Check if this product family is one we want to show
            for family_to_show in software_to_show_families:
                if family_to_show in product_family:
                    existing_installations.append((product_family, version, env))
                    logger.info(f"Found matching installation: {product_family} {version} in {env}")
                    break
        
        # Track compatibility filtering
        compatibility_filtered_count = len(existing_installations)
        filtering_steps.append({
            "stage": "compatibility",
            "count": compatibility_filtered_count,
            "description": f"Found compatible software installations: {compatibility_filtered_count} instances"
        })
        
        # --- Aggregate and format recommendations ---
        threshold = 5  # Lowered from 30 to make recommendations more likely
        logger.info(f"Found {len(existing_installations)} existing installations")
        if existing_installations:
            from collections import defaultdict, Counter
            product_groups = defaultdict(lambda: defaultdict(list))
            for product_family, version, env in existing_installations:
                product_groups[product_family]['versions'].append(version)
                product_groups[product_family]['envs'].append(str(env))
            
            logger.info(f"Grouped into {len(product_groups)} product families")
            qualifying_products = []
            for product_family, data in product_groups.items():
                total_servers = len(data['versions'])
                logger.info(f"Product {product_family}: {total_servers} servers (threshold: {threshold})")
                if total_servers >= threshold:
                    highest_version = "0.0"
                    if data['versions']:
                        try:
                            # Sort versions based on numeric components
                            highest_version = max(data['versions'], key=lambda v: [int(part) for part in v.split('.') if part.isdigit()])
                        except (ValueError, TypeError):
                            pass
                    environments = sorted(list(set(e for e in data['envs'] if e not in ['Unknown', 'Closed', 'nan'])))
                    if environments:
                        qualifying_products.append((product_family, highest_version, environments, total_servers))
                        logger.info(f"Qualified: {product_family} {highest_version} with {total_servers} servers")
            
            logger.info(f"Found {len(qualifying_products)} qualifying products")
            if qualifying_products:
                qualifying_products.sort(key=lambda x: x[3], reverse=True)

                # Filter out all products in the same major family as the primary software
                primary_family = get_software_family(change_request.software_name)
                filtered_products = [
                    prod for prod in qualifying_products
                    if get_software_family(prod[0]) != primary_family
                ]
                logger.info(f"After filtering out {primary_family}: {len(filtered_products)} products")

                # If no related products left, recommend top N other major products in the environment
                if not filtered_products:
                    # Count all product families in the environment
                    family_counts = {}
                    for product_family, version, envs, count in qualifying_products:
                        fam = get_software_family(product_family)
                        if fam != primary_family:
                            family_counts[fam] = family_counts.get(fam, 0) + count
                    # Get top N families
                    top_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    alt_recs = []
                    for fam, fam_count in top_families:
                        alt_recs.append(f"Consider reviewing {fam} products: {fam_count} server(s) in your environment.")
                    if alt_recs:
                        recommendations.extend(alt_recs)
                    else:
                        recommendations.append("No other major product families found in your environment.")
                else:
                    # List all related products, not just the top 5
                    product_strs = [
                        f"{product} {version}: {count} server(s) across {', '.join(envs)}"
                        for (product, version, envs, count) in filtered_products
                    ]
                    summary = "\n  • " + "\n  • ".join(product_strs)
                    # Use clearer language for the recommendation
                    if change_request.action == 'upgrade':
                        recommendations.append(f"Found related software to consider for upgrade:{summary}")
                    else: # for install, etc.
                        warnings.append(f"Found related existing software:{summary}")

            else:
                if change_request.action == 'install':
                    warnings.append("No significant existing installations found (threshold: 10 servers)")
                elif change_request.action == 'upgrade':
                    recommendations.append("No significant existing installations found for this upgrade (threshold: 10 servers)")
        else:
            # No installations found
            filtering_steps.append({
                "stage": "threshold",
                "count": 0,
                "description": f"No software installations found meeting threshold (≥{threshold} servers)"
            })
        
        logger.info(f"Final recommendations: {len(recommendations)} items")
        for i, rec in enumerate(recommendations):
            logger.info(f"Recommendation {i+1}: {rec}")
        
        # Add personalized recommendations based on user configuration
        if target_os or user_database or user_web_servers:
            personalized_recs = self._generate_personalized_recommendations(
                change_request, target_os, user_database, user_web_servers
            )
            recommendations.extend(personalized_recs)

        # --- Inject co-upgrade recommendations (testable) ---
        # Get co-upgrades for main software
        main_co_upgrades = []
        if change_request.catalogid:
            main_co_upgrades = get_co_upgrades(change_request.catalogid, top_n=3)
            if main_co_upgrades:
                co_upgrade_lines = [
                    f"- {item['model']} (seen together {item['count']} times)"
                    for item in main_co_upgrades if item['model'] != change_request.software_name
                ]
                if co_upgrade_lines:
                    recommendations.append(
                        f"Also consider upgrading these related products (based on historical co-upgrade patterns for {change_request.software_name}):\n" + "\n".join(co_upgrade_lines)
                    )
                    logger.info(f"Injected co-upgrade recommendations for main software catalogid {change_request.catalogid}: {co_upgrade_lines}")
        
        # Get co-upgrades for user configuration software
        user_config_catalogids = []
        
        # Extract catalogids from user configuration
        if target_os and ' - ' in target_os:
            os_catalogid = target_os.split(' - ')[0].strip()
            if os_catalogid and os_catalogid != 'None':
                user_config_catalogids.append(('OS', os_catalogid))
        
        if user_database and ' - ' in user_database:
            db_catalogid = user_database.split(' - ')[0].strip()
            if db_catalogid and db_catalogid != 'None':
                user_config_catalogids.append(('Database', db_catalogid))
        
        if user_web_servers:
            for ws in user_web_servers:
                if ' - ' in ws:
                    ws_catalogid = ws.split(' - ')[0].strip()
                    if ws_catalogid and ws_catalogid != 'None':
                        user_config_catalogids.append(('Web Server', ws_catalogid))
        
        # Get co-upgrades for user configuration
        if user_config_catalogids:
            all_user_co_upgrades = []
            for config_type, catalogid in user_config_catalogids:
                if catalogid != change_request.catalogid:  # Avoid duplicates with main software
                    config_co_upgrades = get_co_upgrades(catalogid, top_n=2)
                    if config_co_upgrades:
                        config_lines = [
                            f"- {item['model']} (seen together {item['count']} times)"
                            for item in config_co_upgrades
                        ]
                        if config_lines:
                            all_user_co_upgrades.append(f"Based on your {config_type} configuration:\n" + "\n".join(config_lines))
            
            if all_user_co_upgrades:
                recommendations.append(
                    "Additional recommendations based on your system configuration:\n" + "\n".join(all_user_co_upgrades)
                )
                logger.info(f"Injected co-upgrade recommendations for user config: {user_config_catalogids}")
        
        if not change_request.catalogid and not user_config_catalogids:
            logger.info(f"No catalogid found for change request or user configuration")
        
        # Track final recommendations
        final_recommendation_count = len(recommendations)
        filtering_steps.append({
            "stage": "final",
            "count": final_recommendation_count,
            "description": f"Final recommendations generated: {final_recommendation_count} items"
        })
        
        # Determine overall compatibility
        is_compatible = len(conflicts) == 0
        
        # Create analysis result object for confidence scoring
        analysis_result = CompatibilityResult(
            is_compatible=is_compatible,
            confidence=0.0,  # Will be calculated by confidence scorer
            affected_servers=affected_servers,
            conflicts=conflicts,
            recommendations=recommendations,
            warnings=warnings,
            alternative_versions=alternative_versions
        )
        
        # Calculate compatibility confidence using simple rule-based approach
        confidence = self._calculate_confidence(change_request, analysis_result, rag_results)
        
        # Update the analysis result with the calculated confidence
        analysis_result.confidence = confidence
        
        # Add filtering steps to the result for transparency
        analysis_result.filtering_steps = filtering_steps
        
        return analysis_result
    
    def _extract_version_from_os(self, os_name: str) -> Optional[str]:
        """Extract version from OS name."""
        if not os_name or not isinstance(os_name, str):
            return None
        match = re.search(r'(\d+\.\d+)', os_name)
        return match.group(1) if match else None
    
    def _has_software_installed(self, server: Dict[str, Any], software_name: str) -> bool:
        """Check if server has specific software installed."""
        server_info = server.get('server_info', {})
        manufacturer = server_info.get('manufacturer', '').upper()
        product_type = server_info.get('product_type', '').upper()
        
        # Simple matching logic - can be enhanced
        return software_name.upper() in manufacturer or software_name.upper() in product_type

    def _calculate_confidence(self, change_request: ChangeRequest, analysis_result: CompatibilityResult, rag_results: Optional[List[Dict[str, Any]]] = None) -> float:
        """Calculate confidence score using RAG similarity scores."""
        try:
            # Debug logging to see what we're receiving
            logger.info(f"Debug: change_request type: {type(change_request)}")
            logger.info(f"Debug: rag_results type: {type(rag_results)}")
            if rag_results:
                logger.info(f"Debug: rag_results length: {len(rag_results)}")
                if len(rag_results) > 0:
                    logger.info(f"Debug: first rag_result type: {type(rag_results[0])}")
                    logger.info(f"Debug: first rag_result value: {rag_results[0]}")

            # Handle case where change_request might be a list instead of ChangeRequest object
            if isinstance(change_request, list):
                logger.warning(f"change_request is a list, not a ChangeRequest object. Using fallback query.")
                query_text = "compatibility analysis"
            else:
                query_text = change_request.raw_text

            # Handle case where rag_results might be an integer (length) instead of a list
            if isinstance(rag_results, int):
                logger.warning(f"rag_results is an integer ({rag_results}), not a list. Using fallback query.")
                rag_results = None

            # Use existing RAG results if available, otherwise perform a new query
            if rag_results and isinstance(rag_results, list) and len(rag_results) > 0:
                # Use the actual similarity scores from RAG
                scores = []
                for result in rag_results[:3]:
                    if isinstance(result, dict):
                        score = result.get('similarity_score', 0.0)
                        scores.append(score)
                    elif isinstance(result, (int, float)):
                        # Handle case where result is a number (might be similarity score directly)
                        logger.warning(f"RAG result is a number: {result}, treating as similarity score")
                        scores.append(float(result))
                    else:
                        logger.warning(f"Unexpected result type: {type(result)}")
                
                if scores:
                    # Average the top similarity scores (already normalized to 0-1)
                    confidence = sum(scores) / len(scores)
                    logger.info(f"Confidence from RAG similarity scores: {confidence:.3f} (avg_score: {confidence:.3f})")
                    return confidence
            
            # Fallback to a simple query if no RAG results available
            rag_results = self.query_engine.query(query_text, top_k=5)
            if rag_results and len(rag_results) > 0:
                scores = [result.get('similarity_score', 0.0) for result in rag_results[:3]]
                if scores:
                    confidence = sum(scores) / len(scores)
                    logger.info(f"Confidence from fallback RAG query: {confidence:.3f} (avg_score: {confidence:.3f})")
                    return confidence
            
            # If no RAG results at all, return neutral confidence
            logger.warning("No RAG results available for confidence calculation")
            return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Neutral confidence on error

    
    def _generate_personalized_recommendations(self, change_request: ChangeRequest, target_os: Optional[str], 
                                             user_database: Optional[str], user_web_servers: Optional[List[str]]) -> List[str]:
        """Generate personalized recommendations based on user configuration."""
        recommendations = []
        
        # Don't add personalized config info for general information questions
        if change_request.action == "info":
            return recommendations
        
        config_parts = []
        if target_os:
            config_parts.append(f"OS: {target_os}")
        if user_database:
            config_parts.append(f"DB: {user_database}")
        if user_web_servers:
            config_parts.append(f"Web: {', '.join(user_web_servers)}")
        
        if config_parts:
            recommendations.append(f"Analysis based on your configuration: {', '.join(config_parts)}")
        return recommendations
    
    def format_analysis_result(self, result: CompatibilityResult, change_request: ChangeRequest) -> str:
        print("DEBUG: result.recommendations:", result.recommendations)
        print("DEBUG: type of each recommendation:", [type(r) for r in result.recommendations])
        output = []

        # Overall result
        status = "✅ COMPATIBLE" if result.is_compatible else "❌ INCOMPATIBLE"
        output.append(f"Status: {status} (Confidence: {result.confidence:.1%})")
        output.append("")

        # Affected models
        affected_models_lines = ["📊 Affected Models:"]
        if result.affected_servers:
            model_env_map = {}
            for server in result.affected_servers:
                model = server.get('server_info', {}).get('model', 'Unknown')
                product_type = server.get('server_info', {}).get('product_type', 'Unknown')
                env = server.get('environment', 'Unknown')
                if model in ['Unknown', 'Closed'] or product_type in ['Unknown', 'Closed'] or str(env) in ['Unknown', 'Closed', 'nan']:
                    continue
                key = f"{model} ({product_type})"
                if key not in model_env_map:
                    model_env_map[key] = set()
                model_env_map[key].add(env)
            if model_env_map:
                for model, envs in list(model_env_map.items())[:5]:
                    envs_str = ', '.join(sorted([str(e) for e in envs]))
                    affected_models_lines.append(f"  • {model} [{envs_str}]")
                if len(model_env_map) > 5:
                    affected_models_lines.append(f"  ... and {len(model_env_map) - 5} more")
            else:
                affected_models_lines.append("  • No specific models identified")
        affected_models_lines.append("")
        output.append("\n".join(affected_models_lines))

        # Conflicts
        if result.conflicts:
            output.append("❌ Conflicts Found:")
            for conflict in result.conflicts:
                output.append(f"  • {conflict}")
            output.append("")

        # Warnings
        if result.warnings:
            output.append("⚠️ Warnings:")
            for warning in result.warnings:
                output.append(f"  • {warning}")
            output.append("")

        # Recommendations
        if result.recommendations:
            output.append("💡 Recommendations:")
            for rec in result.recommendations:
                if not isinstance(rec, str):
                    rec = str(rec)
                output.append(f"  • {rec}")
            output.append("")

        # Alternative versions
        if result.alternative_versions:
            output.append("🔄 Alternative Versions:")
            for version in result.alternative_versions:
                output.append(f"  • {version}")
            output.append("")

        return "\n".join(output)

    async def parse_multiple_change_requests(self, text: str) -> List[ChangeRequest]:
        """Parse a natural language request into multiple ChangeRequest objects (one per software/version), with per-software action detection using QueryParser's intent patterns."""
        parser = QueryParser()
        context = parser.parse_query(text)
        detected_software = context.get('detected_software', [])
        detected_versions = context.get('detected_versions', [])
        primary_intent = context.get('primary_intent', 'upgrade')
        text_lower = text.lower()

        # Automatically extract action keywords from QueryParser's intent_patterns
        action_intents = ['upgrade', 'install', 'remove', 'downgrade', 'rollback']
        all_action_words = []
        for intent in action_intents:
            for pattern in parser.intent_patterns.get(intent, []):
                all_action_words.append((intent, pattern))
        # Fallback: if no action keywords found, use all patterns
        if not all_action_words:
            for intent, patterns in parser.intent_patterns.items():
                for pattern in patterns:
                    all_action_words.append((intent, pattern))

        # Find all software and their positions
        software_positions = []
        for software in detected_software:
            for match in re.finditer(re.escape(software.lower()), text_lower):
                software_positions.append((software, match.start()))

        # Find all version numbers and their positions
        version_pattern = r'\d+\.\d+(?:\.\d+)?'
        version_positions = [(m.group(), m.start()) for m in re.finditer(version_pattern, text_lower)]

        # Find all action keywords and their positions
        action_positions = []
        for canonical, word in all_action_words:
            for match in re.finditer(re.escape(word), text_lower):
                action_positions.append((canonical, match.start()))

        # Sort by position in text
        software_positions.sort(key=lambda x: x[1])
        version_positions.sort(key=lambda x: x[1])
        action_positions.sort(key=lambda x: x[1])

        # Pair each software with the nearest version (after it), and nearest action (before or after)
        change_requests = []
        for i, (software, s_pos) in enumerate(software_positions):
            # Find nearest version after software
            version = None
            for v, v_pos in version_positions:
                if v_pos > s_pos:
                    version = v
                    break
            # Find nearest action keyword (before or after software)
            nearest_action = primary_intent
            min_dist = None
            for action, a_pos in action_positions:
                dist = abs(a_pos - s_pos)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    nearest_action = action
            
            # Create a sub-query for this software to get better LLM classification
            sub_query = f"{nearest_action} {software}"
            if version:
                sub_query += f" {version}"
            
            # Use async LLM classification for better intent detection
            cr = await self.parse_change_request_async(sub_query)
            change_requests.append(cr)
        
        # If no software detected, fallback to single parse
        if not change_requests:
            cr = await self.parse_change_request_async(text)
            change_requests = [cr] if cr else []
        return change_requests

    async def analyze_multiple_compatibility(self, change_requests: List[ChangeRequest], target_os: Optional[str] = None,
                                    user_database: Optional[str] = None, user_web_servers: Optional[List[str]] = None) -> List[Tuple[ChangeRequest, CompatibilityResult]]:
        """Analyze compatibility for multiple change requests, optionally filtering by user configuration."""
        results = []
        for cr in change_requests:
            result = self.analyze_compatibility(cr, target_os=target_os, user_database=user_database, user_web_servers=user_web_servers)
            print("DEBUG: analyze_compatibility result type:", type(result))
            results.append((cr, result))
        return results

    def format_multiple_results(self, results: list) -> str:
        """Format multiple compatibility results for display, with defensive checks."""
        output = []
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                cr, result = item
                output.append(self.format_analysis_result(result, cr))
            else:
                print("DEBUG: Unexpected item in results:", item)
                output.append(str(item))
            output.append("\n" + ("-" * 80) + "\n")
        return "\n".join(output)
