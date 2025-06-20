#!/usr/bin/env python3
"""
Compatibility Analyzer for RAG Pipeline
Analyzes software change requests against existing infrastructure
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from collections import Counter
from collections import defaultdict
from src.rag.query_engine import QueryEngine
from src.models.query_parser import QueryParser

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

class CompatibilityAnalyzer:
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

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def parse_change_request(self, text: str) -> ChangeRequest:
        """Parse a natural language change request into structured data.
        
        Args:
            text: Natural language request (e.g., "I want to upgrade Apache to 2.4.50")
            
        Returns:
            Parsed ChangeRequest object
        """
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
            raw_text=text
        )
    
    def analyze_compatibility(self, change_request: ChangeRequest) -> CompatibilityResult:
        """Analyze compatibility of a change request.
        
        Args:
            change_request: Parsed change request
            
        Returns:
            CompatibilityResult with analysis
        """
        affected_servers = []
        conflicts = []
        recommendations = []
        warnings = []
        alternative_versions = []
        
        # --- Dynamically find dependencies using the RAG Query Engine ---
        software_to_show_families = [change_request.software_name.upper()]
        try:
            dependency_query = f"What are the dependencies and compatible software for {change_request.software_name}?"
            rag_results = self.query_engine.query(dependency_query, top_k=3)
            if rag_results:
                context = self.query_engine.format_results_for_llm(rag_results)
                # Parse context for other known software families (using the dynamic list)
                import re
                # Use the dynamically generated list of patterns
                found_deps = re.findall(r'|'.join(re.escape(f) for f in self.known_software_families), context.upper())
                software_to_show_families.extend(found_deps)
                software_to_show_families = list(set(software_to_show_families))
            logger.info(f"Analyzing compatibility for: {software_to_show_families}")
        except Exception as e:
            logger.error(f"Could not query RAG for dependencies: {e}")
        
        # Find affected servers
        if change_request.environment:
            affected_servers = [s for s in self.analysis.get('servers', []) if s['environment'] == change_request.environment]
        else:
            affected_servers = self.analysis.get('servers', [])
        
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
                        server_os = self._extract_os_from_server(server)
                        if server_os and server_os not in compatible_os:
                            conflicts.append(f"Server {server['name']} has incompatible OS: {server_os}")
                
                # Check version constraints
                min_version = version_rules.get('min_os_version')
                max_version = version_rules.get('max_os_version')
                if min_version or max_version:
                    for server in affected_servers:
                        server_os = self._extract_os_from_server(server)
                        if server_os:
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
        existing_installations = []
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
                    break
        
        # --- Aggregate and format recommendations ---
        threshold = 50
        if existing_installations:
            from collections import defaultdict, Counter
            product_groups = defaultdict(lambda: defaultdict(list))
            for product_family, version, env in existing_installations:
                product_groups[product_family]['versions'].append(version)
                product_groups[product_family]['envs'].append(str(env))
            
            qualifying_products = []
            for product_family, data in product_groups.items():
                total_servers = len(data['versions'])
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
            
            if qualifying_products:
                qualifying_products.sort(key=lambda x: x[3], reverse=True)

                # Filter out the software the user is already asking about
                primary_software_family = change_request.software_name.upper()
                filtered_products = [
                    prod for prod in qualifying_products
                    if primary_software_family not in prod[0]
                ]

                if not filtered_products:
                    # If nothing is left after filtering, the recommendation is different
                    if change_request.action == 'upgrade':
                        recommendations.append("No other significant dependent software installations were found.")
                else:
                    top_products = filtered_products[:5]
                    product_strs = [
                        f"{product} {version}: {count} server(s) across {', '.join(envs)}"
                        for (product, version, envs, count) in top_products
                    ]
                    more_count = len(filtered_products) - len(top_products)
                    if more_count > 0:
                        product_strs.append(f"...and {more_count} more related product(s) found")
                    
                    summary = "\n  • " + "\n  • ".join(product_strs)
                    
                    # Use clearer language for the recommendation
                    if change_request.action == 'upgrade':
                        recommendations.append(f"Found related software to consider for upgrade:{summary}")
                    else: # for install, etc.
                        warnings.append(f"Found related existing software:{summary}")

            else:
                if change_request.action == 'install':
                    warnings.append("No significant existing installations found (threshold: 50 servers)")
                elif change_request.action == 'upgrade':
                    recommendations.append("No significant existing installations found for this upgrade (threshold: 50 servers)")
        
        # Calculate compatibility confidence
        confidence = self._calculate_confidence(conflicts, warnings, len(affected_servers))
        
        # Determine overall compatibility
        is_compatible = len(conflicts) == 0
        
        return CompatibilityResult(
            is_compatible=is_compatible,
            confidence=confidence,
            affected_servers=affected_servers,
            conflicts=conflicts,
            recommendations=recommendations,
            warnings=warnings,
            alternative_versions=alternative_versions
        )
    
    def _extract_os_from_server(self, server: Dict[str, Any]) -> Optional[str]:
        """Extract OS information from server data."""
        # This would need to be customized based on your actual data structure
        # For now, return None as placeholder
        return None
    
    def _extract_version_from_os(self, os_name: str) -> Optional[str]:
        """Extract version from OS name."""
        match = re.search(r'(\d+\.\d+)', os_name)
        return match.group(1) if match else None
    
    def _has_software_installed(self, server: Dict[str, Any], software_name: str) -> bool:
        """Check if server has specific software installed."""
        server_info = server.get('server_info', {})
        manufacturer = server_info.get('manufacturer', '').upper()
        product_type = server_info.get('product_type', '').upper()
        
        # Simple matching logic - can be enhanced
        return software_name.upper() in manufacturer or software_name.upper() in product_type
    
    def _calculate_confidence(self, conflicts: List[str], warnings: List[str], total_servers: int) -> float:
        """Calculate confidence score for compatibility."""
        if total_servers == 0:
            return 0.0
        
        # Base confidence
        confidence = 1.0
        
        # Reduce for conflicts
        confidence -= len(conflicts) * 0.3
        
        # Reduce for warnings
        confidence -= len(warnings) * 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def format_analysis_result(self, result: CompatibilityResult, change_request: ChangeRequest) -> str:
        """Format analysis result for display."""
        output = []
        
        output.append(f"🔍 Compatibility Analysis for: {change_request.raw_text}")
        output.append("=" * 60)
        
        # Overall result
        status = "✅ COMPATIBLE" if result.is_compatible else "❌ INCOMPATIBLE"
        output.append(f"Status: {status} (Confidence: {result.confidence:.1%})")
        output.append("")
        
        # Affected models
        output.append(f"📊 Affected Models: ")
        if result.affected_servers:
            # Group by model and version
            model_env_map = {}
            for server in result.affected_servers:
                model = server.get('server_info', {}).get('model', 'Unknown')
                product_type = server.get('server_info', {}).get('product_type', 'Unknown')
                env = server.get('environment', 'Unknown')
                
                # Skip entries with "Unknown" or "Closed" values
                if model in ['Unknown', 'Closed'] or product_type in ['Unknown', 'Closed'] or str(env) in ['Unknown', 'Closed', 'nan']:
                    continue
                    
                key = f"{model} ({product_type})"
                if key not in model_env_map:
                    model_env_map[key] = set()
                model_env_map[key].add(env)
            
            if model_env_map:
                for model, envs in list(model_env_map.items())[:5]:
                    # Ensure all envs are strings before sorting
                    envs_str = ', '.join(sorted([str(e) for e in envs]))
                    output.append(f"  • {model} [{envs_str}]")
                if len(model_env_map) > 5:
                    output.append(f"  ... and {len(model_env_map) - 5} more")
            else:
                output.append("  • No specific models identified")
        output.append("")
        
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
                output.append(f"  • {rec}")
            output.append("")
        
        # Alternative versions
        if result.alternative_versions:
            output.append("🔄 Alternative Versions:")
            for version in result.alternative_versions:
                output.append(f"  • {version}")
            output.append("")
        
        return "\n".join(output)

    def parse_multiple_change_requests(self, text: str) -> List[ChangeRequest]:
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
            change_requests.append(ChangeRequest(
                software_name=software,
                version=version,
                action=nearest_action,
                raw_text=text
            ))
        # If no software detected, fallback to single parse
        if not change_requests:
            cr = self.parse_change_request(text)
            change_requests = [cr] if cr else []
        return change_requests

    def analyze_multiple_compatibility(self, change_requests: List[ChangeRequest]) -> List[Tuple[ChangeRequest, CompatibilityResult]]:
        """Analyze compatibility for multiple change requests."""
        results = []
        for cr in change_requests:
            result = self.analyze_compatibility(cr)
            results.append((cr, result))
        return results

    def format_multiple_results(self, results: List[Tuple[ChangeRequest, CompatibilityResult]]) -> str:
        """Format multiple compatibility results for display."""
        output = []
        for cr, result in results:
            output.append(self.format_analysis_result(result, cr))
            output.append("\n" + ("-" * 80) + "\n")
        return "\n".join(output)

def main():
    """Test the compatibility analyzer."""
    analyzer = CompatibilityAnalyzer()
    
    # Example queries for multi-upgrade, multi-action parsing and analysis
    test_requests = [
        "Upgrade Apache 2.4.50 and Tomcat 9.0.0 in production",  # two upgrades
        "Install NGINX 1.18 and remove Apache HTTPD 2.4 from dev",  # install + remove
        "Rollback WebSphere 8.5, upgrade Apache HTTPD 2.4, and install Python 3.11 on UAT",  # rollback + upgrade + install
        "Remove Tomcat and downgrade MySQL 8.0 in staging",  # remove + downgrade
        "Upgrade IBM HTTP SERVER 9.0.0 and APACHE HTTP SERVER 2.2.24",  # two upgrades, different phrasing
        "Uninstall Apache, add NGINX, and update Python to 3.10 in dev",  # remove + install + upgrade
        "Upgrade Apache and Tomcat",  # two upgrades, no versions
        "Install WebSphere and Python 3.9 in production",  # install, one with version
        "Rollback Java 11 and remove Node.js from UAT",  # rollback + remove
        "Upgrade Apache, remove Tomcat, and install NGINX in prod"  # upgrade + remove + install
    ]
    
    for request_text in test_requests:
        print("\n" + "="*80)
        
        # Parse requests
        change_requests = analyzer.parse_multiple_change_requests(request_text)
        print(f"Parsed Requests: {change_requests}")
        
        # Analyze compatibility
        results = analyzer.analyze_multiple_compatibility(change_requests)
        
        # Format and display results
        formatted_results = analyzer.format_multiple_results(results)
        print(formatted_results)

if __name__ == "__main__":
    main() 