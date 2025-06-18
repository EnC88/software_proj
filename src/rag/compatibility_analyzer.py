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
        
        # Find affected servers
        if change_request.environment:
            # Filter by environment
            affected_servers = [
                server for server in self.analysis.get('servers', [])
                if server['environment'] == change_request.environment
            ]
        else:
            # Check all servers
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
        
        # Check for existing installations
        existing_installations = []
        for server in affected_servers:
            if self._has_software_installed(server, change_request.software_name):
                existing_installations.append(server['name'])
        
        if existing_installations:
            if change_request.action == 'install':
                warnings.append(f"Software already installed on: {', '.join(existing_installations)}")
            elif change_request.action == 'upgrade':
                recommendations.append(f"Upgrading existing installations on: {', '.join(existing_installations)}")
        
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
        
        # Affected servers
        output.append(f"📊 Affected Servers: {len(result.affected_servers)}")
        if result.affected_servers:
            for server in result.affected_servers[:5]:  # Show first 5
                output.append(f"  • {server['name']} ({server['environment']})")
            if len(result.affected_servers) > 5:
                output.append(f"  ... and {len(result.affected_servers) - 5} more")
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

def main():
    """Test the compatibility analyzer."""
    analyzer = CompatibilityAnalyzer()
    
    # Test cases
    test_requests = [
        "I want to upgrade Apache to 2.4.50",
        "Can I install Python 3.11 on my servers?",
        "I need to upgrade WebSphere to 8.5 in production",
        "Install nginx on development servers"
    ]
    
    for request_text in test_requests:
        print("\n" + "="*80)
        
        # Parse request
        change_request = analyzer.parse_change_request(request_text)
        print(f"Parsed Request: {change_request}")
        
        # Analyze compatibility
        result = analyzer.analyze_compatibility(change_request)
        
        # Format and display result
        formatted_result = analyzer.format_analysis_result(result, change_request)
        print(formatted_result)

if __name__ == "__main__":
    main() 