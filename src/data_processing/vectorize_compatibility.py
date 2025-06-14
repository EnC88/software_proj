import json
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from tqdm import tqdm
import faiss
import pickle
from pathlib import Path
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompatibilityRule:
    def __init__(self, 
                 software_type: str,
                 min_version: str,
                 max_version: str,
                 required_dependencies: List[str],
                 os_compatibility: List[str],
                 db_compatibility: List[str]):
        self.software_type = software_type
        self.min_version = min_version
        self.max_version = max_version
        self.required_dependencies = required_dependencies
        self.os_compatibility = os_compatibility
        self.db_compatibility = db_compatibility

class SoftwareComponent:
    def __init__(self, 
                 name: str,
                 software_type: str,
                 version: str,
                 environment: str,
                 os: str,
                 database: Optional[str] = None):
        self.name = name
        self.software_type = software_type
        self.version = version
        self.environment = environment
        self.os = os
        self.database = database
        self.dependencies: List[str] = []
        self.last_updated = datetime.now()

class CompatibilityRecommender:
    def __init__(self, batch_size=1000, index_path='data/processed/recommendation_index'):
        self.batch_size = batch_size
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.server_data = []
        self.software_data = []
        self.compatibility_matrix = None
        self.compatibility_rules: Dict[str, CompatibilityRule] = {}
        self.software_components: Dict[str, SoftwareComponent] = {}
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Initialize default compatibility rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default compatibility rules for common software types."""
        self.compatibility_rules = {
            'HTTP SERVER': CompatibilityRule(
                software_type='HTTP SERVER',
                min_version='2.4.0',
                max_version='2.4.57',
                required_dependencies=['OpenSSL', 'mod_ssl'],
                os_compatibility=['Linux', 'Windows Server'],
                db_compatibility=['MySQL', 'PostgreSQL']
            ),
            'APPLICATION SERVER': CompatibilityRule(
                software_type='APPLICATION SERVER',
                min_version='9.0.0',
                max_version='9.0.65',
                required_dependencies=['JDK', 'JRE'],
                os_compatibility=['Linux', 'Windows Server', 'AIX'],
                db_compatibility=['Oracle', 'DB2', 'MySQL']
            )
        }
    
    def add_compatibility_rule(self, rule: CompatibilityRule):
        """Add a new compatibility rule."""
        self.compatibility_rules[rule.software_type] = rule
    
    def load_data(self, filepath='data/processed/compatibility_analysis.json'):
        """Load and prepare the data for analysis."""
        logger.info(f"Loading data from {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.server_data = data['servers']
        logger.info(f"Loaded {len(self.server_data)} server entries")
        
        # Extract software information
        self.software_data = self._extract_software_info()
        
        # Initialize software components
        self._initialize_software_components()
    
    def _initialize_software_components(self):
        """Initialize software components from the loaded data."""
        for server in self.server_data:
            component = SoftwareComponent(
                name=server['name'],
                software_type=server['server_info']['product_class'],
                version=self._extract_version(server['server_info']['model']),
                environment=server['environment'],
                os=self._extract_os_info(server),
                database=self._extract_database_info(server)
            )
            self.software_components[server['name']] = component
    
    def _extract_os_info(self, server: dict) -> str:
        """Extract OS information from server data."""
        # This would need to be implemented based on your data structure
        return "Linux"  # Placeholder
    
    def _extract_database_info(self, server: dict) -> Optional[str]:
        """Extract database information from server data."""
        # This would need to be implemented based on your data structure
        return None  # Placeholder
    
    def check_compatibility(self, component1: SoftwareComponent, 
                          component2: SoftwareComponent) -> Tuple[bool, List[str]]:
        """Check compatibility between two software components."""
        issues = []
        
        # Check version compatibility
        if component1.software_type == component2.software_type:
            rule = self.compatibility_rules.get(component1.software_type)
            if rule:
                if not self._is_version_compatible(component1.version, rule):
                    issues.append(f"Version {component1.version} is not compatible with {component1.software_type}")
                if not self._is_version_compatible(component2.version, rule):
                    issues.append(f"Version {component2.version} is not compatible with {component2.software_type}")
        
        # Check OS compatibility
        if component1.os != component2.os:
            issues.append(f"OS mismatch: {component1.os} vs {component2.os}")
        
        # Check database compatibility
        if component1.database and component2.database and component1.database != component2.database:
            issues.append(f"Database mismatch: {component1.database} vs {component2.database}")
        
        return len(issues) == 0, issues
    
    def _is_version_compatible(self, version: str, rule: CompatibilityRule) -> bool:
        """Check if a version is compatible with a rule."""
        try:
            version_parts = [int(x) for x in version.split('.')]
            min_parts = [int(x) for x in rule.min_version.split('.')]
            max_parts = [int(x) for x in rule.max_version.split('.')]
            
            return (version_parts >= min_parts and version_parts <= max_parts)
        except ValueError:
            return False
    
    def get_upgrade_recommendations(self, software_name: str, target_version: Optional[str] = None) -> List[dict]:
        """Get upgrade recommendations for a specific software component."""
        if self.compatibility_matrix is None:
            self.build_compatibility_matrix()
        
        component = self.software_components.get(software_name)
        if not component:
            logger.error(f"Software {software_name} not found")
            return []
        
        # Get compatibility scores
        software_idx = next(
            (i for i, s in enumerate(self.software_data) if s['name'] == software_name),
            None
        )
        compatibility_scores = self.compatibility_matrix[software_idx]
        
        # Get top recommendations
        top_indices = np.argsort(compatibility_scores)[-5:][::-1]
        
        recommendations = []
        for idx in top_indices:
            software = self.software_data[idx]
            target_component = self.software_components.get(software['name'])
            
            if target_component:
                is_compatible, issues = self.check_compatibility(component, target_component)
                
                recommendations.append({
                    'name': software['name'],
                    'manufacturer': software['manufacturer'],
                    'product_class': software['product_class'],
                    'model': software['model'],
                    'version': software['version'],
                    'compatibility_score': float(compatibility_scores[idx]),
                    'is_compatible': is_compatible,
                    'compatibility_issues': issues,
                    'dependencies': self._get_required_dependencies(software['product_class'])
                })
        
        return recommendations
    
    def _get_required_dependencies(self, software_type: str) -> List[str]:
        """Get required dependencies for a software type."""
        rule = self.compatibility_rules.get(software_type)
        return rule.required_dependencies if rule else []
    
    def analyze_upgrade_path(self, current_software: str, target_version: str) -> Optional[List[dict]]:
        """Analyze the upgrade path from current version to target version."""
        current_component = self.software_components.get(current_software)
        if not current_component:
            logger.error(f"Software {current_software} not found")
            return None
        
        # Find all intermediate versions
        all_versions = sorted(set(
            s['version'] for s in self.software_data
            if s['manufacturer'] == current_component.software_type
        ))
        
        try:
            current_version_idx = all_versions.index(current_component.version)
            target_version_idx = all_versions.index(target_version)
        except ValueError:
            logger.error("Version not found in available versions")
            return None
        
        if current_version_idx > target_version_idx:
            logger.error("Target version is older than current version")
            return None
        
        # Get intermediate versions
        intermediate_versions = all_versions[current_version_idx+1:target_version_idx+1]
        
        # Get recommendations for each intermediate version
        upgrade_path = []
        for version in intermediate_versions:
            compatible_software = [
                s for s in self.software_data
                if s['version'] == version and
                s['manufacturer'] == current_component.software_type
            ]
            
            if compatible_software:
                step_recommendations = []
                for software in compatible_software:
                    target_component = self.software_components.get(software['name'])
                    if target_component:
                        is_compatible, issues = self.check_compatibility(current_component, target_component)
                        step_recommendations.append({
                            'name': software['name'],
                            'manufacturer': software['manufacturer'],
                            'model': software['model'],
                            'is_compatible': is_compatible,
                            'compatibility_issues': issues,
                            'dependencies': self._get_required_dependencies(software['product_class'])
                        })
                
                upgrade_path.append({
                    'version': version,
                    'recommendations': step_recommendations
                })
        
        return upgrade_path

    def _extract_software_info(self):
        """Extract software information from server data."""
        software_info = []
        for server in self.server_data:
            info = {
                'name': server['name'],
                'environment': server['environment'],
                'manufacturer': server['server_info']['manufacturer'],
                'product_class': server['server_info']['product_class'],
                'product_type': server['server_info']['product_type'],
                'model': server['server_info']['model'],
                'version': self._extract_version(server['server_info']['model']),
                'status': server['server_info']['status'],
                'install_path': server['deployment_info']['install_path'],
                'dependencies': self._get_required_dependencies(server['server_info']['product_class'])
            }
            software_info.append(info)
        return software_info

    def _extract_version(self, model):
        """Extract version number from model string."""
        version_match = re.search(r'\d+\.\d+(\.\d+)?', model)
        return version_match.group(0) if version_match else None

    def build_compatibility_matrix(self):
        """Build compatibility matrix between different software components."""
        logger.info("Building compatibility matrix")
        
        # Create feature vectors for each software component
        software_descriptions = [
            f"{s['manufacturer']} {s['product_class']} {s['product_type']} {s['model']} {' '.join(s['dependencies'])}"
            for s in self.software_data
        ]
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(software_descriptions)
        
        # Calculate compatibility scores
        self.compatibility_matrix = cosine_similarity(tfidf_matrix)
        
        # Save the matrix and vectorizer
        with open(self.index_path / 'compatibility_matrix.npy', 'wb') as f:
            np.save(f, self.compatibility_matrix)
        with open(self.index_path / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

def main():
    recommender = CompatibilityRecommender()
    
    # Load data
    recommender.load_data()
    
    # Example: Get upgrade recommendations
    example_software = "websrv01"
    logger.info(f"\nUpgrade recommendations for {example_software}:")
    recommendations = recommender.get_upgrade_recommendations(example_software)
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\nRecommendation {i}:")
        logger.info(f"Software: {rec['name']}")
        logger.info(f"Manufacturer: {rec['manufacturer']}")
        logger.info(f"Model: {rec['model']}")
        logger.info(f"Version: {rec['version']}")
        logger.info(f"Compatibility Score: {rec['compatibility_score']:.4f}")
        logger.info(f"Compatible: {rec['is_compatible']}")
        if rec['compatibility_issues']:
            logger.info(f"Compatibility Issues: {', '.join(rec['compatibility_issues'])}")
        logger.info(f"Dependencies: {', '.join(rec['dependencies'])}")
    
    # Example: Analyze upgrade path
    logger.info("\nUpgrade path analysis:")
    upgrade_path = recommender.analyze_upgrade_path("websrv01", "2.4")
    if upgrade_path:
        for step in upgrade_path:
            logger.info(f"\nUpgrade to version {step['version']}:")
            for rec in step['recommendations']:
                logger.info(f"- {rec['name']} ({rec['model']})")
                logger.info(f"  Compatible: {rec['is_compatible']}")
                if rec['compatibility_issues']:
                    logger.info(f"  Issues: {', '.join(rec['compatibility_issues'])}")
                logger.info(f"  Dependencies: {', '.join(rec['dependencies'])}")

if __name__ == "__main__":
    main() 