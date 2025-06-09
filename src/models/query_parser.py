import re
from typing import Dict, Any, List, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryParser:
    def __init__(self):
        """Initialize the query parser with enhanced intent detection."""
        self.intent_patterns = {
            'upgrade': [
                r'upgrade',
                r'update',
                r'install',
                r'new version',
                r'latest version',
                r'how to upgrade',
                r'upgrading',
                r'version change'
            ],
            'rollback': [
                r'rollback',
                r'revert',
                r'downgrade',
                r'previous version',
                r'old version',
                r'undo',
                r'back to',
                r'restore'
            ],
            'issue': [
                r'problem',
                r'error',
                r'issue',
                r'bug',
                r'fail',
                r'not working',
                r'broken',
                r'trouble',
                r'fix',
                r'resolve'
            ],
            'compatibility': [
                r'compatible',
                r'compatibility',
                r'work with',
                r'support',
                r'require',
                r'dependency',
                r'prerequisite'
            ],
            'performance': [
                r'performance',
                r'slow',
                r'fast',
                r'speed',
                r'optimize',
                r'efficient',
                r'resource',
                r'memory',
                r'cpu'
            ],
            'security': [
                r'security',
                r'vulnerability',
                r'patch',
                r'secure',
                r'protect',
                r'exploit',
                r'attack'
            ]
        }
        
        # Load common software names and versions
        self.software_info = self._load_software_info()
        
        # Initialize TF-IDF vectorizer for better matching
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # Initialize with common queries
        self._initialize_vectorizer()
    
    def _load_software_info(self) -> Dict[str, Any]:
        """Load software information from JSON file."""
        try:
            software_file = os.path.join(os.path.dirname(__file__), 'software_info.json')
            if os.path.exists(software_file):
                with open(software_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Error loading software info: {str(e)}")
            return {}
    
    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer with common queries."""
        try:
            common_queries = [
                "how to upgrade software version",
                "how to rollback to previous version",
                "fix version compatibility issues",
                "resolve performance problems after upgrade",
                "security patch installation guide",
                "version upgrade best practices",
                "troubleshoot upgrade failures",
                "check software compatibility",
                "optimize system performance",
                "apply security updates"
            ]
            self.vectorizer.fit(common_queries)
        except Exception as e:
            logger.warning(f"Error initializing vectorizer: {str(e)}")
    
    def _detect_software(self, query: str) -> List[str]:
        """Detect software names mentioned in the query."""
        detected_software = []
        query_lower = query.lower()
        
        for software, info in self.software_info.items():
            # Check software name
            if software.lower() in query_lower:
                detected_software.append(software)
            # Check aliases
            for alias in info.get('aliases', []):
                if alias.lower() in query_lower:
                    detected_software.append(software)
                    break
        
        return detected_software
    
    def _detect_versions(self, query: str) -> List[str]:
        """Detect version numbers mentioned in the query."""
        version_pattern = r'\d+\.\d+(?:\.\d+)?'
        return re.findall(version_pattern, query)
    
    def _calculate_intent_scores(self, query: str) -> Dict[str, float]:
        """Calculate intent scores using both pattern matching and TF-IDF."""
        scores = {intent: 0.0 for intent in self.intent_patterns.keys()}
        
        # Pattern matching scores
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query.lower()):
                    scores[intent] += 1
        
        # TF-IDF scores
        try:
            query_vector = self.vectorizer.transform([query])
            for intent, patterns in self.intent_patterns.items():
                pattern_vector = self.vectorizer.transform(patterns)
                similarity = cosine_similarity(query_vector, pattern_vector).max()
                scores[intent] += similarity
        except Exception as e:
            logger.warning(f"Error calculating TF-IDF scores: {str(e)}")
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query with enhanced context understanding."""
        try:
            # Detect intents
            intent_scores = self._calculate_intent_scores(query)
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            
            # Detect software and versions
            detected_software = self._detect_software(query)
            detected_versions = self._detect_versions(query)
            
            # Build context
            context = {
                'primary_intent': primary_intent,
                'intent_scores': intent_scores,
                'detected_software': detected_software,
                'detected_versions': detected_versions,
                'is_version_specific': len(detected_versions) > 0,
                'is_software_specific': len(detected_software) > 0
            }
            
            # Add software-specific context
            if detected_software:
                software_context = {}
                for software in detected_software:
                    if software in self.software_info:
                        software_context[software] = {
                            'type': self.software_info[software].get('type', ''),
                            'common_issues': self.software_info[software].get('common_issues', []),
                            'upgrade_paths': self.software_info[software].get('upgrade_paths', [])
                        }
                context['software_context'] = software_context
            
            return context
            
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            return {
                'primary_intent': 'unknown',
                'intent_scores': {'unknown': 1.0},
                'detected_software': [],
                'detected_versions': [],
                'is_version_specific': False,
                'is_software_specific': False
            }
    
    def get_query_context(self, query: str) -> Dict[str, Any]:
        """Get detailed context for the query."""
        parsed = self.parse_query(query)
        
        # Add additional context based on intent
        if parsed['primary_intent'] == 'upgrade':
            parsed['context'] = {
                'is_major_upgrade': any(
                    len(v.split('.')) >= 2 and int(v.split('.')[0]) < int(v.split('.')[1])
                    for v in parsed['detected_versions']
                ),
                'requires_backup': True,
                'suggested_steps': [
                    'Check system requirements',
                    'Backup current configuration',
                    'Review release notes',
                    'Test in staging environment',
                    'Plan maintenance window'
                ]
            }
        elif parsed['primary_intent'] == 'rollback':
            parsed['context'] = {
                'requires_backup': False,
                'suggested_steps': [
                    'Verify backup availability',
                    'Check rollback compatibility',
                    'Plan data migration',
                    'Schedule maintenance window',
                    'Prepare rollback script'
                ]
            }
        elif parsed['primary_intent'] == 'issue':
            parsed['context'] = {
                'suggested_steps': [
                    'Check error logs',
                    'Verify system requirements',
                    'Review recent changes',
                    'Test in isolation',
                    'Check known issues'
                ]
            }
        
        return parsed 