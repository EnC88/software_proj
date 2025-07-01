import pandas as pd
import logging
from collections import defaultdict
import json
from typing import List, Dict, Any
import os
import operator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompatibilityAnalyzer:
    def __init__(self):
        self.webserver_data = None
        self.sor_hist_data = None
        self.osi_patterns = {}
        self.db_installation_patterns = {}
        self.user_preferences = {}

    def load_data(self):
        """Load and prepare the data for analysis."""
        logger.info("Loading data files...")
        
        CHUNK_SIZE = 100000

        # Load webserver mapping data in chunks
        webserver_chunks = pd.read_csv('data/processed/Webserver_OS_Mapping.csv', chunksize=CHUNK_SIZE)
        webserver_list = []
        for i, chunk in enumerate(webserver_chunks):
            logger.info(f"Processing webserver chunk {i+1}")
            webserver_list.append(chunk)
        self.webserver_data = pd.concat(webserver_list, ignore_index=True)
        # Remove rows with missing or invalid environment
        self.webserver_data = self.webserver_data.dropna(subset=['ENVIRONMENT'])
        self.webserver_data = self.webserver_data[~self.webserver_data['ENVIRONMENT'].isin(['Unknown', 'UNKNOWN', 'Closed'])]
        
        # Remove rows with missing or invalid manufacturer/product info
        self.webserver_data = self.webserver_data.dropna(subset=['MANUFACTURER_x', 'PRODUCTCLASS_x', 'PRODUCTTYPE_x'])
        self.webserver_data = self.webserver_data[~self.webserver_data['MANUFACTURER_x'].isin(['Unknown', 'UNKNOWN', 'Closed'])]
        self.webserver_data = self.webserver_data[~self.webserver_data['PRODUCTCLASS_x'].isin(['Unknown', 'UNKNOWN', 'Closed'])]
        self.webserver_data = self.webserver_data[~self.webserver_data['PRODUCTTYPE_x'].isin(['Unknown', 'UNKNOWN', 'Closed'])]
        
        # Remove rows with missing or invalid model
        self.webserver_data = self.webserver_data.dropna(subset=['MODEL_x'])
        self.webserver_data = self.webserver_data[~self.webserver_data['MODEL_x'].isin(['Unknown', 'UNKNOWN', 'Closed'])]
        
        # Remove rows with "Closed" status
        self.webserver_data = self.webserver_data[~self.webserver_data['STATUS_x'].isin(['Closed'])]

        # Load SOR history data in chunks
        sor_hist_chunks = pd.read_csv('data/processed/sor_hist.csv', chunksize=CHUNK_SIZE)
        sor_hist_list = []
        for i, chunk in enumerate(sor_hist_chunks):
            logger.info(f"Processing SOR history chunk {i+1}")
            sor_hist_list.append(chunk)
        self.sor_hist_data = pd.concat(sor_hist_list, ignore_index=True)
        logger.info(f"Loaded {len(self.sor_hist_data)} SOR history records")
        logger.info(f"SOR history columns: {list(self.sor_hist_data.columns)}")

    def analyze_osi_patterns(self):
        """Analyze OSI patterns based on verumcreatedby column."""
        logger.info("\nAnalyzing OSI patterns from verumcreatedby...")
        
        if 'VERUMCREATEDBY' not in self.sor_hist_data.columns:
            logger.warning("VERUMCREATEDBY column not found in SOR history")
            return
        
        # Group by verumcreatedby to identify OSI patterns
        osi_groups = self.sor_hist_data.groupby('VERUMCREATEDBY')
        
        for osi_user, group in osi_groups:
            if pd.isna(osi_user) or osi_user == '':
                continue
                
            logger.info(f"\nAnalyzing OSI user: {osi_user}")
            
            # Analyze what this OSI user typically installs
            user_patterns = {
                'osi_user': osi_user,
                'total_changes': len(group),
                'common_attributes': group['ATTRIBUTENAME'].value_counts().head(5).to_dict(),
                'common_old_values': group['OLDVALUE'].value_counts().head(5).to_dict(),
                'common_new_values': group['NEWVALUE'].value_counts().head(5).to_dict(),
                'database_versions': {
                    'old_mapped': group['OLD_MAPPED'].dropna().unique().tolist(),
                    'new_mapped': group['NEW_MAPPED'].dropna().unique().tolist()
                },
                'change_frequency': group.groupby('ATTRIBUTENAME').size().to_dict()
            }
            
            self.osi_patterns[osi_user] = user_patterns
            logger.info(f"Found {len(user_patterns['database_versions']['old_mapped'])} old and {len(user_patterns['database_versions']['new_mapped'])} new database versions for {osi_user}")

    def analyze_database_installation_patterns(self):
        """Analyze database installation patterns and history."""
        logger.info("\nAnalyzing database installation patterns...")
        
        if 'OLD_MAPPED' not in self.sor_hist_data.columns or 'NEW_MAPPED' not in self.sor_hist_data.columns:
            logger.warning("OLD_MAPPED or NEW_MAPPED columns not found")
            return
        
        # Filter for database-related changes
        db_changes = self.sor_hist_data[
            (self.sor_hist_data['OLD_MAPPED'].notna()) | 
            (self.sor_hist_data['NEW_MAPPED'].notna())
        ].copy()
        
        if len(db_changes) == 0:
            logger.warning("No database changes found in SOR history")
            return
        
        # Analyze database upgrade patterns
        db_patterns = {
            'total_db_changes': len(db_changes),
            'upgrade_patterns': {},
            'downgrade_patterns': {},
            'common_versions': {},
            'osi_user_preferences': {}
        }
        
        # Analyze upgrade patterns (old -> new)
        for _, row in db_changes.iterrows():
            old_ver = row['OLD_MAPPED']
            new_ver = row['NEW_MAPPED']
            osi_user = row.get('VERUMCREATEDBY', 'Unknown')
            
            if pd.notna(old_ver) and pd.notna(new_ver) and old_ver != new_ver:
                upgrade_key = f"{old_ver} -> {new_ver}"
                if upgrade_key not in db_patterns['upgrade_patterns']:
                    db_patterns['upgrade_patterns'][upgrade_key] = {
                        'count': 0,
                        'osi_users': set(),
                        'dates': []
                    }
                db_patterns['upgrade_patterns'][upgrade_key]['count'] += 1
                db_patterns['upgrade_patterns'][upgrade_key]['osi_users'].add(osi_user)
                db_patterns['upgrade_patterns'][upgrade_key]['dates'].append(row.get('VERUMMODIFIEDDATE', 'Unknown'))
        
        # Convert sets to lists for JSON serialization
        for upgrade_key in db_patterns['upgrade_patterns']:
            db_patterns['upgrade_patterns'][upgrade_key]['osi_users'] = list(
                db_patterns['upgrade_patterns'][upgrade_key]['osi_users']
            )
        
        # Analyze OSI user preferences
        osi_db_prefs = db_changes.groupby('VERUMCREATEDBY').agg({
            'OLD_MAPPED': lambda x: x.dropna().unique().tolist(),
            'NEW_MAPPED': lambda x: x.dropna().unique().tolist()
        }).to_dict('index')
        
        db_patterns['osi_user_preferences'] = osi_db_prefs
        
        # Get most common versions
        all_versions = []
        all_versions.extend(db_changes['OLD_MAPPED'].dropna().tolist())
        all_versions.extend(db_changes['NEW_MAPPED'].dropna().tolist())
        
        version_counts = pd.Series(all_versions).value_counts()
        db_patterns['common_versions'] = version_counts.head(10).to_dict()
        
        self.db_installation_patterns = db_patterns
        logger.info(f"Analyzed {len(db_changes)} database changes")
        logger.info(f"Found {len(db_patterns['upgrade_patterns'])} upgrade patterns")

    def analyze_temporal_bundles(self, time_window_days: int = 1):
        """Analyze temporal installation/upgrade bundles per user and aggregate them across users."""
        logger.info(f"\nAnalyzing temporal bundles with a window of {time_window_days} day(s)...")
        if 'VERUMCREATEDBY' not in self.sor_hist_data.columns or 'VERUMCREATEDDATE' not in self.sor_hist_data.columns:
            logger.warning("VERUMCREATEDBY or VERUMCREATEDDATE column not found in SOR history")
            return
        
        # Convert date column to datetime
        self.sor_hist_data['VERUMCREATEDDATE'] = pd.to_datetime(self.sor_hist_data['VERUMCREATEDDATE'], errors='coerce')
        
        bundle_patterns = defaultdict(int)
        user_bundles = defaultdict(list)
        
        for user, group in self.sor_hist_data.groupby('VERUMCREATEDBY'):
            group = group.sort_values('VERUMCREATEDDATE')
            times = pd.to_datetime(
                group['VERUMCREATEDDATE'],
                format="%d-%b-%y %I.%M.%S.%f %p UTC",
                errors='coerce'
            ).tolist()
            actions = group.to_dict('records')
            n = len(actions)
            i = 0
            while i < n:
                current_time = times[i]
                # Find all actions within the time window
                bundle = [actions[i]]
                j = i + 1
                while j < n and pd.notna(times[j]) and pd.notna(current_time) and (times[j] - current_time).days <= time_window_days:
                    bundle.append(actions[j])
                    j += 1
                # Represent bundle as a tuple of (attribute, old, new) for each action
                bundle_signature = tuple(
                    (a.get('ATTRIBUTENAME'), a.get('OLD_MAPPED'), a.get('NEW_MAPPED')) for a in bundle
                )
                if bundle_signature:
                    bundle_patterns[bundle_signature] += 1
                    user_bundles[user].append(bundle_signature)
                i = j
        self.temporal_bundles = bundle_patterns
        self.user_bundles = user_bundles
        logger.info(f"Found {len(bundle_patterns)} unique temporal bundles across all users.")

    def generate_user_recommendations(self, osi_user: str = None, current_db_version: str = None) -> Dict[str, Any]:
        """Generate recommendations based on OSI user patterns, database history, and temporal bundles."""
        logger.info(f"\nGenerating recommendations for OSI user: {osi_user}, current DB: {current_db_version}")
        
        recommendations = {
            'osi_user': osi_user,
            'current_db_version': current_db_version,
            'recommended_upgrades': [],
            'compatible_versions': [],
            'user_patterns': {},
            'risk_assessment': {},
            'temporal_bundles': []
        }
        
        # Get user-specific patterns
        if osi_user and osi_user in self.osi_patterns:
            user_patterns = self.osi_patterns[osi_user]
            recommendations['user_patterns'] = user_patterns
            
            # Find similar upgrade patterns for this user
            user_db_versions = user_patterns['database_versions']
            if current_db_version in user_db_versions['old_mapped']:
                # User has upgraded from this version before
                recommendations['recommended_upgrades'] = user_db_versions['new_mapped']
        
        # Get general upgrade patterns
        if current_db_version and current_db_version in self.db_installation_patterns.get('upgrade_patterns', {}):
            for upgrade_pattern, details in self.db_installation_patterns['upgrade_patterns'].items():
                if upgrade_pattern.startswith(current_db_version + " ->"):
                    target_version = upgrade_pattern.split(" -> ")[1]
                    recommendations['recommended_upgrades'].append({
                        'target_version': target_version,
                        'frequency': details['count'],
                        'osi_users': details['osi_users']
                    })
        
        # Get compatible versions based on patterns
        if current_db_version:
            compatible_versions = []
            for upgrade_pattern, details in self.db_installation_patterns.get('upgrade_patterns', {}).items():
                if current_db_version in upgrade_pattern:
                    compatible_versions.append({
                        'version': upgrade_pattern.split(" -> ")[1],
                        'success_rate': details['count'],
                        'recommended_by_users': len(details['osi_users'])
                    })
            recommendations['compatible_versions'] = compatible_versions
        
        # Add temporal bundle recommendations
        if osi_user and hasattr(self, 'user_bundles'):
            user_bundles = self.user_bundles.get(osi_user, [])
            # Find the most common bundles for this user
            bundle_counts = defaultdict(int)
            for bundle in user_bundles:
                bundle_counts[bundle] = self.temporal_bundles.get(bundle, 0)
            # Sort bundles by frequency
            sorted_bundles = sorted(bundle_counts.items(), key=lambda x: x[1], reverse=True)
            recommendations['temporal_bundles'] = [
                {'bundle': b, 'count': c} for b, c in sorted_bundles if c > 1
            ]
        
        return recommendations

    def analyze_historical_changes(self):
        """Analyze historical changes from SOR history."""
        logger.info("\nAnalyzing historical changes...")
        
        # Clean column names
        self.sor_hist_data.columns = self.sor_hist_data.columns.str.strip().str.replace('"', '').str.replace(' ', '')
        
        # Group changes by attribute
        changes_by_attribute = self.sor_hist_data.groupby('ATTRIBUTENAME')
        
        for attr, changes in changes_by_attribute:
            logger.info(f"\nChanges in {attr}:")
            logger.info(f"Total changes: {len(changes)}")
            
            # Analyze common changes
            common_changes = changes.groupby(['OLDVALUE', 'NEWVALUE']).size()
            logger.info("Most common changes:")
            logger.info(common_changes.head())

    def identify_compatibility_patterns(self):
        """Identify compatibility patterns from the data."""
        logger.info("\nIdentifying compatibility patterns...")
        
        # Analyze successful combinations
        successful_combinations = self.webserver_data.groupby(
            ['MANUFACTURER_x', 'PRODUCTCLASS_x', 'PRODUCTTYPE_x', 'ENVIRONMENT']
        ).filter(lambda x: x['STATUS_x'].str.contains('Installed').all())
        
        logger.info("\nSuccessful combinations by environment:")
        for env in successful_combinations['ENVIRONMENT'].unique():
            env_data = successful_combinations[successful_combinations['ENVIRONMENT'] == env]
            logger.info(f"\n{env} environment:")
            logger.info(env_data[['MANUFACTURER_x', 'PRODUCTCLASS_x', 'PRODUCTTYPE_x']].value_counts())
            
            # Add summary statistics
            logger.info(f"Total successful combinations in {env}: {len(env_data)}")

    def build_upgrade_pattern_map(self, time_window_days: int = 1):
        """Build a map of upgrade patterns: (old_version, new_version, environment) -> stats, affected models, co-changes, recommendations."""
        logger.info(f"\nBuilding upgrade pattern map with a window of {time_window_days} day(s)...")
        if 'VERUMCREATEDDATE' not in self.sor_hist_data.columns:
            logger.warning("VERUMCREATEDDATE column not found in SOR history")
            return {}
        # Parse dates with explicit format
        self.sor_hist_data['PARSED_DATE'] = pd.to_datetime(
            self.sor_hist_data['VERUMCREATEDDATE'],
            format="%d-%b-%y %I.%M.%S.%f %p UTC",
            errors='coerce'
        )
        pattern_map = {}
        # Group by environment for context
        for env, env_group in self.sor_hist_data.groupby('ENVIRONMENT'):
            # Sort by date
            env_group = env_group.sort_values('PARSED_DATE')
            for idx, row in env_group.iterrows():
                old_ver = row.get('OLD_MAPPED', 'Unknown')
                new_ver = row.get('NEW_MAPPED', 'Unknown')
                model = row.get('MODEL_x', 'Unknown') if 'MODEL_x' in row else row.get('MODEL', 'Unknown')
                parsed_date = row['PARSED_DATE']
                if pd.isna(parsed_date) or old_ver == 'Unknown' or new_ver == 'Unknown':
                    continue
                key = (old_ver, new_ver, env)
                if key not in pattern_map:
                    pattern_map[key] = {
                        'count': 0,
                        'affected_models': set(),
                        'co_changes': [],
                        'dates': [],
                        'recommendations': set()
                    }
                pattern_map[key]['count'] += 1
                pattern_map[key]['affected_models'].add(model)
                pattern_map[key]['dates'].append(parsed_date)
                # Find co-changes within the time window
                window_start = parsed_date - pd.Timedelta(days=time_window_days)
                window_end = parsed_date + pd.Timedelta(days=time_window_days)
                co_changes = env_group[
                    (env_group['PARSED_DATE'] >= window_start) &
                    (env_group['PARSED_DATE'] <= window_end) &
                    (env_group.index != idx)
                ]
                for _, co_row in co_changes.iterrows():
                    co_attr = co_row.get('ATTRIBUTENAME', 'Unknown')
                    co_old = co_row.get('OLD_MAPPED', 'Unknown')
                    co_new = co_row.get('NEW_MAPPED', 'Unknown')
                    pattern_map[key]['co_changes'].append((co_attr, co_old, co_new))
                    # Add to recommendations if it's a different upgrade
                    if co_old != old_ver or co_new != new_ver:
                        pattern_map[key]['recommendations'].add(f"Also upgrade {co_attr} from {co_old} to {co_new}")
        # Convert sets to lists for JSON serialization
        for key in pattern_map:
            pattern_map[key]['affected_models'] = list(pattern_map[key]['affected_models'])
            pattern_map[key]['recommendations'] = list(pattern_map[key]['recommendations'])
        logger.info(f"Built {len(pattern_map)} upgrade patterns.")
        return pattern_map

    def save_analysis(self, filepath='data/processed/compatibility_analysis.json'):
        """Save the analysis results to a JSON file in a RAG-friendly format."""
        def tuple_key_to_str(d):
            """Convert tuple keys to strings for JSON serialization."""
            return {str(k): v for k, v in d.items()}

        def create_server_entry(row):
            """Create a detailed server entry with metadata."""
            return {
                "id": str(row['CATALOGID']),
                "name": row['ASSETNAME'],
                "environment": row['ENVIRONMENT'],
                "objectname": row['OBJECTNAME'] if 'OBJECTNAME' in row and pd.notnull(row['OBJECTNAME']) else None,
                "server_info": {
                    "manufacturer": row['MANUFACTURER_x'],
                    "product_class": row['PRODUCTCLASS_x'],
                    "product_type": row['PRODUCTTYPE_x'],
                    "model": row['MODEL_x'],
                    "status": row['STATUS_x'],
                    "substatus": row['SUBSTATUS']
                },
                "deployment_info": {
                    "instance_name": row['INSTANCENAME']
                },
                "metadata": {
                    "load_date": row['LOAD_DT'],
                    "inventory_number": row['INVNO'],
                    "osi_inventory_number": row['OSIINVNO']
                },
                "old_mapped": row['OLD_MAPPED'] if 'OLD_MAPPED' in row and pd.notnull(row['OLD_MAPPED']) else None,
                "new_mapped": row['NEW_MAPPED'] if 'NEW_MAPPED' in row and pd.notnull(row['NEW_MAPPED']) else None,
                "osi_user": row['VERUMCREATEDBY'] if 'VERUMCREATEDBY' in row and pd.notnull(row['VERUMCREATEDBY']) else None
            }

        # Create detailed server entries
        server_entries = [create_server_entry(row) for _, row in self.webserver_data.iterrows()]

        # Export SOR history as a list of dicts (including OLD_MAPPED and NEW_MAPPED)
        sor_history = self.sor_hist_data.fillna("").to_dict(orient="records")        
        print("SOR History: ")
        for entry in sor_history[:5]:
            print(entry)

        env_summaries = {}
        for env in self.webserver_data['ENVIRONMENT'].unique():
            env_data = self.webserver_data[self.webserver_data['ENVIRONMENT'] == env]
            env_summaries[env] = {
                "total_servers": len(env_data),
                "server_types": tuple_key_to_str(env_data.groupby(['MANUFACTURER_x', 'PRODUCTCLASS_x', 'PRODUCTTYPE_x']).size().to_dict()),
                "status_distribution": env_data['STATUS_x'].value_counts().to_dict()
            }

        # Create manufacturer summaries
        manufacturer_summaries = {}
        for manufacturer in self.webserver_data['MANUFACTURER_x'].unique():
            mfr_data = self.webserver_data[self.webserver_data['MANUFACTURER_x'] == manufacturer]
            manufacturer_summaries[manufacturer] = {
                "total_servers": len(mfr_data),
                "environments": mfr_data['ENVIRONMENT'].value_counts().to_dict(),
                "product_types": tuple_key_to_str(mfr_data.groupby(['PRODUCTCLASS_x', 'PRODUCTTYPE_x']).size().to_dict())
            }

        # Create the final analysis structure with enhanced patterns
        analysis_results = {
            "metadata": {
                "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_servers": len(self.webserver_data),
                "environments": list(self.webserver_data['ENVIRONMENT'].unique()),
                "manufacturers": list(self.webserver_data['MANUFACTURER_x'].unique())
            },
            "servers": server_entries,
            "environment_summaries": env_summaries,
            "manufacturer_summaries": manufacturer_summaries,
            "compatibility_patterns": {
                "by_environment": env_summaries,
                "by_manufacturer": manufacturer_summaries
            },
            "osi_patterns": self.osi_patterns,
            "database_installation_patterns": self.db_installation_patterns,
            "sor_history": sor_history,
            "upgrade_patterns": self.build_upgrade_pattern_map()
        }
        print("Analysis Results: ")
        print(json.dumps(analysis_results, indent=4)[:500])
        
        with open(filepath, 'w') as f:
            json.dump(analysis_results, f, indent=4)
        logger.info(f"Analysis results saved to {filepath}")

        with open(filepath,'r') as f:
            json_content = json.load(f)
            print("First entry in JSON file: ")
            print(json_content['sor_history'][0])

    def enrich_sor_history_with_server_info(self):
        """Enrich each SOR history change with server/environment info from webserver_os_mapped using OLD/NEW values and mappings."""
        logger.info("\nEnriching SOR history with server/environment info from webserver mapping...")
        enriched_changes = []
        # Adjust these field names as needed to match your actual data
        for _, row in self.sor_hist_data.iterrows():
            # Lookup for OLD state
            old_info = self.webserver_data[
                (self.webserver_data['MODEL_x'] == row.get('OLD_VALUE')) &
                (self.webserver_data['MODEL_x'] == row.get('OLD_MAPPED'))
            ] if pd.notna(row.get('OLD_VALUE')) and pd.notna(row.get('OLD_MAPPED')) else pd.DataFrame()
            # Lookup for NEW state
            new_info = self.webserver_data[
                (self.webserver_data['MODEL_x'] == row.get('NEW_VALUE')) &
                (self.webserver_data['MODEL_x'] == row.get('NEW_MAPPED'))
            ] if pd.notna(row.get('NEW_VALUE')) and pd.notna(row.get('NEW_MAPPED')) else pd.DataFrame()
            enriched_changes.append({
                **row,
                'old_server_info': old_info.to_dict('records') if not old_info.empty else None,
                'new_server_info': new_info.to_dict('records') if not new_info.empty else None
            })
        self.enriched_sor_history = enriched_changes
        logger.info(f"Enriched {len(enriched_changes)} SOR history changes with server/environment info.")

def get_db_models_from_sor_history() -> List[Dict[str, Any]]:
    """Get all database versions from SOR history in compatibility_analysis.json."""
    try:
        # Load compatibility analysis JSON
        analysis_path = 'data/processed/compatibility_analysis.json'
        if not os.path.exists(analysis_path):
            logger.warning(f"Compatibility analysis file not found: {analysis_path}")
            return []
        
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Get SOR history from the JSON
        sor_history = analysis_data.get('sor_history', [])
        if not sor_history:
            logger.warning("No SOR history found in compatibility analysis")
            return []
        
        # Filter for DATABASEINSTANCE records
        db_versions = []
        seen_versions = set()
        
        for record in sor_history:
            if record.get('OBJECTNAME') == 'DATABASEINSTANCE':
                old_mapped = record.get('OLD_MAPPED', '').strip()
                new_mapped = record.get('NEW_MAPPED', '').strip()
                
                # Add old version if it exists and not already seen
                if old_mapped:
                    if old_mapped not in seen_versions:
                        seen_versions.add(old_mapped)
                        db_versions.append({
                            'version': old_mapped,
                            'software_name': old_mapped,
                            'change_type': 'old_version'
                        })
                
                # Add new version if it exists and not already seen
                if new_mapped:
                    if new_mapped not in seen_versions:
                        seen_versions.add(new_mapped)
                        db_versions.append({
                            'version': new_mapped,
                            'software_name': new_mapped,
                            'change_type': 'new_version'
                        })
        
        # Sort by software name and version
        db_versions.sort(key=operator.itemgetter('software_name', 'version'))
        
        logger.info(f"Found {len(db_versions)} unique database versions from DATABASEINSTANCE records")
        return db_versions
        
    except Exception as e:
        logger.error(f"Error getting database versions from compatibility analysis: {e}")
        return []

def get_database_options() -> List[str]:
    """Get database options formatted for dropdown."""
    try:
        db_versions = get_db_models_from_sor_history()
        db_options = []
        for v in db_versions:
            if isinstance(v, dict) and 'version' in v:
                db_options.append(v['version'])
            elif isinstance(v, str):
                db_options.append(v)
        
        if not db_options:
            db_options = ["No database versions found"]
        return db_options
    except Exception as e:
        logger.error(f"Error in get_database_options: {e}")
        return ["No database versions found"]

def get_osi_models_from_sor_history() -> List[Dict[str, Any]]:
    """Get all OSI versions from SOR history in compatibility_analysis.json."""
    try:
        # Load compatibility analysis JSON
        analysis_path = 'data/processed/compatibility_analysis.json'
        if not os.path.exists(analysis_path):
            logger.warning(f"Compatibility analysis file not found: {analysis_path}")
            return []
        
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Get SOR history from the JSON
        sor_history = analysis_data.get('sor_history', [])
        if not sor_history:
            logger.warning("No SOR history found in compatibility analysis")
            return []
        
        # Filter for OSI records
        osi_versions = []
        seen_versions = set()
        
        for record in sor_history:
            if record.get('OBJECTNAME') == 'OSI':
                old_mapped = record.get('OLD_MAPPED', '').strip()
                new_mapped = record.get('NEW_MAPPED', '').strip()
                
                # Add old version if it exists and not already seen
                if old_mapped:
                    if old_mapped not in seen_versions:
                        seen_versions.add(old_mapped)
                        osi_versions.append({
                            'version': old_mapped,
                            'software_name': old_mapped,
                            'change_type': 'old_version'
                        })
                
                # Add new version if it exists and not already seen
                if new_mapped:
                    if new_mapped not in seen_versions:
                        seen_versions.add(new_mapped)
                        osi_versions.append({
                            'version': new_mapped,
                            'software_name': new_mapped,
                            'change_type': 'new_version'
                        })
        
        # Sort by software name and version
        osi_versions.sort(key=operator.itemgetter('software_name', 'version'))
        
        logger.info(f"Found {len(osi_versions)} unique OSI versions from OSI records")
        return osi_versions
        
    except Exception as e:
        logger.error(f"Error getting OSI versions from compatibility analysis: {e}")
        return []

def get_osi_options() -> List[str]:
    """Get OSI options formatted for dropdown."""
    try:
        osi_versions = get_osi_models_from_sor_history()
        osi_options = []
        for v in osi_versions:
            if isinstance(v, dict) and 'version' in v:
                osi_options.append(v['version'])
            elif isinstance(v, str):
                osi_options.append(v)
        
        if not osi_options:
            osi_options = ["No OSI versions found"]
        return osi_options
    except Exception as e:
        logger.error(f"Error in get_osi_options: {e}")
        return ["No OSI versions found"]

def main():
    """Main function to run the compatibility analysis."""
    analyzer = CompatibilityAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Run analyses
    analyzer.analyze_osi_patterns()
    analyzer.analyze_database_installation_patterns()
    analyzer.analyze_temporal_bundles()
    analyzer.analyze_historical_changes()
    analyzer.identify_compatibility_patterns()
    
    # Save results
    analyzer.save_analysis()
    
    # Example: Generate recommendations for a specific user
    example_recommendations = analyzer.generate_user_recommendations(
        osi_user="example_user", 
        current_db_version="12.1.0.2.0"
    )
    print("\nExample recommendations:")
    print(json.dumps(example_recommendations, indent=2))
    
    logger.info("Compatibility analysis completed successfully!")

if __name__ == "__main__":
    main() 