import pandas as pd
import logging
from collections import defaultdict
import json
import os
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load SOR history data globally
sor_hist_data = None
sor_hist_path = 'data/processed/sor_pcat_mapped.csv'  # Use the newly processed data with mappings
if os.path.exists(sor_hist_path):
    sor_hist_data = pd.read_csv(sor_hist_path)

class CompatibilityAnalyzer:
    def __init__(self):
        self.webserver_data = None
        self.sor_hist_data = None
        self.pcat_data = None
        self.compatibility_patterns = defaultdict(lambda: defaultdict(list))

    def load_data(self):
        """Load and prepare the data for analysis."""
        logger.info("Loading data files...")
        
        # Load the newly processed data with PRODUCTTYPE mappings
        self.webserver_data = pd.read_csv('data/processed/sor_pcat_mapped.csv')
        
        # Filter out rows with missing, "Unknown", or "Closed" values
        initial_count = len(self.webserver_data)
        
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
        
        final_count = len(self.webserver_data)
        filtered_count = initial_count - final_count
        
        logger.info(f"Filtered out {filtered_count} rows with missing, 'Unknown', or 'Closed' values")
        logger.info(f"Loaded {final_count} valid webserver records")
        logger.info(f"Environment distribution:\n{self.webserver_data['ENVIRONMENT'].value_counts()}")
        
        # Load SOR history data
        self.sor_hist_data = pd.read_csv('data/processed/Change_History.csv')
        logger.info(f"Loaded {len(self.sor_hist_data)} SOR history records")
        
        # Load PCat data
        self.pcat_data = pd.read_csv('data/raw/PCat.csv')
        logger.info(f"Loaded {len(self.pcat_data)} PCat records")

    def analyze_current_deployments(self):
        """Analyze current deployment patterns."""
        logger.info("\nAnalyzing current deployment patterns...")
        
        # Group by environment and analyze patterns
        for env in self.webserver_data['ENVIRONMENT'].unique():
            env_data = self.webserver_data[self.webserver_data['ENVIRONMENT'] == env]
            
            # Analyze web server patterns
            web_server_patterns = env_data.groupby(['MANUFACTURER_x', 'PRODUCTCLASS_x', 'PRODUCTTYPE_x']).size()
            logger.info(f"\nWeb server patterns in {env} environment:")
            logger.info(web_server_patterns)
            
            # Log the count of servers in this environment
            logger.info(f"Total servers in {env}: {len(env_data)}")

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
                "server_info": {
                    "manufacturer": row['MANUFACTURER_x'],
                    "product_class": row['PRODUCTCLASS_x'],
                    "product_type": row['PRODUCTTYPE_x'],
                    "model": row['MODEL_x'],
                    "status": row['STATUS_x'],
                    "substatus": row['SUBSTATUS']
                },
                "deployment_info": {
                    "install_path": row['INSTALLPATH'],
                    "instance_name": row['INSTANCENAME']
                },
                "metadata": {
                    "load_date": row['LOAD_DT'],
                    "inventory_number": row['INVNO'],
                    "osi_inventory_number": row['OSIINVNO']
                }
            }

        # Create detailed server entries
        server_entries = [create_server_entry(row) for _, row in self.webserver_data.iterrows()]
        
        # Create environment summaries
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

        # Create the final analysis structure
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
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_results, f, indent=4)
        logger.info(f"Analysis results saved to {filepath}")

def load_platform_catalog() -> pd.DataFrame:
    catalog_path = 'data/raw/Platform_Catalog.csv'
    if not os.path.exists(catalog_path):
        logger.error(f"Platform catalog file not found at {catalog_path}")
        return pd.DataFrame()
    platform_df = pd.read_csv(catalog_path)
    platform_df.columns = platform_df.columns.str.strip().str.replace('"', '')
    platform_df.columns['CATALOGID'] = platform_df.columns['CATALOGID'].astype(str).str.strip()
    return platform_df

def get_osi_models() -> List[Dict[str, Any]]:
    try: 
        platform_df = load_platform_catalog()
        if platform_df.empty:
            logger.error("No platform catalog data found")
            return []
        os_data = platform_df[(platform_df['PRODUCTTYPE'] == 'OPERATING SYSTEM') & (platform_df['EBFIRMWIDERATING'] != 'Prohibited')]
        os_data = os_data[~os_data['MODEL'].str.contains('Unknown', case = False, na = False)]
        os_models = os_data[['MODEL', 'MANUFACTURER']].drop_duplicates()
        logger.info(f"Found {len(os_models)} OS models")
        return os_models
    except Exception as e:
        logger.error(f"Error getting OS models: {str(e)}")
        return []
    
def get_db_options() -> list:
    try:
        if sor_hist_data is None:
            return ["No database found"]
        # Old database options with catalogid
        old_db = sor_hist_data[sor_hist_data['OLDPRODUCTTYPE'] == 'DATABASE'][['OLDVALUE', 'OLD_MAPPED']].dropna()
        old_db_options = [f"{row['OLDVALUE']} - {row['OLD_MAPPED']}" for _, row in old_db.iterrows()]
        
        # New database options with catalogid
        new_db = sor_hist_data[sor_hist_data['NEWPRODUCTTYPE'] == 'DATABASE'][['NEWVALUE', 'NEW_MAPPED']].dropna()
        new_db_options = [f"{row['NEWVALUE']} - {row['NEW_MAPPED']}" for _, row in new_db.iterrows()]
        
        # Combine and deduplicate
        db_options = sorted(set(old_db_options + new_db_options))
        if not db_options:
            db_options = ['No database found']
        db_options.insert(0, 'None')
        return db_options
    except Exception as e:
        logger.error(f"Error in get_db_options: {e}")
        return ["No database found"]


def get_web_server_options() -> list:
    try:
        if sor_hist_data is None:
            return ["No web server found"]
        # Old web server options with catalogid
        old_ws = sor_hist_data[sor_hist_data['OLDPRODUCTTYPE'] == 'WEB SERVER'][['OLDVALUE', 'OLD_MAPPED']].dropna()
        old_ws_options = [f"{row['OLDVALUE']} - {row['OLD_MAPPED']}" for _, row in old_ws.iterrows()]
        
        # New web server options with catalogid
        new_ws = sor_hist_data[sor_hist_data['NEWPRODUCTTYPE'] == 'WEB SERVER'][['NEWVALUE', 'NEW_MAPPED']].dropna()
        new_ws_options = [f"{row['NEWVALUE']} - {row['NEW_MAPPED']}" for _, row in new_ws.iterrows()]
        
        # Combine and deduplicate
        ws_options = sorted(set(old_ws_options + new_ws_options))
        if not ws_options:
            ws_options = ['No web server found']
        ws_options.insert(0, 'None')
        return ws_options
    except Exception as e:
        logger.error(f"Error in get_web_server_options: {e}")
        return ["No web server found"]


def get_osi_options() -> list:
    try:
        if sor_hist_data is None:
            return ["No operating system found"]
        # Old OS options with catalogid
        old_os = sor_hist_data[sor_hist_data['OLDPRODUCTTYPE'] == 'OPERATING SYSTEM'][['OLDVALUE', 'OLD_MAPPED']].dropna()
        old_os_options = [f"{row['OLDVALUE']} - {row['OLD_MAPPED']}" for _, row in old_os.iterrows()]
        
        # New OS options with catalogid
        new_os = sor_hist_data[sor_hist_data['NEWPRODUCTTYPE'] == 'OPERATING SYSTEM'][['NEWVALUE', 'NEW_MAPPED']].dropna()
        new_os_options = [f"{row['NEWVALUE']} - {row['NEW_MAPPED']}" for _, row in new_os.iterrows()]
        
        # Combine and deduplicate
        os_options = sorted(set(old_os_options + new_os_options))
        if not os_options:
            os_options = ['No operating system found']
        os_options.insert(0, 'None')
        return os_options
    except Exception as e:
        logger.error(f"Error in get_osi_options: {e}")
        return ["No operating system found"]
    
def main():
    analyzer = CompatibilityAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Perform analysis
    analyzer.analyze_current_deployments()
    analyzer.analyze_historical_changes()
    analyzer.identify_compatibility_patterns()
    
    # Save results
    analyzer.save_analysis()

if __name__ == "__main__":
    main() 