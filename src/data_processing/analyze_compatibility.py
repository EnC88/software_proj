import pandas as pd
import logging
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompatibilityAnalyzer:
    def __init__(self):
        self.webserver_data = None
        self.sor_hist_data = None
        self.pcat_data = None
        self.compatibility_patterns = defaultdict(lambda: defaultdict(list))

    def load_data(self):
        """Load and prepare the data for analysis."""
        logger.info("Loading data files...")
        
        # Load webserver mapping data
        self.webserver_data = pd.read_csv('data/processed/Webserver_OS_Mapping.csv')
        
        # Clean environment data
        self.webserver_data['ENVIRONMENT'] = self.webserver_data['ENVIRONMENT'].fillna('UNKNOWN')
        logger.info(f"Loaded {len(self.webserver_data)} webserver records")
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