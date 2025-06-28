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

        # Export SOR history as a list of dicts (including OBJECTNAME)
        sor_history = self.sor_hist_data.fillna("").to_dict(orient="records")

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
            },
            "sor_history": sor_history,
            "database_models_from_sor_history": self.get_database_models_from_sor_history()
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_results, f, indent=4)
        logger.info(f"Analysis results saved to {filepath}")

    def get_database_models(self):
        """Return sorted list of unique models where OBJECTNAME == 'DATABASEINSTANCE'."""
        if self.webserver_data is not None and 'OBJECTNAME' in self.webserver_data.columns and 'MODEL_x' in self.webserver_data.columns:
            db_models = self.webserver_data[self.webserver_data['OBJECTNAME'] == 'DATABASEINSTANCE']['MODEL_x'].dropna().unique()
            return sorted(db_models)
        return []

    def build_catalogid_to_model_mapping_from_pcat(self):
        """Build a mapping from CATALOGID to model+version from PCat.csv."""
        if self.pcat_data is not None and 'CATALOGID' in self.pcat_data.columns:
            # Assuming PCat has MODEL and VERSION columns
            self.catalogid_to_pcat_model = {}
            for _, row in self.pcat_data.iterrows():
                cat_id = row['CATALOGID']
                model = row.get('MODEL', '')
                version = row.get('VERSION', '')
                # Form the model string (e.g., "APACHE HTTPD 2.4")
                model_str = f"{model} {version}".strip()
                if model_str:
                    self.catalogid_to_pcat_model[cat_id] = model_str
        else:
            self.catalogid_to_pcat_model = {}

    def get_database_models_from_sor_history(self):
        """Return sorted list of unique model names for DATABASEINSTANCE changes."""
        models = set()
        if self.sor_hist_data is not None and 'OBJECTNAME' in self.sor_hist_data.columns:
            db_rows = self.sor_hist_data[self.sor_hist_data['OBJECTNAME'] == 'DATABASEINSTANCE']

            # 1. Use OLD_MAPPED/NEW_MAPPED for MODEL changes
            model_rows = db_rows[db_rows['ATTRIBUTENAME'] == 'MODEL']
            if 'OLD_MAPPED' in model_rows.columns:
                models.update(model_rows['OLD_MAPPED'].dropna().unique())
            if 'NEW_MAPPED' in model_rows.columns:
                models.update(model_rows['NEW_MAPPED'].dropna().unique())

            # 2. For CATALOGID changes, get model from column next to CATALOGID in PCat
            catalogid_rows = db_rows[db_rows['ATTRIBUTENAME'] == 'CATALOGID']
            if self.pcat_data is not None and 'CATALOGID' in self.pcat_data.columns:
                columns = list(self.pcat_data.columns)
                cat_idx = columns.index('CATALOGID')
                # Defensive: if MODEL is not the next column, fallback to 'MODEL'
                model_col = columns[cat_idx + 1] if cat_idx + 1 < len(columns) else 'MODEL'
                for col in ['OLDVALUE', 'NEWVALUE']:
                    if col in catalogid_rows.columns:
                        for cid in catalogid_rows[col].dropna().unique():
                            match = self.pcat_data[self.pcat_data['CATALOGID'].astype(str) == str(cid)]
                            if not match.empty:
                                model = match.iloc[0][model_col] if model_col in match.columns else match.iloc[0].get('MODEL', None)
                                if pd.notnull(model) and str(model).strip():
                                    models.add(str(model).strip())
        return sorted(m for m in models if m)

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