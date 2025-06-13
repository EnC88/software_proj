import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def group_swmap_data():
    """Group PCat mapping data by CATALOGID and aggregate relevant fields."""
    logger.info("Reading pcat_mapping.csv...")
    df = pd.read_csv('data/processed/pcat_mapping.csv')
    
    # Group by CATALOGID and aggregate fields
    grouped = df.groupby('CATALOGID').agg({
        'MODEL': lambda x: '|'.join(x.unique()),
        'PRODUCTTYPE': lambda x: '|'.join(x.unique()),
        'PRODUCTCLASS': lambda x: '|'.join(x.unique()),
        'MANUFACTURER': lambda x: '|'.join(x.unique()),
        'STATUS': lambda x: '|'.join(x.unique()),
        'LIFECYCLESTATUS': lambda x: '|'.join(x.unique()),
        'RELEASEVERSION': lambda x: '|'.join(x.unique()),
        'PRODUCTCATEGORY': lambda x: '|'.join(x.unique()),
        'ARCHITECTURE': lambda x: '|'.join(x.unique()),
        'CREATED_DATE': 'min',
        'MODIFIED_DATE': 'max'
    }).reset_index()
    
    # Save grouped data
    output_file = 'data/processed/pcat_mapping_grouped.csv'
    logger.info(f"Saving grouped data to {output_file}...")
    grouped.to_csv(output_file, index=False)
    
    logger.info(f"Grouped {len(df)} records into {len(grouped)} unique CATALOGIDs")
    return grouped

if __name__ == '__main__':
    group_swmap_data() 