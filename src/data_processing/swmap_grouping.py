import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def group_by_catalogid():
    logger.info("Reading swecomponentmapping.csv...")
    df = pd.read_csv('data/raw/swecomponentmapping.csv', encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace('"', '')
    logger.info(f"Read {len(df)} records from swecomponentmapping.csv.")
    
    # Group by CATALOGID and aggregate SWNAME and INVNO
    grouped = df.groupby('CATALOGID').agg({
        'SWNAME': lambda x: list(x),
        'INVNO': lambda x: list(x)
    }).reset_index()
    
    logger.info(f"Grouped into {len(grouped)} unique CATALOGIDs.")
    
    # Save the grouped data
    output_file = 'data/processed/swecomponentmapping_grouped.csv'
    grouped.to_csv(output_file, index=False)
    logger.info(f"Grouped data saved to {output_file}.")

if __name__ == '__main__':
    group_by_catalogid() 