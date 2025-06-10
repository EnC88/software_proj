import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _clean_text(text):
    """Clean text by removing special characters and converting to uppercase."""
    if pd.isna(text):
        return text
    return str(text).strip().upper()

def _validate_data(df, required_columns):
    """Validate that required columns exist in the dataframe."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

def process_webserver_data():
    """Process web server data and map software installations."""
    start_time = datetime.now()
    logger.info("Starting web server data processing...")
    
    # Read input files
    webserver_df = pd.read_csv('data/raw/WebServer.csv')
    mapping_df = pd.read_csv('data/raw/swecomponentmapping.csv')
    
    # Clean column names
    webserver_df.columns = webserver_df.columns.str.strip()
    mapping_df.columns = mapping_df.columns.str.strip().str.replace('"', '')
    
    # Log the number of records read
    logger.info(f"Read {len(webserver_df)} records from data/raw/WebServer.csv")
    logger.info(f"Read {len(mapping_df)} records from data/raw/swecomponentmapping.csv")
    
    # Log the column names for debugging
    logger.info("\nWebServer columns:")
    logger.info(webserver_df.columns.tolist())
    logger.info("\nMapping columns:")
    logger.info(mapping_df.columns.tolist())
    
    # Validate required columns
    required_columns = [
        'ASSETNAME', 'ENVIRONMENT', 'INSTALLPATH', 'INSTANCENAME',
        'MANUFACTURER', 'MODEL', 'PRODUCTCLASS', 'OSIINVNO', 'VERUMIDENTIFIER'
    ]
    _validate_data(webserver_df, required_columns)
    
    # Clean text fields
    text_columns = ['ASSETNAME', 'ENVIRONMENT', 'INSTALLPATH', 'INSTANCENAME',
                   'MANUFACTURER', 'MODEL', 'PRODUCTCLASS']
    for col in text_columns:
        webserver_df[col] = webserver_df[col].apply(_clean_text)
    
    # Create mapping dictionaries for additional software information
    catalog_mapping = dict(zip(mapping_df['CATALOGID'], mapping_df['SWNAME']))
    invno_mapping = dict(zip(mapping_df['INVNO'], mapping_df['SWNAME']))
    
    # Add additional software information if available
    webserver_df['ADDITIONAL_SOFTWARE'] = webserver_df.apply(
        lambda row: list(filter(None, set([
            catalog_mapping.get(row['CATALOGID']),
            invno_mapping.get(row['INVNO']),
            invno_mapping.get(row['OSIINVNO'])
        ]))),
        axis=1
    )
    
    # Select and reorder columns for output
    output_columns = [
        'ASSETNAME', 'ENVIRONMENT', 'INSTALLPATH', 'INSTANCENAME',
        'MANUFACTURER', 'MODEL', 'PRODUCTCLASS', 'OSIINVNO', 'VERUMIDENTIFIER',
        'ADDITIONAL_SOFTWARE'
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed data
    output_file = 'data/processed/WebServer_Merged.csv'
    webserver_df[output_columns].to_csv(output_file, index=False)
    
    # Calculate processing time
    processing_time = datetime.now() - start_time
    logger.info(f"Saved processed data to {output_file}")
    logger.info(f"Data processing completed in {processing_time.total_seconds():.2f} seconds")
    
    # Print summary statistics
    logger.info("\nData Summary:")
    logger.info(f"Total Records: {len(webserver_df)}")
    
    # Count environments
    env_counts = webserver_df['ENVIRONMENT'].value_counts()
    logger.info("\nEnvironment Distribution:")
    for env, count in env_counts.items():
        logger.info(f"{env}: {count} servers")
    
    # Count manufacturers and their models
    logger.info("\nManufacturer and Model Distribution:")
    for mfr in webserver_df['MANUFACTURER'].unique():
        mfr_servers = webserver_df[webserver_df['MANUFACTURER'] == mfr]
        logger.info(f"\n{mfr}:")
        for model, count in mfr_servers['MODEL'].value_counts().items():
            logger.info(f"  {model}: {count} servers")
    
    # Count product classes
    class_counts = webserver_df['PRODUCTCLASS'].value_counts()
    logger.info("\nProduct Class Distribution:")
    for cls, count in class_counts.items():
        logger.info(f"{cls}: {count} servers")
    
    return {
        'total_records': len(webserver_df),
        'environment_distribution': env_counts.to_dict(),
        'manufacturer_distribution': webserver_df.groupby('MANUFACTURER')['MODEL'].value_counts().to_dict(),
        'product_class_distribution': class_counts.to_dict()
    }

if __name__ == '__main__':
    process_webserver_data() 