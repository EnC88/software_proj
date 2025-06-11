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

def process_software_catalog():
    """Process software catalog data and match with component mapping."""
    start_time = datetime.now()
    logger.info("Starting software catalog processing...")
    
    # Read input files
    catalog_df = pd.read_csv('data/raw/softwarecatalog.csv')
    mapping_df = pd.read_csv('data/raw/swecomponentmapping.csv')
    
    # Clean column names
    catalog_df.columns = catalog_df.columns.str.strip()
    mapping_df.columns = mapping_df.columns.str.strip().str.replace('"', '')
    
    # Log the number of records read
    logger.info(f"Read {len(catalog_df)} records from data/raw/softwarecatalog.csv")
    logger.info(f"Read {len(mapping_df)} records from data/raw/swecomponentmapping.csv")
    
    # Validate required columns
    catalog_required = ['CATALOGID', 'SOFTWARE', 'MANUFACTURER', 'LCCURRENTENDDATE', 
                       'LCCURRENTSTARTDATE', 'EDITION', 'SWNMAME', 'STATUS']
    mapping_required = ['CATALOGID', 'SWNAME', 'INVNO']
    _validate_data(catalog_df, catalog_required)
    _validate_data(mapping_df, mapping_required)
    
    # Clean text fields in catalog
    text_columns = ['SOFTWARE', 'MANUFACTURER', 'EDITION', 'SWNMAME', 'STATUS', 
                   'PRODUCTFAMILY', 'COMPONENT', 'PRODUCTCLASS', 'PRODUCTTYPE']
    for col in text_columns:
        if col in catalog_df.columns:
            catalog_df[col] = catalog_df[col].apply(_clean_text)
    
    # Clean text fields in mapping
    mapping_df['SWNAME'] = mapping_df['SWNAME'].apply(_clean_text)
    
    # Convert CATALOGID to string in both dataframes for consistent matching
    catalog_df['CATALOGID'] = catalog_df['CATALOGID'].astype(str)
    mapping_df['CATALOGID'] = mapping_df['CATALOGID'].astype(str)
    
    # Parse and standardize date columns
    catalog_df['LCCURRENTENDDATE'] = pd.to_datetime(catalog_df['LCCURRENTENDDATE'], errors='coerce')
    catalog_df['LCCURRENTSTARTDATE'] = pd.to_datetime(catalog_df['LCCURRENTSTARTDATE'], errors='coerce')
    today = pd.Timestamp(datetime.today().date())
    catalog_df['DAYS_TO_EXPIRY'] = (catalog_df['LCCURRENTENDDATE'] - today).dt.days
    catalog_df['LICENSE_STATUS'] = catalog_df['DAYS_TO_EXPIRY'].apply(
        lambda x: 'Expired' if pd.isna(x) or x < 0 else ('Expiring Soon' if x < 30 else 'Valid')
    )
    
    # Find matching records
    matching_catalog_ids = set(mapping_df['CATALOGID'])
    matched_df = catalog_df[catalog_df['CATALOGID'].isin(matching_catalog_ids)].copy()
    
    # Add mapping information
    mapping_dict = dict(zip(mapping_df['CATALOGID'], mapping_df['SWNAME']))
    matched_df['MAPPED_SOFTWARE'] = matched_df['CATALOGID'].map(mapping_dict)
    
    # Select and reorder columns for output
    output_columns = [
        'CATALOGID', 'SOFTWARE', 'SWNMAME', 'EDITION', 'MANUFACTURER',
        'PRODUCTFAMILY', 'COMPONENT', 'PRODUCTCLASS', 'PRODUCTTYPE',
        'MAJORVERSION', 'MINORVERSION', 'VERSIONSERVICEPACK',
        'LCCURRENTSTARTDATE', 'LCCURRENTENDDATE', 'DAYS_TO_EXPIRY',
        'LICENSE_STATUS', 'STATUS', 'MAPPED_SOFTWARE'
    ]
    matched_df = matched_df[output_columns]
    
    # Create output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed data
    output_file = 'data/processed/Software_Catalog_Matched.csv'
    matched_df.to_csv(output_file, index=False)
    
    # Calculate processing time
    processing_time = datetime.now() - start_time
    logger.info(f"Saved processed data to {output_file}")
    logger.info(f"Data processing completed in {processing_time.total_seconds():.2f} seconds")
    
    # Print summary statistics
    logger.info("\nData Summary:")
    logger.info(f"Total Records in Catalog: {len(catalog_df)}")
    logger.info(f"Total Records in Mapping: {len(mapping_df)}")
    logger.info(f"Matched Records: {len(matched_df)}")
    
    # Count matches by license status
    if 'LICENSE_STATUS' in matched_df.columns:
        status_counts = matched_df['LICENSE_STATUS'].value_counts()
        logger.info("\nMatched Records by License Status:")
        for status, count in status_counts.items():
            logger.info(f"{status}: {count} records")
    
    # Count matches by product family
    if 'PRODUCTFAMILY' in matched_df.columns:
        family_counts = matched_df['PRODUCTFAMILY'].value_counts()
        logger.info("\nMatched Records by Product Family:")
        for family, count in family_counts.items():
            logger.info(f"{family}: {count} records")
    
    return {
        'total_catalog_records': len(catalog_df),
        'total_mapping_records': len(mapping_df),
        'matched_records': len(matched_df),
        'license_status_distribution': status_counts.to_dict() if 'LICENSE_STATUS' in matched_df.columns else {},
        'product_family_distribution': family_counts.to_dict() if 'PRODUCTFAMILY' in matched_df.columns else {}
    }

if __name__ == '__main__':
    process_software_catalog() 