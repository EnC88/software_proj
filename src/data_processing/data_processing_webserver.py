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
    
    # Read mapping file (usually smaller, can be read at once)
    mapping_df = pd.read_csv('data/raw/swecomponentmapping.csv')
    mapping_df.columns = mapping_df.columns.str.strip().str.replace('"', '')
    
    # Validate mapping file columns
    mapping_required = ['CATALOGID', 'SWNAME', 'INVNO']
    _validate_data(mapping_df, mapping_required)
    
    # Create mapping dictionary
    mapping_dict = dict(zip(mapping_df['CATALOGID'].astype(str), mapping_df['SWNAME'].apply(_clean_text)))
    matching_catalog_ids = set(mapping_df['CATALOGID'].astype(str))
    
    # Initialize counters and lists for chunk processing
    total_catalog_records = 0
    matched_records = []
    chunk_size = 10000  # Adjust this based on your memory constraints
    
    # Process catalog file in chunks
    logger.info(f"Processing catalog file in chunks of {chunk_size} records...")
    
    # First, get the column names from the first chunk
    first_chunk = pd.read_csv('data/raw/softwarecatalog.csv', nrows=1)
    catalog_required = ['CATALOGID', 'SOFTWARE', 'MANUFACTURER', 'LCCURRENTENDDATE', 
                       'LCCURRENTSTARTDATE', 'EDITION', 'SWNMAME', 'STATUS']
    _validate_data(first_chunk, catalog_required)
    
    # Process the file in chunks
    for chunk in pd.read_csv('data/raw/softwarecatalog.csv', chunksize=chunk_size):
        # Clean column names
        chunk.columns = chunk.columns.str.strip()
        
        # Clean text fields
        text_columns = ['SOFTWARE', 'MANUFACTURER', 'EDITION', 'SWNMAME', 'STATUS', 
                       'PRODUCTFAMILY', 'COMPONENT', 'PRODUCTCLASS', 'PRODUCTTYPE']
        for col in text_columns:
            if col in chunk.columns:
                chunk[col] = chunk[col].apply(_clean_text)
        
        # Convert CATALOGID to string
        chunk['CATALOGID'] = chunk['CATALOGID'].astype(str)
        
        # Parse dates
        chunk['LCCURRENTENDDATE'] = pd.to_datetime(chunk['LCCURRENTENDDATE'], errors='coerce')
        chunk['LCCURRENTSTARTDATE'] = pd.to_datetime(chunk['LCCURRENTSTARTDATE'], errors='coerce')
        
        # Calculate days to expiry
        today = pd.Timestamp(datetime.today().date())
        chunk['DAYS_TO_EXPIRY'] = (chunk['LCCURRENTENDDATE'] - today).dt.days
        chunk['LICENSE_STATUS'] = chunk['DAYS_TO_EXPIRY'].apply(
            lambda x: 'Expired' if pd.isna(x) or x < 0 else ('Expiring Soon' if x < 30 else 'Valid')
        )
        
        # Find matching records
        matched_chunk = chunk[chunk['CATALOGID'].isin(matching_catalog_ids)].copy()
        
        # Add mapping information
        matched_chunk['MAPPED_SOFTWARE'] = matched_chunk['CATALOGID'].map(mapping_dict)
        
        # Select output columns
        output_columns = [
            'CATALOGID', 'SOFTWARE', 'SWNMAME', 'EDITION', 'MANUFACTURER',
            'PRODUCTFAMILY', 'COMPONENT', 'PRODUCTCLASS', 'PRODUCTTYPE',
            'MAJORVERSION', 'MINORVERSION', 'VERSIONSERVICEPACK',
            'LCCURRENTSTARTDATE', 'LCCURRENTENDDATE', 'DAYS_TO_EXPIRY',
            'LICENSE_STATUS', 'STATUS', 'MAPPED_SOFTWARE'
        ]
        matched_chunk = matched_chunk[output_columns]
        
        # Append matched records
        matched_records.append(matched_chunk)
        total_catalog_records += len(chunk)
        
        # Log progress
        logger.info(f"Processed {len(chunk)} records, found {len(matched_chunk)} matches")
    
    # Combine all matched records
    matched_df = pd.concat(matched_records, ignore_index=True)
    
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
    logger.info(f"Total Records in Catalog: {total_catalog_records}")
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
        'total_catalog_records': total_catalog_records,
        'total_mapping_records': len(mapping_df),
        'matched_records': len(matched_df),
        'license_status_distribution': status_counts.to_dict() if 'LICENSE_STATUS' in matched_df.columns else {},
        'product_family_distribution': family_counts.to_dict() if 'PRODUCTFAMILY' in matched_df.columns else {}
    }

if __name__ == '__main__':
    process_software_catalog() 