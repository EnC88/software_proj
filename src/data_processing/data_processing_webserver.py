import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import sys
import psutil
import gc
import warnings

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

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def robust_read_csv(filepath, **kwargs):
    """Try reading a CSV with utf-8, then fallback to latin1 if needed."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8', **kwargs)
        logger.info(f"Read {filepath} with utf-8 encoding.")
        return df
    except UnicodeDecodeError as e:
        logger.warning(f"utf-8 decode error for {filepath}: {e}. Trying latin1 encoding...")
        df = pd.read_csv(filepath, encoding='latin1', **kwargs)
        logger.info(f"Read {filepath} with latin1 encoding.")
        return df

def process_software_catalog():
    """Process software catalog data and match with component mapping."""
    start_time = datetime.now()
    logger.info("Starting software catalog processing...")
    log_memory_usage()
    
    try:
        # Read mapping file (usually smaller, can be read at once)
        logger.info("Reading mapping file...")
        mapping_df = robust_read_csv('data/raw/swecomponentmapping.csv', dtype={'CATALOGID': str})
        
        # Clean column names (remove quotes and spaces)
        mapping_df.columns = mapping_df.columns.str.strip().str.replace('"', '')
        
        # Validate mapping file columns
        mapping_required = ['CATALOGID', 'SWNAME', 'INVNO']
        _validate_data(mapping_df, mapping_required)
        
        # Create mapping dictionary
        mapping_dict = dict(zip(mapping_df['CATALOGID'], mapping_df['SWNAME'].apply(_clean_text)))
        matching_catalog_ids = set(mapping_df['CATALOGID'])
        
        # Clear mapping dataframe from memory
        del mapping_df
        gc.collect()
        log_memory_usage()
        
        # Initialize counters and lists for chunk processing
        total_catalog_records = 0
        matched_records = []
        chunk_size = 100000  # Increased chunk size for 2M+ records
        
        # Define columns to read from catalog
        catalog_columns = [
            'CATALOGID', 'SOFTWARE', 'MANUFACTURER', 'LCCURRENTENDDATE',
            'LCCURRENTSTARTDATE', 'EDITION', 'SWNMAME', 'STATUS',
            'PRODUCTFAMILY', 'COMPONENT', 'PRODUCTCLASS', 'PRODUCTTYPE',
            'MAJORVERSION', 'MINORVERSION', 'VERSIONSERVICEPACK'
        ]
        
        # Define data types for catalog columns
        dtype_dict = {
            'CATALOGID': str,
            'SOFTWARE': str,
            'MANUFACTURER': str,
            'EDITION': str,
            'SWNMAME': str,
            'STATUS': str,
            'PRODUCTFAMILY': str,
            'COMPONENT': str,
            'PRODUCTCLASS': str,
            'PRODUCTTYPE': str,
            'MAJORVERSION': str,
            'MINORVERSION': str,
            'VERSIONSERVICEPACK': str
        }
        
        # Process catalog file in chunks
        logger.info(f"Processing catalog file in chunks of {chunk_size} records...")
        
        # First, get the column names from the first chunk
        try:
            first_chunk = robust_read_csv('data/raw/softwarecatalog.csv', nrows=1, on_bad_lines='warn')
            # Clean column names
            first_chunk.columns = first_chunk.columns.str.strip().str.replace('"', '')
            
            catalog_required = ['CATALOGID', 'SOFTWARE', 'MANUFACTURER', 'LCCURRENTENDDATE', 
                              'LCCURRENTSTARTDATE', 'EDITION', 'SWNMAME', 'STATUS']
            _validate_data(first_chunk, catalog_required)
        except Exception as e:
            logger.error(f"Error reading first chunk: {str(e)}")
            raise
        
        # Process the file in chunks with error handling
        try:
            chunk_iterator = robust_read_csv('data/raw/softwarecatalog.csv',
                                       chunksize=chunk_size,
                                       on_bad_lines='warn',
                                       dtype=dtype_dict)
            
            for chunk_num, chunk in enumerate(chunk_iterator, 1):
                try:
                    chunk.columns = chunk.columns.str.strip().str.replace('"', '')
                    # Convert all object/mixed type columns to string to avoid mixed type warnings
                    for col in chunk.select_dtypes(include=['object']).columns:
                        chunk[col] = chunk[col].astype(str)
                    text_columns = ['SOFTWARE', 'MANUFACTURER', 'EDITION', 'SWNMAME', 'STATUS', 
                                  'PRODUCTFAMILY', 'COMPONENT', 'PRODUCTCLASS', 'PRODUCTTYPE']
                    for col in text_columns:
                        if col in chunk.columns:
                            chunk[col] = chunk[col].apply(_clean_text)
                    # Explicitly parse date columns with a specific format
                    chunk['LCCURRENTENDDATE'] = pd.to_datetime(chunk['LCCURRENTENDDATE'], errors='coerce')
                    chunk['LCCURRENTSTARTDATE'] = pd.to_datetime(chunk['LCCURRENTSTARTDATE'], errors='coerce')
                    today = pd.Timestamp(datetime.today().date())
                    chunk['DAYS_TO_EXPIRY'] = (chunk['LCCURRENTENDDATE'] - today).dt.days
                    chunk['LICENSE_STATUS'] = chunk['DAYS_TO_EXPIRY'].apply(
                        lambda x: 'Expired' if pd.isna(x) or x < 0 else ('Expiring Soon' if x < 30 else 'Valid')
                    )
                    matched_chunk = chunk[chunk['CATALOGID'].isin(matching_catalog_ids)].copy()
                    matched_chunk['MAPPED_SOFTWARE'] = matched_chunk['CATALOGID'].map(mapping_dict)
                    output_columns = [
                        'CATALOGID', 'SOFTWARE', 'SWNMAME', 'EDITION', 'MANUFACTURER',
                        'PRODUCTFAMILY', 'COMPONENT', 'PRODUCTCLASS', 'PRODUCTTYPE',
                        'MAJORVERSION', 'MINORVERSION', 'VERSIONSERVICEPACK',
                        'LCCURRENTSTARTDATE', 'LCCURRENTENDDATE', 'DAYS_TO_EXPIRY',
                        'LICENSE_STATUS', 'STATUS', 'MAPPED_SOFTWARE'
                    ]
                    matched_chunk = matched_chunk[output_columns]
                    matched_records.append(matched_chunk)
                    total_catalog_records += len(chunk)
                    logger.info(f"Processed chunk {chunk_num}: {len(chunk)} records, found {len(matched_chunk)} matches")
                    log_memory_usage()
                    del chunk
                    del matched_chunk
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_num}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise
        
        # Combine all matched records
        if not matched_records:
            logger.warning("No matching records found!")
            matched_df = pd.DataFrame(columns=output_columns)
        else:
            logger.info("Combining matched records...")
            matched_df = pd.concat(matched_records, ignore_index=True)
            # Clear matched_records from memory
            del matched_records
            gc.collect()
        
        # Create output directory if it doesn't exist
        os.makedirs('data/processed', exist_ok=True)
        
        # Save processed data
        output_file = 'data/processed/Software_Catalog_Matched.csv'
        logger.info(f"Saving {len(matched_df)} records to {output_file}...")
        matched_df.to_csv(output_file, index=False)
        
        # Calculate processing time
        processing_time = datetime.now() - start_time
        logger.info(f"Data processing completed in {processing_time.total_seconds():.2f} seconds")
        log_memory_usage()
        
        # Print summary statistics
        logger.info("\nData Summary:")
        logger.info(f"Total Records in Catalog: {total_catalog_records:,}")
        logger.info(f"Total Records in Mapping: {len(mapping_dict):,}")
        logger.info(f"Matched Records: {len(matched_df):,}")
        
        # Count matches by license status
        if 'LICENSE_STATUS' in matched_df.columns:
            status_counts = matched_df['LICENSE_STATUS'].value_counts()
            logger.info("\nMatched Records by License Status:")
            for status, count in status_counts.items():
                logger.info(f"{status}: {count:,} records")
        
        # Count matches by product family
        if 'PRODUCTFAMILY' in matched_df.columns:
            family_counts = matched_df['PRODUCTFAMILY'].value_counts()
            logger.info("\nMatched Records by Product Family:")
            for family, count in family_counts.items():
                logger.info(f"{family}: {count:,} records")
        
        return {
            'total_catalog_records': total_catalog_records,
            'total_mapping_records': len(mapping_dict),
            'matched_records': len(matched_df),
            'license_status_distribution': status_counts.to_dict() if 'LICENSE_STATUS' in matched_df.columns else {},
            'product_family_distribution': family_counts.to_dict() if 'PRODUCTFAMILY' in matched_df.columns else {}
        }
    
    except Exception as e:
        logger.error(f"Fatal error in processing: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        process_software_catalog()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1) 