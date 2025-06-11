import pandas as pd
import logging
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
    return str(text).strip().upper().replace('"', '')

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

def process_os_mapping():
    """Process WebServer data and match with component mapping based on OSIINVNO."""
    start_time = pd.Timestamp.now()
    logger.info("Starting OS mapping processing...")
    log_memory_usage()
    
    try:
        # Read mapping file
        logger.info("Reading mapping file...")
        mapping_df = robust_read_csv('data/raw/swecomponentmapping.csv', dtype={'CATALOGID': str, 'INVNO': str})
        
        # Clean column names
        mapping_df.columns = mapping_df.columns.str.strip().str.replace('"', '')
        
        # Validate mapping file columns
        mapping_required = ['CATALOGID', 'SWNAME', 'INVNO']
        _validate_data(mapping_df, mapping_required)
        
        # Create a set of INVNOs for quick lookup
        invno_set = set(mapping_df['INVNO'].apply(_clean_text))
        
        # Clear mapping dataframe from memory
        del mapping_df
        gc.collect()
        log_memory_usage()
        
        # Read the entire WebServer.csv file at once
        logger.info("Reading WebServer.csv file...")
        webserver_df = robust_read_csv('data/raw/WebServer.csv')
        
        # Clean column names
        webserver_df.columns = webserver_df.columns.str.strip().str.replace('"', '')
        
        # Initialize a list to store matched records
        matched_records = []
        
        # Iterate through each row in WebServer.csv
        for _, row in webserver_df.iterrows():
            osiinvno = _clean_text(row['OSIINVNO'])
            if osiinvno in invno_set:
                matched_records.append(row)
        
        # Create a DataFrame from matched records
        matched_df = pd.DataFrame(matched_records)
        
        # Create output directory if it doesn't exist
        os.makedirs('data/processed', exist_ok=True)
        
        # Save processed data
        output_file = 'data/processed/Server_Software_Mapping.csv'
        logger.info(f"Saving {len(matched_df)} records to {output_file}...")
        matched_df.to_csv(output_file, index=False)
        
        # Calculate processing time
        processing_time = pd.Timestamp.now() - start_time
        logger.info(f"Data processing completed in {processing_time.total_seconds():.2f} seconds")
        log_memory_usage()
        
        # Print summary statistics
        logger.info("\nData Summary:")
        logger.info(f"Total Records in WebServer: {len(webserver_df):,}")
        logger.info(f"Total Unique INVNOs in Mapping: {len(invno_set):,}")
        logger.info(f"Matched Records: {len(matched_df):,}")
        
        return {
            'total_webserver_records': len(webserver_df),
            'total_mapping_invnos': len(invno_set),
            'matched_records': len(matched_df),
            'processing_time': processing_time.total_seconds()
        }
    
    except Exception as e:
        logger.error(f"Fatal error in processing: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        process_os_mapping()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1) 