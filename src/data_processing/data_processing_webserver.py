import pandas as pd
import numpy as np
import time
import os
import re
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from src.data_processing.data_utils import (
    DataValidationError,
    DataProcessingError,
    clean_text,
    remove_empty_columns,
    standardize_date,
    parse_version
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
logging.getLogger('pandas').setLevel(logging.ERROR)

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

WEB_CSV = os.path.join(RAW_DIR, 'WebServer.csv')
MAP_CSV = os.path.join(RAW_DIR, 'swecomponentmapping.csv')
MERGED_CSV = os.path.join(PROCESSED_DIR, 'WebServer_Merged.csv')
MISMATCH_LOG = os.path.join(PROCESSED_DIR, 'WebServer_Mismatches.log')

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def _clean_text(text):
    """Clean text by removing special characters and standardizing format."""
    if pd.isna(text):
        return ""
    return str(text).strip().upper()

def _clean_column_names(df):
    """Clean column names by removing spaces and standardizing format."""
    df.columns = [col.strip() for col in df.columns]
    return df

def _validate_data(df):
    """Validate input data structure."""
    required_columns = [
        'ASSETNAME', 'MANUFACTURER', 'MODEL', 'ENVIRONMENT',
        'INSTALLPATH', 'INSTANCENAME', 'STATUS', 'SUBSTATUS',
        'PRODUCTCLASS', 'PRODUCTTYPE'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def process_webserver_data():
    """Process web server data from CSV files."""
    start_time = time.time()
    
    # Define file paths
    current_dir = Path(__file__).parent.parent.parent
    raw_data_dir = current_dir / 'data' / 'raw'
    processed_data_dir = current_dir / 'data' / 'processed'
    
    # Create directories if they don't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Input and output file paths
    webserver_file = raw_data_dir / 'WebServer.csv'
    webserver_merged_file = processed_data_dir / 'WebServer_Merged.csv'
    
    try:
        # Load data
        logger.info(f"Loading data from {webserver_file}")
        df = pd.read_csv(webserver_file)
        
        # Clean column names
        df = _clean_column_names(df)
        
        # Validate data
        if not _validate_data(df):
            raise ValueError("Data validation failed")
        
        # Clean text fields
        text_columns = ['MANUFACTURER', 'MODEL', 'ENVIRONMENT', 'STATUS', 
                       'SUBSTATUS', 'PRODUCTCLASS', 'PRODUCTTYPE']
        for col in text_columns:
            df[col] = df[col].apply(_clean_text)
        
        # Clean paths and instance names
        df['INSTALLPATH'] = df['INSTALLPATH'].apply(lambda x: str(x).strip())
        df['INSTANCENAME'] = df['INSTANCENAME'].apply(lambda x: str(x).strip())
        
        # Save processed data
        logger.info(f"Saving processed data to {webserver_merged_file}")
        df.to_csv(webserver_merged_file, index=False)
        
        # Log statistics
        logger.info(f"Processed {len(df)} web server records")
        logger.info(f"Data saved to {webserver_merged_file}")
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing web server data: {str(e)}")
        raise

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize column names."""
    df.columns = df.columns.str.strip().str.upper()
    return df

def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean string values in the dataframe."""
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def validate_input_data(df: pd.DataFrame) -> None:
    """Validate input data structure and content."""
    required_columns = ['ASSETNAME', 'CATALOGID', 'ENVIRONMENT', 'INSTALLPATH', 'INSTANCENAME', 'INVNO', 'MANUFACTURER', 'MODEL', 'PRODUCTCLASS', 'PRODUCTTYPE', 'STATUS', 'SUBSTATUS']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    
    # Check for empty dataframe
    if df.empty:
        raise DataValidationError("Input dataframe is empty")
    
    # Validate data types
    if not pd.api.types.is_string_dtype(df['ASSETNAME']):
        raise DataValidationError("ASSETNAME must be string type")
    if not pd.api.types.is_string_dtype(df['CATALOGID']):
        raise DataValidationError("CATALOGID must be string type")
    
    # Check for null values in critical columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        raise DataValidationError(f"Found null values in required columns:\n{null_counts[null_counts > 0]}")

def standardize_version(text: Any) -> str:
    """Standardize version numbers for different software types with improved error handling."""
    if pd.isna(text):
        return ""
    try:
        text = str(text).strip()
        
        if re.match(r'^\d+\.\d+$', text):
            text += ".0"
        
        text = text.upper()
        
        # Handle specific software types
        software_patterns = {
            "APACHE TOMCAT": r'(\d+\.\d+\.\d+)',
            "APACHE HTTPD": r'(\d+\.\d+)',
            "WEBSPHERE": r'(\d+\.\d+)',
            "NGINX": r'(\d+\.\d+)'
        }
        
        for software, pattern in software_patterns.items():
            if software in text:
                version_match = re.search(pattern, text)
                if version_match:
                    version = version_match.group(1)
                    return f"{software} {version}"
        
        return text
    except Exception as e:
        logger.warning(f"Error standardizing version '{text}': {str(e)}")
        return str(text)

def analyze_software_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze software patterns and add context to the data."""
    try:
        # Standardize software names and versions
        df['MODEL'] = df['MODEL'].apply(standardize_version)
        
        # Add environment context
        df['ENVIRONMENT_TYPE'] = df['ENVIRONMENT'].map({
            'PROD': 'Production',
            'UAT': 'User Acceptance Testing',
            'DEV': 'Development'
        })
        
        # Add software category
        df['SOFTWARE_CATEGORY'] = df['PRODUCTCLASS'].map({
            'APPLICATION SERVER': 'App Server',
            'HTTP SERVER': 'Web Server'
        })
        
        # Add status context
        df['STATUS_CONTEXT'] = df['STATUS'] + ' - ' + df['SUBSTATUS']
        
        # Create columns needed for vectorizer
        df['OBJECTNAME'] = df['MODEL']  # Use MODEL as OBJECTNAME
        df['OLDVALUE'] = df['MODEL'].str.extract(r'(\d+\.\d+(?:\.\d+)?)')  # Extract version number
        df['NEWVALUE'] = df['OLDVALUE']  # For inventory data, old and new are the same
        df['CHUNK_TEXT'] = df.apply(
            lambda row: f"{row['MANUFACTURER']} {row['MODEL']} installed in {row['ENVIRONMENT_TYPE']} environment. "
                       f"Path: {row['INSTALLPATH']}, Instance: {row['INSTANCENAME']}. "
                       f"Status: {row['STATUS_CONTEXT']}",
            axis=1
        )
        
        return df
    except Exception as e:
        raise DataProcessingError(f"Error analyzing software patterns: {str(e)}")

def process_data() -> Dict[str, Any]:
    """Main function to process the data through all cleaning and analysis steps."""
    start_time = time.time()
    logger.info("Starting data processing...")
    
    try:
        # Read and merge data
        logger.info('Reading WebServer.csv...')
        web_df = pd.read_csv(WEB_CSV, dtype=str, skipinitialspace=True)
        web_df = clean_column_names(web_df)
        web_df = clean_values(web_df)
        logger.info(f'WebServer.csv columns: {web_df.columns.tolist()}')

        logger.info('Reading swecomponentmapping.csv...')
        map_df = pd.read_csv(MAP_CSV, dtype=str, skipinitialspace=True, quotechar='"')
        map_df = clean_column_names(map_df)
        map_df = clean_values(map_df)
        logger.info(f'swecomponentmapping.csv columns: {map_df.columns.tolist()}')

        # Validate input data
        validate_input_data(web_df)
        
        # Basic cleaning
        web_df = remove_empty_columns(web_df, ["OSIINVNO", "VERUMIDENTIFIER", "LOAD_DT"])
        logger.info(f"After cleaning: {len(web_df)} records")
        
        # Merge data
        logger.info('Merging on CATALOGID and INVNO...')
        merged = pd.merge(
            web_df, 
            map_df, 
            on=['CATALOGID', 'INVNO'], 
            how='outer', 
            indicator=True, 
            suffixes=('_WEB', '_MAP')
        )

        # Log mismatches
        mismatches = merged[merged['_merge'] != 'both']
        if not mismatches.empty:
            logger.info(f'Found {len(mismatches)} mismatches. Logging to {MISMATCH_LOG}')
            mismatches.to_csv(MISMATCH_LOG, index=False)
        else:
            logger.info('No mismatches found.')

        # Process matched data
        matched = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])
        matched = analyze_software_patterns(matched)
        
        # Save processed data
        matched.to_csv(MERGED_CSV, index=False)
        logger.info(f"Saved processed data to {MERGED_CSV}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Data processing completed in {processing_time:.2f} seconds")
        
        # Return analysis results
        return {
            'total_records': len(matched),
            'environment_counts': matched['ENVIRONMENT'].value_counts().to_dict(),
            'software_counts': matched['PRODUCTCLASS'].value_counts().to_dict(),
            'manufacturer_counts': matched['MANUFACTURER'].value_counts().to_dict()
        }

    except Exception as e:
        logger.info("Error processing data. Please check your input files.")
        return {
            'total_records': 0,
            'environment_counts': {},
            'software_counts': {},
            'manufacturer_counts': {}
        }

if __name__ == "__main__":
    try:
        df = process_webserver_data()
        # --- Analysis summary ---
        logger.info("\n=== DATA SUMMARY ===")
        logger.info(f"Total Records: {len(df)}")
        logger.info(f"Columns: {list(df.columns)}")
        key_columns = ['ENVIRONMENT', 'MANUFACTURER', 'PRODUCTCLASS', 'STATUS']
        for col in key_columns:
            if col in df.columns:
                logger.info(f"\nValue counts for {col}:")
                logger.info(f"\n{df[col].value_counts()}")
        logger.info("\nFirst 3 rows:\n" + str(df.head(3)))
    except Exception as e:
        logger.error(f"Failed to process web server data: {str(e)}")
        raise 