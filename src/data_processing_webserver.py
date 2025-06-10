import pandas as pd
import numpy as np
from datetime import datetime
import re
from dateutil import parser
from packaging import version
import logging
from typing import Dict, List, Any, Optional
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

WEB_CSV = os.path.join(RAW_DIR, 'WebServer.csv')
MAP_CSV = os.path.join(RAW_DIR, 'swecomponentmapping.csv')
MERGED_CSV = os.path.join(PROCESSED_DIR, 'WebServer_Merged.csv')
MISMATCH_LOG = os.path.join(PROCESSED_DIR, 'WebServer_Mismatches.log')

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

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

def clean_text(text: Any) -> str:
    """Clean and standardize text values with improved error handling."""
    if pd.isna(text):
        return ""
    try:
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    except Exception as e:
        logger.warning(f"Error cleaning text '{text}': {str(e)}")
        return str(text)

def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty columns and specific columns that are not needed."""
    try:
        df = df.replace("", np.nan)
        df = df.dropna(axis=1, how="all")
        
        columns_to_drop = ["OSIINVNO", "VERUMIDENTIFIER", "LOAD_DT"]
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            df.drop(columns=existing_columns, inplace=True)
        return df
    except Exception as e:
        raise DataProcessingError(f"Error removing empty columns: {str(e)}")

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
        web_df = remove_empty_columns(web_df)
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
            logger.warning(f'Found {len(mismatches)} mismatches. Logging to {MISMATCH_LOG}')
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
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        results = process_data()
        logger.info("Processing Results:")
        logger.info(f"Total Records: {results['total_records']}")
        logger.info(f"Environment Distribution: {results['environment_counts']}")
        logger.info(f"Software Distribution: {results['software_counts']}")
        logger.info(f"Manufacturer Distribution: {results['manufacturer_counts']}")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise 