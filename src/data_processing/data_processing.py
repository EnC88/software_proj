import pandas as pd
import numpy as np
from datetime import datetime
import re
from dateutil import parser
from packaging import version
import logging
from typing import Dict, List, Any, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

def validate_input_data(df: pd.DataFrame) -> None:
    """Validate input data structure and content."""
    required_columns = ['VERUMCREATEDBY', 'VERUMCREATEDDATE', 'OBJECTNAME', 'NEWVALUE', 'OLDVALUE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    
    # Check for empty dataframe
    if df.empty:
        raise DataValidationError("Input dataframe is empty")
    
    # Validate data types
    if not pd.api.types.is_string_dtype(df['VERUMCREATEDBY']):
        raise DataValidationError("VERUMCREATEDBY must be string type")
    if not pd.api.types.is_string_dtype(df['OBJECTNAME']):
        raise DataValidationError("OBJECTNAME must be string type")
    
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
        
        columns_to_drop = ["RELATEDOBJECTID", "VERUMIDENTIFIER", "ENTITLEMENTUSED"]
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            df.drop(columns=existing_columns, inplace=True)
        return df
    except Exception as e:
        raise DataProcessingError(f"Error removing empty columns: {str(e)}")

def remove_duplicate_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate changes for each user with improved performance."""
    try:
        # Use groupby and transform for better performance
        df['is_duplicate'] = df.groupby(['VERUMCREATEDBY', 'NEWVALUE', 'OLDVALUE']).cumcount() > 0
        return df[~df['is_duplicate']].drop(columns=['is_duplicate'])
    except Exception as e:
        raise DataProcessingError(f"Error removing duplicate changes: {str(e)}")

def standardize_date(date_str: Any) -> Optional[str]:
    """Standardize date format across the dataset with improved error handling."""
    if pd.isna(date_str):
        return None
    try:
        date_str = str(date_str).strip()
        date_str = re.sub(r'\s+', ' ', date_str)
        date_str = re.sub(r'(\d{2})\.(\d{2})\.(\d{2})', r'\1:\2:\3', date_str)
        date_str = re.sub(r'(:\d{2})\.(\d+)(?=\s*(AM|PM|UTC))', r'\1', date_str)
        date_str = re.sub(r' UTC$', '', date_str)
        date_obj = parser.parse(date_str)
        return date_obj.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logger.warning(f"Error parsing date '{date_str}': {str(e)}")
        return None

def organize_user_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """Organize data by user and time with improved performance."""
    try:
        df.columns = df.columns.str.strip()
        
        # Use vectorized operations for better performance
        df["NEWVALUE"] = df["NEWVALUE"].str.lower()
        df["OLDVALUE"] = df["OLDVALUE"].str.lower()
        
        df["VERUMCREATEDDATE"] = df["VERUMCREATEDDATE"].apply(standardize_date)
        df["VERUMCREATEDDATE"] = pd.to_datetime(df["VERUMCREATEDDATE"], errors="coerce")
        
        # Sort using numpy for better performance
        sort_idx = np.lexsort((df["VERUMCREATEDDATE"].values, df["VERUMCREATEDBY"].values))
        return df.iloc[sort_idx]
    except Exception as e:
        raise DataProcessingError(f"Error organizing user and time data: {str(e)}")

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
            "ORACLE DATABASE": r'(\d+\.\d+\.\d+\.\d+)',
            "MONGODB": r'(\d+\.\d+)',
            "APACHE TOMCAT": r'(\d+\.\d+\.\d+)',
            "APACHE CASSANDRA": r'(\d+\.\d+)',
            "SQL SERVER": r'(\d{4})'
        }
        
        for software, pattern in software_patterns.items():
            if software in text:
                version_match = re.search(pattern, text)
                if version_match:
                    version = version_match.group(1)
                    if software == "SQL SERVER":
                        return f"{software} {version} - LINUX"
                    return f"{software} {version}"
        
        return text
    except Exception as e:
        logger.warning(f"Error standardizing version '{text}': {str(e)}")
        return str(text)

def parse_version(ver_str: Any) -> Optional[version.Version]:
    """Parse version strings into comparable version objects with improved error handling."""
    if pd.isna(ver_str) or ver_str == "":
        return None
    try:
        version_match = re.search(r'(\d+(?:\.\d+)*)', str(ver_str))
        if version_match:
            return version.parse(version_match.group(1))
        return None
    except Exception as e:
        logger.warning(f"Error parsing version '{ver_str}': {str(e)}")
        return None

def analyze_version_chains(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze version upgrade chains and redundant logs with improved performance."""
    try:
        # Use groupby with named aggregations for better performance
        version_chains = df.groupby(['OBJECTNAME', 'NEWVALUE']).agg(
            OLDVALUE=('OLDVALUE', list),
            VERUMCREATEDDATE=('VERUMCREATEDDATE', list),
            VERUMCREATEDBY=('VERUMCREATEDBY', list)
        ).reset_index()
        
        return version_chains[version_chains['OLDVALUE'].str.len() > 1]
    except Exception as e:
        raise DataProcessingError(f"Error analyzing version chains: {str(e)}")

def analyze_upgrade_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze upgrade patterns and add context to the data with improved performance."""
    try:
        # Sort using numpy for better performance
        sort_idx = np.lexsort((
            df['VERUMCREATEDDATE'].values,
            df['OBJECTNAME'].values,
            df['VERUMCREATEDBY'].values
        ))
        df = df.iloc[sort_idx]
        
        # Calculate time differences using vectorized operations
        df['TIME_SINCE_LAST_UPGRADE'] = df.groupby(['VERUMCREATEDBY', 'OBJECTNAME'])['VERUMCREATEDDATE'].diff()
        df['IS_CLUSTERED_UPGRADE'] = df['TIME_SINCE_LAST_UPGRADE'] < pd.Timedelta(minutes=5)
        df['USER_UPGRADE_COUNT'] = df.groupby(['VERUMCREATEDBY', 'OBJECTNAME']).cumcount() + 1
        
        # Identify version rollbacks using vectorized operations
        df['IS_ROLLBACK'] = False
        mask = (
            (df['OBJECTNAME'] == df['OBJECTNAME'].shift(1)) &
            (df['VERUMCREATEDBY'] == df['VERUMCREATEDBY'].shift(1))
        )
        
        # Apply version comparison only where mask is True
        df.loc[mask, 'IS_ROLLBACK'] = df[mask].apply(
            lambda row: (
                parse_version(row['NEWVALUE']) is not None and
                parse_version(row['OLDVALUE']) is not None and
                parse_version(row['NEWVALUE']) < parse_version(row['OLDVALUE'])
            ),
            axis=1
        )
        
        return df
    except Exception as e:
        raise DataProcessingError(f"Error analyzing upgrade patterns: {str(e)}")

def process_data(input_file: str, output_file: str) -> Dict[str, Any]:
    """Main function to process the data through all cleaning and analysis steps."""
    start_time = time.time()
    logger.info("Starting data processing...")
    
    try:
        # Read data
        df = pd.read_csv(input_file)
        logger.info(f"Read {len(df)} records from input file")
        
        # Validate input data
        validate_input_data(df)
        
        # Basic cleaning
        df = remove_empty_columns(df)
        df = remove_duplicate_changes(df)
        logger.info(f"After cleaning: {len(df)} records")
        
        # Standardize data
        df["NEWVALUE"] = df["NEWVALUE"].apply(standardize_version)
        df["OLDVALUE"] = df["OLDVALUE"].apply(standardize_version)
        df = organize_user_and_time(df)
        
        # Analysis
        df = analyze_upgrade_patterns(df)
        
        # Save processed data
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Data processing completed in {processing_time:.2f} seconds")
        
        # Return analysis results
        return {
            'redundant_upgrades': analyze_version_chains(df),
            'clustered_upgrades': df[df['IS_CLUSTERED_UPGRADE']],
            'rollbacks': df[df['IS_ROLLBACK']],
            'processing_time': processing_time,
            'total_records': len(df)
        }
        
    except DataValidationError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise
    except DataProcessingError as e:
        logger.error(f"Data processing error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data processing: {str(e)}")
        raise

if __name__ == "__main__":
    input_file = "data/raw/rawdata.csv"
    output_file = "data/processed/processed_data.csv"
    
    try:
        results = process_data(input_file, output_file)
        
        # Print analysis results
        logger.info("\nAnalysis Results:")
        logger.info(f"Total records processed: {results['total_records']}")
        logger.info(f"Processing time: {results['processing_time']:.2f} seconds")
        logger.info(f"Number of redundant upgrades: {len(results['redundant_upgrades'])}")
        logger.info(f"Number of clustered upgrades: {len(results['clustered_upgrades'])}")
        logger.info(f"Number of rollbacks: {len(results['rollbacks'])}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise