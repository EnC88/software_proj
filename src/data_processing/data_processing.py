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
import sys
import psutil
import gc
import warnings
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

def _clean_text(text):
    """Clean text by removing quotes and extra spaces."""
    if pd.isna(text):
        return text
    return str(text).strip().replace('"', '').replace("'", '').strip()

def _parse_date(date_str):
    """Parse date string with better error handling."""
    if pd.isna(date_str):
        return None
    date_str = _clean_text(date_str)
    try:
        # Try parsing with the expected format
        return pd.to_datetime(date_str, format='%d-%b-%y %I.%M.%S.%f %p %Z')
    except:
        try:
            # Try parsing with pandas' flexible parser
            return pd.to_datetime(date_str)
        except:
            logger.warning(f"Could not parse date with any format: {date_str}")
            return None

def _standardize_version(version_str):
    """Standardize version numbers for better comparison."""
    if pd.isna(version_str):
        return version_str
    version_str = str(version_str).strip()
    # Extract version numbers (e.g., 9.0.95 from "APACHE TOMCAT 9.0.95")
    match = re.search(r'(\d+(?:\.\d+)*)', version_str)
    if match:
        return match.group(1)
    return version_str

def process_sor_history():
    """Process SOR history data and save to CSV."""
    start_time = time.time()
    logger.info("Starting SOR history processing...")
    logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    try:
        # Read SOR history file with proper CSV parameters
        logger.info("Reading SOR history file...")
        sor_hist_path = os.path.join('data', 'raw', 'sor_hist.csv')
        sor_hist_df = pd.read_csv(
            sor_hist_path,
            encoding='utf-8',
            quotechar='"',
            skipinitialspace=True,
            on_bad_lines='warn'
        )
        logger.info(f"Read {sor_hist_path} with utf-8 encoding.")

        # Clean column names
        sor_hist_df.columns = sor_hist_df.columns.str.strip().str.replace('"', '')

        # Log raw values for debugging
        logger.info("\nRaw values in sor_hist.csv:")
        logger.info(sor_hist_df[['ATTRIBUTENAME', 'OLDVALUE', 'NEWVALUE', 'VERUMCREATEDDATE']].head())

        # Read mapping file
        logger.info("\nReading swecomponentmapping file...")
        mapping_path = os.path.join('data', 'raw', 'swecomponentmapping.csv')
        mapping_df = pd.read_csv(
            mapping_path,
            encoding='utf-8',
            quotechar='"',
            skipinitialspace=True
        )
        logger.info(f"Read {mapping_path} with utf-8 encoding.")

        # Clean and standardize mapping data
        mapping_df.columns = mapping_df.columns.str.strip().str.replace('"', '')
        mapping_df['CATALOGID'] = mapping_df['CATALOGID'].apply(_clean_text)
        mapping_df['SWNAME'] = mapping_df['SWNAME'].apply(_clean_text)

        # Log mapping file contents for debugging
        logger.info("\nMapping file contents:")
        logger.info(mapping_df[['CATALOGID', 'SWNAME']].head())

        # Create catalogid to swname mapping
        catalogid_to_swname = dict(zip(mapping_df['CATALOGID'], mapping_df['SWNAME']))
        # Clean and standardize SOR history data
        sor_hist_df['ATTRIBUTENAME'] = sor_hist_df['ATTRIBUTENAME'].apply(_clean_text)
        sor_hist_df['OLDVALUE'] = sor_hist_df['OLDVALUE'].apply(_clean_text)
        sor_hist_df['NEWVALUE'] = sor_hist_df['NEWVALUE'].apply(_clean_text)
        sor_hist_df['VERUMCREATEDDATE'] = sor_hist_df['VERUMCREATEDDATE'].apply(_parse_date)

        # Create a copy of the dataframe for all changes
        all_changes = sor_hist_df.copy()

        # Add columns for version tracking
        all_changes['CHANGE_TYPE'] = all_changes['ATTRIBUTENAME']
        all_changes['IS_VERSION_CHANGE'] = all_changes['ATTRIBUTENAME'].isin(['CATALOGID', 'MODEL'])
        all_changes['OLD_VERSION'] = ''
        all_changes['NEW_VERSION'] = ''
        all_changes['OLD_CATALOGID'] = ''
        all_changes['NEW_CATALOGID'] = ''
        all_changes['OLD_SWNAME'] = ''
        all_changes['NEW_SWNAME'] = ''

        # Process CATALOGID changes
        catalogid_mask = all_changes['ATTRIBUTENAME'] == 'CATALOGID'
        all_changes.loc[catalogid_mask, 'OLD_CATALOGID'] = all_changes.loc[catalogid_mask, 'OLDVALUE']
        all_changes.loc[catalogid_mask, 'NEW_CATALOGID'] = all_changes.loc[catalogid_mask, 'NEWVALUE']
        all_changes.loc[catalogid_mask, 'OLD_SWNAME'] = all_changes.loc[catalogid_mask, 'OLDVALUE'].map(catalogid_to_swname)
        all_changes.loc[catalogid_mask, 'NEW_SWNAME'] = all_changes.loc[catalogid_mask, 'NEWVALUE'].map(catalogid_to_swname)
        all_changes.loc[catalogid_mask, 'OLD_VERSION'] = all_changes.loc[catalogid_mask, 'OLD_SWNAME'].apply(_standardize_version)
        all_changes.loc[catalogid_mask, 'NEW_VERSION'] = all_changes.loc[catalogid_mask, 'NEW_SWNAME'].apply(_standardize_version)
        # Sort by date
        all_changes = all_changes.sort_values('VERUMCREATEDDATE')

        # In process_sor_history(), after creating all_changes, add a new column for installation tracking
        all_changes['IS_INSTALLATION'] = (all_changes['OLDVALUE'].isna() | (all_changes['OLDVALUE'] == '')) & (~all_changes['NEWVALUE'].isna() & (all_changes['NEWVALUE'] != ''))

        # Create output directory if it doesn't exist
        os.makedirs(os.path.join('data', 'processed'), exist_ok=True)

        # Save to CSV
        output_path = os.path.join('data', 'processed', 'Change_History.csv')
        logger.info(f"\nSaving {len(all_changes)} records to {output_path}...")
        all_changes.to_csv(output_path, index=False, quoting=1)  # QUOTE_MINIMAL

        # Log summary
        end_time = time.time()
        logger.info(f"Data processing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
        logger.info("\nData Summary:")
        logger.info(f"Total Records in SOR History: {len(sor_hist_df)}")
        logger.info(f"Total Changes: {len(all_changes)}")
        logger.info(f"Version Changes (CATALOGID + MODEL): {len(all_changes[all_changes['IS_VERSION_CHANGE']])}")
        logger.info(f"CATALOGID Changes: {len(all_changes[catalogid_mask])}")
        logger.info(f"Installation Changes: {len(all_changes[all_changes['IS_INSTALLATION']])}")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    process_sor_history()