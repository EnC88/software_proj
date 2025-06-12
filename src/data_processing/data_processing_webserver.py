import pandas as pd
import logging
import time
import os
from datetime import datetime
import re
import psutil
from os_mapping import process_os_mapping, map_os_to_software  # Import both functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def process_webserver_os_mapping():
    """Process and map webservers to operating systems using swecomponentmapping, webserver, and softwarecatalog data."""
    start_time = time.time()
    logger.info("Starting webserver-OS mapping processing...")
    logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    try:
        # Use the existing function to process OS mapping
        process_os_mapping()

        # Read webserver data file
        logger.info("\nReading webserver data file...")
        webserver_path = os.path.join('data', 'raw', 'WebServer.csv')
        webserver_df = pd.read_csv(
            webserver_path,
            encoding='utf-8',
            quotechar='"',
            skipinitialspace=True
        )
        logger.info(f"Read {webserver_path} with utf-8 encoding.")

        # Clean column names
        webserver_df.columns = webserver_df.columns.str.strip().str.replace('"', '')

        # Log raw values for debugging
        logger.info("\nRaw values in webserver_data.csv:")
        logger.info(webserver_df.head())

        # Prepare mapping from swecomponentmapping
        mapping_df = pd.read_csv(os.path.join('data', 'raw', 'swecomponentmapping.csv'), encoding='utf-8', quotechar='"', skipinitialspace=True)
        mapping_df.columns = mapping_df.columns.str.strip().str.replace('"', '')
        catalogid_to_swname = mapping_df.set_index('CATALOGID')['SWNAME'].to_dict()

        # Prepare catalog data
        catalog_df = pd.read_csv(os.path.join('data', 'raw', 'softwarecatalog.csv'), encoding='utf-8', quotechar='"', skipinitialspace=True)
        catalog_df.columns = catalog_df.columns.str.strip().str.replace('"', '')

        # Merge webserver with catalog
        merged_df = pd.merge(webserver_df, catalog_df, on='CATALOGID', how='left', suffixes=('', '_CATALOG'))

        # Add swname from mapping
        merged_df['swname'] = merged_df['CATALOGID'].map(catalogid_to_swname)
        # mapped_software is the same as swname
        merged_df['mapped_software'] = merged_df['swname']

        # Extract major/minor version from swname (or software if swname is missing)
        def extract_major_minor(text):
            if pd.isna(text):
                return None, None
            match = re.search(r'(\d+)\.(\d+)', str(text))
            if match:
                return match.group(1), match.group(2)
            return None, None
        merged_df[['majorversion', 'minorversion']] = merged_df.apply(
            lambda row: pd.Series(extract_major_minor(row['swname'] if pd.notna(row['swname']) else row['SOFTWARE'])), axis=1)

        # Select relevant columns
        output_columns = [
            'ASSETNAME', 'INSTANCENAME', 'CATALOGID', 'SOFTWARE', 'swname', 'EDITION', 'MANUFACTURER', 'PRODUCTFAMILY', 'PRODUCTCLASS', 'PRODUCTTYPE', 'ENVIRONMENT', 'STATUS', 'SUBSTATUS', 'INSTALLPATH', 'OSIINVNO', 'INVNO', 'LOAD_DT', 'majorversion', 'minorversion', 'VERSIONSERVICEPACK', 'mapped_software'
        ]
        # Some columns may be missing, so filter to those present
        output_columns = [col for col in output_columns if col in merged_df.columns]
        final_df = merged_df[output_columns]

        # Create output directory if it doesn't exist
        os.makedirs(os.path.join('data', 'processed'), exist_ok=True)

        # Save to CSV
        output_path = os.path.join('data', 'processed', 'Webserver_OS_Mapping.csv')
        logger.info(f"\nSaving {len(final_df)} records to {output_path}...")
        final_df.to_csv(output_path, index=False, quoting=1)  # QUOTE_MINIMAL

        # Log summary
        end_time = time.time()
        logger.info(f"Data processing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
        logger.info("\nData Summary:")
        logger.info(f"Total Records in Webserver Data: {len(webserver_df)}")
        logger.info(f"Total Records in Software Catalog: {len(catalog_df)}")
        logger.info(f"Total Records in Webserver-OS Mapping: {len(merged_df)}")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    process_webserver_os_mapping() 