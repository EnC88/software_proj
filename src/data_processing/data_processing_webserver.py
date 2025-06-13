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
    """Process and map webservers to operating systems using PCat.csv and webserver data."""
    start_time = time.time()
    logger.info("Starting webserver-OS mapping processing...")
    logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

    try:
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

        # Read PCat data with proper handling of quotes and spaces
        logger.info("\nReading PCat data...")
        pcat_path = os.path.join('data', 'raw', 'PCat.csv')
        import os as _os
        logger.info(f"Current working directory: {_os.getcwd()}")
        logger.info(f"Absolute path to PCat.csv: {_os.path.abspath(pcat_path)}")
        with open(pcat_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        logger.info(f"Read {len(lines)} lines from PCat.csv")
        # Clean header
        header = lines[0].replace('"', '').replace(' ', '').strip().split(',')
        # Write cleaned temp file
        import tempfile
        with tempfile.NamedTemporaryFile('w+', delete=False, newline='', encoding='utf-8') as tmp:
            tmp.write(','.join(header) + '\n')
            for line in lines[1:]:
                tmp.write(line)
            tmp_path = tmp.name
        # Now read with pandas
        pcat_df = pd.read_csv(tmp_path, dtype=str)
        # Clean up temp file
        _os.remove(tmp_path)
        
        # Clean column names
        pcat_df.columns = pcat_df.columns.str.strip().str.replace('"', '')
        
        # Log PCat data for debugging
        logger.info("\nPCat data sample:")
        logger.info(pcat_df[['CATALOGID', 'MODEL', 'PRODUCTTYPE', 'PRODUCTCLASS']].head())

        # Create a mapping from OSIINVNO to PCat data
        # First, get unique OSIINVNOs from webserver data
        osiinvnos = webserver_df['OSIINVNO'].unique()
        logger.info(f"\nUnique OSIINVNOs in webserver data: {len(osiinvnos)}")
        
        # Create a mapping dataframe with OSIINVNO as CATALOGID
        mapping_df = pcat_df.copy()
        mapping_df['CATALOGID'] = mapping_df['CATALOGID'].astype(str)
        
        # Ensure both merge columns are string type
        webserver_df['OSIINVNO'] = webserver_df['OSIINVNO'].astype(str)
        mapping_df['CATALOGID'] = mapping_df['CATALOGID'].astype(str)
        # Merge webserver with PCat data using OSIINVNO as the key
        merged_df = pd.merge(
            webserver_df,
            mapping_df,
            left_on='OSIINVNO',
            right_on='CATALOGID',
            how='left',
            suffixes=('', '_PCAT')
        )

        # Extract major/minor version from MODEL (or SOFTWARE if MODEL is missing)
        def extract_major_minor(text):
            if pd.isna(text):
                return None, None
            match = re.search(r'(\d+)\.(\d+)', str(text))
            if match:
                return match.group(1), match.group(2)
            return None, None
        merged_df[['majorversion', 'minorversion']] = merged_df.apply(
            lambda row: pd.Series(extract_major_minor(row['MODEL'] if pd.notna(row['MODEL']) else row['SOFTWARE'])), axis=1)

        # Select relevant columns
        output_columns = [
            # Original webserver columns
            'ASSETNAME', 'INSTANCENAME', 'CATALOGID', 'SOFTWARE', 'EDITION', 
            'MANUFACTURER', 'PRODUCTFAMILY', 'PRODUCTCLASS', 'PRODUCTTYPE', 
            'ENVIRONMENT', 'STATUS', 'SUBSTATUS', 'INSTALLPATH', 'OSIINVNO', 
            'INVNO', 'LOAD_DT', 'majorversion', 'minorversion', 'VERSIONSERVICEPACK',
            # Additional PCat columns
            'MODEL', 'LIFECYCLESTATUS', 'RELEASEVERSION', 'PRODUCTCATEGORY',
            'ARCHITECTURE', 'VERUMCREATEDDATE', 'VERUMMODIFIEDDATE'
        ]
        # Some columns may be missing, so filter to those present
        output_columns = [col for col in output_columns if col in merged_df.columns]
        final_df = merged_df[output_columns]

        # Read softwarecatalog data
        logger.info("\nReading softwarecatalog data...")
        softwarecatalog_path = os.path.join('data', 'raw', 'softwarecatalog.csv')
        softwarecatalog_df = pd.read_csv(softwarecatalog_path, dtype=str)
        logger.info(f"Read {len(softwarecatalog_df)} records from softwarecatalog.csv.")

        # Ensure both merge columns are string type
        final_df['CATALOGID'] = final_df['CATALOGID'].astype(str)
        softwarecatalog_df['CATALOGID'] = softwarecatalog_df['CATALOGID'].astype(str)
        # Merge with softwarecatalog data
        final_df = pd.merge(
            final_df,
            softwarecatalog_df,
            on='CATALOGID',
            how='left',
            suffixes=('', '_SC')
        )

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
        logger.info(f"Total Records in PCat Data: {len(pcat_df)}")
        logger.info(f"Total Records in Softwarecatalog Data: {len(softwarecatalog_df)}")
        logger.info(f"Total Records in Webserver-OS Mapping: {len(merged_df)}")
        logger.info(f"Matched Records: {len(final_df)}")

        # Validate the merge
        logger.info("\nValidating merge...")
        unmatched = merged_df[merged_df['CATALOGID'].isna()]
        logger.info(f"Unmatched records: {len(unmatched)}")
        if len(unmatched) > 0:
            logger.info("Unmatched OSIINVNOs:")
            logger.info(unmatched[['OSIINVNO', 'ASSETNAME']].head())
        else:
            logger.info("All records matched successfully.")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def prepare_for_vectorization():
    """
    Prepares the processed output for vectorization by selecting relevant columns and filling missing values.
    """
    logger.info("Preparing data for vectorization...")
    # Read the processed output file
    df = pd.read_csv('data/processed/Webserver_OS_Mapping.csv')
    # Select columns needed for vectorization
    columns_for_vectorization = [
        'ASSETNAME', 'INSTANCENAME', 'CATALOGID', 'MANUFACTURER', 'PRODUCTCLASS', 'PRODUCTTYPE',
        'ENVIRONMENT', 'STATUS', 'SUBSTATUS', 'INSTALLPATH', 'OSIINVNO', 'INVNO', 'LOAD_DT',
        'majorversion', 'minorversion', 'MODEL', 'LIFECYCLESTATUS', 'RELEASEVERSION', 'PRODUCTCATEGORY',
        'ARCHITECTURE', 'VERUMCREATEDDATE', 'VERUMMODIFIEDDATE', 'EDITION', 'SOFTWARE', 'PRODUCTFAMILY',
        'COMPONENT', 'SWNMAME', 'ISLICENSIBLE', 'MAJORVERSION', 'MINORVERSION', 'VERSIONSERVICEPACK',
        'CATSERVICEPACK', 'SUIT', 'ISSUITCOMPONENT', 'GADATE', 'VERUMIDENTIFIER', 'VERUMSOR',
        'VERUMMODIFIEDDATE_SC', 'VERUMLASTMODIFIEDBY', 'VERUMCREATEDBY', 'SOFTWAREMMVERSION',
        'LCCURRENTENDDATE', 'LCCURRENTSTARTDATE', 'SWSERVICEPACK', 'GSPCEOL', 'SWID', 'MANUFACTURER_SC',
        'MODEL_SC', 'PRODUCTCLASS_SC', 'PRODUCTTYPE_SC', 'STATUS_SC'
    ]
    df_vectorization = df[columns_for_vectorization].fillna('')
    # Save the refined output
    output_path = 'data/processed/Webserver_OS_Mapping_For_Vectorization.csv'
    df_vectorization.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df_vectorization)} records to {output_path} for vectorization.")

if __name__ == "__main__":
    process_webserver_os_mapping()
    prepare_for_vectorization() 