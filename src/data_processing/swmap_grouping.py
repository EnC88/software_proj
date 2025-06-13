import pandas as pd
import logging
import os
import time
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_webserver_os_mapping():
    """
    Process the webserver-OS mapping by reading WebServer.csv and PCat.csv, merging them on CATALOGID,
    and saving the merged data to data/processed/Webserver_OS_Mapping.csv.
    """
    start_time = time.time()
    logger.info("Starting webserver-OS mapping processing...")
    logger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")

    # Read webserver data
    logger.info("\nReading webserver data file...")
    webserver_df = pd.read_csv('data/raw/WebServer.csv', encoding='utf-8')
    # Clean column names
    webserver_df.columns = webserver_df.columns.str.strip().str.replace('"', '').str.replace(' ', '')
    logger.info(f"WebServer columns: {list(webserver_df.columns)}")
    logger.info(f"Read data/raw/WebServer.csv with utf-8 encoding.")
    logger.info("\nRaw values in webserver_data.csv:")
    logger.info(webserver_df.head())

    # Read PCat data
    logger.info("\nReading PCat data...")
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    pcat_path = os.path.join(current_dir, 'data/raw/PCat.csv')
    logger.info(f"Absolute path to PCat.csv: {pcat_path}")
    pcat_df = pd.read_csv(pcat_path, quotechar='"', skipinitialspace=True, dtype=str)
    # Clean column names
    pcat_df.columns = pcat_df.columns.str.strip().str.replace('"', '').str.replace(' ', '')
    logger.info(f"PCat columns: {list(pcat_df.columns)}")
    logger.info(f"Read {len(pcat_df)} lines from PCat.csv")
    logger.info("\nPCat data sample:")
    logger.info(pcat_df.head())

    # Log unique CATALOGIDs in webserver data
    logger.info(f"\nUnique CATALOGIDs in webserver data: {webserver_df['CATALOGID'].nunique()}")

    # Ensure CATALOGID is string in both dataframes
    webserver_df['CATALOGID'] = webserver_df['CATALOGID'].astype(str)
    pcat_df['CATALOGID'] = pcat_df['CATALOGID'].astype(str)
    # Merge webserver data with PCat data
    final_df = pd.merge(webserver_df, pcat_df, on='CATALOGID', how='left')

    # Save the merged data
    output_path = 'data/processed/Webserver_OS_Mapping.csv'
    final_df.to_csv(output_path, index=False)
    logger.info(f"\nSaving {len(final_df)} records to {output_path}...")

    # Log data summary
    logger.info(f"Data processing completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    logger.info("\nData Summary:")
    logger.info(f"Total Records in Webserver Data: {len(webserver_df)}")
    logger.info(f"Total Records in PCat Data: {len(pcat_df)}")
    logger.info(f"Total Records in Webserver-OS Mapping: {len(final_df)}")
    logger.info(f"Matched Records: {len(final_df.dropna(subset=['CATALOGID']))}")

    # Validate merge
    logger.info("\nValidating merge...")
    unmatched = final_df[final_df['CATALOGID'].isna()]
    logger.info(f"Unmatched records: {len(unmatched)}")
    if len(unmatched) > 0:
        logger.info("Unmatched CATALOGIDs:")
        logger.info(unmatched[['CATALOGID', 'ASSETNAME']].head())
    else:
        logger.info("All records matched successfully.")

if __name__ == "__main__":
    process_webserver_os_mapping() 