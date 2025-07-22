import pandas as pd
import logging
import os 
import time
import psutil
import unicodedata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_sor_history():
    start_time = time.time()
    logger.info("Starting SOR history processing...")
    logger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    try:
        logger.info("\nReading SOR history file...")
        mapping_path = os.path.join('data', 'processed', 'pcat.csv')
        platform_df = pd.read_csv(
            mapping_path, 
            encoding='latin1',
            quotechar='"',
            skipinitialspace = True
            )
        logger.info(f"Read {mapping_path} with latin1 encoding.")
        
        platform_df.columns = platform_df.columns.str.strip().str.replace('"', '').str.replace(' ', '')
        # Normalize platform_df['CATALOGID'] once outside the chunk loop
        platform_df['CATALOGID'] = platform_df['CATALOGID'].astype(str).str.strip().str.upper().apply(lambda x: unicodedata.normalize('NFKC', x))
        
        logger.info("Reading SOR history file in chunks...")
        sor_hist_path = os.path.join('data', 'processed', 'sor_hist.csv')
        chunk_size = 100000
        sor_hist_chunks = pd.read_csv(
            sor_hist_path,
            encoding='latin1',
            quotechar='"',
            skipinitialspace = True,
            chunksize = chunk_size,
            on_bad_lines = 'warn'
        )
        processed_chunks = []
        for i, chunk in enumerate(sor_hist_chunks):
            logger.info(f"Processing chunk {i+1}")
            chunk.columns = chunk.columns.str.strip().str.replace('"', '')
            columns_to_drop = ['VERUMLASTMODIFIEDBY', 'VERUMMODIFIEDDATE','VERUMSTATUS', 'VERUMRETIREDDATE']
            chunk = chunk.drop(columns=columns_to_drop, errors='ignore')
            # Normalize chunk['NEWVALUE'] and chunk['OLDVALUE'] inside the loop
            chunk['NEWVALUE'] = chunk['NEWVALUE'].astype(str).str.strip().str.upper().apply(lambda x: unicodedata.normalize('NFKC', x))
            chunk['OLDVALUE'] = chunk['OLDVALUE'].astype(str).str.strip().str.upper().apply(lambda x: unicodedata.normalize('NFKC', x))
            logger.debug(f"Chunk {i+1} head:\n{chunk.head()}")

            # Debug: print the head of the chunk
            print(f"\n--- Chunk {i+1} head ---")
            print(chunk.head())

            catalogid_rows = chunk[chunk['ATTRIBUTENAME'] == 'CATALOGID'].copy()
            catalogid_rows['OLDVALUE'] = catalogid_rows['OLDVALUE'].apply(lambda x: x if pd.notna(x) else 'UNKNOWN')
            catalogid_rows['OLDVALUE'] = catalogid_rows['OLDVALUE'].astype(str).str.strip().str.upper().apply(lambda x: unicodedata.normalize('NFKC', x))
            catalogid_rows['NEWVALUE'] = catalogid_rows['NEWVALUE'].astype(str).str.strip().str.upper().apply(lambda x: unicodedata.normalize('NFKC', x))
            platform_df['CATALOGID'] = platform_df['CATALOGID'].astype(str).str.strip().str.upper().apply(lambda x: unicodedata.normalize('NFKC', x))

            # Merge for OLD_MAPPED and OLD_PRODUCTTYPE
            old_mapped_catalogid = catalogid_rows[['OLDVALUE']].merge(
                platform_df[['CATALOGID', 'MODEL', 'PRODUCTTYPE']],
                how='left',
                left_on='OLDVALUE',
                right_on='CATALOGID'
            ).rename(columns={'MODEL':'OLD_MAPPED', 'PRODUCTTYPE': 'OLD_PRODUCTTYPE'})

            # Merge for NEW_MAPPED and NEW_PRODUCTTYPE
            new_mapped_catalogid = catalogid_rows[['NEWVALUE']].merge(
                platform_df[['CATALOGID', 'MODEL', 'PRODUCTTYPE']],
                how='left',
                left_on='NEWVALUE',
                right_on='CATALOGID'
            ).rename(columns={'MODEL': 'NEW_MAPPED', 'PRODUCTTYPE': 'NEW_PRODUCTTYPE'})

            # Assign OLD mappings
            catalogid_rows['OLD_MAPPED'] = old_mapped_catalogid['OLD_MAPPED'].values
            catalogid_rows['OLD_PRODUCTTYPE'] = old_mapped_catalogid['OLD_PRODUCTTYPE'].values

            # Assign NEW mappings
            catalogid_rows['NEW_MAPPED'] = new_mapped_catalogid['NEW_MAPPED'].values
            catalogid_rows['NEW_PRODUCTTYPE'] = new_mapped_catalogid['NEW_PRODUCTTYPE'].values

            print(f"Old mapped catalogid head:\n{old_mapped_catalogid.head()}")
            print(f"New mapped catalogid head:\n{new_mapped_catalogid.head()}")

            platform_df['PRODUCTTYPE'] = platform_df['PRODUCTTYPE'].astype(str).str.strip().str.upper().apply(lambda x: unicodedata.normalize('NFKC', x))

            # Debug: print the head of the CATALOGID mapping (after assignment)
            print(f"\n--- Chunk {i+1} CATALOGID mapping head ---")
            print(catalogid_rows[['OLDVALUE', 'OLD_MAPPED', 'OLD_PRODUCTTYPE', 'NEWVALUE', 'NEW_MAPPED', 'NEW_PRODUCTTYPE']].head())

            model_rows = chunk[chunk['ATTRIBUTENAME'] == 'MODEL'].copy()
            model_rows.loc[:, 'OLD_MAPPED'] = model_rows['OLDVALUE']
            model_rows.loc[:, 'NEW_MAPPED'] = model_rows['NEWVALUE']

            model_rows['OLDPRODUCTTYPE'] = None
            model_rows['NEWPRODUCTTYPE'] = None

            chunk.loc[catalogid_rows.index, ['OLD_MAPPED', 'NEW_MAPPED', 'OLD_PRODUCTTYPE', 'NEW_PRODUCTTYPE']] = catalogid_rows[['OLD_MAPPED', 'NEW_MAPPED', 'OLD_PRODUCTTYPE', 'NEW_PRODUCTTYPE']]
            chunk.loc[model_rows.index, ['OLD_MAPPED', 'NEW_MAPPED', 'OLD_PRODUCTTYPE', 'NEW_PRODUCTTYPE']] = model_rows[['OLD_MAPPED', 'NEW_MAPPED', 'OLD_PRODUCTTYPE', 'NEW_PRODUCTTYPE']]

            processed_chunks.append(chunk)
        sor_hist_df = pd.concat(processed_chunks, ignore_index=True)

        os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
        output_path = os.path.join('data', 'processed', 'sor_pcat_mapped.csv')
        logger.info(f"\n Saving {len(sor_hist_df)} rows to {output_path}")
        sor_hist_df.to_csv(output_path, index=False, quoting=1)

        end_time = time.time()
        logger.info(f"Data processing completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    except Exception as e:
        logger.error(f"Error processing SOR history: {e}")
        raise

if __name__ == "__main__":
    process_sor_history()