import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> None:
    """Validate the input data structure and content."""
    # Clean column names by removing spaces and quotes
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    required_columns = ['ASSETNAME', 'CATALOGID', 'INVNO', 'OSIINVNO']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("Input dataframe is empty")

def clean_text(text: Any) -> str:
    """Clean and standardize text values."""
    if pd.isna(text):
        return ""
    try:
        text = str(text).strip().replace('"', '')
        return text
    except Exception as e:
        logger.warning(f"Error cleaning text '{text}': {str(e)}")
        return str(text)

def process_webserver_data(webserver_file: str = 'data/raw/WebServer.csv',
                         mapping_file: str = 'data/raw/swecomponentmapping.csv',
                         output_file: str = 'data/processed/WebServer_Merged.csv') -> Dict[str, Any]:
    """Process web server data and map software installations."""
    start_time = time.time()
    logger.info("Starting web server data processing...")
    
    try:
        # Read data
        webserver_df = pd.read_csv(webserver_file)
        mapping_df = pd.read_csv(mapping_file)
        logger.info(f"Read {len(webserver_df)} records from {webserver_file}")
        logger.info(f"Read {len(mapping_df)} records from {mapping_file}")
        
        # Clean column names
        webserver_df.columns = webserver_df.columns.str.strip().str.replace('"', '')
        mapping_df.columns = mapping_df.columns.str.strip().str.replace('"', '')
        
        # Convert ID columns to string type for consistent matching
        id_columns = ['CATALOGID', 'INVNO', 'OSIINVNO']
        for col in id_columns:
            if col in webserver_df.columns:
                webserver_df[col] = webserver_df[col].astype(str).apply(clean_text)
            if col in mapping_df.columns:
                mapping_df[col] = mapping_df[col].astype(str).apply(clean_text)
        
        # Validate data
        validate_data(webserver_df)
        
        # Clean text fields
        text_columns = ['ASSETNAME', 'CATALOGID', 'INVNO', 'OSIINVNO']
        for col in text_columns:
            if col in webserver_df.columns:
                webserver_df[col] = webserver_df[col].apply(clean_text)
        
        # Clean mapping data
        mapping_df['CATALOGID'] = mapping_df['CATALOGID'].apply(clean_text)
        mapping_df['INVNO'] = mapping_df['INVNO'].apply(clean_text)
        mapping_df['SWNAME'] = mapping_df['SWNAME'].apply(clean_text)
        
        # Log the first few rows of both dataframes for debugging
        logger.info("\nFirst few rows of webserver_df:")
        logger.info(webserver_df[['CATALOGID', 'INVNO', 'OSIINVNO']].head())
        logger.info("\nFirst few rows of mapping_df:")
        logger.info(mapping_df[['CATALOGID', 'INVNO', 'SWNAME']].head())
        
        # Map software installations
        # First, map by CATALOGID
        catalog_mapping = mapping_df.set_index('CATALOGID')['SWNAME'].to_dict()
        webserver_df['INSTALLED_SOFTWARE'] = webserver_df['CATALOGID'].map(catalog_mapping)
        
        # Then, map by INVNO
        invno_mapping = mapping_df.set_index('INVNO')['SWNAME'].to_dict()
        webserver_df['INVNO_SOFTWARE'] = webserver_df['INVNO'].map(invno_mapping)
        
        # Finally, map by OSIINVNO (using INVNO as key)
        osiinvno_mapping = mapping_df.set_index('INVNO')['SWNAME'].to_dict()
        webserver_df['OSIINVNO_SOFTWARE'] = webserver_df['OSIINVNO'].map(osiinvno_mapping)
        
        # Log the mapping results for debugging
        logger.info("\nMapping results for first few rows:")
        logger.info(webserver_df[['CATALOGID', 'INVNO', 'OSIINVNO', 'INSTALLED_SOFTWARE', 'INVNO_SOFTWARE', 'OSIINVNO_SOFTWARE']].head())
        
        # Combine all software installations, excluding None and empty strings
        webserver_df['ALL_INSTALLED_SOFTWARE'] = webserver_df.apply(
            lambda row: list(filter(None, set([
                row['INSTALLED_SOFTWARE'],
                row['INVNO_SOFTWARE'],
                row['OSIINVNO_SOFTWARE']
            ]))),
            axis=1
        )
        
        # Drop unnecessary columns
        columns_to_keep = ['ASSETNAME', 'CATALOGID', 'INVNO', 'OSIINVNO', 'ALL_INSTALLED_SOFTWARE']
        webserver_df = webserver_df[columns_to_keep]
        
        # Save processed data
        webserver_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Data processing completed in {processing_time:.2f} seconds")
        
        # Return statistics
        software_counts = {}
        for software_list in webserver_df['ALL_INSTALLED_SOFTWARE']:
            for software in software_list:
                if software and not pd.isna(software):  # Only count non-empty, non-null values
                    software_counts[software] = software_counts.get(software, 0) + 1
        
        # Calculate mapping coverage
        total_mappings = len(webserver_df)
        successful_mappings = sum(1 for x in webserver_df['ALL_INSTALLED_SOFTWARE'] if len(x) > 0)
        mapping_coverage = (successful_mappings / total_mappings) * 100
        
        return {
            'total_records': len(webserver_df),
            'software_installation_counts': software_counts,
            'mapping_coverage': mapping_coverage,
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results = process_webserver_data()
        
        # Print statistics
        logger.info("\nData Summary:")
        logger.info(f"Total Records: {results['total_records']}")
        logger.info(f"Mapping Coverage: {results['mapping_coverage']:.1f}%")
        logger.info("\nSoftware Installation Distribution:")
        for software, count in sorted(results['software_installation_counts'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{software}: {count} installations")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise 