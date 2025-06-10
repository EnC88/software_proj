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
    # Clean column names by removing spaces
    df.columns = df.columns.str.strip()
    
    required_columns = ['ASSETNAME', 'MANUFACTURER', 'MODEL', 'ENVIRONMENT', 
                       'INSTALLPATH', 'INSTANCENAME', 'STATUS', 'SUBSTATUS', 
                       'PRODUCTCLASS', 'PRODUCTTYPE']
    
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
        text = str(text).strip()
        return text
    except Exception as e:
        logger.warning(f"Error cleaning text '{text}': {str(e)}")
        return str(text)

def process_webserver_data(input_file: str = 'data/raw/WebServer.csv', 
                         output_file: str = 'data/processed/WebServer_Merged.csv') -> Dict[str, Any]:
    """Process web server data and return statistics."""
    start_time = time.time()
    logger.info("Starting web server data processing...")
    
    try:
        # Read data
        df = pd.read_csv(input_file)
        logger.info(f"Read {len(df)} records from {input_file}")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Validate data
        validate_data(df)
        
        # Clean text fields
        text_columns = ['ASSETNAME', 'MANUFACTURER', 'MODEL', 'ENVIRONMENT', 
                       'INSTALLPATH', 'INSTANCENAME', 'STATUS', 'SUBSTATUS', 
                       'PRODUCTCLASS', 'PRODUCTTYPE']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
        
        # Save processed data
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Data processing completed in {processing_time:.2f} seconds")
        
        # Return statistics
        return {
            'total_records': len(df),
            'environment_counts': df['ENVIRONMENT'].value_counts().to_dict(),
            'manufacturer_counts': df['MANUFACTURER'].value_counts().to_dict(),
            'product_class_counts': df['PRODUCTCLASS'].value_counts().to_dict(),
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
        logger.info("\nEnvironment Distribution:")
        for env, count in results['environment_counts'].items():
            logger.info(f"{env}: {count}")
        logger.info("\nManufacturer Distribution:")
        for manufacturer, count in results['manufacturer_counts'].items():
            logger.info(f"{manufacturer}: {count}")
        logger.info("\nProduct Class Distribution:")
        for product_class, count in results['product_class_counts'].items():
            logger.info(f"{product_class}: {count}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise 