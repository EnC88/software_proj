import pandas as pd
import numpy as np
import re
import logging
from typing import Any, Optional
from dateutil import parser
from packaging import version

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

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

def remove_empty_columns(df: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
    """Remove empty columns and specific columns that are not needed."""
    try:
        df = df.replace("", np.nan)
        df = df.dropna(axis=1, how="all")
        
        if columns_to_drop:
            existing_columns = [col for col in columns_to_drop if col in df.columns]
            if existing_columns:
                df.drop(columns=existing_columns, inplace=True)
        return df
    except Exception as e:
        raise DataProcessingError(f"Error removing empty columns: {str(e)}")

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