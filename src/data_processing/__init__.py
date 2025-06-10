"""
Data processing package for software compatibility project.
Contains utilities and processors for both Verum and WebServer data.
"""

from .data_utils import (
    DataValidationError,
    DataProcessingError,
    clean_text,
    remove_empty_columns,
    standardize_date,
    parse_version
)

from src.data_processing.data_processing import process_data as process_verum_data
from src.data_processing.data_processing_webserver import process_data as process_webserver_data

__all__ = [
    'DataValidationError',
    'DataProcessingError',
    'clean_text',
    'remove_empty_columns',
    'standardize_date',
    'parse_version',
    'process_verum_data',
    'process_webserver_data'
] 