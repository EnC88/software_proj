import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_processing.data_processing_webserver import process_webserver_data

if __name__ == "__main__":
    try:
        df = process_webserver_data()
        print("\n=== DATA SUMMARY ===")
        print(f"Total Records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        key_columns = ['ENVIRONMENT', 'MANUFACTURER', 'PRODUCTCLASS', 'STATUS']
        for col in key_columns:
            if col in df.columns:
                print(f"\nValue counts for {col}:")
                print(df[col].value_counts())
        print("\nFirst 3 rows:")
        print(df.head(3))
    except Exception as e:
        print(f"Error: {str(e)}")
        raise 