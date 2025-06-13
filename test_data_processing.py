import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can import from src.data_processing
from src.data_processing.data_processing import process_data

if __name__ == "__main__":
    try:
        results = process_data()
        print("\n=== DATA SUMMARY ===")
        print(f"Total Records: {results['total_records']}")
        print("\nEnvironment Distribution:")
        for env, count in results['environment_counts'].items():
            print(f"{env}: {count}")
        print("\nSoftware Distribution:")
        for software, count in results['software_counts'].items():
            print(f"{software}: {count}")
        print("\nManufacturer Distribution:")
        for manufacturer, count in results['manufacturer_counts'].items():
            print(f"{manufacturer}: {count}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise 