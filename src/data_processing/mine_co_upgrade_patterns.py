import pandas as pd
import json
from collections import defaultdict

def mine_co_upgrade_patterns(df, time_window_days=7):
    # Build mapping from catalog ID to model name
    catalogid_to_model = df.set_index('NEWVALUE')['NEW_MAPPED'].to_dict()
    co_upgrade_counts = defaultdict(lambda: defaultdict(int))
    # Parse dates after cleaning
    df['VERUMCREATEDDATE'] = pd.to_datetime(
        df['VERUMCREATEDDATE'],
        format='%y-%b-%d %I.%M.%S.%f %p %Z',
        errors='coerce'
    )
    df = df.dropna(subset=['VERUMCREATEDDATE'])
    # Debug: print a sample of the preprocessed data
    print("Sample of preprocessed data:")
    print(df[['VERUMCREATEDBY', 'OLDVALUE', 'NEWVALUE', 'NEW_MAPPED', 'VERUMCREATEDDATE']].head(20))
    print("\nTop NEWVALUE values:")
    print(df['NEWVALUE'].value_counts().head(10))
    # Pre-sort for efficient join
    df = df.sort_values(['VERUMCREATEDBY', 'OLDVALUE', 'VERUMCREATEDDATE'])
    # Group by user and asset
    group_count = 0
    multi_event_groups = 0
    MAX_GROUP_SIZE = 2000
    for (user, asset), group in df.groupby(['VERUMCREATEDBY', 'OLDVALUE']):
        group_count += 1
        if len(group) > MAX_GROUP_SIZE:
            print(f"Truncating group with {len(group)} events to {MAX_GROUP_SIZE} (User: {user}, Asset: {asset})")
            group = group.head(MAX_GROUP_SIZE)
        if len(group) > 1:
            multi_event_groups += 1
            print(f"User: {user}, Asset: {asset}, Events: {len(group)}")
        times = group['VERUMCREATEDDATE'].values
        prods = group['NEWVALUE'].astype(str).values
        n = len(group)
        left = 0
        for right in range(n):
            if right % 10000 == 0 and right > 0:
                print(f"  Processed {right} events in current group (User: {user}, Asset: {asset})")
            while times[right] - times[left] > pd.Timedelta(days=time_window_days):
                left += 1
            for i in range(left, right):
                if prods[i] != prods[right]:
                    co_upgrade_counts[prods[right]][prods[i]] += 1
                    co_upgrade_counts[prods[i]][prods[right]] += 1  # symmetric
    print(f"\nTotal groups: {group_count}, Groups with >1 event: {multi_event_groups}")
    # Filter out co-upgrade pairs that only occur once
    print("Starting filtering of co-upgrade pairs...")
    for prod in list(co_upgrade_counts.keys()):
        co_upgrade_counts[prod] = {k: v for k, v in co_upgrade_counts[prod].items() if v >= 2}
    print("Finished filtering.")
    # Build structured output
    output = {}
    for prod, co_dict in co_upgrade_counts.items():
        model = catalogid_to_model.get(prod, "Unknown")
        output[prod] = {
            "model": model,
            "co_upgrades": {
                co_prod: {"model": catalogid_to_model.get(co_prod, "Unknown"), "count": count}
                for co_prod, count in co_dict.items()
            }
        }
    return output

def main():
    df = pd.read_csv('data/processed/Change_History.csv')
    df = df.head(200)  # Only use the first 200 rows for testing
    print("Raw columns:", df.columns.tolist())
    print("First 5 rows (raw):")
    print(df.head())
    # Standardize column names and remove quotes
    df.columns = df.columns.str.strip().str.replace('"', '').str.upper()
    print("\nColumns after standardization:", df.columns.tolist())
    # Strip quotes from all string/object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.replace('"', '')
    # Print a sample of the date column before parsing
    print("Sample VERUMCREATEDDATE values:", df['VERUMCREATEDDATE'].head(10).tolist())
    co_upgrade_patterns = mine_co_upgrade_patterns(df, time_window_days=7)
    print("Starting JSON dump...")
    with open('data/processed/co_upgrade_patterns.json', 'w') as f:
        json.dump(co_upgrade_patterns, f, indent=2)
    print("Finished JSON dump.")
    print("Co-upgrade patterns saved to data/processed/co_upgrade_patterns.json")

if __name__ == "__main__":
    main() 