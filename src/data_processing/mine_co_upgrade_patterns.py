import pandas as pd
import json

def mine_co_upgrade_patterns(df, time_window_days=1):
    """
    Find products that are often upgraded together on the same server within a time window.
    Returns a dict: {product: {co_product: count, ...}, ...}
    """
    from collections import defaultdict
    co_upgrade_counts = defaultdict(lambda: defaultdict(int))
    # Only consider version changes
    upgrades = df[df['IS_VERSION_CHANGE']].copy()
    upgrades['VERUMCREATEDDATE'] = pd.to_datetime(upgrades['VERUMCREATEDDATE'])
    # Group by server and user
    for (server, user), group in upgrades.groupby(['CATALOGID', 'VERUMCREATEDBY']):
        group = group.sort_values('VERUMCREATEDDATE')
        for idx, row in group.iterrows():
            prod = row['NEW_SWNAME']
            time = row['VERUMCREATEDDATE']
            # Find other upgrades within the time window
            window = group[(group['VERUMCREATEDDATE'] >= time) &
                           (group['VERUMCREATEDDATE'] <= time + pd.Timedelta(days=time_window_days))]
            for _, co_row in window.iterrows():
                co_prod = co_row['NEW_SWNAME']
                if co_prod != prod:
                    co_upgrade_counts[prod][co_prod] += 1
    return {k: dict(v) for k, v in co_upgrade_counts.items()}

def main():
    # Load your processed SOR history
    df = pd.read_csv('data/processed/Change_History.csv')
    # Mine co-upgrade patterns
    co_upgrade_patterns = mine_co_upgrade_patterns(df)
    # Save to JSON
    with open('data/processed/co_upgrade_patterns.json', 'w') as f:
        json.dump(co_upgrade_patterns, f, indent=2)
    print("Co-upgrade patterns saved to data/processed/co_upgrade_patterns.json")

if __name__ == "__main__":
    main() 