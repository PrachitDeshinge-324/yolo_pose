import pandas as pd
import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Map track IDs to person identities")
    parser.add_argument("--features", type=str, required=True, 
                       help="Path to gait features CSV file")
    parser.add_argument("--output", type=str, default="person_identities.json",
                       help="Output JSON file for identity mappings")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load features file
    if not os.path.exists(args.features):
        print(f"Error: File {args.features} not found")
        return
    
    features_df = pd.read_csv(args.features)
    track_ids = features_df['track_id'].unique()
    
    print(f"Found {len(track_ids)} unique track IDs in the features file")
    
    # Load existing mappings if available
    identities = {}
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            try:
                identities = json.load(f)
                # Convert string keys to integers
                identities = {int(k): v for k, v in identities.items()}
                print(f"Loaded {len(identities)} existing identity mappings")
            except json.JSONDecodeError:
                print("Error reading existing identity file. Starting fresh.")
    
    # Interactive mapping
    print("\nEnter person names for each track ID (or leave blank to skip):")
    for track_id in sorted(track_ids):
        if track_id in identities:
            current = identities[track_id]
            new_name = input(f"Track ID {track_id} (currently '{current}'): ")
            if new_name:
                identities[track_id] = new_name
        else:
            name = input(f"Track ID {track_id}: ")
            if name:
                identities[track_id] = name
    
    # Save mappings
    with open(args.output, 'w') as f:
        json.dump(identities, f, indent=2)
    
    print(f"\nSaved {len(identities)} identity mappings to {args.output}")

if __name__ == "__main__":
    main()