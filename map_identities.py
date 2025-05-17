import pandas as pd
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # You may need to install this: pip install tabulate

def parse_args():
    parser = argparse.ArgumentParser(description="Map track IDs to person identities")
    parser.add_argument("--features", type=str, required=True, 
                       help="Path to gait features CSV file")
    parser.add_argument("--output", type=str, default="person_identities.json",
                       help="Output JSON file for identity mappings")
    parser.add_argument("--show_plots", action="store_true",
                       help="Show feature plots for each track (requires GUI)")
    return parser.parse_args()

def display_track_summary(df_track, track_id, show_plots=False):
    """Display a summary of track information to help with identification"""
    print(f"\n===== Track ID: {track_id} =====")
    
    # Display frame range where this track appears
    if 'frame_idx' in df_track.columns:
        min_frame = df_track['frame_idx'].min()
        max_frame = df_track['frame_idx'].max()
        print(f"Appears in frames: {min_frame} to {max_frame} (duration: {max_frame - min_frame + 1})")
    
    # Display movement direction if available
    if 'movement_direction' in df_track.columns:
        directions = df_track['movement_direction'].value_counts()
        main_direction = directions.index[0] if not directions.empty else "unknown"
        print(f"Primary movement direction: {main_direction}")
    
    # Show most distinctive features for identification
    key_features = [
        'height_width_ratio', 'leg_torso_ratio', 'upper_lower_leg_ratio',
        'arm_swing_symmetry', 'spine_angle_mean', 'stride_length_mean',
        'stride_frequency', 'walking_speed'
    ]
    
    # Filter to keep only columns that exist in the dataframe
    valid_features = [f for f in key_features if f in df_track.columns]
    
    if valid_features:
        # Calculate mean values for numerical features
        feature_data = []
        for feature in valid_features:
            if pd.api.types.is_numeric_dtype(df_track[feature]):
                mean_val = df_track[feature].mean()
                feature_data.append([feature, f"{mean_val:.4f}"])
        
        # Display as a table
        print("\nDistinctive features:")
        print(tabulate(feature_data, headers=["Feature", "Mean Value"], tablefmt="simple"))
    
    # Plot some key features if requested
    if show_plots and len(df_track) > 5:
        try:
            plt.figure(figsize=(12, 8))
            
            # Select a subset of columns to plot
            plot_columns = [col for col in df_track.columns 
                            if pd.api.types.is_numeric_dtype(df_track[col]) 
                            and col != 'track_id' 
                            and col != 'frame_idx'
                            and not df_track[col].isnull().all()][:6]  # Limit to 6 features
            
            if plot_columns:
                for i, col in enumerate(plot_columns):
                    if len(plot_columns) > 1:
                        plt.subplot(2, 3, i+1)
                    plt.plot(df_track[col].values)
                    plt.title(col)
                    plt.tight_layout()
                
                plt.suptitle(f"Track ID {track_id} Features")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()
        except Exception as e:
            print(f"Couldn't plot features: {e}")

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
    print("\nDisplaying sample data for each track ID to help with identification")
    for track_id in sorted(track_ids):
        # Convert NumPy int64 to native Python int
        track_id = int(track_id)
        
        # Display summary information for this track
        df_track = features_df[features_df['track_id'] == track_id]
        display_track_summary(df_track, track_id, show_plots=args.show_plots)
        
        # Ask for identity
        if track_id in identities:
            current = identities[track_id]
            new_name = input(f"Track ID {track_id} (currently '{current}'): ")
            if new_name:
                identities[track_id] = new_name
        else:
            name = input(f"Track ID {track_id}: ")
            if name:
                identities[track_id] = name
    
    # Convert any remaining NumPy int64 keys to Python int
    identities = {int(k): v for k, v in identities.items()}
    
    # Save mappings
    with open(args.output, 'w') as f:
        json.dump(identities, f, indent=2)
    
    print(f"\nSaved {len(identities)} identity mappings to {args.output}")

if __name__ == "__main__":
    main()