import pandas as pd
import json
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description="Map track IDs to person identities")
    parser.add_argument("--features", type=str, required=True, 
                       help="Path to gait features CSV file with aggregate statistics")
    parser.add_argument("--sequential", type=str, default=None,
                       help="Path to sequential NPY data file (gait_features_flat.npy)")
    parser.add_argument("--output", type=str, default="results/person_identities.json",
                       help="Output JSON file for identity mappings")
    parser.add_argument("--video", type=str, required=True,
                       help="Original video file to extract sample frames")
    parser.add_argument("--bbox_info", type=str, default="results/bbox_info.json",
                       help="JSON file containing bounding box information")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of sample frames per track ID")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory for output files")
    parser.add_argument("--show_plots", action="store_true", default=True,
                       help="Show feature plots for each track")
    parser.add_argument("--headless", action="store_true", 
                      help="Run without displaying plots or images")
    parser.add_argument("--save_sequences", action="store_true",
                       help="Save keypoint sequences for each identity")
    parser.add_argument("--merged_output", type=str, default=None,
                       help="Output path for merged numpy file")
    
    args = parser.parse_args()
    
    # Set default sequential data path if not provided
    if not args.sequential and os.path.exists(args.features.replace('.csv', '_flat.npy')):
        args.sequential = args.features.replace('.csv', '_flat.npy')
        print(f"Using sequential data from: {args.sequential}")
    
    # Override show_plots if headless is specified
    if args.headless:
        args.show_plots = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default merged output path if not specified
    if args.save_sequences and not args.merged_output:
        args.merged_output = os.path.join(args.output_dir, "gait_features_flat_merged.npy")
    
    return args

def extract_crops_from_bbox(video_path, bbox_data, track_id, num_samples=3):
    """Extract crops from original video using saved bounding box coordinates"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []
    
    if not bbox_data or str(track_id) not in bbox_data:
        print(f"No bounding box data found for track {track_id}")
        return []
    
    # Get bbox data for this track
    track_boxes = bbox_data[str(track_id)]
    
    # Select frames to sample
    if len(track_boxes) <= num_samples:
        selected_boxes = track_boxes
    else:
        # Evenly space the samples
        indices = np.linspace(0, len(track_boxes)-1, num_samples, dtype=int)
        selected_boxes = [track_boxes[i] for i in indices]
    
    # Open the video and extract crops
    crops_data = []
    cap = cv2.VideoCapture(video_path)
    
    for box_info in selected_boxes:
        frame_idx = box_info['frame_idx']
        x1, y1, x2, y2 = box_info['original_box'] if 'original_box' in box_info else [box_info['x1'], box_info['y1'], box_info['x2'], box_info['y2']]
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            crops_data.append({
                'crop': crop,
                'frame_idx': frame_idx,
                'track_id': track_id
            })
    
    cap.release()
    return crops_data

def display_track_summary(stats_df, seq_data, track_id, bbox_info, video_path, show_plots=False, num_samples=3):
    """Display a summary of track information to help with identification"""
    print(f"\n===== Track ID: {track_id} =====")
    
    # Get frame range and count from sequential data if available
    frame_range = "N/A"
    frame_count = 0
    if seq_data is not None:
        track_frames = seq_data[seq_data[:, 0] == track_id, 1]
        if len(track_frames) > 0:
            frame_range = f"{int(track_frames.min())} to {int(track_frames.max())}"
            frame_count = len(track_frames)
    
    # Get sample frames from video using bbox info
    samples = []
    if bbox_info and str(track_id) in bbox_info:
        samples = extract_crops_from_bbox(video_path, bbox_info, track_id, num_samples)
    
    # Display basic information in tabular format
    print(tabulate([
        ["Frame Range", frame_range],
        ["Frame Count", frame_count],
        ["Samples Found", len(samples)]
    ], tablefmt="grid"))
    
    # Extract feature statistics from stats_df for this track
    if stats_df is not None and not stats_df.empty:
        track_stats = stats_df[stats_df['track_id'] == track_id]
        
        if not track_stats.empty:
            # Create a table of the statistics
            track_stats_dict = track_stats.iloc[0].to_dict()
            stats_table = []
            
            for key, value in track_stats_dict.items():
                if key != 'track_id':
                    try:
                        # Try to format as float if it's numeric
                        val_str = f"{float(value):.4f}"
                    except (ValueError, TypeError):
                        # Otherwise use as is
                        val_str = str(value)
                    
                    stats_table.append([key, val_str])
            
            print("\nTrack Statistics:")
            print(tabulate(stats_table, headers=["Feature", "Value"], tablefmt="grid"))
    
    # Show samples if available and not in headless mode
    if samples and show_plots:
        fig = plt.figure(figsize=(15, 4))
        for i, sample in enumerate(samples):
            plt.subplot(1, len(samples), i+1)
            plt.imshow(cv2.cvtColor(sample['crop'], cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {sample['frame_idx']}")
            plt.axis('off')
        plt.suptitle(f"Track ID: {track_id}")
        plt.tight_layout()
        plt.show()
    
    # Show sequential feature plots if available
    if seq_data is not None and show_plots:
        track_data = seq_data[seq_data[:, 0] == track_id]
        
        if len(track_data) > 10:  # Only if we have enough data points
            # Plot keypoint trajectories or other features
            # Assuming columns after track_id and frame_idx are keypoints and features
            
            # First plot the path (x/y coordinates) if available
            # We'll look for keypoints that might represent the body center (like hip keypoints)
            keypoint_indices = range(2, min(50, track_data.shape[1]))  # Limit to first few columns
            
            plt.figure(figsize=(12, 8))
            
            # Plot trajectory (X vs. Y coordinates) if we have paired coordinates
            if len(keypoint_indices) >= 4:  # At least one x,y pair
                plt.subplot(2, 1, 1)
                
                # Try to find keypoints that might represent the body center
                for i in range(2, min(track_data.shape[1]-1, 30), 2):
                    x_coords = track_data[:, i]
                    y_coords = track_data[:, i+1]
                    
                    # Filter out invalid coordinates (zeros or extreme values)
                    valid_indices = (x_coords > 0) & (y_coords > 0) & (x_coords < 2000) & (y_coords < 2000)
                    if np.sum(valid_indices) > 10:
                        plt.plot(x_coords[valid_indices], y_coords[valid_indices], '.-', alpha=0.5, 
                                label=f"Keypoint {i//2}")
                
                plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
                plt.title("Keypoint Trajectories")
                plt.xlabel("X coordinate")
                plt.ylabel("Y coordinate")
                if (i-2)//2 < 5:  # Only show legend if we have few keypoints
                    plt.legend()
                
            # Plot keypoint vertical positions over time
            plt.subplot(2, 1, 2)
            frames = track_data[:, 1]
            
            # Plot Y coordinates for a few keypoints over time
            for i in range(3, min(track_data.shape[1], 20), 2):
                y_coords = track_data[:, i]
                valid_indices = (y_coords > 0) & (y_coords < 2000)
                if np.sum(valid_indices) > 10:
                    plt.plot(frames[valid_indices], y_coords[valid_indices], '.-', alpha=0.7, 
                            label=f"Keypoint {i//2}")
            
            plt.title("Keypoint Vertical Positions Over Time")
            plt.xlabel("Frame")
            plt.ylabel("Y coordinate")
            plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
            if (i-2)//2 < 5:  # Only show legend if we have few keypoints
                plt.legend()
                
            plt.tight_layout()
            plt.show()
    
    return True

def merge_identity_tracks(identity_tracks_dict):
    """Display summary of merged tracks by identity"""
    # Display summary of identities
    print("\n===== Identity Summary =====")
    for identity, tracks in identity_tracks_dict.items():
        print(f"Identity: {identity} -> Tracks: {tracks}")
    
    return identity_tracks_dict

def merge_tracks_in_flat_npy(flat_npy_path, identity_tracks, output_path):
    """Merge tracks in flat numpy array based on identity mapping"""
    if not os.path.exists(flat_npy_path):
        print(f"Flat numpy file not found: {flat_npy_path}")
        return False
    
    try:
        # Load the flat numpy array
        data = np.load(flat_npy_path, allow_pickle=True)
        print(f"Loaded data with shape: {data.shape}")
        
        # Create a new array with updated track IDs
        new_data = []
        for row in data:
            track_id = int(row[0])
            # Find which identity this track belongs to
            for identity, tracks in identity_tracks.items():
                if track_id in tracks:
                    # Replace track_id with identity name or number
                    # If identity is numeric, convert to int
                    try:
                        identity_id = int(identity)
                    except ValueError:
                        # Use hash of string if identity is not numeric
                        identity_id = hash(identity) % 10000  # Limit to reasonable size
                    
                    new_row = np.copy(row)
                    new_row[0] = identity_id
                    new_data.append(new_row)
                    break
            else:
                # If track not found in any identity, keep original
                new_data.append(row)
        
        # Convert to numpy array and save
        new_data = np.array(new_data)
        np.save(output_path, new_data)
        print(f"Merged data saved to: {output_path}")
        
        return True
    except Exception as e:
        print(f"Error merging tracks: {str(e)}")
        return False

def main():
    args = parse_args()
    
    # Load CSV features (statistical summaries)
    stats_df = None
    if args.features.endswith('.csv'):
        try:
            stats_df = pd.read_csv(args.features)
            print(f"Loaded CSV statistics with {len(stats_df)} entries")
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    # Load sequential data from NPY file
    seq_data = None
    if args.sequential and os.path.exists(args.sequential):
        try:
            seq_data = np.load(args.sequential, allow_pickle=True)
            print(f"Loaded sequential data with shape {seq_data.shape}")
        except Exception as e:
            print(f"Error loading sequential data: {e}")
    
    # Load bbox information if available
    bbox_info = None
    if os.path.exists(args.bbox_info):
        try:
            with open(args.bbox_info, 'r') as f:
                bbox_info = json.load(f)
            print(f"Loaded bounding box info with {len(bbox_info)} tracks")
        except Exception as e:
            print(f"Error loading bbox info: {e}")
    
    # Get unique track IDs from any available source
    unique_tracks = set()
    
    if stats_df is not None and 'track_id' in stats_df.columns:
        unique_tracks.update(stats_df['track_id'].unique())
    
    if seq_data is not None and seq_data.shape[1] > 0:
        unique_tracks.update(np.unique(seq_data[:, 0]))
    
    if bbox_info is not None:
        unique_tracks.update([int(track_id) for track_id in bbox_info.keys()])
    
    unique_tracks = sorted(unique_tracks)
    print(f"Found {len(unique_tracks)} unique tracks")
    
    # Interactive identification process
    identities = {}
    
    for track_id in unique_tracks:
        # Display track information and samples
        display_track_summary(stats_df, seq_data, track_id, bbox_info, args.video, 
                             show_plots=args.show_plots, num_samples=args.num_samples)
        
        # Ask user to identify the track
        identity = input(f"Enter identity for Track {track_id} (or 's' to skip): ")
        if identity.lower() == 's':
            continue
        
        identities[str(track_id)] = identity
    
    # Save identity mappings
    with open(args.output, 'w') as f:
        json.dump(identities, f, indent=2)
    print(f"Saved identity mappings to {args.output}")
    
    # Create identity to track mapping
    identity_tracks = {}
    for track_id, identity in identities.items():
        if identity not in identity_tracks:
            identity_tracks[identity] = []
        identity_tracks[identity].append(int(track_id))
    
    # Display identity summary
    merge_identity_tracks(identity_tracks)
    
    # Merge tracks in flat numpy array if requested
    if args.save_sequences and args.sequential and os.path.exists(args.sequential):
        if not merge_tracks_in_flat_npy(args.sequential, identity_tracks, args.merged_output):
            print("Failed to merge tracks in flat numpy array")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()