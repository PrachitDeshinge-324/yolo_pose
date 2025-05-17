import pandas as pd
import json
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import defaultdict
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Map track IDs to person identities")
    parser.add_argument("--features", type=str, required=True, 
                       help="Path to gait features CSV file")
    parser.add_argument("--output", type=str, default="person_identities.json",
                       help="Output JSON file for identity mappings")
    parser.add_argument("--video", type=str, default="../Person_New/input/3c.mp4",
                       help="Original video file to extract sample frames")
    parser.add_argument("--frames_dir", type=str, default="identity_samples",
                       help="Directory to save sample frames")
    parser.add_argument("--save_sequences", action="store_true",
                       help="Save keypoint sequences for each identity")
    parser.add_argument("--sequences_dir", type=str, default="identity_sequences",
                       help="Directory to save identity sequences")
    parser.add_argument("--show_plots", action="store_true",
                       help="Show feature plots for each track (requires GUI)")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of sample frames to show for each track")
    # Add to the argument parser:
    parser.add_argument("--crops_dir", type=str, default="person_crops",
                    help="Directory to find saved person crops")
    parser.add_argument("--use_saved_crops", action="store_true",
                    help="Use pre-saved crops instead of extracting new frames")
    return parser.parse_args()

def extract_frame_samples(video_path, track_data, frames_dir, num_samples=3):
    """Extract sample frames where a track appears"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []
    
    # Create directory for frame samples if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get frame range for the track
    min_frame = int(track_data['frame_idx'].min())
    max_frame = int(track_data['frame_idx'].max())
    
    # Select frames to sample
    total_frames = max_frame - min_frame + 1
    if total_frames <= num_samples:
        sample_frames = list(range(min_frame, max_frame + 1))
    else:
        # Evenly space the sample frames
        step = total_frames // num_samples
        sample_frames = [min_frame + i * step for i in range(num_samples)]
    
    # Get track ID
    track_id = int(track_data['track_id'].iloc[0])
    
    # Open the video and extract frames
    sample_paths = []
    cap = cv2.VideoCapture(video_path)
    
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Save the frame
            frame_path = os.path.join(frames_dir, f"track_{track_id}_frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            sample_paths.append(frame_path)
    
    cap.release()
    return sample_paths

def display_track_summary(df_track, track_id, show_plots=False, 
                         video_path=None, frames_dir=None, num_samples=3,
                         crops_dir=None, use_saved_crops=False):
    """Display a summary of track information to help with identification"""
    print(f"\n===== Track ID: {track_id} =====")
    
    # Extract and display sample frames/crops
    sample_paths = []
    
    # Check for saved crops first if requested
    if use_saved_crops and crops_dir and os.path.exists(crops_dir):
        # Look for any crops for this track ID
        crop_pattern = f"track_{track_id}_frame_*.jpg"
        crop_files = glob.glob(os.path.join(crops_dir, crop_pattern))
        
        if crop_files:
            # Sort by frame number
            crop_files.sort(key=lambda x: int(os.path.basename(x).split("_frame_")[1].split(".")[0]))
            
            # Select a subset if there are too many
            if len(crop_files) > num_samples:
                indices = np.linspace(0, len(crop_files)-1, num_samples, dtype=int)
                sample_paths = [crop_files[i] for i in indices]
            else:
                sample_paths = crop_files
                
            print(f"Found {len(sample_paths)} saved crops for track {track_id}")
    
    # Fall back to extracting frames if no crops found or not using saved crops
    if not sample_paths and video_path and frames_dir:
        sample_paths = extract_frame_samples(video_path, df_track, frames_dir, num_samples)
        if sample_paths:
            print(f"Extracted {len(sample_paths)} sample frames from video")
    
    # Display the frames if show_plots is enabled and we have samples
    if show_plots and sample_paths:
        plt.figure(figsize=(15, 5))
        for i, img_path in enumerate(sample_paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
            plt.subplot(1, len(sample_paths), i+1)
            plt.imshow(img)
            
            # Get frame number from filename
            frame_num = os.path.basename(img_path).split('_frame_')[1].split('.')[0]
            plt.title(f"Frame {frame_num}")
            plt.axis('off')
        
        plt.suptitle(f"Track ID {track_id} {'Crops' if 'crops' in sample_paths[0] else 'Frames'}")
        plt.tight_layout()
        plt.show()

def merge_identity_tracks(features_df, identities):
    """
    Group tracks by identity, creating merged data for each person
    Returns a dictionary mapping person names to lists of track IDs
    """
    identity_tracks = defaultdict(list)
    
    # Group track IDs by identity
    for track_id, name in identities.items():
        if name:  # Skip empty names
            identity_tracks[name].append(int(track_id))
    
    return identity_tracks

def save_identity_sequences(features_df, identity_tracks, sequences_dir):
    """Save keypoint sequences for each identity"""
    if not os.path.exists(sequences_dir):
        os.makedirs(sequences_dir, exist_ok=True)
    
    # Check if we have keypoint data
    keypoint_cols = [col for col in features_df.columns if col.startswith('keypoint_')]
    if not keypoint_cols:
        print("No keypoint data found in features file. Cannot save sequences.")
        return False
    
    # Save sequences for each identity
    for identity, track_ids in identity_tracks.items():
        # Get all frames for this identity
        identity_data = features_df[features_df['track_id'].isin(track_ids)]
        
        if len(identity_data) == 0:
            continue
        
        # Sort by frame index
        identity_data = identity_data.sort_values('frame_idx')
        
        # Extract keypoints
        keypoints_data = identity_data[keypoint_cols].values
        
        # Save to file
        sequence_path = os.path.join(sequences_dir, f"{identity}_sequence.npy")
        np.save(sequence_path, keypoints_data)
        
        print(f"Saved {len(keypoints_data)} frames of keypoint data for {identity}")
    
    return True

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
        display_track_summary(
            df_track, track_id, 
            show_plots=args.show_plots,
            video_path=args.video,
            frames_dir=args.frames_dir,
            num_samples=args.num_samples,
            crops_dir=args.crops_dir,
            use_saved_crops=args.use_saved_crops
        )
        
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
    
    # Group tracks by identity
    identity_tracks = merge_identity_tracks(features_df, identities)
    
    # Print identity statistics
    print("\nIdentity summary:")
    for name, track_ids in identity_tracks.items():
        print(f"- {name}: {len(track_ids)} tracks ({', '.join(map(str, track_ids))})")
    
    # Save sequences if requested
    if args.save_sequences:
        print("\nSaving identity sequences...")
        save_identity_sequences(features_df, identity_tracks, args.sequences_dir)
    
    # Save mappings
    with open(args.output, 'w') as f:
        json.dump(identities, f, indent=2)
    
    print(f"\nSaved {len(identities)} identity mappings to {args.output}")

if __name__ == "__main__":
    main()