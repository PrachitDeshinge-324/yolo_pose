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
    parser.add_argument("--show_plots", action="store_true", default=True,
                       help="Show feature plots for each track (requires GUI)")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of sample frames to show for each track")
    parser.add_argument("--crops_dir", type=str, default="person_crops",
                    help="Directory to find saved person crops")
    parser.add_argument("--use_saved_crops", action="store_true", default=True,
                    help="Use pre-saved crops instead of extracting new frames")
    parser.add_argument("--bbox_info", type=str, default="bbox_info.json",
                    help="JSON file containing bounding box information")
    parser.add_argument("--use_bbox_info", action="store_true", default=True,
                    help="Use bounding box information to extract crops on-demand")
    parser.add_argument("--merged_output", type=str, default=None,
                    help="Output path for merged numpy file (default: features_flat_merged.npy)")
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
    
    # Verify that track IDs match (defensive check)
    verified_boxes = []
    for box in track_boxes:
        # If track_id is stored in the box info, verify it matches
        if 'track_id' in box and int(box['track_id']) == int(track_id):
            verified_boxes.append(box)
        # If no track_id in the box info (for backward compatibility)
        elif 'track_id' not in box:
            verified_boxes.append(box)
    
    # If we filtered out all boxes, use the original list (backward compatibility)
    if not verified_boxes and track_boxes:
        verified_boxes = track_boxes
    
    # Select frames to sample
    if len(verified_boxes) <= num_samples:
        selected_boxes = verified_boxes
    else:
        # Evenly space the samples
        indices = np.linspace(0, len(verified_boxes)-1, num_samples, dtype=int)
        selected_boxes = [verified_boxes[i] for i in indices]
    
    # Open the video and extract crops
    crops_data = []
    cap = cv2.VideoCapture(video_path)
    
    for box_info in selected_boxes:
        frame_idx = box_info['frame_idx']
        x1, y1, x2, y2 = box_info['x1'], box_info['y1'], box_info['x2'], box_info['y2']
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            # Generate a temporary path to return
            crops_data.append({
                'crop': crop,
                'frame_idx': frame_idx,
                'track_id': track_id  # Ensure track_id is included here
            })
    
    cap.release()
    return crops_data

def display_track_summary(df_track, track_id, show_plots=False, 
                         video_path=None, frames_dir=None, num_samples=3,
                         crops_dir=None, use_saved_crops=False,
                         bbox_info=None, use_bbox_info=False):
    """Display a summary of track information to help with identification"""
    print(f"\n===== Track ID: {track_id} =====")
    
    # Extract and display sample frames/crops
    sample_paths = []
    crops_data = []
    
    # First try using bbox info if requested
    if use_bbox_info and bbox_info and video_path:
        crops_data = extract_crops_from_bbox(video_path, bbox_info, track_id, num_samples)
        if crops_data:
            print(f"Extracted {len(crops_data)} crops using bbox information for track {track_id}")
    
    # If no bbox data, check for saved crops
    if not crops_data and use_saved_crops and crops_dir and os.path.exists(crops_dir):
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
    
    # Fall back to extracting frames only if no other option worked
    if not crops_data and not sample_paths and video_path and frames_dir:
        sample_paths = extract_frame_samples(video_path, df_track, frames_dir, num_samples)
        if sample_paths:
            print(f"Extracted {len(sample_paths)} sample frames from video")
    
    # Display the frames/crops if show_plots is enabled
    if show_plots:
        plt.figure(figsize=(15, 5))
        
        # Display either crops from bbox data or from files
        if crops_data:
            for i, crop_info in enumerate(crops_data):
                crop = crop_info['crop']
                frame_idx = crop_info['frame_idx']
                # Use the track_id from crop_info if available
                crop_track_id = crop_info.get('track_id', track_id)
                
                # Convert BGR to RGB for matplotlib
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                plt.subplot(1, len(crops_data), i+1)
                plt.imshow(crop_rgb)
                plt.title(f"ID:{crop_track_id} Frame:{frame_idx}")
                plt.axis('off')
            
            plt.suptitle(f"Track ID {track_id} Crops")
        
        elif sample_paths:
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
        
        # Check for Colab environment by looking for common environment variables
        is_colab = 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ
        
        if is_colab:
            # For Colab, we need to make the plot display non-blocking
            try:
                plt.show(block=False)
                # Small pause to ensure the plot is displayed
                import time
                time.sleep(0.5)
            except Exception:
                plt.show()
        else:
            # Normal display for non-Colab environments
            plt.show()
        
    return len(crops_data) > 0 or len(sample_paths) > 0

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

def save_identity_sequences_from_flat(flat_npy_path, identity_tracks, sequences_dir):
    """Save merged keypoint+feature sequences for each identity from the flat npy array."""
    if not os.path.exists(sequences_dir):
        os.makedirs(sequences_dir, exist_ok=True)
    # Load the flat numpy array
    data = np.load(flat_npy_path)
    # track_id is always the first column
    for identity, track_ids in identity_tracks.items():
        # Select rows where track_id is in this identity's tracks
        mask = np.isin(data[:, 0], track_ids)
        merged = data[mask].copy()
        # Change track_id column to a unique value for this identity (e.g., use min(track_ids) or a hash)
        merged[:, 0] = min(track_ids)  # or use: hash(identity) % 1e6
        # Save merged array
        out_path = os.path.join(sequences_dir, f"{identity}_sequence.npy")
        np.save(out_path, merged)
        print(f"Saved {merged.shape[0]} frames for identity '{identity}' to {out_path}")
    return True

def merge_tracks_in_flat_npy(flat_npy_path, identity_tracks, output_path=None):
    """
    Merge track IDs in the flat numpy array based on identity mappings
    and save a new numpy file with merged data.
    
    Parameters:
    - flat_npy_path: Path to the flat numpy array
    - identity_tracks: Dictionary mapping identity names to lists of track IDs
    - output_path: Path to save the merged numpy array (if None, uses _merged suffix)
    
    Returns:
    - Path to the saved merged numpy array
    """
    if not os.path.exists(flat_npy_path):
        print(f"Error: Flat numpy file {flat_npy_path} not found")
        return None
    
    # Set default output path if not provided
    if output_path is None:
        basename = os.path.basename(flat_npy_path)
        dirname = os.path.dirname(flat_npy_path)
        name_without_ext = os.path.splitext(basename)[0]
        output_path = os.path.join(dirname, f"{name_without_ext}_merged.npy")
    
    # Load the flat numpy array
    print(f"Loading data from {flat_npy_path}...")
    data = np.load(flat_npy_path)
    print(f"Loaded array with shape {data.shape}")
    
    # Make a copy to avoid modifying the original
    merged_data = data.copy()
    
    # Create a mapping from original track IDs to merged IDs
    track_to_merged_id = {}
    identity_to_merged_id = {}
    
    # For each identity, assign a merged ID (using min track ID as the merged ID)
    for identity, track_ids in identity_tracks.items():
        if not track_ids:
            continue
        
        # Use minimum track ID as the merged ID for this identity
        merged_id = min(track_ids)
        identity_to_merged_id[identity] = merged_id
        
        # Map all track IDs to this merged ID
        for track_id in track_ids:
            track_to_merged_id[track_id] = merged_id
    
    # Update the track IDs in the data
    # First column is assumed to be the track_id
    for i in range(len(merged_data)):
        track_id = int(merged_data[i, 0])
        if track_id in track_to_merged_id:
            merged_data[i, 0] = track_to_merged_id[track_id]
    
    # Save the merged data
    print(f"Saving merged data to {output_path}...")
    np.save(output_path, merged_data)
    print(f"Saved merged data with shape {merged_data.shape}")
    
    # Print a summary of the merges
    print("\nMerge summary:")
    for identity, track_ids in identity_tracks.items():
        if not track_ids:
            continue
        merged_id = identity_to_merged_id[identity]
        print(f"- {identity}: Merged {len(track_ids)} tracks to ID {merged_id}")
        
    return output_path

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
    
    # Load bbox information if requested
    bbox_info = None
    if args.use_bbox_info and os.path.exists(args.bbox_info):
        try:
            with open(args.bbox_info, 'r') as f:
                bbox_info = json.load(f)
            print(f"Loaded bounding box information for {len(bbox_info)} tracks")
        except json.JSONDecodeError:
            print(f"Error reading bbox information file: {args.bbox_info}")
    elif args.use_bbox_info:
        print(f"Bounding box information file not found: {args.bbox_info}")
    
    # Check if running in Colab by looking for environment variables
    is_colab = 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ
    if is_colab:
        print("Running in Google Colab environment")
        
        # In Colab, make sure the user is ready before starting the interactive part
        print("\n" + "="*70)
        print("IMPORTANT: You'll be shown images for each track and asked to name them.")
        print("Make sure you can see both the images AND the input prompt.")
        print("="*70)
        input("Press Enter when you're ready to start the identity mapping process...")
    
    # Interactive mapping
    print("\nDisplaying sample data for each track ID to help with identification")
    print("Enter a name for each track ID or press Enter to skip")
    
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
            use_saved_crops=args.use_saved_crops,
            bbox_info=bbox_info,
            use_bbox_info=args.use_bbox_info
        )
        
        # In Colab, make the input prompt more visible
        if is_colab:
            print("\n" + "="*50)
            print(f"PLEASE ENTER A NAME FOR TRACK ID {track_id}:")
            print("="*50)
        
        # Ask for identity
        if track_id in identities:
            current = identities[track_id]
            new_name = input(f"Track ID {track_id} (currently '{current}'): ")
            if new_name:
                identities[track_id] = new_name
                print(f"Set identity for track {track_id} to: {new_name}")
        else:
            name = input(f"Track ID {track_id}: ")
            if name:
                identities[track_id] = name
                print(f"Set identity for track {track_id} to: {name}")
            else:
                print(f"No identity provided for track {track_id}")
        
        # If in Colab, close the figure to avoid overwhelming the browser
        if is_colab:
            plt.close('all')
            
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
        # Use the new flat npy file if it exists
        flat_npy = args.features.replace('.csv', '_flat.npy')
        if os.path.exists(flat_npy):
            save_identity_sequences_from_flat(flat_npy, identity_tracks, args.sequences_dir)
        else:
            save_identity_sequences(features_df, identity_tracks, args.sequences_dir)
    
    # Always merge tracks in the flat numpy file if it exists
    flat_npy = args.features.replace('.csv', '_flat.npy')
    if os.path.exists(flat_npy):
        print("\nMerging tracks in flat numpy file...")
        merged_path = merge_tracks_in_flat_npy(flat_npy, identity_tracks, args.merged_output)
        print(f"Saved merged data to {merged_path}")
    
    # Save mappings
    with open(args.output, 'w') as f:
        json.dump(identities, f, indent=2)
    
    print(f"\nSaved {len(identities)} identity mappings to {args.output}")

if __name__ == "__main__":
    main()