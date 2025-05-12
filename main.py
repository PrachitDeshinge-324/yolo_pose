from ultralytics import YOLO
import cv2 as cv
import torch
import numpy as np
import logging
from collections import defaultdict
from utils.skeleton_gait import GaitFeatureExtractor, visualize_gait_features
from utils.gait_validator import GaitValidator
import argparse
import os

# Suppress Ultralytics YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Gait-based Person Identification")
    parser.add_argument("--video", type=str, default="../Person_New/input/3c.mp4",
                       help="Path to input video")
    parser.add_argument("--start_frame", type=int, default=150,
                       help="Starting frame number")
    parser.add_argument("--output_features", type=str, default="industrial_gait_features.csv",
                       help="Path to save extracted features")
    parser.add_argument("--model", type=str, default="gait_validation_results/gait_classifier_model.pkl",
                       help="Path to pre-trained gait classifier model")
    parser.add_argument("--identify", action="store_true",
                       help="Perform real-time identification")
    parser.add_argument("--output_video", type=str, default="",
                       help="Path to save output video (if provided, no window will be shown)")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode (no window display)")
    return parser.parse_args()


# Configuration - Device priority: CUDA > MPS > CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# Always use CPU for pose on Mac due to MPS bug
POSE_DEVICE = 'cpu' if DEVICE == 'mps' else DEVICE
BUFFER_RATIO = 0.1  # Buffer around person detection for better pose estimation
MIN_CONFIDENCE = 0.35  # Slightly increased confidence threshold for detection
POSE_CONFIDENCE = 0.45  # Higher confidence for pose

# Initialize models with tracking
model_det = YOLO('model/yolo11x.pt').to(DEVICE)
model_pose = YOLO('model/yolo11x-pose.pt').to(POSE_DEVICE)

if __name__ == "__main__":
    args = parse_args()
    
    # Video capture
    cap = cv.VideoCapture(args.video)
    start_frame = args.start_frame
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    
    # Get video properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv.CAP_PROP_FPS)
    
    # Set up video writer if output path is provided
    video_writer = None
    if args.output_video:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        video_writer = cv.VideoWriter(
            args.output_video, 
            fourcc, 
            fps_video, 
            (frame_width, frame_height)
        )
        print(f"Writing output to {args.output_video}")
    
    # Initialize gait analyzer
    gait_analyzer = GaitFeatureExtractor()
    
    # Set up identification if requested
    person_identifier = None
    if args.identify and os.path.exists(args.model):
        person_identifier = GaitValidator()
        if person_identifier.load_model(args.model):
            print("Loaded gait classifier for person identification")
        else:
            person_identifier = None
        
    # Define colors for better visualization - unique colors for each person
    np.random.seed(42)  # For reproducibility
    TRACK_COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()
    JOINT_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    # Store keypoints history for smoothing
    keypoints_history = defaultdict(lambda: [])
    HISTORY_LENGTH = 5  # Number of frames to keep for smoothing

    # COCO keypoint connections for skeleton
    SKELETON = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5]]


    # For frame rate calculation
    prev_time = cv.getTickCount()
    frame_count = 0
    fps = 0

    def smooth_keypoints(history, current):
        """Apply temporal smoothing to keypoints"""
        if not history or len(history) < 2:
            return current
        
        # Weight recent frames more heavily
        weights = np.linspace(0.5, 1.0, len(history) + 1)
        weights = weights / np.sum(weights)
        
        # Add current frame to weighted average calculation
        all_keypoints = history + [current]
        
        # Only smooth valid keypoints (where both x and y > 0)
        result = np.zeros_like(current)
        for i in range(len(current)):
            # Check if keypoint exists and is valid in each frame's history
            valid_pts = []
            for kp in all_keypoints:
                if i < len(kp) and kp[i].size >= 2 and kp[i][0] > 0 and kp[i][1] > 0:
                    valid_pts.append(kp[i])
            
            if valid_pts:
                valid_weights = weights[-len(valid_pts):]
                valid_weights = valid_weights / np.sum(valid_weights)
                result[i] = np.average(valid_pts, axis=0, weights=valid_weights)
            else:
                result[i] = current[i]
        
        return result
    
    # Display information about running mode
    if video_writer or args.headless:
        print("Running in headless mode (no window display)")
    else:
        print("Running with window display (press 'q' to quit)")
    
    # Process video with tracking
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = cv.getTickCount()
            fps = 30 / ((current_time - prev_time) / cv.getTickFrequency())
            prev_time = current_time
        
        # 1. Detect and track people (with tracking enabled)
        det_results = model_det.track(frame, conf=MIN_CONFIDENCE, classes=[0], 
                                    device=DEVICE, verbose=False,
                                    tracker="bytetrack.yaml", persist=True)
        
        # Display FPS on frame
        cv.putText(frame, f"FPS: {int(fps)}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
        
        for result in det_results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            # Get tracking IDs
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id
            
            if track_ids is None:
                continue
                
            track_ids = track_ids.int().cpu().numpy()
            
            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate buffer around person for better pose detection
                w, h = x2 - x1, y2 - y1
                x1_buf = max(0, x1 - int(w * BUFFER_RATIO))
                y1_buf = max(0, y1 - int(h * BUFFER_RATIO))
                x2_buf = min(frame.shape[1], x2 + int(w * BUFFER_RATIO))
                y2_buf = min(frame.shape[0], y2 + int(h * BUFFER_RATIO))
                
                # Get the buffered crop
                person_crop = frame[y1_buf:y2_buf, x1_buf:x2_buf]
                if person_crop.size == 0:
                    continue
                    
                # Get color for this track ID
                color = TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]
                
                # Draw the tracking box with ID
                cv.rectangle(frame, (x1_buf, y1_buf), (x2_buf, y2_buf), color, 2)
                cv.putText(frame, f"ID: {int(track_id)}", (x1_buf, y1_buf - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Run pose estimation on the buffered crop
                pose_results = model_pose.predict(person_crop, conf=POSE_CONFIDENCE, 
                                                device=POSE_DEVICE, verbose=False)
                
                # Process pose results
                for pose_result in pose_results:
                    if pose_result.keypoints is None:
                        continue
                        
                    keypoints = pose_result.keypoints.xy.cpu().numpy()[0]
                    
                    # Store keypoints for this track
                    track_key = int(track_id)
                    if len(keypoints_history[track_key]) >= HISTORY_LENGTH:
                        keypoints_history[track_key].pop(0)
                    keypoints_history[track_key].append(keypoints.copy())
                    
                    # Apply temporal smoothing to reduce jitter
                    smoothed_keypoints = smooth_keypoints(keypoints_history[track_key][:-1], keypoints)
                    
                    gait_analyzer.update_track(track_key, smoothed_keypoints, frame_count)
                    
                    # Visualize gait features (optional)
                    # frame = visualize_gait_features(frame, track_key, gait_analyzer.gait_features, x1_buf, y2_buf + 20)
                    
                    # If identification is enabled, identify the person
                    if person_identifier and frame_count % 15 == 0:  # Check every 15 frames
                        feature_vector = gait_analyzer.get_feature_vector(track_key)
                        if feature_vector is not None:
                            identity, confidence = person_identifier.identify_person(feature_vector)
                            
                            # Draw the identity and confidence
                            id_text = f"{identity}: {confidence:.2f}"
                            cv.putText(frame, id_text, (x1_buf, y1_buf - 30),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
                    # Draw all keypoints
                    for i, point in enumerate(smoothed_keypoints):
                        px, py = int(point[0]), int(point[1])
                        
                        # Convert from crop coordinates to original frame
                        frame_px, frame_py = px + x1_buf, py + y1_buf
                        
                        # Draw the point if it's valid
                        if px > 0 and py > 0:
                            # Color by keypoint type
                            joint_color = JOINT_COLORS[i % len(JOINT_COLORS)]
                            cv.circle(frame, (frame_px, frame_py), 5, joint_color, -1)
                            
                    # Connect keypoints with lines to form skeleton
                    for pair in SKELETON:
                        p1, p2 = pair[0]-1, pair[1]-1  # COCO format to 0-indexed
                        if p1 >= len(smoothed_keypoints) or p2 >= len(smoothed_keypoints):
                            continue
                        if (smoothed_keypoints[p1][0] > 0 and smoothed_keypoints[p1][1] > 0 and 
                            smoothed_keypoints[p2][0] > 0 and smoothed_keypoints[p2][1] > 0):
                            cv.line(frame, 
                                    (int(smoothed_keypoints[p1][0]) + x1_buf, int(smoothed_keypoints[p1][1]) + y1_buf), 
                                    (int(smoothed_keypoints[p2][0]) + x1_buf, int(smoothed_keypoints[p2][1]) + y1_buf),
                                    color, 2)

        # Write frame to video if output is specified
        if video_writer:
            video_writer.write(frame)
            
        # Display the frame if not in headless mode
        if not args.headless and not args.output_video:
            cv.imshow('Person Tracking & Pose Detection', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Export features
    gait_analyzer.export_features_csv(args.output_features)
    print(f"Exported gait features to {args.output_features}")

    # Clean up resources
    cap.release()
    if video_writer:
        video_writer.release()
    if not args.headless and not args.output_video:
        cv.destroyAllWindows()