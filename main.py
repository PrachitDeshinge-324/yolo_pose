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

# Import DeepSORT libraries
from deep_sort_realtime.deepsort_tracker import DeepSort

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
    parser.add_argument("--show_buffer", action="store_true",
                       help="Show buffered bounding box used for pose estimation")
    parser.add_argument("--buffer_size", type=float, default=0.1,
                       help="Buffer size ratio around detected person (default: 0.1)")
    return parser.parse_args()


# Configuration - Device priority: CUDA > MPS > CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# Always use CPU for pose on Mac due to MPS bug
POSE_DEVICE = 'cpu' if torch.backends.mps.is_available() else DEVICE

# Initialize YOLO models with tracking capability
model_det = YOLO('model/yolo11x.pt').to(DEVICE)
model_pose = YOLO('model/yolo11x-pose.pt').to(POSE_DEVICE)

# Initialize DeepSORT
def init_deepsort():
    # Initialize with default parameters
    deepsort = DeepSort(
        max_age=30,            # Maximum frames object can be tracked without detection
        n_init=3,              # Minimum detections to initialize track
        max_iou_distance=0.7,  # Maximum IOU distance for track association
        max_cosine_distance=0.2, # Maximum cosine distance for appearance association
        nn_budget=100,         # Maximum samples to retain per track
        override_track_class=None, # Optional custom track class
        embedder="mobilenet",  # Feature extractor to use
        half=True,             # Use half precision for inference
        bgr=True,              # Whether images are in BGR format (OpenCV default)
        embedder_gpu=DEVICE=='cuda',  # Use GPU for embedder if available
        embedder_model_name=None, # Optional custom embedder model
        embedder_wts=None,     # Optional custom embedder weights
        polygon=False,         # Whether detections are polygons
        today=None             # Optional timestamp
    )
    return deepsort

# COCO keypoint skeleton connections
SKELETON = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5]]


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


if __name__ == "__main__":
    args = parse_args()
    
    # Set buffer ratio from command line
    BUFFER_RATIO = args.buffer_size
    MIN_CONFIDENCE = 0.45  # Higher confidence threshold for detection
    POSE_CONFIDENCE = 0.45  # Confidence for pose
    
    print(f"Using device: {DEVICE}, Pose device: {POSE_DEVICE}")
    
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
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(
            args.output_video, 
            fourcc, 
            fps_video, 
            (frame_width, frame_height)
        )
        print(f"Writing output to {args.output_video}")
    
    # Initialize gait analyzer
    gait_analyzer = GaitFeatureExtractor()
    
    # Initialize DeepSORT tracker
    deepsort_tracker = init_deepsort()
    
    # Set up identification if requested
    person_identifier = None
    if args.identify and os.path.exists(args.model):
        person_identifier = GaitValidator()
        if person_identifier.load_model(args.model):
            print("Loaded gait classifier for person identification")
        else:
            person_identifier = None
        
    # Define colors for visualization
    np.random.seed(42)  # For reproducibility
    TRACK_COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()
    JOINT_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    # Store keypoints history for smoothing
    keypoints_history = defaultdict(lambda: [])
    HISTORY_LENGTH = 5  # Number of frames to keep for smoothing

    # For frame rate calculation
    prev_time = cv.getTickCount()
    frame_count = 0
    fps = 0
    
    # Display information about running mode
    if video_writer or args.headless:
        print("Running in headless mode (no window display)")
    else:
        print("Running with window display (press 'q' to quit)")
    
    # Track ID mapping
    next_track_id = 1
    yolo_id_to_track_id = {}
    
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
        
        # 1. Run YOLO detection in tracking mode
        results = model_det.track(frame, persist=True, conf=MIN_CONFIDENCE, 
                                classes=[0], device=DEVICE, verbose=False)
        
        # Display FPS on frame
        cv.putText(frame, f"FPS: {int(fps)}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
        
        # Process detection results
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id
            
            if track_ids is not None:
                track_ids = track_ids.int().cpu().numpy()
                
                # Process each detection
                for i, (box, conf, yolo_id) in enumerate(zip(boxes, confs, track_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Convert YOLO tracking IDs to our stable tracking IDs
                    if yolo_id not in yolo_id_to_track_id:
                        yolo_id_to_track_id[yolo_id] = next_track_id
                        next_track_id += 1
                    
                    track_id = yolo_id_to_track_id[yolo_id]
                    
                    # Draw the tracking box (tight around person)
                    color = TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]
                    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Calculate buffer around person for better pose detection
                    w, h = x2 - x1, y2 - y1
                    x1_buf = max(0, x1 - int(w * BUFFER_RATIO))
                    y1_buf = max(0, y1 - int(h * BUFFER_RATIO))
                    x2_buf = min(frame.shape[1], x2 + int(w * BUFFER_RATIO))
                    y2_buf = min(frame.shape[0], y2 + int(h * BUFFER_RATIO))
                            
                    # Optionally show the buffered box with dashed line
                    if args.show_buffer:
                        # Draw dashed rectangle for buffer area
                        dash_length = 10
                        for i in range(0, 2 * (x2_buf - x1_buf + y2_buf - y1_buf), dash_length * 2):
                            # Top edge
                            if i < (x2_buf - x1_buf):
                                p1 = (x1_buf + i, y1_buf)
                                p2 = (min(x1_buf + i + dash_length, x2_buf), y1_buf)
                            # Right edge
                            elif i < (x2_buf - x1_buf) + (y2_buf - y1_buf):
                                p1 = (x2_buf, y1_buf + i - (x2_buf - x1_buf))
                                p2 = (x2_buf, min(y1_buf + i - (x2_buf - x1_buf) + dash_length, y2_buf))
                            # Bottom edge
                            elif i < 2 * (x2_buf - x1_buf) + (y2_buf - y1_buf):
                                p1 = (x2_buf - (i - ((x2_buf - x1_buf) + (y2_buf - y1_buf))), y2_buf)
                                p2 = (max(x2_buf - (i - ((x2_buf - x1_buf) + (y2_buf - y1_buf))) - dash_length, x1_buf), y2_buf)
                            # Left edge
                            else:
                                p1 = (x1_buf, y2_buf - (i - (2 * (x2_buf - x1_buf) + (y2_buf - y1_buf))))
                                p2 = (x1_buf, max(y2_buf - (i - (2 * (x2_buf - x1_buf) + (y2_buf - y1_buf))) - dash_length, y1_buf))
                            
                            cv.line(frame, p1, p2, (128, 128, 128), 1)
                    
                    # Get the buffered crop for pose estimation
                    person_crop = frame[y1_buf:y2_buf, x1_buf:x2_buf]
                    if person_crop.size == 0:
                        continue
                    
                    # Run pose estimation on the buffered crop
                    pose_results = model_pose.predict(person_crop, conf=POSE_CONFIDENCE, 
                                                    device=POSE_DEVICE, verbose=False)
                    
                    # Process pose results
                    for pose_result in pose_results:
                        if pose_result.keypoints is None:
                            continue
                            
                        keypoints = pose_result.keypoints.xy.cpu().numpy()[0]
                        
                        # Store keypoints for this track
                        if len(keypoints_history[track_id]) >= HISTORY_LENGTH:
                            keypoints_history[track_id].pop(0)
                        keypoints_history[track_id].append(keypoints.copy())
                        
                        # Apply temporal smoothing to reduce jitter
                        smoothed_keypoints = smooth_keypoints(keypoints_history[track_id][:-1], keypoints)
                        
                        gait_analyzer.update_track(track_id, smoothed_keypoints, frame_count)
                        
                        # If identification is enabled, identify the person
                        if person_identifier and frame_count % 15 == 0:  # Check every 15 frames
                            feature_vector = gait_analyzer.get_feature_vector(track_id)
                            if feature_vector is not None:
                                identity, confidence = person_identifier.identify_person(feature_vector)
                                
                                # Draw the identity and confidence
                                id_text = f"{identity}: {confidence:.2f}"
                                cv.putText(frame, id_text, (x1, y1 - 30),
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
                                cv.circle(frame, (frame_px, frame_py), 3, joint_color, -1)
                                
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
                                        color, 1)

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