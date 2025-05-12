# yolo_pose_demo_improved.py
from ultralytics import YOLO
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import time 
from tqdm import tqdm
import sys
import logging

# Suppress Ultralytics YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Configuration - Device priority: CUDA > MPS > CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
BUFFER_RATIO = 0.2
MIN_CONFIDENCE = 0.5
MAX_AGE = 90
POSE_SMOOTHING = 5

# Initialize models
model_det = YOLO('yolo11x.pt').to(DEVICE)  # Use appropriate model version
model_pose = YOLO('yolo11x-pose.pt').to(DEVICE)

# Tracker configuration with device-aware settings
tracker = DeepSort(
    max_age=MAX_AGE,
    n_init=2,
    nms_max_overlap=0.9,
    embedder="torchreid",
    embedder_model_name="osnet_x1_0",
    max_cosine_distance=0.35,
    nn_budget=150,
    half=True,
    embedder_gpu=True if DEVICE == 'cuda' else False  # Only use GPU embedder for CUDA
)

# Track management
track_history = {}
pose_history = {}
last_seen = {}
last_appearances = {}
id_mapping = {}

cap = cv2.VideoCapture('/Users/prachit/self/Working/Person_New/input/3c.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress = tqdm(total=frame_count, desc="Processing frames", file=sys.stdout)

def pose_similarity(pose1, pose2):
    """Calculate pose similarity using stable keypoints"""
    if len(pose1) == 0 or len(pose2) == 0:
        return 0
    stable_points = [5, 6, 11, 12]
    valid_points1 = [i for i in stable_points if i < len(pose1) and pose1[i].sum() > 0]
    valid_points2 = [i for i in stable_points if i < len(pose2) and pose2[i].sum() > 0]
    common_points = set(valid_points1).intersection(set(valid_points2))
    if not common_points:
        return 0
    similarity = 0
    for i in common_points:
        dist = np.linalg.norm(pose1[i] - pose2[i])
        similarity += np.exp(-dist / 100)
    return similarity / len(common_points)

# Video writer initialization
out_writer = None
frame_size = None

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])
            # Try multiple codecs for Colab compatibility
            for codec in ['mp4v', 'avc1', 'X264', 'MJPG']:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out_writer = cv2.VideoWriter(f'output_pose_tracking_{codec}.mp4', fourcc, 30, frame_size)
                if out_writer.isOpened():
                    break
            if not out_writer.isOpened():
                # Fallback to AVI
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_writer = cv2.VideoWriter('output_pose_tracking.avi', fourcc, 30, frame_size)

        progress.update(1)

        # Detection and tracking pipeline
        det_results = model_det.predict(frame, conf=MIN_CONFIDENCE, classes=[0], device=DEVICE, verbose=False)
        detections = []
        
        for result in det_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                x1_buf = max(0, x1 - int(w * BUFFER_RATIO))
                y1_buf = max(0, y1 - int(h * BUFFER_RATIO))
                x2_buf = min(frame.shape[1], x2 + int(w * BUFFER_RATIO))
                y2_buf = min(frame.shape[0], y2 + int(h * BUFFER_RATIO))
                detections.append(([x1_buf, y1_buf, x2_buf - x1_buf, y2_buf - y1_buf], box.conf.item(), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        current_ids = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            # ID mapping and recovery logic
            if hasattr(track, 'features') and track.features:
                last_appearances[track_id] = track.features[-1]
            if track.age < 5 and track_id not in id_mapping:
                for old_id, features in last_appearances.items():
                    if old_id not in current_ids and old_id not in id_mapping.values():
                        if track.features and len(track.features) > 0:
                            similarity = 1 - np.linalg.norm(track.features[-1] - features)
                            if similarity > 0.4:
                                id_mapping[track_id] = old_id
                                break
            if track_id in id_mapping:
                track_id = id_mapping[track_id]

            current_ids.append(track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Update track history
            track_history.setdefault(track_id, []).append(center)
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # Pose estimation
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            pose_results = model_pose.predict(crop, device=DEVICE)
            if pose_results[0].keypoints is not None:
                current_pose = pose_results[0].keypoints.xy.cpu().numpy()[0]
                if track_id in pose_history and pose_history[track_id]:
                    expected_shape = pose_history[track_id][0].shape
                else:
                    expected_shape = current_pose.shape
                
                if current_pose.shape == expected_shape:
                    pose_history.setdefault(track_id, []).append(current_pose)
                    if len(pose_history[track_id]) > POSE_SMOOTHING:
                        pose_history[track_id].pop(0)
                    
                    # Pose-based recovery
                    for old_id in list(pose_history.keys()):
                        if old_id not in current_ids and old_id != track_id:
                            if pose_history[old_id] and pose_history[old_id][-1].shape == expected_shape:
                                pose_sim = pose_similarity(current_pose, pose_history[old_id][-1])
                                if pose_sim > 0.7:
                                    id_mapping[track_id] = old_id
                                    break
                    
                    # Visualize pose
                    pose_img = pose_results[0].plot()
                    pose_img = cv2.resize(pose_img, (x2-x1, y2-y1))
                    # Fix shape mismatch before assignment
                    h_img, w_img = pose_img.shape[:2]
                    h_frame = y2 - y1
                    w_frame = x2 - x1
                    h = min(h_img, h_frame)
                    w = min(w_img, w_frame)
                    frame[y1:y1+h, x1:x1+w] = pose_img[:h, :w]

            # Draw trajectory
            history = track_history[track_id]
            for i in range(1, len(history)):
                thickness = int(np.sqrt(30 / (i + 1)) * 1)
                cv2.line(frame, history[i-1], history[i], (0,255,0), thickness)

        # Cleanup old tracks
        current_time = time.time()
        for track_id in list(track_history.keys()):
            if track_id not in current_ids:
                if track_id not in last_seen:
                    last_seen[track_id] = current_time
                elif current_time - last_seen[track_id] > 10:
                    for d in [track_history, pose_history, last_appearances]:
                        if track_id in d:
                            del d[track_id]

        # Write frame
        out_writer.write(frame)

except Exception as e:
    print(f"\nError occurred: {str(e)}")
finally:
    progress.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()
    print("Processing completed. Output video saved.")