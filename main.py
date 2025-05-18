from ultralytics import YOLO
import cv2 as cv
import torch
import numpy as np
import logging
from collections import defaultdict
from utils.skeleton_gait import GaitFeatureExtractor
from utils.gait_validator import GaitValidator
import argparse
import os
import tqdm
import json

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
    parser.add_argument("--end_frame", type=int, default=300,
                       help="Ending frame number")
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
    parser.add_argument("--buffer_size", type=float, default=0.1,
                       help="Buffer size ratio around detected person (default: 0.1)")
    parser.add_argument("--save_bbox_info", action="store_true", default=False,
                       help="Save bounding box information to JSON")
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
    deepsort = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        nn_budget=100,
        override_track_class=None,
        embedder="torchreid",
        half=True,
        bgr=True,
        embedder_gpu=DEVICE=='cuda',
        embedder_model_name="osnet_ain_x1_0",
        embedder_wts=None,
        polygon=False,
        today=None
    )
    return deepsort

# COCO keypoint skeleton connections
SKELETON = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5]]


def smooth_keypoints(history, current):
    """Apply temporal smoothing to keypoints"""
    if not history or len(history) < 2:
        return current
    
    weights = np.linspace(0.5, 1.0, len(history) + 1)
    weights = weights / np.sum(weights)
    all_keypoints = history + [current]
    result = np.zeros_like(current)
    for i in range(len(current)):
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


def process_frame(frame, frame_count, args, model_det, model_pose, gait_analyzer, keypoints_history, yolo_id_to_track_id, next_track_id, bbox_info, person_identifier, BUFFER_RATIO, MIN_CONFIDENCE, POSE_CONFIDENCE, TRACK_COLORS, JOINT_COLORS, SKELETON, HISTORY_LENGTH,fps=0):
    results = model_det.track(frame, persist=True, conf=MIN_CONFIDENCE, 
                              classes=[0], device=DEVICE, verbose=False)
    cv.putText(frame, f"FPS: {int(fps)}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 
               1, (0, 255, 0), 2)
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id
        if track_ids is not None:
            track_ids = track_ids.int().cpu().numpy()
            for i, (box, conf, yolo_id) in enumerate(zip(boxes, confs, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                if yolo_id not in yolo_id_to_track_id:
                    yolo_id_to_track_id[yolo_id] = next_track_id
                    next_track_id += 1
                track_id = yolo_id_to_track_id[yolo_id]
                color = TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                w, h = x2 - x1, y2 - y1
                x1_buf = max(0, x1 - int(w * BUFFER_RATIO))
                y1_buf = max(0, y1 - int(h * BUFFER_RATIO))
                x2_buf = min(frame.shape[1], x2 + int(w * BUFFER_RATIO))
                y2_buf = min(frame.shape[0], y2 + int(h * BUFFER_RATIO))
                if args.save_bbox_info:
                    bbox_info[int(track_id)].append({
                        'track_id': int(track_id),
                        'frame_idx': frame_count,
                        'x1': x1_buf,
                        'y1': y1_buf, 
                        'x2': x2_buf,
                        'y2': y2_buf,
                        'original_box': [int(x) for x in [x1, y1, x2, y2]]
                    })
                person_crop = frame[y1_buf:y2_buf, x1_buf:x2_buf]
                if person_crop.size == 0:
                    continue
                pose_results = model_pose.predict(person_crop, conf=POSE_CONFIDENCE, 
                                                  device=POSE_DEVICE, verbose=False)
                for pose_result in pose_results:
                    if pose_result.keypoints is None:
                        continue
                    keypoints = pose_result.keypoints.xy.cpu().numpy()[0]
                    if len(keypoints_history[track_id]) >= HISTORY_LENGTH:
                        keypoints_history[track_id].pop(0)
                    keypoints_history[track_id].append(keypoints.copy())
                    smoothed_keypoints = smooth_keypoints(keypoints_history[track_id][:-1], keypoints)
                    gait_analyzer.update_track(track_id, smoothed_keypoints, frame_count)
                    if person_identifier and frame_count % 15 == 0:
                        feature_vector = gait_analyzer.get_feature_vector(track_id)
                        if feature_vector is not None:
                            identity, confidence = person_identifier.identify_person(feature_vector)
                            id_text = f"{identity}: {confidence:.2f}"
                            cv.putText(frame, id_text, (x1, y1 - 30),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    for i, point in enumerate(smoothed_keypoints):
                        px, py = int(point[0]), int(point[1])
                        frame_px, frame_py = px + x1_buf, py + y1_buf
                        if px > 0 and py > 0:
                            joint_color = JOINT_COLORS[i % len(JOINT_COLORS)]
                            cv.circle(frame, (frame_px, frame_py), 3, joint_color, -1)
                    for pair in SKELETON:
                        p1, p2 = pair[0]-1, pair[1]-1
                        if p1 >= len(smoothed_keypoints) or p2 >= len(smoothed_keypoints):
                            continue
                        if (smoothed_keypoints[p1][0] > 0 and smoothed_keypoints[p1][1] > 0 and 
                            smoothed_keypoints[p2][0] > 0 and smoothed_keypoints[p2][1] > 0):
                            cv.line(frame, 
                                    (int(smoothed_keypoints[p1][0]) + x1_buf, int(smoothed_keypoints[p1][1]) + y1_buf), 
                                    (int(smoothed_keypoints[p2][0]) + x1_buf, int(smoothed_keypoints[p2][1]) + y1_buf),
                                    color, 1)
    return next_track_id


def export_all_data(gait_analyzer, args, bbox_info, features_path, bbox_json_path, feature_order_path, flat_npy_path):
    gait_analyzer.export_features_csv(features_path)
    print(f"Exported gait features to {features_path}")
    if args.save_bbox_info:
        with open(bbox_json_path, 'w') as f:
            json.dump(bbox_info, f)
        print(f"Saved bounding box information to {bbox_json_path}")
    all_rows = []
    feature_keys = None
    all_feature_keys = set()
    for track_id in gait_analyzer.track_history:
        features = gait_analyzer.get_features(track_id)
        if features is not None:
            all_feature_keys.update([k for k, v in features.items() if isinstance(v, (int, float, np.floating, np.integer))])
    feature_keys = sorted(all_feature_keys)
    with open(feature_order_path, 'w') as f:
        for k in feature_keys:
            f.write(f"{k}\n")
    print(f"Saved feature order to {feature_order_path}")
    for track_id, history in gait_analyzer.track_history.items():
        features = gait_analyzer.get_features(track_id)
        if features is not None:
            feature_vec = np.array([
                features[k] if (k in features and isinstance(features[k], (int, float, np.floating, np.integer))) else np.nan
                for k in feature_keys
            ], dtype=np.float32)
        else:
            feature_vec = np.full(len(feature_keys), np.nan, dtype=np.float32)
        for frame_idx, kpts in history:
            flat_kpts = []
            for pt in kpts:
                if isinstance(pt, np.ndarray) and pt.size == 2:
                    flat_kpts.extend(pt.tolist())
                else:
                    flat_kpts.extend([0, 0])
            row = [int(track_id), int(frame_idx)] + flat_kpts + feature_vec.tolist()
            all_rows.append(row)
    all_rows_np = np.array(all_rows, dtype=np.float32)
    np.save(flat_npy_path, all_rows_np)
    print(f"Saved flat numpy array with id, frame, keypoints, features to {flat_npy_path}")


def main():
    args = parse_args()
    # Define results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Update output paths to use results_dir
    base_features_name = os.path.basename(args.output_features)
    features_path = os.path.join(results_dir, base_features_name)
    flat_npy_path = features_path.replace('.csv', '_flat.npy')
    bbox_json_path = os.path.join(results_dir, "bbox_info.json")
    feature_order_path = features_path.replace('.csv', '_feature_order.txt')

    # If output_video is specified, put it in results_dir
    if args.output_video:
        output_video_path = os.path.join(results_dir, os.path.basename(args.output_video))
    else:
        output_video_path = None

    BUFFER_RATIO = args.buffer_size
    MIN_CONFIDENCE = 0.45
    POSE_CONFIDENCE = 0.45
    print(f"Using device: {DEVICE}, Pose device: {POSE_DEVICE}")
    cap = cv.VideoCapture(args.video)
    start_frame = args.start_frame
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    video_writer = None
    if output_video_path:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(
            output_video_path, 
            fourcc, 
            fps_video, 
            (frame_width, frame_height)
        )
        print(f"Writing output to {output_video_path}")
    gait_analyzer = GaitFeatureExtractor()
    deepsort_tracker = init_deepsort()
    person_identifier = None
    if args.identify and os.path.exists(args.model):
        person_identifier = GaitValidator()
        if person_identifier.load_model(args.model):
            print("Loaded gait classifier for person identification")
        else:
            person_identifier = None
    np.random.seed(42)
    TRACK_COLORS = np.random.randint(0, 255, size=(100, 3)).tolist()
    JOINT_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    keypoints_history = defaultdict(lambda: [])
    HISTORY_LENGTH = 5
    prev_time = cv.getTickCount()
    frame_count = 0
    fps = 0
    if video_writer or args.headless:
        print("Running in headless mode (no window display)")
    else:
        print("Running with window display (press 'q' to quit)")
    next_track_id = 1
    yolo_id_to_track_id = {}
    bbox_info = defaultdict(list)
    with tqdm.tqdm(total=args.end_frame-start_frame, desc="Processing frames") as pbar:
        while cap.isOpened() and frame_count < args.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Calculate FPS for every frame
            current_time = cv.getTickCount()
            time_diff = (current_time - prev_time) / cv.getTickFrequency()
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_time = current_time
            next_track_id = process_frame(
                frame, frame_count, args, model_det, model_pose, gait_analyzer, keypoints_history, yolo_id_to_track_id, next_track_id, bbox_info, person_identifier, BUFFER_RATIO, MIN_CONFIDENCE, POSE_CONFIDENCE, TRACK_COLORS, JOINT_COLORS, SKELETON, HISTORY_LENGTH, fps
            )
            if video_writer:
                video_writer.write(frame)
            if not args.headless and not args.output_video:
                cv.imshow('Person Tracking & Pose Detection', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            pbar.update(1)
    export_all_data(gait_analyzer, args, bbox_info, features_path, bbox_json_path, feature_order_path, flat_npy_path)
    cap.release()
    if video_writer:
        video_writer.release()
    if not args.headless and not args.output_video:
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()