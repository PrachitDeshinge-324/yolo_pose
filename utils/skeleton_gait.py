import numpy as np
import cv2 as cv
from collections import defaultdict
from scipy.signal import find_peaks, savgol_filter
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GaitFeatureExtractor:
    """
    Extract unique gait features from skeleton keypoints to identify individuals
    even when wearing the same uniform in industrial settings.
    Handles multi-directional movement including toward/away from camera.
    """

    def __init__(self):
        # COCO keypoint indices for reference
        self.nose = 0
        self.left_eye, self.right_eye = 1, 2
        self.left_ear, self.right_ear = 3, 4
        self.left_shoulder, self.right_shoulder = 5, 6
        self.left_elbow, self.right_elbow = 7, 8
        self.left_wrist, self.right_wrist = 9, 10
        self.left_hip, self.right_hip = 11, 12
        self.left_knee, self.right_knee = 13, 14
        self.left_ankle, self.right_ankle = 15, 16
        
        # Historical data storage for each track
        self.track_history = defaultdict(list)
        self.gait_features = defaultdict(dict)
        self.movement_direction = defaultdict(lambda: "unknown")
        
        # Stride analysis settings
        self.min_frames_for_analysis = 20  # Reduced for quicker analysis
        self.peak_distance = 8  # Minimum frames between steps
        
        # For movement direction calculation
        self.position_history = defaultdict(list)
        self.direction_history = defaultdict(list)
        self.direction_window = 10  # Frames to consider for direction

    def is_valid_keypoint(self, kpt):
        """Check if a keypoint has valid coordinates"""
        return (kpt is not None and 
                isinstance(kpt, np.ndarray) and 
                kpt.size >= 2 and 
                kpt[0] > 0 and kpt[1] > 0)

    def update_track(self, track_id, keypoints, frame_idx):
        """Add new keypoint data for a tracked person"""
        # Skip empty keypoints
        if keypoints is None or len(keypoints) == 0:
            return
            
        if len(self.track_history[track_id]) > 100:
            # Keep a moving window to avoid excessive memory usage
            self.track_history[track_id].pop(0)
            if len(self.position_history[track_id]) > 0:
                self.position_history[track_id].pop(0)
        
        # Store frame index with keypoints for temporal analysis
        self.track_history[track_id].append((frame_idx, keypoints.copy()))
        
        # Calculate average position (center of mass) and track movement direction
        valid_points = [kp for kp in keypoints if self.is_valid_keypoint(kp)]
        if valid_points:
            avg_position = np.mean(valid_points, axis=0)
            self.position_history[track_id].append(avg_position)
            
            # Determine movement direction if we have enough history
            if len(self.position_history[track_id]) >= 5:
                self._update_movement_direction(track_id)
        
        # Update viewing direction based on shoulders and hips
        self._estimate_viewing_direction(track_id, keypoints)
        
        # Update features if we have enough frames
        if len(self.track_history[track_id]) >= self.min_frames_for_analysis:
            self._extract_all_features(track_id)
    
    def _update_movement_direction(self, track_id):
        """Calculate the movement direction of the person"""
        positions = self.position_history[track_id]
        if len(positions) < 5:
            return
            
        # Get the recent positions and calculate movement vector
        recent_pos = positions[-5:]
        start_pos = np.mean(recent_pos[:2], axis=0)  # Average of oldest positions
        end_pos = np.mean(recent_pos[-2:], axis=0)   # Average of newest positions
        
        # Calculate movement vector and distance
        move_vector = end_pos - start_pos
        distance = np.linalg.norm(move_vector)
        
        # Only update if there's significant movement
        if distance > 5:  # Minimum pixel movement to consider
            # Calculate angle of movement
            angle = np.arctan2(move_vector[1], move_vector[0]) * 180 / np.pi
            
            # Classify direction
            # Note: Camera perspective means y increases downward
            if abs(move_vector[0]) > abs(move_vector[1] * 2):
                # Primarily horizontal movement
                direction = "right" if move_vector[0] > 0 else "left"
            elif abs(move_vector[1]) > abs(move_vector[0] * 2):
                # Primarily vertical movement - in camera space this is toward/away
                direction = "toward" if move_vector[1] > 0 else "away"
            else:
                # Diagonal movement
                if move_vector[0] > 0 and move_vector[1] > 0:
                    direction = "toward-right"
                elif move_vector[0] > 0 and move_vector[1] < 0:
                    direction = "away-right"
                elif move_vector[0] < 0 and move_vector[1] > 0:
                    direction = "toward-left"
                else:
                    direction = "away-left"
                    
            # Store in direction history
            self.direction_history[track_id].append(direction)
            if len(self.direction_history[track_id]) > self.direction_window:
                self.direction_history[track_id].pop(0)
                
            # Set the most common recent direction
            if self.direction_history[track_id]:
                # Count occurrences of each direction
                direction_counts = {}
                for d in self.direction_history[track_id]:
                    if d not in direction_counts:
                        direction_counts[d] = 0
                    direction_counts[d] += 1
                
                # Find most common direction
                self.movement_direction[track_id] = max(
                    direction_counts.items(), key=lambda x: x[1])[0]
    
    def _estimate_viewing_direction(self, track_id, keypoints):
        """Estimate which way the person is facing"""
        if len(keypoints) <= max(self.left_hip, self.right_hip):
            return "unknown"
            
        # Check if we can see shoulders and hips
        shoulders_visible = (self.is_valid_keypoint(keypoints[self.left_shoulder]) and 
                           self.is_valid_keypoint(keypoints[self.right_shoulder]))
        hips_visible = (self.is_valid_keypoint(keypoints[self.left_hip]) and 
                       self.is_valid_keypoint(keypoints[self.right_hip]))
                
        if shoulders_visible and hips_visible:
            # Calculate width of shoulders and hips
            shoulder_width = np.linalg.norm(
                keypoints[self.left_shoulder] - keypoints[self.right_shoulder])
            hip_width = np.linalg.norm(
                keypoints[self.left_hip] - keypoints[self.right_hip])
            
            # Calculate shoulder hip ratio - helps determine if front/back vs side
            if hip_width > 0:
                sh_ratio = shoulder_width / hip_width
                
                # Store the ratio
                self.gait_features[track_id]['shoulder_hip_ratio'] = sh_ratio
                
                # Use this to guess viewing angle
                if 0.8 < sh_ratio < 1.2:
                    view = "front_or_back"  # Front/back typically has similar width
                else:
                    view = "side"  # Side view typically has different widths
                    
                self.gait_features[track_id]['viewing_angle'] = view
        
    def get_features(self, track_id):
        """Get the computed features for a specific track"""
        if track_id in self.gait_features:
            return self.gait_features[track_id]
        return None
    
    def get_movement_direction(self, track_id):
        """Get the current movement direction"""
        return self.movement_direction.get(track_id, "unknown")
    
    def _extract_all_features(self, track_id):
        """Calculate all gait features for the track"""
        history = self.track_history[track_id]
        
        # Extract features only if we have enough frames
        if len(history) < self.min_frames_for_analysis:
            return
        
        # Get all valid keypoints sequences
        keypoints_seq = []
        for _, kpts in history:
            if kpts is not None and len(kpts) > 0:
                keypoints_seq.append(kpts)
        
        # Skip if we don't have any valid keypoints
        if not keypoints_seq:
            return
        
        # Get movement direction for context
        movement_dir = self.get_movement_direction(track_id)
        self.gait_features[track_id]['movement_direction'] = movement_dir
            
        # Extract core biomechanical features with direction awareness
        self._calculate_relative_proportions(track_id, keypoints_seq)
        self._analyze_walking_pattern(track_id, keypoints_seq)
        self._calculate_joint_angles(track_id, keypoints_seq)
        self._analyze_posture(track_id, keypoints_seq)
        self._extract_gait_cycle_features(track_id, keypoints_seq)
        self._analyze_center_of_mass(track_id, keypoints_seq)
        self._analyze_weight_transfer(track_id, keypoints_seq)
        self._analyze_head_movement(track_id, keypoints_seq)
        self._analyze_motion_dynamics(track_id, keypoints_seq)
        self._analyze_movement_periodicity(track_id, keypoints_seq)
        
    def _calculate_relative_proportions(self, track_id, keypoints_seq):
        """Calculate relative body proportions that are scale-invariant"""
        features = {}
        
        # Calculate relative proportions that remain consistent regardless of distance from camera
        # These are more reliable than absolute pixel measurements
        
        # Height to width ratio (stays consistent at different distances)
        height_width_ratios = []
        
        for keypoints in keypoints_seq:
            # Find the bounding dimensions of the person
            valid_x = []
            valid_y = []
            
            for i, kp in enumerate(keypoints):
                if len(keypoints) > i and self.is_valid_keypoint(kp):
                    valid_x.append(kp[0])
                    valid_y.append(kp[1])
            
            if valid_x and valid_y:
                height = max(valid_y) - min(valid_y)
                width = max(valid_x) - min(valid_x)
                
                if width > 0:  # Avoid division by zero
                    height_width_ratios.append(height / width)
        
        if height_width_ratios:
            features['height_width_ratio'] = np.median(height_width_ratios)
        
        # Leg to torso ratio (more reliable than absolute heights)
        leg_torso_ratios = []
        
        for keypoints in keypoints_seq:
            # Try with left side first
            if (len(keypoints) > self.left_hip and 
                len(keypoints) > self.left_ankle and 
                len(keypoints) > self.left_shoulder and 
                self.is_valid_keypoint(keypoints[self.left_hip]) and 
                self.is_valid_keypoint(keypoints[self.left_ankle]) and
                self.is_valid_keypoint(keypoints[self.left_shoulder])):
                
                torso_length = np.linalg.norm(keypoints[self.left_shoulder] - keypoints[self.left_hip])
                leg_length = np.linalg.norm(keypoints[self.left_hip] - keypoints[self.left_ankle])
                
                if torso_length > 0:  # Avoid division by zero
                    leg_torso_ratios.append(leg_length / torso_length)
            
            # If left side fails, try right side
            elif (len(keypoints) > self.right_hip and 
                  len(keypoints) > self.right_ankle and 
                  len(keypoints) > self.right_shoulder and 
                  self.is_valid_keypoint(keypoints[self.right_hip]) and 
                  self.is_valid_keypoint(keypoints[self.right_ankle]) and
                  self.is_valid_keypoint(keypoints[self.right_shoulder])):
                
                torso_length = np.linalg.norm(keypoints[self.right_shoulder] - keypoints[self.right_hip])
                leg_length = np.linalg.norm(keypoints[self.right_hip] - keypoints[self.right_ankle])
                
                if torso_length > 0:  # Avoid division by zero
                    leg_torso_ratios.append(leg_length / torso_length)
        
        if leg_torso_ratios:
            features['leg_torso_ratio'] = np.median(leg_torso_ratios)
        
        # Upper to lower leg ratio (highly individual)
        upper_lower_leg_ratios = []
        
        for keypoints in keypoints_seq:
            # Left leg
            if (len(keypoints) > self.left_hip and 
                len(keypoints) > self.left_knee and 
                len(keypoints) > self.left_ankle and 
                self.is_valid_keypoint(keypoints[self.left_hip]) and 
                self.is_valid_keypoint(keypoints[self.left_knee]) and
                self.is_valid_keypoint(keypoints[self.left_ankle])):
                
                upper_leg = np.linalg.norm(keypoints[self.left_hip] - keypoints[self.left_knee])
                lower_leg = np.linalg.norm(keypoints[self.left_knee] - keypoints[self.left_ankle])
                
                if lower_leg > 0:  # Avoid division by zero
                    upper_lower_leg_ratios.append(upper_leg / lower_leg)
            
            # Right leg
            elif (len(keypoints) > self.right_hip and 
                  len(keypoints) > self.right_knee and 
                  len(keypoints) > self.right_ankle and 
                  self.is_valid_keypoint(keypoints[self.right_hip]) and 
                  self.is_valid_keypoint(keypoints[self.right_knee]) and
                  self.is_valid_keypoint(keypoints[self.right_ankle])):
                
                upper_leg = np.linalg.norm(keypoints[self.right_hip] - keypoints[self.right_knee])
                lower_leg = np.linalg.norm(keypoints[self.right_knee] - keypoints[self.right_ankle])
                
                if lower_leg > 0:  # Avoid division by zero
                    upper_lower_leg_ratios.append(upper_leg / lower_leg)
        
        if upper_lower_leg_ratios:
            features['upper_lower_leg_ratio'] = np.median(upper_lower_leg_ratios)
        
        # Update feature dictionary
        self.gait_features[track_id].update(features)
        
    def _analyze_walking_pattern(self, track_id, keypoints_seq):
        """Analyze walking patterns including stride length and frequency"""
        features = {}
        
        # Get the movement direction to contextualize pattern analysis
        movement_dir = self.get_movement_direction(track_id)
        
        # Extract ankle positions over time
        left_ankle_y = [kpts[self.left_ankle][1] 
                        if len(kpts) > self.left_ankle and self.is_valid_keypoint(kpts[self.left_ankle]) 
                        else np.nan for kpts in keypoints_seq]
        right_ankle_y = [kpts[self.right_ankle][1] 
                         if len(kpts) > self.right_ankle and self.is_valid_keypoint(kpts[self.right_ankle]) 
                         else np.nan for kpts in keypoints_seq]
        
        # Also get X positions for lateral movement analysis
        left_ankle_x = [kpts[self.left_ankle][0] 
                        if len(kpts) > self.left_ankle and self.is_valid_keypoint(kpts[self.left_ankle]) 
                        else np.nan for kpts in keypoints_seq]
        right_ankle_x = [kpts[self.right_ankle][0] 
                         if len(kpts) > self.right_ankle and self.is_valid_keypoint(kpts[self.right_ankle]) 
                         else np.nan for kpts in keypoints_seq]
        
        # Clean nan values
        left_ankle_y = np.array([y for y in left_ankle_y if not np.isnan(y)])
        right_ankle_y = np.array([y for y in right_ankle_y if not np.isnan(y)])
        left_ankle_x = np.array([x for x in left_ankle_x if not np.isnan(x)])
        right_ankle_x = np.array([x for x in right_ankle_x if not np.isnan(x)])
        
        # Select which coordinate to use based on movement direction
        if movement_dir in ["left", "right", "toward-right", "toward-left", "away-right", "away-left"]:
            # For side movement, x-coordinate shows stride pattern better
            left_signal = left_ankle_x
            right_signal = right_ankle_x
        else:
            # For toward/away movement, y-coordinate shows stride pattern better
            left_signal = left_ankle_y
            right_signal = right_ankle_y
        
        # Apply signal processing if we have enough points
        if len(left_signal) > 10:
            try:
                # Smooth the signal first to reduce noise
                left_signal_smooth = savgol_filter(left_signal, min(9, len(left_signal) - (len(left_signal) % 2 - 1)), 3)
                # Find peaks (steps) in ankle movement
                left_peaks, _ = find_peaks(left_signal_smooth, distance=self.peak_distance)
                if len(left_peaks) >= 2:
                    features['left_stride_frequency'] = len(left_peaks) / len(left_signal)
                    # Calculate stride length variation (consistency of steps)
                    if len(left_peaks) >= 3:
                        stride_lengths = np.diff(left_peaks)
                        features['left_stride_consistency'] = 1.0 - (np.std(stride_lengths) / np.mean(stride_lengths))
            except Exception:
                pass
                
        if len(right_signal) > 10:
            try:
                # Smooth the signal
                right_signal_smooth = savgol_filter(right_signal, min(9, len(right_signal) - (len(right_signal) % 2 - 1)), 3)
                # Find peaks
                right_peaks, _ = find_peaks(right_signal_smooth, distance=self.peak_distance)
                if len(right_peaks) >= 2:
                    features['right_stride_frequency'] = len(right_peaks) / len(right_signal)
                    if len(right_peaks) >= 3:
                        stride_lengths = np.diff(right_peaks)
                        features['right_stride_consistency'] = 1.0 - (np.std(stride_lengths) / np.mean(stride_lengths))
            except Exception:
                pass
                
        # Calculate arm swing characteristics - normalized to body size
        arm_swing_features = self._calculate_arm_swing_features(keypoints_seq)
        features.update(arm_swing_features)
            
        # Update feature dictionary
        self.gait_features[track_id].update(features)
    
    def _calculate_arm_swing_features(self, keypoints_seq):
        """Calculate normalized arm swing patterns"""
        features = {}
        
        # Extract wrist positions relative to hip positions
        left_wrist_rel = []
        right_wrist_rel = []
        
        for keypoints in keypoints_seq:
            # Left arm relative to hip
            if (len(keypoints) > self.left_wrist and 
                len(keypoints) > self.left_hip and 
                self.is_valid_keypoint(keypoints[self.left_wrist]) and 
                self.is_valid_keypoint(keypoints[self.left_hip])):
                
                rel_pos = keypoints[self.left_wrist] - keypoints[self.left_hip]
                left_wrist_rel.append(rel_pos)
            
            # Right arm relative to hip
            if (len(keypoints) > self.right_wrist and 
                len(keypoints) > self.right_hip and 
                self.is_valid_keypoint(keypoints[self.right_wrist]) and 
                self.is_valid_keypoint(keypoints[self.right_hip])):
                
                rel_pos = keypoints[self.right_wrist] - keypoints[self.right_hip]
                right_wrist_rel.append(rel_pos)
        
        # Calculate swing characteristics
        if len(left_wrist_rel) > 5:
            left_wrist_rel = np.array(left_wrist_rel)
            features['left_arm_swing_x'] = np.std(left_wrist_rel[:, 0])
            features['left_arm_swing_y'] = np.std(left_wrist_rel[:, 1])
            
            # Calculate arm swing area (approximation of swing volume)
            features['left_arm_swing_area'] = features['left_arm_swing_x'] * features['left_arm_swing_y']
        
        if len(right_wrist_rel) > 5:
            right_wrist_rel = np.array(right_wrist_rel)
            features['right_arm_swing_x'] = np.std(right_wrist_rel[:, 0])
            features['right_arm_swing_y'] = np.std(right_wrist_rel[:, 1])
            
            # Calculate arm swing area
            features['right_arm_swing_area'] = features['right_arm_swing_x'] * features['right_arm_swing_y']
        
        # Calculate arm swing symmetry if both arms are visible
        if ('left_arm_swing_area' in features and 'right_arm_swing_area' in features and
            features['left_arm_swing_area'] > 0 and features['right_arm_swing_area'] > 0):
            
            # Ratio closer to 1.0 means more symmetric arm swing
            swing_ratio = min(features['left_arm_swing_area'], features['right_arm_swing_area']) / \
                          max(features['left_arm_swing_area'], features['right_arm_swing_area'])
            features['arm_swing_symmetry'] = swing_ratio
        
        return features

    def _calculate_joint_angles(self, track_id, keypoints_seq):
        """Calculate characteristic joint angles during movement"""
        features = {}
        
        # Elbow angles over time (characteristic of individual movement)
        left_elbow_angles = []
        right_elbow_angles = []
        
        # Knee angles over time
        left_knee_angles = []
        right_knee_angles = []
        
        for keypoints in keypoints_seq:
            # Left elbow angle
            if (len(keypoints) > self.left_shoulder and
                len(keypoints) > self.left_elbow and
                len(keypoints) > self.left_wrist and
                self.is_valid_keypoint(keypoints[self.left_shoulder]) and
                self.is_valid_keypoint(keypoints[self.left_elbow]) and
                self.is_valid_keypoint(keypoints[self.left_wrist])):
                
                angle = self._calculate_angle(
                    keypoints[self.left_shoulder],
                    keypoints[self.left_elbow],
                    keypoints[self.left_wrist]
                )
                left_elbow_angles.append(angle)
            
            # Right elbow angle
            if (len(keypoints) > self.right_shoulder and
                len(keypoints) > self.right_elbow and
                len(keypoints) > self.right_wrist and
                self.is_valid_keypoint(keypoints[self.right_shoulder]) and
                self.is_valid_keypoint(keypoints[self.right_elbow]) and
                self.is_valid_keypoint(keypoints[self.right_wrist])):
                
                angle = self._calculate_angle(
                    keypoints[self.right_shoulder],
                    keypoints[self.right_elbow],
                    keypoints[self.right_wrist]
                )
                right_elbow_angles.append(angle)
                
            # Left knee angle
            if (len(keypoints) > self.left_hip and
                len(keypoints) > self.left_knee and
                len(keypoints) > self.left_ankle and
                self.is_valid_keypoint(keypoints[self.left_hip]) and
                self.is_valid_keypoint(keypoints[self.left_knee]) and
                self.is_valid_keypoint(keypoints[self.left_ankle])):
                
                angle = self._calculate_angle(
                    keypoints[self.left_hip],
                    keypoints[self.left_knee],
                    keypoints[self.left_ankle]
                )
                left_knee_angles.append(angle)
            
            # Right knee angle
            if (len(keypoints) > self.right_hip and
                len(keypoints) > self.right_knee and
                len(keypoints) > self.right_ankle and
                self.is_valid_keypoint(keypoints[self.right_hip]) and
                self.is_valid_keypoint(keypoints[self.right_knee]) and
                self.is_valid_keypoint(keypoints[self.right_ankle])):
                
                angle = self._calculate_angle(
                    keypoints[self.right_hip],
                    keypoints[self.right_knee],
                    keypoints[self.right_ankle]
                )
                right_knee_angles.append(angle)
        
        # Store statistical features of joint angles
        if left_elbow_angles:
            features['left_elbow_angle_mean'] = np.mean(left_elbow_angles)
            features['left_elbow_angle_std'] = np.std(left_elbow_angles)
            features['left_elbow_angle_range'] = max(left_elbow_angles) - min(left_elbow_angles)
            
        if right_elbow_angles:
            features['right_elbow_angle_mean'] = np.mean(right_elbow_angles)
            features['right_elbow_angle_std'] = np.std(right_elbow_angles)
            features['right_elbow_angle_range'] = max(right_elbow_angles) - min(right_elbow_angles)
            
        if left_knee_angles:
            features['left_knee_angle_mean'] = np.mean(left_knee_angles)
            features['left_knee_angle_std'] = np.std(left_knee_angles)
            features['left_knee_angle_range'] = max(left_knee_angles) - min(left_knee_angles)
            
        if right_knee_angles:
            features['right_knee_angle_mean'] = np.mean(right_knee_angles)
            features['right_knee_angle_std'] = np.std(right_knee_angles)
            features['right_knee_angle_range'] = max(right_knee_angles) - min(right_knee_angles)
            
        # Calculate angle symmetry between left and right sides
        if left_elbow_angles and right_elbow_angles:
            features['elbow_angle_symmetry'] = 1.0 - abs(
                features['left_elbow_angle_mean'] - features['right_elbow_angle_mean']) / 180.0
                
        if left_knee_angles and right_knee_angles:
            features['knee_angle_symmetry'] = 1.0 - abs(
                features['left_knee_angle_mean'] - features['right_knee_angle_mean']) / 180.0
            
        # Update feature dictionary
        self.gait_features[track_id].update(features)
    
    def _analyze_posture(self, track_id, keypoints_seq):
        """Analyze posture characteristics"""
        features = {}
        
        # Calculate spine angles (hip to shoulder)
        spine_angles = []
        
        for keypoints in keypoints_seq:
            # Get midpoints of shoulders and hips
            if (len(keypoints) > self.left_shoulder and
                len(keypoints) > self.right_shoulder and
                len(keypoints) > self.left_hip and
                len(keypoints) > self.right_hip and
                self.is_valid_keypoint(keypoints[self.left_shoulder]) and
                self.is_valid_keypoint(keypoints[self.right_shoulder]) and
                self.is_valid_keypoint(keypoints[self.left_hip]) and
                self.is_valid_keypoint(keypoints[self.right_hip])):
                
                mid_shoulder = (keypoints[self.left_shoulder] + keypoints[self.right_shoulder]) / 2
                mid_hip = (keypoints[self.left_hip] + keypoints[self.right_hip]) / 2
                
                # Calculate angle from vertical
                spine_vector = mid_shoulder - mid_hip
                vertical_vector = np.array([0, -1])  # Pointing up
                
                cos_angle = np.dot(spine_vector, vertical_vector) / (
                    np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                
                spine_angles.append(angle)
        
        if spine_angles:
            features['spine_angle_mean'] = np.mean(spine_angles)
            features['spine_angle_std'] = np.std(spine_angles)
        
        # Update feature dictionary
        self.gait_features[track_id].update(features)
    
    def _extract_gait_cycle_features(self, track_id, keypoints_seq):
        """Extract timing features of the gait cycle"""
        features = {}
        
        # Track foot positions over time
        left_foot_y = []
        right_foot_y = []
        frames = []
        
        for i, keypoints in enumerate(keypoints_seq):
            left_ankle_valid = (len(keypoints) > self.left_ankle and 
                               self.is_valid_keypoint(keypoints[self.left_ankle]))
            right_ankle_valid = (len(keypoints) > self.right_ankle and 
                                self.is_valid_keypoint(keypoints[self.right_ankle]))
                
            if left_ankle_valid:
                left_foot_y.append(keypoints[self.left_ankle][1])
                frames.append(i)
                
            if right_ankle_valid:
                right_foot_y.append(keypoints[self.right_ankle][1])
        
        # Need enough frames for analysis
        if len(left_foot_y) > self.min_frames_for_analysis and len(right_foot_y) > self.min_frames_for_analysis:
            try:
                # Smooth signals
                left_smooth = savgol_filter(left_foot_y, min(9, len(left_foot_y) - (len(left_foot_y) % 2 - 1)), 3)
                right_smooth = savgol_filter(right_foot_y, min(9, len(right_foot_y) - (len(right_foot_y) % 2 - 1)), 3)
                
                # Find peaks for heel strike detection
                left_peaks, _ = find_peaks(left_smooth, distance=self.peak_distance)
                right_peaks, _ = find_peaks(right_smooth, distance=self.peak_distance)
                
                if len(left_peaks) > 1 and len(right_peaks) > 1:
                    # Calculate stride time (time between consecutive heel strikes)
                    left_stride_times = np.diff(left_peaks)
                    right_stride_times = np.diff(right_peaks)
                    
                    # Store stride time statistics if available
                    if len(left_stride_times) > 0:
                        features['left_stride_time_mean'] = np.mean(left_stride_times)
                        features['left_stride_time_std'] = np.std(left_stride_times)
                        
                    if len(right_stride_times) > 0:
                        features['right_stride_time_mean'] = np.mean(right_stride_times)
                        features['right_stride_time_std'] = np.std(right_stride_times)
                    
                    # Calculate phase difference between feet (normalized to stride cycle)
                    if len(left_peaks) > 0 and len(right_peaks) > 0:
                        # Find the closest peaks to calculate phase difference
                        min_dist = float('inf')
                        phase_diff = 0.0
                        
                        for lp in left_peaks:
                            for rp in right_peaks:
                                dist = abs(lp - rp)
                                if dist < min_dist:
                                    min_dist = dist
                                    # Calculate as a fraction of stride cycle
                                    if 'left_stride_time_mean' in features and features['left_stride_time_mean'] > 0:
                                        phase_diff = dist / features['left_stride_time_mean']
                        
                        features['step_phase_difference'] = min(phase_diff, 1.0 - phase_diff)
            except Exception:
                pass
                
        # Update feature dictionary
        self.gait_features[track_id].update(features)
    
    def _analyze_center_of_mass(self, track_id, keypoints_seq):
        """Analyze center of mass movement patterns"""
        features = {}
        
        # Track center of mass positions over time
        com_positions = []
        
        for keypoints in keypoints_seq:
            # Get all valid keypoints
            valid_keypoints = []
            for i in range(len(keypoints)):
                if len(keypoints) > i and self.is_valid_keypoint(keypoints[i]):
                    valid_keypoints.append(keypoints[i])
                    
            if valid_keypoints:
                # Calculate center of mass
                com = np.mean(valid_keypoints, axis=0)
                com_positions.append(com)
        
        if len(com_positions) >= 10:
            com_positions = np.array(com_positions)
            
            # Calculate vertical oscillation (distinctive walking characteristic)
            y_positions = com_positions[:, 1]
            features['com_vertical_oscillation'] = np.std(y_positions)
            
            # Calculate lateral stability (how much side-to-side movement)
            x_positions = com_positions[:, 0]
            features['com_lateral_stability'] = np.std(x_positions)
            
            # Calculate path smoothness using velocity changes
            if len(com_positions) > 1:
                velocities = np.diff(com_positions, axis=0)
                acc = np.diff(velocities, axis=0)
                features['path_smoothness'] = 1.0 / (1.0 + np.mean(np.linalg.norm(acc, axis=1)))
        
        self.gait_features[track_id].update(features)

    def _analyze_weight_transfer(self, track_id, keypoints_seq):
        """Analyze how weight transfers between feet"""
        features = {}
        
        # Track hip position relative to ankles
        hip_to_ankle_distances = []
        
        for keypoints in keypoints_seq:
            if (len(keypoints) > self.left_hip and 
                len(keypoints) > self.right_hip and
                len(keypoints) > self.left_ankle and
                len(keypoints) > self.right_ankle and
                self.is_valid_keypoint(keypoints[self.left_hip]) and
                self.is_valid_keypoint(keypoints[self.right_hip]) and
                self.is_valid_keypoint(keypoints[self.left_ankle]) and
                self.is_valid_keypoint(keypoints[self.right_ankle])):
                
                # Calculate mid-hip position
                mid_hip = (keypoints[self.left_hip] + keypoints[self.right_hip]) / 2
                
                # Calculate distances from mid-hip to each ankle (proxy for weight distribution)
                left_dist = np.linalg.norm(mid_hip - keypoints[self.left_ankle])
                right_dist = np.linalg.norm(mid_hip - keypoints[self.right_ankle])
                
                # Calculate left-right ratio (value close to 1 means even weight distribution)
                if min(left_dist, right_dist) > 0:
                    ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
                    hip_to_ankle_distances.append(ratio)
        
        if hip_to_ankle_distances:
            # Weight transfer stability - lower std means more consistent weight transfer
            features['weight_transfer_stability'] = 1.0 - np.std(hip_to_ankle_distances)
            # Weight distribution symmetry - higher value means more symmetric weight distribution
            features['weight_distribution_symmetry'] = np.mean(hip_to_ankle_distances)
        
        self.gait_features[track_id].update(features)

    def _analyze_head_movement(self, track_id, keypoints_seq):
        """Analyze head position and movement relative to body"""
        features = {}
        
        # Track head position relative to shoulders
        head_angles = []
        head_heights = []
        
        for keypoints in keypoints_seq:
            if (len(keypoints) > self.nose and
                len(keypoints) > self.left_shoulder and
                len(keypoints) > self.right_shoulder and
                self.is_valid_keypoint(keypoints[self.nose]) and
                self.is_valid_keypoint(keypoints[self.left_shoulder]) and
                self.is_valid_keypoint(keypoints[self.right_shoulder])):
                
                # Calculate mid-shoulder position
                mid_shoulder = (keypoints[self.left_shoulder] + keypoints[self.right_shoulder]) / 2
                
                # Vector from mid-shoulder to nose
                head_vector = keypoints[self.nose] - mid_shoulder
                vertical = np.array([0, -1])  # Pointing up
                
                # Calculate head angle from vertical
                cos_angle = np.dot(head_vector, vertical) / (np.linalg.norm(head_vector) * np.linalg.norm(vertical))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                head_angles.append(angle)
                
                # Calculate relative head height (how high the person holds their head)
                shoulder_width = np.linalg.norm(keypoints[self.left_shoulder] - keypoints[self.right_shoulder])
                if shoulder_width > 0:  # Avoid division by zero
                    relative_height = np.linalg.norm(head_vector) / shoulder_width
                    head_heights.append(relative_height)
        
        if head_angles:
            features['head_angle_mean'] = np.mean(head_angles)
            features['head_angle_std'] = np.std(head_angles)
        
        if head_heights:
            features['head_height_ratio'] = np.mean(head_heights)
        
        self.gait_features[track_id].update(features)

    def _analyze_motion_dynamics(self, track_id, keypoints_seq):
        """Analyze acceleration/deceleration patterns in movement"""
        features = {}
        
        # Track hip center position
        hip_positions = []
        
        for keypoints in keypoints_seq:
            if (len(keypoints) > self.left_hip and 
                len(keypoints) > self.right_hip and
                self.is_valid_keypoint(keypoints[self.left_hip]) and
                self.is_valid_keypoint(keypoints[self.right_hip])):
                
                # Calculate mid-hip position
                mid_hip = (keypoints[self.left_hip] + keypoints[self.right_hip]) / 2
                hip_positions.append(mid_hip)
        
        if len(hip_positions) >= 3:  # Need at least 3 points for acceleration
            hip_positions = np.array(hip_positions)
            
            # Calculate velocities and accelerations
            velocities = np.diff(hip_positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            
            # Speed changes (how quickly the person changes walking speed)
            speed_changes = np.diff(np.linalg.norm(velocities, axis=1))
            
            if len(speed_changes) > 0:
                features['acceleration_capacity'] = np.percentile(speed_changes, 95)
                features['deceleration_capacity'] = np.percentile(speed_changes, 5)
                features['speed_stability'] = 1.0 / (1.0 + np.std(np.linalg.norm(velocities, axis=1)))
        
        self.gait_features[track_id].update(features)

    def _analyze_movement_periodicity(self, track_id, keypoints_seq):
        """Analyze the periodicity of movement using frequency analysis"""
        features = {}
        
        # Get ankles y-positions (vertical movement)
        left_ankle_y = []
        right_ankle_y = []
        
        for keypoints in keypoints_seq:
            if len(keypoints) > self.left_ankle and self.is_valid_keypoint(keypoints[self.left_ankle]):
                left_ankle_y.append(keypoints[self.left_ankle][1])
            
            if len(keypoints) > self.right_ankle and self.is_valid_keypoint(keypoints[self.right_ankle]):
                right_ankle_y.append(keypoints[self.right_ankle][1])
        
        # Need enough points for frequency analysis
        if len(left_ankle_y) >= 20 and len(right_ankle_y) >= 20:
            try:
                from scipy.fft import fft
                
                # Apply FFT to ankle movements
                left_fft = np.abs(fft(left_ankle_y - np.mean(left_ankle_y)))
                right_fft = np.abs(fft(right_ankle_y - np.mean(right_ankle_y)))
                
                # Get dominant frequencies (excluding DC component)
                left_freq_idx = np.argsort(left_fft[1:len(left_fft)//2])[-3:] + 1
                right_freq_idx = np.argsort(right_fft[1:len(right_fft)//2])[-3:] + 1
                
                # Store dominant frequencies and their amplitudes
                features['left_dominant_freq'] = left_freq_idx[0] / len(left_ankle_y)
                features['right_dominant_freq'] = right_freq_idx[0] / len(right_ankle_y)
                features['gait_rhythm_ratio'] = min(features['left_dominant_freq'], features['right_dominant_freq']) / \
                                            max(features['left_dominant_freq'], features['right_dominant_freq'])
            except Exception:
                pass
        
        self.gait_features[track_id].update(features)

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points in degrees"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        
        return angle
    
    def export_features_csv(self, filename):
        """Export all gait features to CSV file"""
        data = []
        
        for track_id, features in self.gait_features.items():
            # Ensure all expected features are present
            features_with_id = {'track_id': track_id}
            features_with_id.update(features)
            data.append(features_with_id)
        
        if data:
            df = pd.DataFrame(data)
            
            # Check feature count
            expected_features = 26  # Update this to match your expected count
            actual_features = len(df.columns) - 1  # -1 for track_id
            
            if actual_features < expected_features:
                print(f"WARNING: Only extracted {actual_features}/{expected_features} features")
                # Identify missing features
                print(f"Missing features may affect identification accuracy")
                
            df.to_csv(filename, index=False)
            print(f"Exported {len(data)} gait feature records with {actual_features} features")
            return True
        else:
            print("No gait features to export")
            return False

    def get_feature_vector(self, track_id):
        """Get a normalized feature vector for machine learning models"""
        if track_id not in self.gait_features:
            return None
            
        features = self.gait_features[track_id]
        
        # Extract the features we want to use for identification
        # Prioritize direction-invariant features
        vector_keys = [
            # Body proportions (view invariant)
            'height_width_ratio', 'leg_torso_ratio', 'upper_lower_leg_ratio',
            # Walking pattern features
            'left_stride_consistency', 'right_stride_consistency',
            'arm_swing_symmetry', 'left_arm_swing_area', 'right_arm_swing_area',
            # Joint angles
            'left_elbow_angle_range', 'right_elbow_angle_range',
            'left_knee_angle_range', 'right_knee_angle_range',
            'elbow_angle_symmetry', 'knee_angle_symmetry',
            # Posture
            'spine_angle_mean', 'spine_angle_std',
            # Gait timing
            'step_phase_difference'
        ]
        vector_keys.extend([
            'com_vertical_oscillation', 'com_lateral_stability', 'path_smoothness',
            'weight_transfer_stability', 'weight_distribution_symmetry',
            'head_angle_mean', 'head_height_ratio',
            'speed_stability', 'gait_rhythm_ratio'
        ])
        
        # Create vector using available features
        vector = []
        for key in vector_keys:
            if key in features:
                vector.append(features[key])
            else:
                vector.append(np.nan)
        
        # Handle missing values
        vector = np.array(vector)
        vector = np.nan_to_num(vector, nan=0.0)
        
        return vector
    
    def get_keypoint_sequences(self):
        """Extract raw keypoint sequences for each track ID"""
        sequences = {}
        
        for track_id, history in self.track_history.items():
            if len(history) < self.min_frames_for_analysis:
                continue
            
            # Extract keypoints in sequence
            keypoints_seq = []
            for _, kpts in history:
                if kpts is not None and len(kpts) > 0:
                    # Flatten keypoints to a single vector [x1,y1,x2,y2,...,xn,yn]
                    flattened = []
                    for i in range(len(kpts)):
                        if self.is_valid_keypoint(kpts[i]):
                            flattened.extend(kpts[i])
                        else:
                            # Add zeros for missing keypoints
                            flattened.extend([0, 0])
                    keypoints_seq.append(flattened)
            
            if keypoints_seq:
                sequences[track_id] = keypoints_seq
        
        return sequences

    def export_sequences_for_lstm(self, filename):
        """Export skeleton sequences to a file for LSTM training"""
        sequences = self.get_keypoint_sequences()
        
        if sequences:
            # Save using torch.save or numpy
            np.save(filename, sequences)
            print(f"Exported {len(sequences)} skeleton sequences")
            return True
        else:
            print("No sequences to export")
            return False

    def normalize_keypoints(self, keypoints_seq):
        """Normalize keypoints for neural network input"""
        # Center keypoints around hip center
        normalized = []
        
        for keypoints in keypoints_seq:
            if len(keypoints) > self.left_hip and len(keypoints) > self.right_hip:
                hip_center = (keypoints[self.left_hip] + keypoints[self.right_hip]) / 2
                
                # Shift to center
                centered = [kp - hip_center if self.is_valid_keypoint(kp) else np.zeros(2) for kp in keypoints]
                
                # Scale by the hip-shoulder distance for size invariance
                if (len(keypoints) > self.left_shoulder and 
                    len(keypoints) > self.right_shoulder and
                    self.is_valid_keypoint(keypoints[self.left_shoulder]) and
                    self.is_valid_keypoint(keypoints[self.right_shoulder])):
                    
                    shoulder_center = (keypoints[self.left_shoulder] + keypoints[self.right_shoulder]) / 2
                    scale_factor = np.linalg.norm(shoulder_center - hip_center)
                    
                    if scale_factor > 0:
                        normalized_kpts = [kp / scale_factor for kp in centered]
                        normalized.append(normalized_kpts)
        
        return normalized


class SkeletonSequence(Dataset):
    """Dataset class for skeleton sequences with corresponding person IDs"""
    
    def __init__(self, sequences, labels, seq_length=30, transform=None):
        """
        Args:
            sequences: Dictionary mapping track_id to list of keypoints sequences
            labels: Dictionary mapping track_id to person identity
            seq_length: Fixed sequence length for LSTM input
            transform: Optional transform to apply to sequences
        """
        self.sequences = []
        self.labels = []
        self.label_map = {}  # Maps string labels to integers
        self.inv_label_map = {}  # Inverse mapping
        self.seq_length = seq_length
        self.transform = transform
        
        # Process sequences and labels
        label_idx = 0
        for track_id, seq in sequences.items():
            if track_id in labels:
                # Create fixed-length sequences
                for i in range(0, max(1, len(seq) - seq_length + 1), seq_length // 2):  # 50% overlap
                    # Extract sequence chunk
                    end_idx = min(i + seq_length, len(seq))
                    if end_idx - i < seq_length // 2:  # Skip if too short
                        continue
                        
                    chunk = seq[i:end_idx]
                    
                    # Pad if necessary
                    if len(chunk) < seq_length:
                        # Pad with zeros
                        pad_size = seq_length - len(chunk)
                        chunk = chunk + [np.zeros_like(chunk[0])] * pad_size
                    
                    # Add to dataset
                    self.sequences.append(chunk)
                    
                    # Process label
                    label = labels[track_id]
                    if label not in self.label_map:
                        self.label_map[label] = label_idx
                        self.inv_label_map[label_idx] = label
                        label_idx += 1
                    
                    self.labels.append(self.label_map[label])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        seq = np.array(seq)
        
        # Apply transform if available
        if self.transform:
            seq = self.transform(seq)
        
        # Convert to tensor
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return seq_tensor, label_tensor
    
    def get_num_classes(self):
        """Return the number of unique classes"""
        return len(self.label_map)
    
    def get_label_mapping(self):
        """Return the label mapping dictionary"""
        return self.inv_label_map
    
class SkeletonLSTM(nn.Module):
    """LSTM model for skeleton sequence classification"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        """
        Args:
            input_size: Number of features in input (keypoints x 2)
            hidden_size: Size of LSTM hidden states
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(SkeletonLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Classification layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM output
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        # attn_weights shape: (batch_size, sequence_length, 1)
        
        # Apply attention to LSTM output
        context = torch.sum(lstm_out * attn_weights, dim=1)
        # context shape: (batch_size, hidden_size)
        
        # Classification
        output = self.fc(context)
        # output shape: (batch_size, num_classes)
        
        return output

# Helper function to visualize gait features on frame
def visualize_gait_features(frame, track_id, gait_features, x_pos, y_pos):
    """Draw gait features on the frame for a specific track"""
    if track_id not in gait_features:
        return frame
    
    features = gait_features[track_id]
    y_offset = 0
    
    # Show movement direction if available
    if 'movement_direction' in features:
        text = f"Direction: {features['movement_direction']}"
        cv.putText(frame, text, (x_pos, y_pos + y_offset), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20
    
    # Display the most distinctive features for identification
    key_features = [
        'height_width_ratio', 'leg_torso_ratio', 
        'arm_swing_symmetry', 'spine_angle_mean'
    ]
    
    for key in key_features:
        if key in features:
            value = features[key]
            text = f"{key}: {value:.2f}"
            cv.putText(frame, text, (x_pos, y_pos + y_offset), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
    
    return frame