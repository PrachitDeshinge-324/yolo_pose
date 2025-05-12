import cv2
import torch
import numpy as np
import time
from tqdm import tqdm
import logging
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

# Configure logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("deep_sort_realtime").setLevel(logging.WARNING)

class IndustrialPersonTracker:
    def __init__(self, det_model_path='yolo11x.pt', pose_model_path='yolo11x-pose.pt', 
                 min_confidence=0.4, max_age=30, n_init=5, max_cosine_distance=0.4):
        self.device = self._get_device()
        self.det_model = YOLO(det_model_path).to(self.device)
        self.pose_model = YOLO(pose_model_path).to(self.device)
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            embedder='torchreid',
            embedder_model_name='osnet_ain_x1_0',
            embedder_wts='torchreid://osnet_ain_x1_0',
            half=(self.device=='cuda')
        )
        self.tracks = {}
        self.next_id = 1
        self.min_confidence = min_confidence

    def _get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    class TrackedWorker:
        def __init__(self, track_id, features, pose, position):
            self.id = track_id
            self.features = deque([features], maxlen=5)
            self.pose_history = deque([], maxlen=5)
            self.trajectory = deque([position], maxlen=15)
            self.last_seen = time.time()
            self.color = tuple(np.random.randint(0, 255, 3).tolist())
            if pose is not None:
                self.pose_history.append(pose)

        def update(self, features, pose, position):
            if features is not None:
                self.features.append(features)
            if pose is not None and len(pose) >= 17:
                self.pose_history.append(pose)
            self.trajectory.append(position)
            self.last_seen = time.time()

        def similarity(self, other_features, current_pose, position):
            try:
                # Appearance similarity
                app_sim = max(np.dot(f, other_features) for f in self.features) if other_features is not None else 0
                # Pose similarity
                pose_sim = self._pose_similarity(current_pose)
                # Motion consistency
                motion_sim = self._motion_consistency(position)
                return 0.4*app_sim + 0.4*pose_sim + 0.2*motion_sim
            except Exception:
                return 0

        def _pose_similarity(self, current_pose):
            if not self.pose_history or current_pose is None or len(current_pose) < 17:
                return 0
            stable_joints = [5, 6, 11, 12]
            valid_poses = [p for p in self.pose_history if p is not None and len(p) >= 17]
            if not valid_poses:
                return 0
            sim = 0
            for joint in stable_joints:
                hist = np.mean([p[joint] for p in valid_poses], axis=0)
                dist = np.linalg.norm(hist - current_pose[joint])
                sim += np.exp(-dist/50)
            return sim / len(stable_joints)

        def _motion_consistency(self, position):
            if len(self.trajectory) < 2:
                return 1.0
            prev, last = self.trajectory[-2], self.trajectory[-1]
            dx, dy = last[0]-prev[0], last[1]-prev[1]
            pred = (last[0]+dx, last[1]+dy)
            dist = np.linalg.norm(np.array(pred) - np.array(position))
            return np.exp(-dist/20)

    def track_workers(self, video_path, output_path='industrial_tracking_output.mp4'):
        cap = cv2.VideoCapture(video_path)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = self._safe_detect_people(frame)
            tracks = self.tracker.update_tracks(detections, frame=frame)
            self._process_tracks(frame, tracks)
            self._draw_results(frame)
            out.write(frame)
            self._cleanup_tracks()

        cap.release()
        out.release()
        print(f"Processing complete. Output saved to {output_path}")

    def _safe_detect_people(self, frame):
        try:
            results = self.det_model.predict(frame, conf=self.min_confidence, classes=[0], verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    dets.append(([x1, y1, x2-x1, y2-y1], box.conf.item(), 'person'))
            return dets
        except Exception:
            return []

    def _process_tracks(self, frame, tracks):
        current = set()
        for tr in tracks:
            if not tr.is_confirmed(): continue
            x1, y1, x2, y2 = map(int, tr.to_ltrb())
            center = ((x1+x2)//2, (y1+y2)//2)
            features = tr.features[-1] if tr.features else None
            pose = None
            if (x2-x1)>50 and (y2-y1)>50:
                pose = self._get_valid_pose(frame[y1:y2, x1:x2])
            # match existing
            best_id, best_score = None, 0.65
            for tid, worker in self.tracks.items():
                if tid in current: continue
                score = worker.similarity(features, pose, center)
                if score>best_score:
                    best_id, best_score = tid, score
            if best_id:
                self.tracks[best_id].update(features, pose, center)
                current.add(best_id)
            else:
                nid = self._generate_consistent_id(features, pose, center)
                self.tracks[nid] = self.TrackedWorker(nid, features, pose, center)
                current.add(nid)

    def _get_valid_pose(self, crop):
        try:
            if crop.size == 0:
                return None
            res = self.pose_model.predict(crop, verbose=False)
            kps = res[0].keypoints.xy.cpu().numpy() if res and res[0].keypoints else None
            return kps[0] if kps is not None and len(kps)>0 else None
        except Exception:
            return None

    def _generate_consistent_id(self, features, pose, position):
        # always assign a fresh ID
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def _draw_results(self, frame):
        for worker in self.tracks.values():
            for i in range(1, len(worker.trajectory)):
                cv2.line(frame, worker.trajectory[i-1], worker.trajectory[i], worker.color, 2)
            if worker.trajectory:
                x, y = worker.trajectory[-1]
                cv2.putText(frame, f"ID: {worker.id}", (x-10, y-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, worker.color, 2)

    def _cleanup_tracks(self, max_age=30):
        now = time.time()
        to_remove = [tid for tid, w in self.tracks.items() if now - w.last_seen > max_age]
        for tid in to_remove:
            del self.tracks[tid]

if __name__ == "__main__":
    tracker = IndustrialPersonTracker()
    tracker.track_workers('/content/drive/MyDrive/datasets/My Movie.mp4')
