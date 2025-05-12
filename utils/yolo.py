from ultralytics import YOLO
import torch

def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def get_skeleton_points(frame, model=None, model_path=None, device=None):
    """
    Get skeleton points from the frame using YOLO model.
    """
    if model is None:
        if device is None:
            device = get_best_device()
        model = YOLO(model_path).to(device)
    results = model.predict(frame, conf=0.5, device=device)
    skeleton_points = []
    for result in results:
        if hasattr(result.keypoints, "xyn"):
            skeleton_points.extend(result.keypoints.xyn.cpu().numpy())
        else:
            skeleton_points.extend(result.keypoints.xy.cpu().numpy())
    return skeleton_points

def get_people_crops(frame, model=None, model_path=None, device=None):
    """
    Get people crops from the frame using YOLO model.
    """
    if model is None:
        if device is None:
            device = get_best_device()
        model = YOLO(model_path).to(device)
    results = model.predict(frame, conf=0.5, device=device)
    people_crops = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            person_crop = frame[y1:y2, x1:x2]
            people_crops.append(person_crop)
    return people_crops

def get_silhouette(frame, model=None, model_path=None, device=None):
    """
    Get silhouette from the frame using YOLO model.
    """
    if model is None:
        if device is None:
            device = get_best_device()
        model = YOLO(model_path).to(device)
    results = model.predict(frame, conf=0.5, device=device)
    silhouettes = []
    for result in results:
        masks = result.masks
        # Check if masks is None before iterating
        if masks is not None:
            for mask in masks:
                silhouette = mask.data.cpu().numpy()
                silhouettes.append(silhouette)
    return silhouettes