import re
import cv2
import numpy as np

class SpatialFusion:
    """
    Spatial Fusion Module
    - Detector decides WHERE objects are (ground truth)
    - LLaVA describes WHAT the object is (region constrained)
    - Depth model estimates HOW FAR the object is
    """

    def __init__(self, detector, reasoning, perception):
        self.detector = detector      # YOLO / GroundingDINO (GPU 0)
        self.reasoning = reasoning    # LLaVA (GPU 0)
        self.perception = perception  # Depth (GPU 1)

    def run_inference(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {image_path}")

        h, w = frame.shape[:2]

        # 1 object detection (gt boxes)
        detections = self.detector.detect(frame)

        if len(detections) == 0:
            print("No objects detected.")
            return [], frame

        # 2: depth estimation (full frame)
        depth_map = self.perception.get_distance_map(frame)

        results = []

        # 3 per-object Region reasoning
        for det in detections:
            x1, y1, x2, y2 = det["box"]

            # Safety clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # crop object region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # LLaVA: describe ONLY this region
            label = self.reasoning.describe_region(crop)

            # Hard reject hallucinated or useless outputs
            if label in ["unknown", "", None]:
                continue

            # depth dstimation (CENTER REGION ONLY)
            box_h = y2 - y1
            box_w = x2 - x1

            pad_h = int(0.2 * box_h)
            pad_w = int(0.2 * box_w)

            cy1 = y1 + pad_h
            cy2 = y2 - pad_h
            cx1 = x1 + pad_w
            cx2 = x2 - pad_w

            obj_depth = depth_map[cy1:cy2, cx1:cx2]

            distance = -1.0
            valid_depth = obj_depth[obj_depth > 0]

            if valid_depth.size > 0:
                distance = float(np.median(valid_depth))

            # final result entry
            results.append({
                "label": label,
                "detector_label": det["label"],
                "confidence": round(det["confidence"], 2),
                "distance": round(distance, 2),
                "box": [x1, y1, x2, y2]
            })

        return results, frame
        
