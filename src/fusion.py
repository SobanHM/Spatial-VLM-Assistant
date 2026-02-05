import re
import cv2
import numpy as np


class SpatialFusion:
    def __init__(self, perception, reasoning):
        self.perception = perception  # GPU 1
        self.reasoning = reasoning  # GPU 0

    def run_inference(self, image_path):
        # 1. Get raw text from LLaVA
        raw_text = self.reasoning.get_spatial_objects(image_path)
        print(f"\n--- Brain Reasoning ---\n{raw_text}\n----------------------")

        # 2. Get Depth Map
        import cv2
        frame = cv2.imread(image_path)
        h, w, _ = frame.shape
        depth_map = self.perception.get_distance_map(frame)

        # 3. Flexible Regex: Handles [100, 200...] AND [0.1, 0.2...]
        # It looks for 4 numbers (float or int) inside brackets
        pattern = r"([a-zA-Z]+)\s*\[([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\]"
        matches = re.findall(pattern, raw_text)

        results = []
        for name, ymin, xmin, ymax, xmax in matches:
            coords = [float(ymin), float(xmin), float(ymax), float(xmax)]

            # Use 1.0 scale since your output showed decimals
            y1, x1 = int(coords[0] * h), int(coords[1] * w)
            y2, x2 = int(coords[2] * h), int(coords[3] * w)

            # SHRINK the box by 10% (Center Crop)
            # This ensures we don't pick up background pixels or edges
            pad_h = int((y2 - y1) * 0.1)
            pad_w = int((x2 - x1) * 0.1)

            y1_c, y2_c = y1 + pad_h, y2 - pad_h
            x1_c, x2_c = x1 + pad_w, x2 - pad_w

            # Extract Depth from the CENTER of the object
            obj_depth = depth_map[y1_c:y2_c, x1_c:x2_c]

            if obj_depth.size > 0:
                distance = np.percentile(obj_depth, 50)  # Median
                # If median is 0, try the max value in that area
                if distance == 0:
                    distance = np.max(obj_depth)

                results.append({
                    "label": name,
                    "distance": round(float(distance), 2),
                    "box": [x1, y1, x2, y2]
                })

        return results, frame
