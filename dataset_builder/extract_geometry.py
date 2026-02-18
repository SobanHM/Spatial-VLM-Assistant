import numpy as np
from collections import defaultdict
from dataset_builder.utils import GeometryUtils
class GeometryExtractor:
    def __init__(self):
        self.utils = GeometryUtils()
    def extract_objects(self, sample):
        image = sample["image"]
        depth = sample["depth"]
        semantic = sample["semantic"]
        height, width = semantic.shape
        object_ids = np.unique(semantic)
        objects = []
        for obj_id in object_ids:
            if obj_id == 0:
                continue  # skip background
            mask = (semantic == obj_id)
            median_depth = self.utils.compute_median_depth(depth, mask)
            if median_depth is None:
                continue
            cx, cy = self.utils.compute_centroid(mask)
            if cx is None:
                continue
            direction = self.utils.estimate_direction(cx, width)
            distance_label = self.utils.distance_bucket(median_depth)
            objects.append({
                "object_id": int(obj_id),
                "distance_m": median_depth,
                "distance_label": distance_label,
                "direction": direction,
                "centroid": (cx, cy)
            })
        return objects
