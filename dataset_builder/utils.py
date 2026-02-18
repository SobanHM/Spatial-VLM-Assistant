import numpy as np

class GeometryUtils:

    @staticmethod
    def compute_median_depth(depth_map, mask):
        """Compute robust depth using median."""
        valid_pixels = depth_map[mask > 0]
        if len(valid_pixels) == 0:
            return None
        return float(np.median(valid_pixels))

    @staticmethod
    def compute_centroid(mask):
        """ Compute object centroid """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None, None
        return float(np.mean(xs)), float(np.mean(ys))

    @staticmethod
    def estimate_direction(x, image_width):
        """Estimate left/center/right direction."""
        if x < image_width / 3:
            return "left"
        elif x < 2 * image_width / 3:
            return "center"
        else:
            return "right"

    @staticmethod
    def distance_bucket(distance):
        """Convert raw meters into human-friendly category."""
        if distance < 1.0:
            return "very close"
        elif distance < 2.5:
            return "near"
        elif distance < 4.5:
            return "moderately far"
        else:
            return "far"
