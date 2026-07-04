import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

# 2 one level to Spatial-VLM-Assistant
PROJECT_ROOT = CURRENT_DIR.parent

# 3. goes DOWN into 'models/checkpoints' to find the file
weights_path = PROJECT_ROOT / "models" / "checkpoints" / "depth_anything_v2_metric_hypersim_vitl.pth"

SUBMODULE_PATH = CURRENT_DIR / "Depth-Anything-V2"

# Add submodule to sys.path to import from it
if str(SUBMODULE_PATH) not in sys.path:
    sys.path.append(str(SUBMODULE_PATH))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print(f"Error: Submodule not found at {SUBMODULE_PATH}")
    print("run the command: git submodule update --init --recursive")
    sys.exit(1)


class SpatialEyes:
    def __init__(self, device='cuda:1'):
        self.device = device

        # critical fix --
        # telling torch: "For this entire process, cuda means gpu device 1
        if 'cuda' in self.device:
            device_id = int(self.device.split(':')[-1])
            torch.cuda.set_device(device_id)
        # critical fix end
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        print(f"Initializing Depth Anything V2 on {self.device}...")
        self.model = DepthAnythingV2(**model_configs['vitl'])

        project_root = Path(__file__).resolve().parent.parent
        weights_path = project_root / "models" / "checkpoints" / "depth_anything_v2_metric_hypersim_vitl.pth"

        if not weights_path.exists():
            print(f"FATAL: Weights not found at {weights_path}")
            sys.exit(1)

        self.model.load_state_dict(torch.load(str(weights_path), map_location='cpu', weights_only=True))
        self.model.to(self.device).eval()
        print(f"Perception Model loaded on {self.device}")

    def get_metric_depth(self, frame):
        """Takes a BGR frame and returns depth in meters."""
        with torch.no_grad():
            # Because we set torch.cuda.set_device, infer_image
            # will now create internal tensors on cuda:1 automatically.
            depth = self.model.infer_image(frame)
        return depth

if __name__ == "__main__":
    # Test block
    try:
        # Check if CUDA 1 exists, else fallback to CUDA 0 for testing
        target_device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'

        eyes = SpatialEyes(device=target_device)

        # Create a blank test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_map = eyes.get_metric_depth(test_img)

        print(f"SUCCESS! Test inference complete.")
        print(f"Depth Map Shape: {depth_map.shape}")
        print(f"Min depth: {depth_map.min():.2f}m, Max depth: {depth_map.max():.2f}m")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
