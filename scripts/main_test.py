import sys
import os
# add parent directory (root) to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.perception_1 import SpatialEyes
from src.reasoning_1 import  SpatialReasoning
from src.fusion import  SpatialFusion
from src.detection import SpatialDetector
import cv2

print("Starting Dual-GPU Pipeline...")
eyes = SpatialEyes(device='cuda:1')
brain = SpatialReasoning(device_id=0)
fusion = SpatialFusion(eyes, brain)

# process mall image
img_path = r"C:\Users\soban\OneDrive\VLM_Evaluation_Benchmark_Dataset(FYP)\Pakistani_Supermarkets_Dataset\sm_54.jpg"
results, frame = fusion.run_inference(img_path)

# draw and show results
print("\nFinal Spatial Results")
for res in results:
    x1, y1, x2, y2 = res['box']
    label = f"{res['label']}: {res['distance']}m"
    print(f"Detected: {label}")

    # draw box and text on the frame
    detector = SpatialDetector(device="cuda:0")
    eyes = SpatialEyes(device="cuda:1")
    brain = SpatialReasoning(device_id=0)
    fusion = SpatialFusion(detector, brain, eyes)


    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# save image in the same folder as this script
output_path = os.path.join(SCRIPT_DIR, "7elevan_shopping_mall_meatcorner.jpg")

cv2.imwrite(output_path, frame)
print(f"\nProcess complete! Check {output_path}")
