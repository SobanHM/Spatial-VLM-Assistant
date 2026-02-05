import cv2
import torch
import os
import numpy as np
from perception import SpatialEyes
from reasoning import SpatialReasoning

def run_assistant():
    print("Starting Spatial-VLM Assistant..")

    # 1 initialize Perception (GPU 1)
    eyes = SpatialEyes(device='cuda:1')

    # 2 initialize Reasoning (GPU 0)
    brain = SpatialReasoning(device_id=0)

    print("\nsystem ready with depth anything")

    img_path = r"C:\Users\soban\OneDrive\VLM_Evaluation_Benchmark_Dataset(FYP)\Pakistani_Supermarkets_Dataset\sm_38.jpg"

    if not os.path.exists(img_path):
        print(f"error: image not found at {img_path}")
        return

    frame = cv2.imread(img_path)
    if frame is None:
        print("error: can not decode image")
        return

    print(f"image loaded: {img_path}")

    # Step A: depth analysis
    print("perception: analyzing depth (GPU 0)")
    depth_map = eyes.get_metric_depth(frame)

    # statistics calculation
    valid_mask = depth_map > 0.1  # ignore noise
    if np.any(valid_mask):
        valid_depths = depth_map[valid_mask]
        avg_depth = np.mean(valid_depths)
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        spatial_context = f"closest person/object is {min_depth:.2f} meters away. The scene is about {avg_depth:.2f} meters deep."
    else:
        avg_depth = min_depth = max_depth = 0.0
        spatial_context = "Distance information is currently unavailable."
        print("warning: depth map is still returning zero")

    # step B: reasoning with injecting actual values
    prompt = (
        f"USER: <image>\nCONTEXT: {spatial_context}\n"
        "Question: Are people standing in front of me? "
        "Describe the people in the foreground and their distance. "
        "Is the path blocked based on the distance provided?\nAssistant(LLaVA):"
    )

    description = brain.analyze_scene(frame, prompt=prompt)

    # final results
    print("\n" + "=" * 35)
    print("Spatial-vlm analytics:")
    print("=" * 35)
    print(f"Scene: {description.strip()}")
    print("-" * 35)
    print(f"average distance: {avg_depth:.2f}m")
    print(f"closest object: {min_depth:.2f}m")
    print(f"Deepest Point: {max_depth:.2f}m")
    print("=" * 35)

if __name__ == "__main__":
    run_assistant()
