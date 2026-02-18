from datasets import load_dataset
import numpy as np
# import os
# import json

# dataset = load_dataset("jagennath-hari/nyuv2", split="train")
# sample = dataset[0]
# rgb = np.array(sample["rgb"])
# depth_raw = np.array(sample["depth"])
# semantic_raw = np.array(sample["semantic"])
# instance_raw = np.array(sample["instance"])

class NYUv2Loader:
    def __init__(self, split="train"):
        self.dataset = load_dataset("jagennath-hari/nyuv2", split=split)

    def __len__(self):
        return len(self.dataset)

    def get_sample(self, index):
        sample = self.dataset[index]

        image = np.array(sample["rgb"])
        # depth = np.array(sample["depth"])
        depth = np.array(sample["depth"]).astype(np.float32)
        depth = depth / 1000.0  # converted milimeter to meters

        semantic = np.array(sample["semantic"])
        instance = np.array(sample["instance"])

        return {
            "image": image,
            "depth": depth,
            "semantic": semantic,
            "instance": instance
        }
