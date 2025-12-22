# Spatial-VLM-Assistant: Metric Depth-Aware Visual Reasoning

A state-of-the-art Assistive Spatial AI system that integrates **Vision-Language Models (LLaVA v1.5)** with **Metric Depth Transformers (Depth Anything V2)** to provide real-world spatial awareness.

## 🚀 The Vision
Standard VLMs can describe "what" is in a scene but are "space-blind"—they cannot accurately estimate distance. This project bridges that gap for assistive navigation (e.g., helping a user navigate a supermarket) by providing real-time metric distance and orientation of objects.

## 🛠️ Tech Stack & Methodology
- **Brain (Semantic):** LLaVA v1.5 (7B) via 4-bit Quantization (GPU 0)
- **Eyes (Geometric):** Depth Anything V2 - ViT-Large (GPU 1)
- **Dataset:** NYU Depth V2 (Metric Validation)
- **Hardware Strategy:** Dual-GPU Model Parallelism (2x RTX Super 8GB)



## 📈 Methodology Workflow
1. **Metric Calibration:** Validating Depth Anything V2 against NYU Ground Truth `.mat` files to ensure sub-10cm accuracy.
2. **Object-to-Depth Fusion:** Extracting LLaVA bounding boxes and mapping them to Median Metric Depth maps.
3. **Spatial Prompting:** Injecting physical coordinates into the LLM context for "Embodied Reasoning."

## 📂 Project Structure
- `src/`: Core logic for dual-GPU inference and spatial fusion.
- `notebooks/`: NYU Depth Dataset validation and metric benchmarking.
- `models/`: Checkpoints for Depth Anything V2 (Metric Hypersim).