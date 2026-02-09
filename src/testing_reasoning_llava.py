from reasoning_1 import SpatialReasoning

brain = SpatialReasoning()
#
image_path = r"C:\Users\soban\OneDrive\VLM_Evaluation_Benchmark_Dataset(FYP)\Pakistani_Supermarkets_Dataset\sm_12.jpg"
response = brain.ask_about_image(image_path, "List the main objects you see and their general positions (left, right, center).")
print(f"LLaVA says: {response}")
