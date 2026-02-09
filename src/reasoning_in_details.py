import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image


class SpatialReasoning:
    def __init__(self, device_id=0):
        self.model_id = "llava-hf/llava-1.5-7b-hf"
        self.device = f"cuda:{device_id}"

        print(f"Loading LLaVA-1.5-7B onto {self.device}...")

        # 4-bit config to fit inside 8GB VRAM
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map={"": self.device}  # Force GPU 0
        )
        print(f"Reasoning System ({self.device}) is Online.")

    def ask_about_image(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")

        # Standard LLaVA v1.5 prompt format
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=200)

        return self.processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1]

    # ADD THE NEW METHOD HERE:
    def get_spatial_objects(self, image_path):
        prompt = (
            "Describe the scene by identifying 4 main objects. For each, "
            "provide the name and the bounding box in [ymin, xmin, ymax, xmax] format "
            "using decimal coordinates (0.0 to 1.0). "
            "For Example: man [0.1, 0.5, 0.9, 0.8]"
            "For Example: Nearest Chair [0.2, 0.6, 0.8, 0.7]"
        )
        response = self.ask_about_image(image_path, prompt)
        return response

if __name__ == "__main__":
    # Test LLaVA independently
    brain = SpatialReasoning()
