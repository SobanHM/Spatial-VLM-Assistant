import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig


class SpatialReasoning:
    def __init__(self, device_id=0):
        self.model_id = "llava-hf/llava-1.5-7b-hf"
        self.device = f"cuda:{device_id}"

        print(f"Initializing LLaVA on {self.device} (4-bit)...")

        # 4-bit configuration to fit in 8GB VRAM
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map={"": self.device}  # Pins specifically to GPU 0
        )
        print(f"Reasoning Model (LLaVA) loaded on {self.device}")

    def analyze_scene(self, image, prompt="What objects are in front of me?"):
        # formatting for LLaVA v1.5
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=100)

        return self.processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1]


if __name__ == "__main__":
    # test initialization
    brain = SpatialReasoning()