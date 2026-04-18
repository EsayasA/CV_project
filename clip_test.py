import torch
from transformers import CLIPProcessor, CLIPModel

class ClipValidator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_score(self, image, text_query):
        inputs = self.processor(text=[text_query], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits_per_image.softmax(dim=1).item()
