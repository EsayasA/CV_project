from PIL import Image
from sam_masks import SamHandler
import torch

class ProjectPipeline:
    def __init__(self):
        self.sam = SamHandler()

    def run(self, image_input, query):
        # FIX: Check if input is already a PIL Image (from Gradio) or a path (string)
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
        
        results = self.sam.segment_by_text(image, query)
        
        if len(results["masks"]) == 0:
            return image, None, 0.0

        # Pick best mask
        best_idx = torch.argmax(results["scores"]).item()
        mask = results["masks"][best_idx].cpu().numpy()
        score = results["scores"][best_idx].item()

        return image, mask, score
