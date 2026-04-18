import torch
from transformers import Sam3Processor, Sam3Model

class SamHandler:
    def __init__(self):
        # Use 'mps' for Mac M1/M2/M3, 'cuda' for Nvidia, else 'cpu'
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"Loading SAM 3 on {self.device}...")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def segment_by_text(self, image, text_query):
        inputs = self.processor(images=image, text=text_query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.4, target_sizes=[image.size[::-1]]
        )[0]
        return results
