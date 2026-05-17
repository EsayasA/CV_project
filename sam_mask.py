import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from clip_test import ClipValidator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class GroundedSamHandler:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        print(f"Loading models on {self.device}")

        # =========================
        # 1. Grounding DINO
        # =========================
        self.dino_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        )

        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(self.device)

        # =========================
        # 2. SAM 2 (Image Predictor)
        # =========================
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        sam2_model_cfg = "sam2_hiera_l.yaml"
        
        self.sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # =========================
        # 3. CLIP
        # =========================
        self.clip_validator = ClipValidator()


    # =========================
    # DETECT + SEGMENT
    # =========================
    def detect_and_segment(
        self,
        image,
        text_query,
        box_threshold=0.3,
        text_threshold=0.3,
    ):

        text_query = text_query.lower().strip()
        text_query = text_query.replace(" and ", " . ").replace(",", " . ")

        if not text_query.endswith("."):
            text_query += "."

        # --- DINO Detection ---
        inputs = self.dino_processor(
            images=image,
            text=text_query,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results.get("text_labels", results.get("labels"))

        if len(boxes) == 0:
            return [], [], []

        image_np = np.array(image)

        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        # --- CLIP Verification ---
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)

            crop = image_np[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(crop)
            clip_score = self.clip_validator.get_score(crop_pil, text_query)
            dino_score = float(score.cpu())

            combined_score = (dino_score + clip_score) / 2

            if combined_score >= box_threshold:
                filtered_boxes.append(box.cpu().numpy().astype(np.float32).tolist())
                filtered_scores.append(combined_score)
                filtered_labels.append(label)

        if len(filtered_boxes) == 0:
            return [], [], []

        # --- SAM 2 Segmentation ---
        # SAM 2 optimizes by calculating the image embedding once
        self.sam2_predictor.set_image(image_np)

        final_masks = []
        for box in filtered_boxes:
            # Predict the mask using the bounding box
            masks, _, _ = self.sam2_predictor.predict(
                box=np.array(box),
                multimask_output=False # We only want the best single mask per box
            )
            # SAM 2 returns shape (1, H, W) when multimask is False
            final_masks.append(masks[0])

        return final_masks, filtered_scores, filtered_labels

    # =========================
    # SEGMENT FROM BOXES (Helper)
    # =========================
    def segment_from_boxes(self, image, boxes_list):
        if not boxes_list:
            return []

        image_np = np.array(image)
        self.sam2_predictor.set_image(image_np)

        final_masks = []
        for box in boxes_list:
            masks, _, _ = self.sam2_predictor.predict(
                box=np.array(box),
                multimask_output=False
            )
            final_masks.append(masks[0])

        return final_masks