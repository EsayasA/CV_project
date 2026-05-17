import cv2
import numpy as np
import os
import tempfile
import shutil
from PIL import Image
import matplotlib.cm as cm
import torch

from sam_mask import GroundedSamHandler
from utils import get_bbox_from_mask

# Import SAM 2 for video tracking
from sam2.build_sam import build_sam2_video_predictor

class ProjectPipeline:
    def __init__(self):
        self.handler = GroundedSamHandler()
        
        # Initialize SAM 2 specifically for video tracking
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        sam2_model_cfg = "sam2_hiera_l.yaml"
        self.sam2_predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint, device=self.device)

    # =========================
    # IMAGE
    # =========================
    def run_image(self, image_input, query, threshold=0.3):
        image = (
            image_input.convert("RGB")
            if not isinstance(image_input, str)
            else Image.open(image_input).convert("RGB")
        )

        masks, scores, labels = self.handler.detect_and_segment(
            image,
            query,
            box_threshold=threshold,
        )

        return image, masks, scores, labels

    # =========================
    # VIDEO (Fixed with SAM 2)
    # =========================
    def run_video(self, video_path, query, threshold=0.3):
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = "output_video.mp4"

        # 1. SAM 2 requires video frames saved in a temporary folder
        temp_dir = tempfile.mkdtemp()
        frame_idx = 0
        frame_paths = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            p = os.path.join(temp_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(p, frame)
            frame_paths.append(p)
            frame_idx += 1
        cap.release()

        # 2. Find the object in the first frame using Grounding DINO + CLIP (from sam_mask.py)
        frame0_rgb = cv2.cvtColor(cv2.imread(frame_paths[0]), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame0_rgb)
        
        masks, scores, labels = self.handler.detect_and_segment(pil_img, query, box_threshold=threshold)

        if len(masks) == 0:
            shutil.rmtree(temp_dir)
            raise ValueError(f"Could not find '{query}' in the first frame.")

        # 3. Initialize SAM 2 Video Tracker
        inference_state = self.sam2_predictor.init_state(video_path=temp_dir)
        
        tracked_colors = [cm.hsv(i / len(masks)) for i in range(len(masks))]
        
        # Feed all detected objects into SAM 2's memory
        for i, mask in enumerate(masks):
            bbox = get_bbox_from_mask(mask)
            if bbox:
                # Convert [x, y, w, h] to [x1, y1, x2, y2] for SAM 2
                x, y, w, h = bbox
                box_coords = np.array([x, y, x + w, y + h])
                
                _, _, _ = self.sam2_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=i+1, # Object IDs must start at 1
                    box=box_coords
                )

        # 4. Propagate tracking throughout the video using Memory
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
            frame_masks = {}
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                if mask.any():
                    frame_masks[obj_id] = mask
            video_segments[out_frame_idx] = frame_masks

        # 5. Draw Results
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(len(frame_paths)):
            frame = cv2.imread(frame_paths[i])
            
            if i in video_segments:
                for obj_id, mask in video_segments[i].items():
                    idx = obj_id - 1 # Convert back to 0-indexed for colors/labels
                    color = tracked_colors[idx]
                    label = labels[idx]
                    score = scores[idx]

                    bgr_color = (int(color[2]*255), int(color[1]*255), int(color[0]*255))

                    # Apply Solid Mask Overlay
                    overlay = frame.copy()
                    overlay[mask > 0] = np.array(bgr_color)
                    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

                    # Draw Bounding Box and Label
                    bbox = get_bbox_from_mask(mask)
                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
                        label_text = f"{label} {score:.2f}"
                        text_y = y - 10 if y > 20 else y + 20
                        cv2.putText(frame, label_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)

            out.write(frame)

        out.release()
        shutil.rmtree(temp_dir) # Clean up temp frames
        return output_path