import cv2
import logging
import numpy as np
import os
import supervision as sv
import sys
import torch
import torchvision

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))     
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))  

GROUNDED_SAM_ROOT = os.path.join(PROJECT_ROOT, "external", "grounded_sam")
if not os.path.isdir(GROUNDED_SAM_ROOT):
    raise RuntimeError(f"Cannot find grounded-sam repo at {GROUNDED_SAM_ROOT}")

sys.path.insert(0, GROUNDED_SAM_ROOT)
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from typing import Optional


class GroundedSamWrapper:

    detections: Optional[sv.Detections] = None

    def __init__(self,
                 dino_config_relpath: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 dino_checkpoint_relpath: str = "groundingdino_swint_ogc.pth",
                 sam_checkpoint_relpath: str = "sam_vit_h_4b8939.pth",
                 sam_encoder: str = "vit_h",
                 device: torch.device = None):
        self.logger = logging.getLogger(__name__)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        config_path = os.path.join(GROUNDED_SAM_ROOT, dino_config_relpath)
        checkpoint_path = os.path.join(GROUNDED_SAM_ROOT, dino_checkpoint_relpath)
        sam_checkpoint = os.path.join(GROUNDED_SAM_ROOT, sam_checkpoint_relpath)

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")
        if not os.path.isfile(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")

        self.dino = Model(
            model_config_path = config_path,
            model_checkpoint_path = checkpoint_path,
            device = self.device
        )
        self.detections = None

        sam = sam_model_registry[sam_encoder](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    def detect_and_segment(self,
                           image_bgr: np.ndarray,
                           prompt: list,
                           box_threshold: float = 0.25,
                           text_threshold: float = 0.25,
                           nms_threshold: float = 0.5):
        # GroundingDINO detections
        self.detections = self.dino.predict_with_classes(
            image=image_bgr,
            classes=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        boxes_tensor = torch.from_numpy(self.detections.xyxy)
        scores_tensor = torch.from_numpy(self.detections.confidence)
        nms_idx = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_threshold).numpy().tolist()
        self.detections.xyxy = self.detections.xyxy[nms_idx]
        self.detections.confidence = self.detections.confidence[nms_idx]

        # SAM segmentation per box
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(rgb)
        result_masks = []
        for bbox in self.detections.xyxy:
            masks, scores_sam, logits = self.sam_predictor.predict(
                box=np.array(bbox),
                multimask_output=True
            )
            idx = np.argmax(scores_sam)
            result_masks.append(masks[idx])

        self.detections.mask = np.array(result_masks)
        return self.detections.xyxy, self.detections.mask

    def annotate_and_save(self, image_bgr: np.ndarray, output_path: str = "grounded_sam_annotated.jpg"):

        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()

        annotated = mask_annotator.annotate(scene=image_bgr.copy(), detections=self.detections)
        annotated = box_annotator.annotate(scene=annotated, detections=self.detections)

        cv2.imwrite(output_path, annotated)
        self.logger.info(f"Saved annotated image to {output_path}")