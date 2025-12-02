import os
import sys
import cv2
import numpy as np
import torch
import torchvision
import supervision as sv


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))     
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))  

GROUNDED_SAM_ROOT = os.path.join(PROJECT_ROOT, "external", "grounded_sam")
if not os.path.isdir(GROUNDED_SAM_ROOT):
    raise RuntimeError(f"Cannot find grounded-sam repo at {GROUNDED_SAM_ROOT}")

# grounded-sam root for imports 
sys.path.insert(0, GROUNDED_SAM_ROOT)

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

class GroundedSamWrapper:
    def __init__(self,
                 dino_config_relpath: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 dino_checkpoint_relpath: str = "groundingdino_swint_ogc.pth",
                 sam_checkpoint_relpath: str = "sam_vit_h_4b8939.pth",
                 sam_encoder: str = "vit_h",
                 device: torch.device = None):
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
                           nms_threshold: float = 0.8):
        # GroundingDINO detection
        self.detections = self.dino.predict_with_classes(
            image=image_bgr,
            classes=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # parse detections
        labels = []
        for det in self.detections:
            class_id = None
            confidence = None

            if isinstance(det, (list, tuple)):
                if len(det) == 5:
                    x1, y1, x2, y2, confidence = det
                elif len(det) >= 6:
                    x1, y1, x2, y2, confidence, class_id = det[:6]
                else:
                    print("Unexpected detection tuple (len < 5):", det)
                    continue

            else:
                try:
                # this depends on what fields det has
                # adapt if naming differs
                    bbox = det.xyxy  # or something similar
                    confidence = det.confidence if hasattr(det, 'confidence') else det.conf
                    class_id = det.class_id if hasattr(det, 'class_id') else None
                    x1, y1, x2, y2 = bbox
                except Exception as e:
                    print("Cannot parse detection object:", det, "error:", e)
                    continue

            if class_id is not None and class_id < len(prompt):
                label_cls = prompt[class_id]
            else:
                label_cls = prompt[0]  # fallback or generic
            lbl = f"{label_cls} {confidence:0.2f}" if confidence is not None else label_cls
            labels.append(lbl)

        if hasattr(self.detections, 'xyxy') and hasattr(self.detections, 'confidence'):
            print(f"Before NMS: {len(self.detections.xyxy)} boxes")
            try:
                # 2. NMS (non-maximum suppression) on boxes
                boxes_tensor = torch.from_numpy(self.detections.xyxy)
                scores_tensor = torch.from_numpy(self.detections.confidence)
                nms_idx = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_threshold).numpy().tolist()
                self.detections.xyxy = self.detections.xyxy[nms_idx]
                self.detections.confidence = self.detections.confidence[nms_idx]
                if hasattr(self.detections, 'class_id'):
                    self.detections.class_id = self.detections.class_id[nms_idx]
            except Exception as e:
                print("NMS failed:", e)
        else:
            print("Detections object does not have .xyxy / .confidence attributes — skipping NMS + segmentation.")

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
        print("Saved annotated image to", output_path)