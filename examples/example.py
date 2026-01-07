## This code is cited from https://zenn.dev/watanko/articles/ef4370d5d94dac

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def opencv_visualization(
    image: np.ndarray,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float = 0.0,
    color: tuple = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Create an OpenCV visualization with masks and bounding boxes.

    Args:
        image (np.ndarray): RGB image array shaped (H, W, 3).
        masks (torch.Tensor): Boolean masks in shape (N, 1, H, W) or (N, H, W).
        boxes (torch.Tensor): Bounding boxes in xyxy format with shape (N, 4).
        scores (torch.Tensor): Confidence scores for each mask.
        score_threshold (float): Minimum score required to visualize an instance.
        alpha (float): Opacity of the colored mask overlay.

    Returns:
        np.ndarray: BGR image with overlays suitable for cv2.imwrite.

    Raises:
        ValueError: If the number of masks, boxes, and scores does not match.
        ValueError: If mask shapes cannot be aligned with the image.
    """
    if masks.shape[0] != boxes.shape[0] or boxes.shape[0] != scores.shape[0]:
        raise ValueError("masks, boxes, and scores must have the same length.")

    height, width = image.shape[0], image.shape[1]
    overlay = image.copy()

    for idx in range(masks.shape[0]):
        score = float(scores[idx])
        if score < score_threshold:
            continue

        mask_np = masks[idx].detach().cpu().numpy()
        if mask_np.ndim > 2 and mask_np.shape[0] == 1:
            mask_np = np.squeeze(mask_np, axis=0)
        if mask_np.ndim != 2:
            raise ValueError("Each mask must be a 2D array.")
        if mask_np.shape != (height, width):
            mask_np = cv2.resize(
                mask_np.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_region = mask_np > 0.5
        overlay[mask_region] = (
            alpha * np.array(color) + (1 - alpha) * overlay[mask_region]
        ).astype(np.uint8)

        x0, y0, x1, y1 = boxes[idx].detach().cpu().numpy()
        x0_i, y0_i = max(int(x0), 0), max(int(y0), 0)
        x1_i, y1_i = min(int(x1), width - 1), min(int(y1), height - 1)
        cv2.rectangle(
            overlay,
            (x0_i, y0_i),
            (x1_i, y1_i),
            color=color,
            thickness=2,
        )
        label_text = f"{score:.2f}"
        cv2.putText(
            overlay,
            label_text,
            (x0_i, max(y0_i - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    return overlay


def apply_mask_nms(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float,
    mask_iou_threshold: float,
    box_iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply mask-based NMS to filter detections without external dependencies.

    Args:
        masks (torch.Tensor): Masks shaped (N, H, W) or (N, 1, H, W).
        boxes (torch.Tensor): Bounding boxes in xyxy format with shape (N, 4).
        scores (torch.Tensor): Confidence scores for each mask.
        score_threshold (float): Minimum score to consider a detection for NMS.
        mask_iou_threshold (float): IoU threshold for suppressing overlapping masks.
        box_iou_threshold (float): IoU threshold for suppressing overlapping boxes.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Filtered masks, boxes, and scores.

    Raises:
        ValueError: If masks have unsupported dimensions.
    """
    if masks.dim() == 4 and masks.shape[1] == 1:
        masks_for_nms = masks[:, 0]
    elif masks.dim() == 3:
        masks_for_nms = masks
    else:
        raise ValueError("masks must have shape (N, H, W) or (N, 1, H, W).")

    masks_for_nms = masks_for_nms.to("cpu").bool()
    scores_cpu = scores.to("cpu")
    boxes_cpu = boxes.to("cpu")
    order = torch.argsort(scores_cpu, descending=True)
    keep_mask = torch.zeros(len(order), dtype=torch.bool, device="cpu")
    kept_masks: list[torch.Tensor] = []
    kept_boxes: list[torch.Tensor] = []

    for order_idx, det_idx in enumerate(order):
        score = scores_cpu[det_idx]
        if score < score_threshold:
            continue
        current_mask = masks_for_nms[det_idx]
        current_box = boxes_cpu[det_idx]
        if not kept_masks:
            keep_mask[order_idx] = True
            kept_masks.append(current_mask)
            kept_boxes.append(current_box)
            continue
        stacked_kept = torch.stack(kept_masks, dim=0)
        intersection = (stacked_kept & current_mask).sum(dim=(1, 2)).float()
        union = (stacked_kept | current_mask).sum(dim=(1, 2)).float()
        ious = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        kept_boxes_tensor = torch.stack(kept_boxes, dim=0)
        x1 = torch.maximum(current_box[0], kept_boxes_tensor[:, 0])
        y1 = torch.maximum(current_box[1], kept_boxes_tensor[:, 1])
        x2 = torch.minimum(current_box[2], kept_boxes_tensor[:, 2])
        y2 = torch.minimum(current_box[3], kept_boxes_tensor[:, 3])
        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter_area = inter_w * inter_h
        current_area = (current_box[2] - current_box[0]) * (
            current_box[3] - current_box[1]
        )
        kept_areas = (kept_boxes_tensor[:, 2] - kept_boxes_tensor[:, 0]) * (
            kept_boxes_tensor[:, 3] - kept_boxes_tensor[:, 1]
        )
        box_union = current_area + kept_areas - inter_area
        box_ious = torch.where(
            box_union > 0, inter_area / box_union, torch.zeros_like(box_union)
        )

        if torch.all(ious <= mask_iou_threshold) and torch.all(
            box_ious <= box_iou_threshold
        ):
            keep_mask[order_idx] = True
            kept_masks.append(current_mask)
            kept_boxes.append(current_box)

    selected_indices = order[keep_mask].to(masks.device)
    return masks[selected_indices], boxes[selected_indices], scores[selected_indices]


# === EDIT ME ===
prompts = [
    "ground",
    "truck",
]

colors = [(255, 0, 0), (0, 255, 0)]
image_path = "assets/images/truck.jpg"
# === EDIT ME ===

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.3)
# Load an image
image = Image.open(image_path).convert("RGB")

inference_state = processor.set_image(image)

overlay_image = image.copy()
overlay_image = np.array(overlay_image)
overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

for prompt, color in zip(prompts, colors):
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    masks, boxes, scores = apply_mask_nms(
        masks=masks,
        boxes=boxes,
        scores=scores,
        score_threshold=0.3,
        mask_iou_threshold=0.1,
        box_iou_threshold=0.1,
    )

    print("Image Masks shape:", masks.shape)
    print("Image Boxes shape:", boxes.shape)
    print("Image Scores shape:", scores.shape)

    overlay_image = opencv_visualization(
        image=overlay_image,
        masks=masks,
        boxes=boxes,
        scores=scores,
        score_threshold=0.3,
        color=color,
        alpha=0.5,
    )

cv2.imwrite("visualization.png", overlay_image)
