import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from utils.calculate_dice import calculate_dice
from utils.calculate_iou import calculate_iou
import numpy as np

def eval_fn(device, loader, model, score_threshold=0.5, save_dir=None):
    loop = tqdm(loader, desc="Evaluating", leave=False)

    # Track eval predictions and dice score
    eval_predictions = []
    total_dice = 0
    total_iou = 0
    count=0

    # Disable gradient calculation (back prop)
    with torch.no_grad():
        for batch_idx, (images, targets, filenames) in enumerate(loop):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images)

            # Calculate dice score per image
            for img, pred, target, filename in zip(images, preds, targets, filenames):

                # Ground truth: union all instance masks -> (H, W)
                gt_mask = target["masks"]  # (N, H, W)
                if gt_mask.shape[0] == 0:
                    gt_mask = torch.zeros(512, 512, device=gt_mask.device)
                else:
                    gt_mask = (gt_mask > 0.5).float().max(0)[0]  # (H, W)

                # Prediction: union all masks above score_threshold -> (H, W)
                if len(pred["masks"]) == 0:
                    pred_mask = torch.zeros_like(gt_mask)
                else:
                    keep = pred["scores"] > score_threshold
                    if keep.any():
                        masks = pred["masks"][keep]        # (N, 1, H, W)
                        masks = (masks.squeeze(1) > 0.5).float()
                        pred_mask = masks.max(0)[0]        # (H, W)
                    else:
                        pred_mask = torch.zeros_like(gt_mask)

                # Calculate Dice score and IOU
                dice = calculate_dice(pred_mask, gt_mask)
                iou = calculate_iou(pred_mask, gt_mask)

                if gt_mask.sum() > 0:
                    total_dice += dice
                    total_iou += iou
                    count += 1

                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                axes[0].imshow(img.cpu().permute(1, 2, 0).numpy().clip(0, 1))
                axes[0].set_title("Img")
                axes[0].axis("off")
                axes[1].imshow(gt_mask.cpu().numpy(), cmap="gray")
                axes[1].set_title("GT Mask")
                axes[1].axis("off")
                axes[2].imshow(pred_mask.cpu().numpy(), cmap="gray")
                axes[2].set_title(f"Pred Mask (dice={dice:.2f}, iou={iou:.2f}")
                axes[2].axis("off")
                plt.suptitle(filename)
                plt.tight_layout()

                temp_path = os.path.join(save_dir, filename.replace(".png", "_eval.png"))

                
                plt.savefig(temp_path, bbox_inches="tight")
                plt.close(fig)


                """eval_predictions.append({
                    "filename": filename,
                    "pred_mask": pred_mask,
                    "dice": dice
                })"""

                

    avg_dice = total_dice / count
    avg_iou = total_iou / count
    print(f"\tAvg. Dice score: {avg_dice:.4f}")
    print(f"\tAvg. IOU: {avg_iou:.4f}")

    return avg_iou, avg_dice