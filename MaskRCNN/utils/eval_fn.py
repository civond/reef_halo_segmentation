# Import libraries
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchmetrics.detection import MeanAveragePrecision
import numpy as np
from dataclasses import dataclass

# Custom imports
from utils.calculate_dice import calculate_dice
from utils.calculate_iou import calculate_iou

@dataclass
class MetricStats:
    avg: float
    std: float

@dataclass  
class EvalMetrics:
    dice: MetricStats
    iou: MetricStats

def create_summary_plot(metrics, dice_val_array, iou_val_array, save_dir):
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Bar + error
    bars = axes[0].bar(["Dice", "IoU"], [metrics.dice.avg, metrics.iou.avg],
                        yerr=[metrics.dice.std, metrics.iou.std],
                        color=["#5DCAA5", "#AFA9EC"],
                        linewidth=1.5, capsize=8, width=0.6, error_kw={"elinewidth": 1.5})
    for bar, avg, std in zip(bars, [metrics.dice.avg, metrics.iou.avg], [metrics.dice.std, metrics.iou.std]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,  # just above the error bar
            f"{avg:.2f} ± {std:.2f}",
            ha="center", va="bottom",
            fontsize=9, color="black"
        )
    axes[0].margins(x=0.4)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Avg. ± STD (N={len(dice_val_array)})", fontsize=11, fontweight="normal")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].yaxis.grid(True, alpha=0.3)
    axes[0].set_axisbelow(True)
    axes[0].set_ylabel("Average")

    # Distribution
    axes[1].hist([dice_val_array, iou_val_array], bins=20, range=(0,1),
                histtype="bar",
                alpha=0.5,
                color=["#2EAA81", "#A69FF5"],
                edgecolor=["k", "k"],
                linewidth=0.2,
                label=["Dice", "IoU"],
                density=False)
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Frequency")
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].set_xlim(0, 1)
    axes[1].set_title(f"Score Distribution (N={len(dice_val_array)})", fontsize=11, fontweight="normal")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].yaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eval_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def eval_fn(device, loader, model, score_threshold=0.5, save_dir=None, save_logits=False):
    loop = tqdm(loader, desc="Evaluating", leave=False)

    # Track eval predictions and dice score
    dice_val_array = []
    iou_val_array = []

    # AP metric — needs per-instance masks, not unioned
    map_metric = MeanAveragePrecision(iou_type="segm")

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
                        masks = pred["masks"][keep]
                        soft_mask = masks.squeeze(1).max(0)[0]
                        pred_mask = (soft_mask > 0.5).float()
                    else:
                        soft_mask = torch.zeros_like(gt_mask)
                        pred_mask = torch.zeros_like(gt_mask)
                
                # Calculate Dice score and IOU
                dice = calculate_dice(pred_mask, gt_mask)
                iou = calculate_iou(pred_mask, gt_mask)
                
                # Calculate AP metrics
                n_pred = len(pred["masks"])
                n_gt = target["masks"].shape[0]

                H, W = target["masks"].shape[-2], target["masks"].shape[-1]
                map_metric.update(
                    preds=[{
                        "masks": (pred["masks"].squeeze(1) > 0.5).bool().cpu() if n_pred > 0
                                 else torch.zeros(0, H, W, dtype=torch.bool),
                        "scores": pred["scores"].cpu() if n_pred > 0
                                  else torch.zeros(0),
                        "labels": torch.zeros(n_pred, dtype=torch.int).cpu() if n_pred > 0
                                  else torch.zeros(0, dtype=torch.int),
                    }],
                    target=[{
                        "masks": (target["masks"] > 0.5).bool().cpu() if n_gt > 0
                                 else torch.zeros(0, H, W, dtype=torch.bool),
                        "labels": torch.zeros(n_gt, dtype=torch.int).cpu() if n_gt > 0
                                  else torch.zeros(0, dtype=torch.int),
                    }]
                )

                # Save logits
                stem = os.path.splitext(filename)[0]
                torch.save(
                    {"soft_mask": soft_mask.cpu(), "gt_mask": gt_mask.cpu()},
                    os.path.join(save_dir, f"{stem}_logit.pt")
                )
                    
                if gt_mask.sum() > 0:
                    dice_val_array.append(dice)
                    iou_val_array.append(iou)

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

    # Compute average and standard deviation
    dice_avg = np.mean(dice_val_array)
    dice_std = np.std(dice_val_array, ddof=1)
    iou_avg = np.mean(iou_val_array)
    iou_std = np.std(iou_val_array, ddof=1)

    metrics = EvalMetrics(
        dice=MetricStats(avg=dice_avg, std=dice_std),
        iou=MetricStats(avg=iou_avg,   std=iou_std),
    )

    # Compute AP
    ap_results = map_metric.compute()

    # Print statements
    print(f"\tAvg. Precision @50%:     {ap_results['map_50']:.4f}")
    print(f"\tAvg. Precision @75%:     {ap_results['map_75']:.4f}")
    print(f"\tAvg. Precision @50:95%:  {ap_results['map']:.4f}")
    print(f"\tAvg. Dice score:         {dice_avg:.4f} (std={dice_std:.4f})")
    print(f"\tAvg. IoU:                {iou_avg:.4f} (std={iou_std:.4f})")

    # Create summary plot
    create_summary_plot(
        metrics=metrics,
        dice_val_array=dice_val_array,
        iou_val_array=iou_val_array,
        save_dir=save_dir
    )

    return metrics, ap_results