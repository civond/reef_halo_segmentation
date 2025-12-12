import torch
from tqdm import tqdm

from utils.calculate_dice import calculate_dice

def val_fn(device, loader, model, score_threshold=0.5):
    model.eval()
    loop = tqdm(loader, desc="Validating", leave=False)

    # Track loss and dice score
    total_loss = 0
    total_dice = 0
    count=0

    with torch.no_grad():
        for batch_idx, (images, targets, filenames) in enumerate(loop):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute validation loss via forward pass (requires model.train())
            model.train() 
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            
            # Compute predictions (requires model.eval())
            model.eval()
            preds = model(images)

            # Calculate dice score per image
            for pred, target in zip(preds, targets):
                # Convert ground truth mask to binary image in memory
                gt_mask = target["masks"]
                if gt_mask.dim() == 3:
                    gt_mask = gt_mask.squeeze(0)
                gt_mask = (gt_mask > 0.5).float()

                # Unify all predicted masks above score_threshold
                if len(pred["masks"]) == 0:
                    pred_mask = torch.zeros_like(gt_mask)
                else:
                    keep = pred["scores"] > score_threshold
                    if keep.any():
                        masks = pred["masks"][keep]  # [N,1,H,W]
                        masks = (masks.squeeze(1) > 0.5).float()
                        pred_mask = masks.max(0)[0]  # union of predicted masks
                    else:
                        pred_mask = torch.zeros_like(gt_mask)

                total_dice += calculate_dice(pred_mask, gt_mask)
                count += 1
            

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(loader) # calculate avg loss
    avg_dice = total_dice / count if count > 0 else 0.0 # calculate avg dice score

    # update tqdm loop
    loop.set_postfix(loss=loss.item())

    return avg_loss, avg_dice