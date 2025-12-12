# Calculate dice score
import torch
import torch.nn.functional as F

def one_hot(target, num_classes):
    # target: [B, H, W]
    # returns: [B, C, H, W]
    return F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

def calculate_dice(pred, target, eps=1e-7):
    num_classes = pred.shape[1]

    target_one_hot = one_hot(target, num_classes)
    pred_probs = torch.softmax(pred, dim=1)
    
    intersection = (pred_probs * target_one_hot).sum(dim=(2,3))
    union = pred_probs.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))

    # Mask out classes not present in either pred or target
    valid = union > 0

    dice = (2 * intersection + eps) / (union + eps)
    dice = dice * valid  # zero out invalid classes

    # mean only over valid classes
    return dice.sum() / valid.sum()