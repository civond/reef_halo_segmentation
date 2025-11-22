# Calculate dice score

def calculate_dice(pred, target, eps=1e-7):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + eps) / (union + eps)

    return dice.item()