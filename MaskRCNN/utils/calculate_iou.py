
# Calculate intersection over union between predicted and groundtruth mask
def calculate_iou(pred_mask, gt_mask):
    intersection = (pred_mask * gt_mask).sum()
    union = (pred_mask + gt_mask).clamp(0, 1).sum()
    
    if union == 0:
        return 1.0
    
    return (intersection / union).item()