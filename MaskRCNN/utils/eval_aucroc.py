from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

# Create AUC-ROC to evaluate model
def eval_aucroc(save_dir):
    y_true_arr = []
    y_score_arr = []

    for file in os.listdir(save_dir):
        if not file.endswith(".pt"):
            continue
        
        # Load data saved in .pt file
        data = torch.load(os.path.join(save_dir, file))
        y_true  = data["gt_mask"].numpy().flatten()
        y_score = data["soft_mask"].numpy().flatten()

        # Skip items with no halos and append to arr
        if y_true.sum() == 0:
            continue
        
        y_true_arr.append(y_true)
        y_score_arr.append(y_score)

    # Calculate AUC score
    auc = roc_auc_score(
        np.concatenate(y_true_arr),
        np.concatenate(y_score_arr)
    )
    print(f"\tAvg. AUC-ROC: {auc:.4f}")

    # Generate ROC Curve
    y_true_all  = np.concatenate(y_true_arr)
    y_score_all = np.concatenate(y_score_arr)
    fpr, tpr, thresholds = roc_curve(y_true_all, y_score_all)

    # Calculate Optimal Threshold using Youden J Statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\tOptimal threshold: {optimal_threshold:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        fpr, tpr, "b", 
        label=f"halo pixel (AUC={auc:.4f})"
    )
    ax.plot(
        [0, 1], [0, 1], 
        "r--", label="random"
    )
    ax.plot(
        fpr[optimal_idx], tpr[optimal_idx], 
        "ro", 
        label=f"optimal thresh ({optimal_threshold:.4f})"
    )
    ax.set_xlim([0,1])
    ax.set_ylim(0,1) 
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Mask R-CNN ROC Curve")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), bbox_inches="tight")
    plt.close(fig)

    return auc, optimal_threshold