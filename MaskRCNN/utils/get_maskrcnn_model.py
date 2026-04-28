from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
import torch.nn.functional as F
import torch

# --- Focal Loss helpers ---
def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).mean()

def softmax_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(inputs, targets, reduction="none")
    p_t = torch.exp(-ce)
    return (alpha * ((1 - p_t) ** gamma) * ce).mean()

# --- Subclassed ROI head ---
class FocalRoIHeads(RoIHeads):
    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = softmax_focal_loss(class_logits, labels)

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, _ = class_logits.shape
        box_regression = box_regression.reshape(N, -1, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        ) / max(1, labels.numel())

        return classification_loss, box_loss

# --- Subclassed RPN ---
class FocalRPN(RegionProposalNetwork):
    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds])

        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        objectness_loss = sigmoid_focal_loss(
            objectness[sampled_inds],
            labels[sampled_inds].float(),
        )

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / max(1, sampled_pos_inds.numel())

        return objectness_loss, box_loss
    
def get_maskrcnn_model():
    # Import MaskRCNN
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=512,
        max_size=512
    )

    # Replace classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes=2  # Halo + background
    )

    # Replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes=2  # Halo + background
    )

    # Replace RPN
    rpn = model.rpn
    model.rpn = FocalRPN(
        anchor_generator=rpn.anchor_generator,
        head=rpn.head,
        fg_iou_thresh=rpn.proposal_matcher.high_threshold,
        bg_iou_thresh=rpn.proposal_matcher.low_threshold,
        batch_size_per_image=rpn.fg_bg_sampler.batch_size_per_image,
        positive_fraction=rpn.fg_bg_sampler.positive_fraction,
        pre_nms_top_n=rpn._pre_nms_top_n,
        post_nms_top_n=rpn._post_nms_top_n,
        nms_thresh=rpn.nms_thresh,
        score_thresh=rpn.score_thresh,
    )

    # Swap ROI head
    roi = model.roi_heads
    model.roi_heads = FocalRoIHeads(
        box_roi_pool=roi.box_roi_pool,
        box_head=roi.box_head,
        box_predictor=roi.box_predictor,
        fg_iou_thresh=roi.proposal_matcher.high_threshold,
        bg_iou_thresh=roi.proposal_matcher.low_threshold,
        batch_size_per_image=roi.fg_bg_sampler.batch_size_per_image,
        positive_fraction=roi.fg_bg_sampler.positive_fraction,
        bbox_reg_weights=roi.box_coder.weights,
        score_thresh=roi.score_thresh,
        nms_thresh=roi.nms_thresh,
        detections_per_img=roi.detections_per_img,
        mask_roi_pool=roi.mask_roi_pool,
        mask_head=roi.mask_head,
        mask_predictor=roi.mask_predictor,
    )

    return model