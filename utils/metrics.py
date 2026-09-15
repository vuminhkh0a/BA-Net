"""Segmentation metrics for BA-Net (all differentiable-free, batch-averaged)."""
import torch


def dice_coef(y_true, y_pred, smooth=1e-15):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def jaccard_similarity(y_true, y_pred, smooth=1e-15):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def recall_precision(y_true, y_pred):
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    return recall, precision


def compute_hd95(pred, target, spacing=None):
    """95th-percentile Hausdorff distance (batch flattened, in pixels)."""
    pred = pred.bool()
    target = target.bool()

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')

    pred_points = torch.nonzero(pred).float()
    target_points = torch.nonzero(target).float()

    if spacing is not None:
        spacing_tensor = torch.tensor(spacing, device=pred.device).float()
        pred_points = pred_points * spacing_tensor
        target_points = target_points * spacing_tensor

    d_pred_to_target = []
    for batch in pred_points.split(4096):
        d = torch.cdist(batch, target_points)
        min_d, _ = torch.min(d, dim=1)
        d_pred_to_target.append(min_d)
    d_pred_to_target = torch.cat(d_pred_to_target)

    d_target_to_pred = []
    for batch in target_points.split(4096):
        d = torch.cdist(batch, pred_points)
        min_d, _ = torch.min(d, dim=1)
        d_target_to_pred.append(min_d)
    d_target_to_pred = torch.cat(d_target_to_pred)

    return max(
        torch.quantile(d_pred_to_target, 0.95).item(),
        torch.quantile(d_target_to_pred, 0.95).item(),
    )
