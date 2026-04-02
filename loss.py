import torch
import torch.nn as nn
import pytorch_iou, pytorch_ssim
import torch.nn.functional as F

def dice_coef(y_true, y_pred, smooth=1e-15):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def jaccard_similarity(y_true, y_pred, smooth=1e-15):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def jacard_loss(y_true, y_pred):
    return 1.0 - jaccard_similarity(y_true, y_pred)


def Ssim_loss(y_true, y_pred, max_val=1.0):
    mean_true = y_true.mean([1,2,3], keepdim=True)
    mean_pred = y_pred.mean([1,2,3], keepdim=True)
    var_true = y_true.var([1,2,3], keepdim=True)
    var_pred = y_pred.var([1,2,3], keepdim=True)
    covar = (y_true * y_pred).mean([1,2,3], keepdim=True) - mean_true * mean_pred
    c1 = (0.01 * max_val)**2
    c2 = (0.03 * max_val)**2
    ssim = ((2 * mean_true * mean_pred + c1) * (2 * covar + c2)) / ((mean_true**2 + mean_pred**2 + c1) * (var_true + var_pred + c2))
    return 1 - ssim.mean()

def focal_loss(y_true, y_pred, alpha=0.26, gamma=2.3):
    BCE = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE
    return focal_loss.mean()

def joint_loss1(y_true, y_pred):
    f_loss = focal_loss(y_true, y_pred)
    s_loss = Ssim_loss(y_true, y_pred)
    j_loss = jacard_loss(y_true, y_pred)
    return (f_loss + s_loss + j_loss) / 3.0

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    return bce_out + ssim_out + iou_out

def MSE_loss(rawA, rawB):
    num_classes = 2.0
    mse = F.mse_loss(rawA, rawB, reduction='none')
    mse_per_image = mse.mean(dim=[1,2,3])
    return (mse_per_image.mean() / num_classes)

def recall_precision(y_true, y_pred):
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp
    recall = ((tp + 1e-6) / (tp + fn + 1e-6))
    precision = ((tp + 1e-6) / (tp + fp + 1e-6))
    return recall, precision

def compute_hd95(pred, target, spacing=None):
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

    hd95_val = max(
        torch.quantile(d_pred_to_target, 0.95).item(),
        torch.quantile(d_target_to_pred, 0.95).item()
    )

    return hd95_val

def unlabeled_loss(pred0, pred1, target0, target1):
	return bce_loss(pred0, target0) + dice_loss(pred1, target1)

def muti_bce_loss_fusion(s0, s1, s2, s3, s4, labels_v):
	loss0 = bce_ssim_loss(s0,labels_v)
	loss1 = bce_ssim_loss(s1,labels_v)
	loss2 = bce_ssim_loss(s2,labels_v)
	loss3 = bce_ssim_loss(s3,labels_v)
	loss4 = bce_ssim_loss(s4,labels_v)
	loss = loss0 + loss1 + loss2 + loss3 + loss4
	return loss
