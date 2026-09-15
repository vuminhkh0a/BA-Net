"""Training losses for BA-Net.

Consolidates the former ``pytorch_iou`` and ``pytorch_ssim`` packages
(``IOU`` and ``SSIM`` modules) so no extra sub-packages are required.
Evaluation-only scores (Dice, Jaccard, precision/recall, HD95) live in
``utils/metrics.py``.
"""
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.metrics import dice_coef

# ---------------------------------------------------------------------------
# SSIM (from pytorch-ssim: https://github.com/Po-Hsun-Su/pytorch-ssim)
# ---------------------------------------------------------------------------

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = _create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = _create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# ---------------------------------------------------------------------------
# IoU loss (former ``pytorch_iou`` package)
# ---------------------------------------------------------------------------

def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1
        # IoU loss is (1 - IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)


# ---------------------------------------------------------------------------
# Supervised / consistency losses
# ---------------------------------------------------------------------------

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def Ssim_loss(y_true, y_pred, max_val=1.0):
    mean_true = y_true.mean([1, 2, 3], keepdim=True)
    mean_pred = y_pred.mean([1, 2, 3], keepdim=True)
    var_true = y_true.var([1, 2, 3], keepdim=True)
    var_pred = y_pred.var([1, 2, 3], keepdim=True)
    covar = (y_true * y_pred).mean([1, 2, 3], keepdim=True) - mean_true * mean_pred
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim = ((2 * mean_true * mean_pred + c1) * (2 * covar + c2)) / \
        ((mean_true ** 2 + mean_pred ** 2 + c1) * (var_true + var_pred + c2))
    return 1 - ssim.mean()


def jacard_loss(y_true, y_pred):
    from utils.metrics import jaccard_similarity
    return 1.0 - jaccard_similarity(y_true, y_pred)


def focal_loss(y_true, y_pred, alpha=0.26, gamma=2.3):
    BCE = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
    return focal_loss.mean()


def joint_loss1(y_true, y_pred):
    """Validation loss: mean of focal + simplified-SSIM + Jaccard."""
    f_loss = focal_loss(y_true, y_pred)
    s_loss = Ssim_loss(y_true, y_pred)
    j_loss = jacard_loss(y_true, y_pred)
    return (f_loss + s_loss + j_loss) / 3.0


bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = SSIM(window_size=11, size_average=True)
iou_loss = IOU(size_average=True)


def bce_ssim_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    return bce_out + ssim_out + iou_out


def MSE_loss(rawA, rawB):
    num_classes = 2.0
    mse = F.mse_loss(rawA, rawB, reduction='none')
    mse_per_image = mse.mean(dim=[1, 2, 3])
    return (mse_per_image.mean() / num_classes)


def unlabeled_loss(pred0, pred1, target0, target1):
    return bce_loss(pred0, target0) + dice_loss(pred1, target1)


def muti_bce_loss_fusion(s0, s1, s2, s3, s4, labels_v):
    loss0 = bce_ssim_loss(s0, labels_v)
    loss1 = bce_ssim_loss(s1, labels_v)
    loss2 = bce_ssim_loss(s2, labels_v)
    loss3 = bce_ssim_loss(s3, labels_v)
    loss4 = bce_ssim_loss(s4, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4
    return loss
