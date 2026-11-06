"""Binary segmentation losses and dataset-level pixel counts."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import upsample


@dataclass(frozen=True)
class LossWeights:
    aux: float = 0.4
    boundary: float = 0.1
    balance: float = 0.01


def boundary_map(mask):
    return F.max_pool2d(mask, 3, stride=1, padding=1) + F.max_pool2d(-mask, 3, stride=1, padding=1)


def segmentation_loss(output, mask, weights):
    logits = output.logits.float()
    probability = logits.sigmoid()
    intersection = (probability * mask).sum(dim=(1, 2, 3))
    size = (probability + mask).sum(dim=(1, 2, 3))
    loss = F.binary_cross_entropy_with_logits(logits, mask) + (1 - (2 * intersection + 1) / (size + 1)).mean()
    if output.coarse is not None:
        if weights.aux:
            coarse = upsample(output.coarse.float(), mask.shape[-2:])
            loss = loss + weights.aux * F.binary_cross_entropy_with_logits(coarse, mask)
        if weights.boundary:
            loss = loss + weights.boundary * F.l1_loss(boundary_map(probability), boundary_map(mask))
        if weights.balance:
            loss = loss + weights.balance * output.balance
    return loss


class PixelCounts:
    def __init__(self, device):
        self.totals = torch.zeros(3, device=device, dtype=torch.int64)

    def update(self, logits, target, threshold=0.5):
        prediction = logits.float().sigmoid() >= threshold
        target = target.bool()
        self.totals += torch.stack(((prediction & target).sum(), prediction.sum(), target.sum()))

    def scores(self):
        tp, predicted, positive = self.totals.tolist()

        def ratio(numerator, denominator):
            return numerator / denominator if denominator else 0.0

        return {
            "dice": ratio(2 * tp, predicted + positive),
            "iou": ratio(tp, predicted + positive - tp),
            "precision": ratio(tp, predicted),
            "recall": ratio(tp, positive),
        }
