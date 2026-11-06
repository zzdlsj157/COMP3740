"""Diagnostics for natural domain shift and paired synthetic corruptions."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .objectives import PixelCounts


@dataclass
class Calibration:
    bins: int = 15
    count: torch.Tensor = field(init=False)
    probabilities: torch.Tensor = field(init=False)
    positives: torch.Tensor = field(init=False)
    squared_error: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.count = torch.zeros(self.bins, dtype=torch.int64)
        self.probabilities = torch.zeros(self.bins, dtype=torch.float64)
        self.positives = torch.zeros_like(self.probabilities)
        self.squared_error = torch.zeros(2, dtype=torch.float64)

    def update(self, probability, mask):
        p = probability.flatten().double()
        y = mask.flatten().long()
        index = (p * self.bins).long().clamp_max(self.bins - 1)
        self.count += torch.bincount(index, minlength=self.bins)
        self.probabilities += torch.bincount(index, weights=p, minlength=self.bins)
        self.positives += torch.bincount(index, weights=y.double(), minlength=self.bins)
        self.squared_error += torch.bincount(y, weights=(p - y).square(), minlength=2)

    def scores(self):
        count = int(self.count.sum())
        foreground = int(self.positives.sum())
        background = count - foreground
        return {
            "ece": float((self.probabilities - self.positives).abs().sum()) / count,
            "brier": float(self.squared_error.sum()) / count,
            "foreground_brier": float(self.squared_error[1]) / foreground if foreground else None,
            "background_brier": float(self.squared_error[0]) / background if background else None,
            "bins": [
                {
                    "lower": index / self.bins,
                    "upper": (index + 1) / self.bins,
                    "pixels": n,
                    "mean_probability": predicted / n if n else None,
                    "observed_smoke_rate": positive / n if n else None,
                }
                for index, (n, predicted, positive) in enumerate(
                    zip(self.count.tolist(), self.probabilities.tolist(), self.positives.tolist())
                )
            ],
        }


class DomainScores:
    def __init__(self, small_smoke_fraction):
        self.small_smoke_fraction = small_smoke_fraction
        self.calibration = Calibration()
        self.groups = {
            name: PixelCounts("cpu") for name in ("all", "no_smoke", "small_smoke", "larger_smoke")
        }
        self.images = {name: 0 for name in self.groups}
        self.false_alarm_images = 0
        self.uncertainty = []
        self.routing = {}

    def update(self, probabilities, mask, threshold, expert_mass):
        self.calibration.update(probabilities, mask)
        p = probabilities.flatten(1)
        y = mask.bool().flatten(1)
        predicted = p >= threshold
        counts = torch.stack(((predicted & y).sum(1), predicted.sum(1), y.sum(1)), dim=1)
        coverage = y.float().mean(1)
        absent = coverage == 0
        small = (coverage > 0) & (coverage <= self.small_smoke_fraction)
        for name, selector in (
            ("all", torch.ones_like(absent)), ("no_smoke", absent),
            ("small_smoke", small), ("larger_smoke", ~(absent | small)),
        ):
            self.groups[name].totals += counts[selector].sum(0)
            self.images[name] += int(selector.sum())
        self.false_alarm_images += int((absent & predicted.any(dim=1)).sum())

        entropy = (torch.special.entr(p) + torch.special.entr(1 - p)) / math.log(2)
        tail_count = max(1, math.ceil(0.1 * p.shape[1]))
        tail_mean = entropy.topk(tail_count, dim=1, sorted=False).values.mean(1)
        self.uncertainty.extend(tail_mean.tolist())
        for layer, mass in expert_mass.items():
            weighted = mass.detach().cpu().double() * p.shape[0]
            self.routing[layer] = self.routing.get(layer, 0) + weighted

    def scores(self):
        images = self.images["all"]
        negative_images = self.images["no_smoke"]
        return {
            "images": images,
            **self.groups["all"].scores(),
            "false_alarm_rate_on_no_smoke_images": self.false_alarm_images / negative_images if negative_images else None,
            "subgroups": {
                name: {"images": self.images[name], **(group.scores() if self.images[name] else {})}
                for name, group in self.groups.items() if name != "all"
            },
            "calibration": self.calibration.scores(),
            "mean_top_decile_entropy": sum(self.uncertainty) / images,
            "expert_mass": {str(layer): (mass / images).tolist() for layer, mass in self.routing.items()},
        }


def entropy_auroc(id_scores, shifted_scores):
    """Probability that an OOD image outranks an ID image, with half credit for ties."""
    negative = np.sort(id_scores)
    left = np.searchsorted(negative, shifted_scores, side="left")
    right = np.searchsorted(negative, shifted_scores, side="right")
    return float(((left + right) / (2 * len(negative))).mean())


def routing_divergence(reference, shifted):
    """Jensen–Shannon divergence in bits for two distributions over experts."""
    p, q = np.asarray(reference), np.asarray(shifted)
    p, q = p / p.sum(), q / q.sum()
    middle = (p + q) / 2

    def kl(a):
        support = a > 0
        return float((a[support] * np.log2(a[support] / middle[support])).sum())

    return (kl(p) + kl(q)) / 2


def compare_domain(reference, shifted, reference_report):
    report = shifted.scores()
    report["delta_vs_id_pp"] = {
        metric: 100 * (report[metric] - reference_report[metric]) for metric in ("dice", "iou", "precision", "recall")
    }
    report["entropy_auroc_vs_id"] = entropy_auroc(reference.uncertainty, shifted.uncertainty)
    report["routing_jsd_bits_vs_id"] = {
        layer: routing_divergence(reference_report["expert_mass"][layer], mass)
        for layer, mass in report["expert_mass"].items()
    }
    return report


@torch.inference_mode()
def score_domain(model, loader, device, threshold, small_smoke_fraction, flip_tta=False):
    model.eval()
    scores = DomainScores(small_smoke_fraction)
    for images, masks in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        output = model(images, report_routing=True)
        probabilities = output.logits.float().sigmoid()
        routing = output.expert_mass or {}
        if flip_tta:
            flipped = model(images.flip(-1), report_routing=True)
            probabilities = (probabilities + flipped.logits.float().sigmoid().flip(-1)) / 2
            if routing:
                routing = {layer: (mass + flipped.expert_mass[layer]) / 2 for layer, mass in routing.items()}
        probabilities = probabilities.cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("The model produced non-finite probabilities during OOD evaluation.")
        scores.update(probabilities, masks, threshold, routing)
    return scores
