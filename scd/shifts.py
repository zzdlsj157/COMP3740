"""Label-preserving camera corruptions for evaluation and optional training."""

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


CORRUPTIONS = ("haze", "low-light", "blur", "noise")


def corrupt(images, kind, severity, generator):
    """Apply a severity from 1 to 5 to an RGB batch in [0, 1]."""
    if kind not in CORRUPTIONS or not 1 <= severity <= 5:
        raise ValueError("Corruption needs a known kind and a severity from 1 to 5.")
    if kind == "haze":
        vertical = torch.linspace(1, 0.55, images.shape[-2], device=images.device).view(1, 1, -1, 1)
        veil = 0.12 * severity * vertical
        airlight = images.new_tensor([0.84, 0.88, 0.93]).view(1, 3, 1, 1)
        return images.lerp(airlight, veil.to(images.dtype))
    if kind == "low-light":
        return (1 - 0.10 * severity) * images.pow(1 + 0.16 * severity)
    if kind == "blur":
        return F.avg_pool2d(images, 2 * severity + 1, stride=1, padding=severity, count_include_pad=False)
    noise = torch.randn(images.shape, generator=generator, device="cpu", dtype=torch.float32)
    return (images + 0.015 * severity * noise.to(images)).clamp(0, 1)


class CorruptedDataset(Dataset):
    def __init__(self, dataset, kind, severity, seed):
        self.dataset = dataset
        self.kind = kind
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        # Seed per example so results are independent of loader workers and batch size.
        rng = torch.Generator().manual_seed(self.seed + index)
        return corrupt(image.unsqueeze(0), self.kind, self.severity, rng).squeeze(0), mask


class DomainAugment:
    def __init__(self, probability, max_severity, seed):
        self.probability = probability
        self.max_severity = max_severity
        self.rng = torch.Generator().manual_seed(seed)

    def __call__(self, images):
        if not self.probability or torch.rand((), generator=self.rng).item() >= self.probability:
            return None
        kind = CORRUPTIONS[torch.randint(len(CORRUPTIONS), (), generator=self.rng).item()]
        severity = torch.randint(1, self.max_severity + 1, (), generator=self.rng).item()
        return corrupt(images, kind, severity, self.rng)


def consistency_loss(student, teacher):
    """Match the clean probabilities where the clean prediction is confident."""
    clean = teacher.detach().float().sigmoid()
    confidence = (2 * clean - 1).abs()
    error = (student.float().sigmoid() - clean).square()
    return (confidence * error).sum() / confidence.sum().clamp_min(1.0)
