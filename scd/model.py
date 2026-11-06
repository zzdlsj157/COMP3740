"""The SCD encoder, smoke cues, feature fusion, and refinement from Chapter 4."""

import math
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import create_backbone, encoder_blocks


SUPPORTED_ENCODERS = {"vit", "vit_msn", "vit_mae", "beit", "ijepa", "dinov2", "dinov3_vit"}

VARIANTS = (
    "scd", "baseline", "lora", "uniform", "feature-only",
    "no-cues", "no-fusion", "no-uncertainty", "no-balance",
)


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "dinov3"
    weights: str | None = None
    variant: str = "scd"
    rank: int = 8
    experts: int = 4
    top_k: int = 2
    alpha: float = 8.0
    router_width: int = 128
    width: int = 128
    cue_width: int = 16
    taps: tuple[int, ...] | None = None
    seed: int = 42


@dataclass
class RoutingPass:
    grid: tuple[int, int]
    expert_mass: dict | None = None
    updates: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    balance: list = field(default_factory=list)

    @property
    def patch_count(self):
        return self.grid[0] * self.grid[1]


_active_pass: ContextVar[RoutingPass | None] = ContextVar("scd_pass", default=None)


class ExpertQKV(nn.Module):
    def __init__(self, channels, outputs, cfg):
        super().__init__()
        self.experts = 1 if cfg.variant == "lora" else cfg.experts
        self.top_k = cfg.top_k
        self.rank = cfg.rank
        self.scale = cfg.alpha / cfg.rank
        self.spatial = cfg.variant not in ("lora", "uniform", "feature-only")
        self.a = nn.Parameter(torch.empty(self.experts, self.rank, channels))
        self.b = nn.Parameter(torch.zeros(outputs, self.experts, self.rank))
        nn.init.kaiming_uniform_(self.a.flatten(0, 1), a=5 ** 0.5)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels) if self.spatial else None
        self.router = None
        if cfg.variant not in ("lora", "uniform"):
            input_width = 2 * channels + 2 if self.spatial else channels
            self.router = nn.Sequential(
                nn.Linear(input_width, cfg.router_width), nn.GELU(), nn.Linear(cfg.router_width, self.experts),
            )

    def gates(self, tokens, grid):
        if self.router is None:
            return tokens.new_full((*tokens.shape[:2], self.experts), 1.0 / self.experts), None

        features = tokens
        if self.spatial:
            height, width = grid
            patch_map = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], height, width)
            local = self.local(patch_map).flatten(2).transpose(1, 2)
            y, x = torch.meshgrid(
                2 * (torch.arange(height, device=tokens.device, dtype=torch.float32) + 0.5) / height - 1,
                2 * (torch.arange(width, device=tokens.device, dtype=torch.float32) + 0.5) / width - 1,
                indexing="ij",
            )
            xy = torch.stack((x, y), dim=-1).reshape(1, -1, 2).to(tokens.dtype)
            features = torch.cat((tokens, local, xy.expand(tokens.shape[0], -1, -1)), dim=-1)
        logits = self.router(features).float()
        selected, indices = logits.topk(self.top_k, dim=-1)
        # Softmax over the selected logits equals softmax, top-k, and renormalization.
        weights = torch.zeros_like(logits).scatter(-1, indices, selected.softmax(dim=-1))
        load = torch.zeros_like(logits).scatter_(-1, indices, 1).mean(dim=(0, 1)) / self.top_k
        importance = logits.softmax(dim=-1).mean(dim=(0, 1))
        balance = self.experts * (load * importance).sum()
        return weights, balance

    def forward(self, tokens, grid, expert_mass=None, layer=None):
        weights, balance = self.gates(tokens, grid)
        if expert_mass is not None:
            expert_mass[layer] = weights.detach().float().mean(dim=(0, 1))
        low_rank = F.linear(tokens, self.a.flatten(0, 1))
        low_rank = low_rank.unflatten(-1, (self.experts, self.rank))
        # Gating in rank space avoids materializing K full-size QKV tensors.
        gated = low_rank * weights.to(low_rank.dtype).unsqueeze(-1)
        update = F.linear(gated.flatten(-2), self.b.flatten(1)) * self.scale
        return update, balance


def add_qkv_update(module, args, base, *, adapter, output_slice, layer):
    routing = _active_pass.get()
    if routing is None:
        return base
    count = routing.patch_count
    if adapter not in routing.updates:
        if args[0].shape[1] < count:
            raise RuntimeError("The encoder removed spatial tokens before the QKV projection.")
        update, balance = adapter(args[0][:, -count:], routing.grid, routing.expert_mass, layer)
        routing.updates[adapter] = update
        if balance is not None:
            routing.balance.append(balance)
    update = routing.updates[adapter][..., output_slice].to(base.dtype)
    # CLS and register tokens use the frozen projection.
    return torch.cat((base[:, :-count], base[:, -count:] + update), dim=1)


def capture_feature(module, args, output, *, index):
    routing = _active_pass.get()
    if routing is not None:
        routing.features[index] = output


def conv_block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GELU())


class SmokeCues(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1), persistent=False)
        self.layers = nn.Sequential(conv_block(3, width), conv_block(width, width))

    def forward(self, image, grid):
        gray = (image * self.luma).sum(dim=1, keepdim=True)
        mean = F.avg_pool2d(gray, 5, stride=1, padding=2, count_include_pad=False)
        mean_square = F.avg_pool2d(gray.square(), 5, stride=1, padding=2, count_include_pad=False)
        contrast = (mean_square - mean.square()).clamp_min(0).sqrt()
        high_pass = (gray - mean).abs()
        cues = F.interpolate(torch.cat((gray, contrast, high_pass), dim=1), size=grid, mode="area")
        return self.layers(cues)


@dataclass
class SegmentationOutput:
    logits: Tensor
    coarse: Tensor | None
    balance: Tensor
    expert_mass: dict | None = None


class SmokeSegmenter(nn.Module):
    def __init__(self, cfg, backbone, stats):
        super().__init__()
        if cfg.variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {cfg.variant}")
        if min(cfg.rank, cfg.experts, cfg.width, cfg.cue_width, cfg.router_width) < 1 or not math.isfinite(cfg.alpha) or cfg.alpha <= 0:
            raise ValueError("LoRA and head dimensions and alpha must be positive.")
        if cfg.variant not in ("baseline", "lora", "uniform") and not 1 <= cfg.top_k <= cfg.experts:
            raise ValueError("Top-k must be in [1, experts] for dynamic routing.")
        self.cfg = cfg
        self.backbone = backbone.requires_grad_(False).eval()
        self.kind = backbone.config.model_type
        if self.kind not in SUPPORTED_ENCODERS:
            raise ValueError(f"Unsupported dense ViT encoder: {self.kind}")
        self.patch_size = backbone.config.patch_size
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size, self.patch_size)
        for name, values in zip(("mean", "std"), stats):
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1), persistent=False)

        channels = backbone.config.hidden_size
        blocks = encoder_blocks(backbone)
        indices = cfg.taps or tuple(len(blocks) * quarter // 4 - 1 for quarter in range(1, 5))
        if (len(indices) != 4 or tuple(sorted(set(indices))) != tuple(indices)
                or not 0 <= indices[0] <= indices[-1] < len(blocks)):
            raise ValueError("Select four distinct, increasing, zero-based ViT block indices.")
        self.adapters = nn.ModuleDict()
        if cfg.variant == "baseline":
            self.probe = nn.Conv2d(channels, 1, 1)
            return

        for index in indices:
            attention = blocks[index].attention
            projections = (attention.q_proj, attention.k_proj, attention.v_proj)
            adapter = ExpertQKV(channels, sum(p.out_features for p in projections), cfg)
            self.adapters[str(index)] = adapter
            start = 0
            for projection in projections:
                end = start + projection.out_features
                projection.register_forward_hook(partial(
                    add_qkv_update, adapter=adapter, output_slice=slice(start, end), layer=index,
                ))
                start = end

        self.taps = (indices[-1],) if cfg.variant == "no-fusion" else indices
        for index in self.taps:
            blocks[index].register_forward_hook(partial(capture_feature, index=index))
        self.projections = nn.ModuleList(nn.Conv2d(channels, cfg.width, 1) for _ in self.taps)
        self.fusion_logits = nn.Parameter(torch.zeros(len(self.taps))) if len(self.taps) > 1 else None
        self.fusion = conv_block(cfg.width, cfg.width)
        self.cues = None if cfg.variant == "no-cues" else SmokeCues(cfg.cue_width)
        self.uses_uncertainty = cfg.variant != "no-uncertainty"
        self.coarse_head = nn.Conv2d(cfg.width, 1, 1)
        refine_inputs = cfg.width + (cfg.cue_width if self.cues else 0) + int(self.uses_uncertainty)
        self.refine = nn.Sequential(conv_block(refine_inputs, cfg.width), nn.Conv2d(cfg.width, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def patch_map(tokens, routing):
        return tokens[:, -routing.patch_count:].transpose(1, 2).reshape(tokens.shape[0], -1, *routing.grid)

    def forward(self, image, output_size=None, report_routing=False):
        height, width = image.shape[-2:]
        ph, pw = self.patch_size
        if height < ph or width < pw or height % ph or width % pw:
            raise ValueError(f"Input size {(height, width)} must be divisible by patch size {(ph, pw)}.")
        output_size = output_size or (height, width)
        routing = RoutingPass((height // ph, width // pw), expert_mass={} if report_routing else None)
        pixels = (image - self.mean) / self.std
        options = {}
        if self.kind in ("vit", "vit_msn", "vit_mae", "ijepa"):
            options["interpolate_pos_encoding"] = True
        if self.kind == "vit_mae":
            # No masking or permutation. Spatial indices must remain in raster order.
            options["noise"] = torch.arange(routing.patch_count, device=image.device).expand(image.shape[0], -1)

        token = _active_pass.set(routing)
        try:
            encoded = self.backbone(pixel_values=pixels, **options)
        finally:
            _active_pass.reset(token)

        if self.cfg.variant == "baseline":
            logits = self.probe(self.patch_map(encoded.last_hidden_state, routing))
            return SegmentationOutput(upsample(logits, output_size), None, logits.new_zeros(()))

        features = [p(self.patch_map(routing.features[i], routing)) for i, p in zip(self.taps, self.projections)]
        fused = features[0]
        if self.fusion_logits is not None:
            fused = (torch.stack(features, dim=0) * self.fusion_logits.softmax(0).view(-1, 1, 1, 1, 1)).sum(0)
        fused = self.fusion(fused)
        coarse = self.coarse_head(fused)
        refine_features = [fused]
        if self.cues is not None:
            refine_features.append(self.cues(image, routing.grid))
        if self.uses_uncertainty:
            probability = coarse.float().sigmoid()
            refine_features.append((4 * probability * (1 - probability)).to(fused.dtype))
        logits = self.refine(torch.cat(refine_features, dim=1))
        balance = torch.stack(routing.balance).mean() if routing.balance else logits.new_zeros(())
        return SegmentationOutput(upsample(logits, output_size), coarse, balance, routing.expert_mass)


def upsample(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def build_model(cfg, download=False, stats=None):
    backbone, source_stats = create_backbone(cfg.backbone, cfg.weights, download, cfg.seed)
    if backbone.config.model_type == "vit_mae":
        backbone.config.mask_ratio = 0.0
    return SmokeSegmenter(cfg, backbone, stats or source_stats)
