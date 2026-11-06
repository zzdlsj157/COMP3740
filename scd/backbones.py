"""Load frozen encoders while keeping weight acquisition opt-in."""

from pathlib import Path
from pickle import UnpicklingError

import torch
from torch import nn
from transformers import AutoModel, BeitConfig, DINOv3ViTConfig, ViTConfig


HF_MODELS = {
    "dino": "facebook/dino-vitb16",
    "beit": "microsoft/beit-base-patch16-224-pt22k",
    "msn": "facebook/vit-msn-base",
    "mae": "facebook/vit-mae-base",
    "ijepa": "facebook/ijepa_vith14_22k",
    "dinov2": "facebook/dinov2-base",
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}
RAW_MODELS = ("mocov3", "ibot", "simmim")
BACKBONES = ("tiny", *HF_MODELS, *RAW_MODELS)

IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def create_backbone(name, weights=None, download=False, seed=42):
    if name == "tiny":
        if weights:
            raise ValueError("The tiny backbone uses seeded random initialization.")
        config = DINOv3ViTConfig(
            image_size=64, patch_size=8, hidden_size=48, intermediate_size=96,
            num_hidden_layers=4, num_attention_heads=4, num_register_tokens=2,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = AutoModel.from_config(config)
        return model, IMAGENET_STATS

    source = weights or HF_MODELS.get(name)
    if not source:
        raise ValueError(f"{name} requires --weights pointing to a local checkpoint or Transformers model.")
    if Path(source).is_file():
        if name not in RAW_MODELS:
            raise ValueError(f"{name} expects a Transformers model directory or model ID.")
        return load_raw_backbone(name, source), IMAGENET_STATS
    if Path(source).is_dir():
        source = str(Path(source).resolve())

    model, info = AutoModel.from_pretrained(
        source, local_files_only=not download, trust_remote_code=False,
        dtype=torch.float32, output_loading_info=True,
    )
    # A task-specific checkpoint may include an unused classification/pretraining head.
    missing = [
        key for key in info["missing_keys"]
        if not key.startswith("pooler.") and not key.endswith(("mask_token", "mask_tokens"))
    ]
    if missing or info.get("mismatched_keys"):
        raise ValueError(f"Incomplete pretrained encoder at {source}: missing={missing}, "
                         f"mismatched={info.get('mismatched_keys')}")
    stats = ((0.5,) * 3, (0.5,) * 3) if name == "beit" else IMAGENET_STATS
    # Local processor metadata also covers model variants with different statistics.
    from transformers import AutoImageProcessor

    try:
        processor = AutoImageProcessor.from_pretrained(
            source, local_files_only=not download, trust_remote_code=False,
        )
    except (OSError, ValueError):
        pass
    else:
        if processor.image_mean is not None and processor.image_std is not None:
            stats = (processor.image_mean, processor.image_std)
    return model, stats


def encoder_blocks(model):
    for path in ("layers", "encoder.layer", "model.layer"):
        try:
            layers = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList):
            return layers
    raise ValueError(f"Unsupported ViT layer layout: {model.config.model_type}")


def load_raw_backbone(name, path):
    """Convert the official ViT-B/16 tensor names and check all encoder weights."""
    model = AutoModel.from_config(raw_config(name))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except UnpicklingError as error:
        raise ValueError(f"Could not load tensor weights from {path}. Export its encoder state dict first. "
                         "See the checkpoint section in README.md.") from error
    converted = convert_raw_state(name, checkpoint, len(encoder_blocks(model)))
    expected = set(model.state_dict())
    # HF ViT can have a pooler and BEiT can have an unused masked-token parameter.
    optional = {"pooler.dense.weight", "pooler.dense.bias", "embeddings.mask_token"}
    missing = expected - converted.keys() - optional
    if missing:
        raise ValueError(f"{path} is missing {name} ViT-B/16 encoder weights: {sorted(missing)}")
    model.load_state_dict({key: value for key, value in converted.items() if key in expected}, strict=False)
    return model


def raw_config(name, **overrides):
    common = {"image_size": 224, "patch_size": 16, "hidden_size": 768, "num_hidden_layers": 12,
              "num_attention_heads": 12, "intermediate_size": 3072, "layer_norm_eps": 1e-6}
    common.update(overrides)
    if name == "simmim":
        return BeitConfig(
            **common, use_absolute_position_embeddings=False, use_shared_relative_position_bias=True,
            use_relative_position_bias=False, use_mean_pooling=False, layer_scale_init_value=0.1,
        )
    return ViTConfig(**common, qkv_bias=True)


def convert_raw_state(name, checkpoint, depth):
    if name == "mocov3":
        checkpoint = checkpoint.get("state_dict", checkpoint)
        for prefix in ("module.base_encoder.", "base_encoder."):
            if any(key.startswith(prefix) for key in checkpoint):
                checkpoint = {key[len(prefix):]: value for key, value in checkpoint.items() if key.startswith(prefix)}
                break
    elif name == "ibot":
        checkpoint = checkpoint.get("teacher", checkpoint)
    elif name == "simmim":
        checkpoint = checkpoint.get("model", checkpoint)

    source = {}
    for key, value in checkpoint.items():
        for prefix in ("module.", "backbone.", "encoder."):
            key = key.removeprefix(prefix)
        if isinstance(value, torch.Tensor):
            source[key] = value

    pairs = {
        "cls_token": "embeddings.cls_token",
        "pos_embed": "embeddings.position_embeddings",
        "mask_token": "embeddings.mask_token",
        "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
        "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
        "norm.weight": "layernorm.weight",
        "norm.bias": "layernorm.bias",
        "rel_pos_bias.relative_position_bias_table": "shared_position_bias.relative_position_bias_table",
    }
    result = {target: source[key] for key, target in pairs.items() if key in source}
    for index in range(depth):
        src = f"blocks.{index}."
        dst = f"layers.{index}."
        layer_pairs = {
            "norm1": "layernorm_before",
            "norm2": "layernorm_after",
            "attn.proj": "attention.o_proj",
            "mlp.fc1": "mlp.fc1",
            "mlp.fc2": "mlp.fc2",
        }
        for old, new in layer_pairs.items():
            for suffix in (".weight", ".bias"):
                if src + old + suffix in source:
                    result[dst + new + suffix] = source[src + old + suffix]
        for suffix in (".weight", ".bias"):
            fused = source.get(src + "attn.qkv" + suffix)
            if fused is not None:
                for projection, tensor in zip("qkv", fused.chunk(3, dim=0)):
                    result[dst + f"attention.{projection}_proj" + suffix] = tensor
        if name == "simmim":
            for projection in ("q", "v"):
                key = src + f"attn.{projection}_bias"
                if key in source:
                    result[dst + f"attention.{projection}_proj.bias"] = source[key]
            for number in (1, 2):
                key = src + f"gamma_{number}"
                if key in source:
                    result[dst + f"lambda_{number}"] = source[key]
    return result
