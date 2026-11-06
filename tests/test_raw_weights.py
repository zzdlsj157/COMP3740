import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModel

from scd.backbones import RAW_MODELS, load_raw_backbone, raw_config


def upstream_fixture(model, relative_bias):
    state = model.state_dict()
    state_dict = {
        "cls_token": state["embeddings.cls_token"],
        "patch_embed.proj.weight": state["embeddings.patch_embeddings.projection.weight"],
        "patch_embed.proj.bias": state["embeddings.patch_embeddings.projection.bias"],
        "norm.weight": state["layernorm.weight"],
        "norm.bias": state["layernorm.bias"],
    }
    if relative_bias:
        state_dict["rel_pos_bias.relative_position_bias_table"] = state["shared_position_bias.relative_position_bias_table"]
        state_dict["mask_token"] = torch.zeros_like(state["embeddings.cls_token"])
    else:
        state_dict["pos_embed"] = state["embeddings.position_embeddings"]

    for index in range(model.config.num_hidden_layers):
        src, dst = f"layers.{index}.", f"blocks.{index}."
        for base, target in (("layernorm_before", "norm1"), ("layernorm_after", "norm2"),
                             ("attention.o_proj", "attn.proj"), ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
            for suffix in (".weight", ".bias"):
                state_dict[dst + target + suffix] = state[src + base + suffix]
        state_dict[dst + "attn.qkv.weight"] = torch.cat([state[src + f"attention.{p}_proj.weight"] for p in "qkv"])
        if relative_bias:
            for projection in "qv":
                state_dict[dst + f"attn.{projection}_bias"] = state[src + f"attention.{projection}_proj.bias"]
            for number in (1, 2):
                state_dict[dst + f"gamma_{number}"] = state[src + f"lambda_{number}"]
        else:
            state_dict[dst + "attn.qkv.bias"] = torch.cat([state[src + f"attention.{p}_proj.bias"] for p in "qkv"])
    return state_dict


class RawWeightsTest(unittest.TestCase):
    def test_official_checkpoint_layouts(self):
        torch.set_num_threads(2)
        image = torch.rand(1, 3, 24, 24)
        for name in RAW_MODELS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                config = raw_config(
                    name, image_size=24, patch_size=8, hidden_size=32, intermediate_size=64,
                    num_attention_heads=4, num_hidden_layers=4,
                )
                expected = AutoModel.from_config(config).eval()
                raw = upstream_fixture(expected, name == "simmim")
                if name == "mocov3":
                    raw = {"state_dict": {f"module.base_encoder.{key}": value for key, value in raw.items()}}
                elif name == "ibot":
                    raw = {"teacher": {f"module.backbone.{key}": value for key, value in raw.items()}}
                else:
                    raw = {"model": {f"encoder.{key}": value for key, value in raw.items()}}
                path = Path(directory) / "backbone.pth"
                torch.save(raw, path)
                with patch("scd.backbones.raw_config", return_value=config):
                    actual = load_raw_backbone(name, path).eval()
                with torch.no_grad():
                    torch.testing.assert_close(actual(image).last_hidden_state, expected(image).last_hidden_state)
