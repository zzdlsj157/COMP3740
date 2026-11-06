import tempfile
import unittest

import torch
from transformers import (
    AutoModel, BeitConfig, DINOv3ViTConfig, Dinov2Config, IJepaConfig,
    ViTConfig, ViTMAEConfig, ViTMSNConfig,
)

from scd.backbones import IMAGENET_STATS, create_backbone
from scd.model import VARIANTS, ExpertQKV, ModelConfig, SmokeSegmenter, build_model
from scd.objectives import LossWeights, PixelCounts, segmentation_loss


class SCDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_expert_contraction_and_sparse_weights(self):
        torch.manual_seed(1)
        cfg = ModelConfig(rank=3, experts=4, top_k=2, alpha=6, router_width=8)
        adapter = ExpertQKV(8, 24, cfg)
        with torch.no_grad():
            adapter.b.normal_()
        tokens = torch.randn(2, 6, 8)
        gates, balance = adapter.gates(tokens, (2, 3))
        torch.testing.assert_close(gates.sum(dim=-1), torch.ones(2, 6))
        self.assertTrue(torch.all((gates != 0).sum(dim=-1) == 2))
        expected = adapter.scale * torch.einsum("bnk,okr,kri,bni->bno", gates, adapter.b, adapter.a, tokens)
        actual, actual_balance = adapter(tokens, (2, 3))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(balance, actual_balance)

    def test_native_architectures(self):
        # Resize a 3x3 source grid to a 4x5 inference grid; no weight downloads.
        common = dict(image_size=24, patch_size=8, hidden_size=32, intermediate_size=64,
                      num_attention_heads=4, num_hidden_layers=4)
        configs = (
            ViTConfig(**common), ViTMSNConfig(**common), ViTMAEConfig(**common, mask_ratio=0.0),
            BeitConfig(**common, use_shared_relative_position_bias=True), IJepaConfig(**common),
            Dinov2Config(**common), DINOv3ViTConfig(**common, num_register_tokens=3),
        )
        image = torch.rand(2, 3, 32, 40)
        for config in configs:
            with self.subTest(backbone=config.model_type):
                cfg = ModelConfig(width=16, cue_width=4, rank=2, router_width=8)
                model = SmokeSegmenter(cfg, AutoModel.from_config(config), IMAGENET_STATS).train()
                self.assertFalse(model.backbone.training)
                output = model(image)
                self.assertEqual(output.logits.shape, (2, 1, 32, 40))
                self.assertEqual(output.coarse.shape, (2, 1, 4, 5))
                target = (image[:, :1] > 0.5).float()
                segmentation_loss(output, target, LossWeights()).backward()
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.backbone.parameters()))
                for adapter in model.adapters.values():
                    self.assertGreater(adapter.b.grad.abs().sum().item(), 0)
                    self.assertGreater(adapter.router[-1].weight.grad.abs().sum().item(), 0)

    def test_variants(self):
        image = torch.rand(1, 3, 24, 32)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = ModelConfig(backbone="tiny", variant=variant, width=16, cue_width=4, rank=2, router_width=8)
                model = build_model(cfg)
                output = model(image, output_size=(25, 35))
                self.assertEqual(output.logits.shape, (1, 1, 25, 35))
                self.assertEqual(len(model.adapters), 0 if variant == "baseline" else 4)
                if variant in ("baseline", "lora", "uniform"):
                    self.assertEqual(output.balance.item(), 0)
                if variant == "no-fusion":
                    self.assertEqual(len(model.projections), 1)

    def test_offline_local_weights(self):
        config = ViTConfig(image_size=32, patch_size=8, hidden_size=32, intermediate_size=64,
                           num_attention_heads=4, num_hidden_layers=4)
        model = AutoModel.from_config(config)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded, _ = create_backbone("dino", weights=directory)
        torch.testing.assert_close(loaded.state_dict(), model.state_dict())

    def test_pixel_counts(self):
        counts = PixelCounts("cpu")
        counts.update(torch.tensor([1.0, 1.0, 0.0, -1.0]), torch.tensor([1, 0, 1, 1]))
        counts.update(torch.tensor([-1.0]), torch.tensor([0]))
        self.assertEqual(counts.scores(), {"dice": 2 / 3, "iou": 0.5, "precision": 2 / 3, "recall": 2 / 3})


if __name__ == "__main__":
    unittest.main()
