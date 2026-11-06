import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from scd.cli import make_parser
from scd.ood import Calibration, DomainScores, entropy_auroc, routing_divergence
from scd.shifts import CORRUPTIONS, CorruptedDataset, consistency_loss


class OODTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shifts_preserve_mask_and_rng_per_example(self):
        images = torch.linspace(0, 1, 3 * 12 * 12).reshape(1, 3, 12, 12).repeat(2, 1, 1, 1)
        masks = (images[:, :1] > 0.5).float()
        source = TensorDataset(images, masks)
        for kind in CORRUPTIONS:
            with self.subTest(kind=kind):
                dataset = CorruptedDataset(source, kind, 5, seed=17)
                shifted, mask = dataset[1]
                self.assertEqual(shifted.shape, images[1].shape)
                self.assertTrue(((0 <= shifted) & (shifted <= 1)).all())
                self.assertFalse(torch.allclose(shifted, images[1]))
                torch.testing.assert_close(mask, masks[1])
                dataset[0]
                torch.testing.assert_close(shifted, dataset[1][0])

    def test_consistency_only_updates_student(self):
        teacher = torch.tensor([8.0, 0.0], requires_grad=True)
        student = torch.tensor([-8.0, 3.0], requires_grad=True)
        loss = consistency_loss(student, teacher)
        loss.backward()
        self.assertGreater(loss.detach().item(), 0.9)
        self.assertIsNone(teacher.grad)
        self.assertNotEqual(float(student.grad[0]), 0)
        self.assertEqual(float(student.grad[1]), 0)

    def test_calibration_and_drift(self):
        calibration = Calibration(bins=4)
        calibration.update(torch.tensor([0.0, 0.25, 0.75, 1.0]), torch.tensor([0, 0, 1, 1]))
        report = calibration.scores()
        self.assertAlmostEqual(report["ece"], 0.125)
        self.assertAlmostEqual(report["brier"], 0.03125)
        self.assertEqual(sum(bucket["pixels"] for bucket in report["bins"]), 4)
        self.assertAlmostEqual(entropy_auroc([0.1, 0.3], [0.3, 0.8]), 0.875)
        self.assertEqual(routing_divergence([0.5, 0.5], [0.5, 0.5]), 0)
        self.assertEqual(routing_divergence([1, 0], [0, 1]), 1)

    def test_subgroups_and_false_alarms(self):
        target = torch.zeros(3, 1, 4, 4)
        target[1, 0, 0, 0] = 1
        target[2, 0, :2, :] = 1
        p = target.clone()
        p[0, 0, 0, 0] = 1
        scores = DomainScores(small_smoke_fraction=0.1)
        scores.update(p, target, 0.5, {})
        report = scores.scores()
        self.assertEqual(report["false_alarm_rate_on_no_smoke_images"], 1)
        self.assertEqual(report["subgroups"]["small_smoke"]["dice"], 1)
        self.assertEqual(report["subgroups"]["larger_smoke"]["dice"], 1)
        self.assertEqual(report["mean_top_decile_entropy"], 0)

    def test_augmented_training_and_report(self):
        parser = make_parser()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            for split, hue in (("train", 45), ("id", 60), ("target", 90)):
                (root / split / "images").mkdir(parents=True)
                (root / split / "masks").mkdir()
                mask = np.zeros((20, 24), dtype=np.uint8)
                mask[5:12, 7:13] = 255
                image = np.repeat((mask // 2 + hue)[..., None], 3, axis=2)
                Image.fromarray(image).save(root / split / "images" / "sample.png")
                Image.fromarray(mask).save(root / split / "masks" / "sample.png")
            train = parser.parse_args([
                "train", "--backbone", "tiny", "--size", "32", "--device", "cpu", "--epochs", "1",
                "--train", str(root / "train"), "--val", str(root / "id"), "--run", str(root / "run"),
                "--width", "16", "--cue-width", "4", "--router-width", "8", "--domain-augment", "1",
            ])
            checkpoint = train.handler(train)
            history = json.loads((root / "run" / "history.jsonl").read_text())
            self.assertEqual(history["corrupted_train_images"], 1)
            self.assertIsNotNone(history["consistency_loss_on_corruptions"])
            path = root / "report.json"
            ood = parser.parse_args([
                "ood", "--checkpoint", str(checkpoint), "--id", str(root / "id"), "--device", "cpu",
                "--ood", f"another_camera={root / 'target'}", "--corruption", "noise", "--severity", "1", "5",
                "--flip-tta", "--report", str(path),
            ])
            ood.handler(ood)
            report = json.loads(path.read_text())
            self.assertEqual(report["id"]["images"], 1)
            self.assertEqual(len(report["domains"]["another_camera"]["routing_jsd_bits_vs_id"]), 4)
            self.assertEqual(set(report["corruptions"]["noise"]), {"1", "5"})
            for mass in report["id"]["expert_mass"].values():
                self.assertAlmostEqual(sum(mass), 1.0, places=6)
            ood.ood = [f"overlap={root / 'id'}"]
            ood.report = None
            with self.assertRaisesRegex(ValueError, "overlaps"):
                ood.handler(ood)
