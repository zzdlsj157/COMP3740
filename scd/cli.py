"""Commands for training, evaluating dataset splits, and exporting masks."""

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .backbones import BACKBONES
from .data import SmokeDataset, by_stem, read_image, read_pairs, split_train_val
from .model import VARIANTS, ModelConfig, build_model
from .objectives import LossWeights, PixelCounts, segmentation_loss
from .ood import compare_domain, score_domain
from .shifts import CORRUPTIONS, CorruptedDataset, DomainAugment, consistency_loss


def select_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def data_loader(pairs, args, device, shuffle=False, shift=None):
    dataset = SmokeDataset(pairs, args.size, args.mask_threshold)
    if shift is not None:
        dataset = CorruptedDataset(dataset, *shift)
    return DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )


def move_batch(batch, device):
    return (tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch)


def save_checkpoint(path, model, args, epoch, metrics):
    state = {
        "format_version": 1,
        "model": asdict(model.cfg),
        "normalization": (model.mean.flatten().tolist(), model.std.flatten().tolist()),
        "trainable": {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad},
        "size": args.size,
        "mask_threshold": args.mask_threshold,
        "epoch": epoch,
        "validation": metrics,
    }
    # A partially written checkpoint should not replace the previous best.
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_checkpoint(args, device):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported SCD checkpoint format.")
    cfg = ModelConfig(**state["model"])
    if args.weights:
        cfg = replace(cfg, weights=args.weights)
    model = build_model(cfg, args.download, state["normalization"])
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if state["trainable"].keys() != expected.keys():
        raise ValueError("Checkpoint parameter names do not match the SCD architecture.")
    with torch.no_grad():
        for name, value in state["trainable"].items():
            if value.shape != expected[name].shape:
                raise ValueError(f"Checkpoint parameter shape mismatch: {name}")
            expected[name].copy_(value)
    args.size = state["size"] if args.size is None else args.size
    if getattr(args, "mask_threshold", None) is None:
        args.mask_threshold = state["mask_threshold"]
    return model.to(device).eval()


@torch.inference_mode()
def validate(model, loader, device, threshold, weights=None):
    model.eval()
    counts = PixelCounts(device)
    loss_total = torch.zeros((), device=device)
    for batch in loader:
        images, masks = move_batch(batch, device)
        output = model(images)
        counts.update(output.logits, masks, threshold)
        if weights is not None:
            loss_total += segmentation_loss(output, masks, weights) * images.shape[0]
    return counts, loss_total.item() / len(loader.dataset) if weights is not None else None


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    if args.size is None:
        args.size = 64 if args.backbone == "tiny" else 448
    weights_path = args.weights
    if weights_path and Path(weights_path).exists():
        weights_path = str(Path(weights_path).resolve())
    cfg = ModelConfig(
        backbone=args.backbone, weights=weights_path, variant=args.variant,
        rank=args.rank, experts=args.experts, top_k=args.top_k, alpha=args.alpha,
        router_width=args.router_width, width=args.width, cue_width=args.cue_width,
        taps=tuple(args.taps) if args.taps else None, seed=args.seed,
    )
    pairs = read_pairs(args.train)
    validation = read_pairs(args.val) if args.val else None
    train_pairs, val_pairs = split_train_val(pairs, validation, args.val_fraction, args.seed)
    model = build_model(cfg, args.download).to(device)
    train_loader = data_loader(train_pairs, args, device, shuffle=True)
    val_loader = data_loader(val_pairs, args, device)
    weights = LossWeights(args.aux_weight, args.boundary_weight, 0 if args.variant == "no-balance" else args.balance_weight)
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    augment = DomainAugment(args.domain_augment, args.augment_severity, args.seed)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "best.pt"
    settings = {key: value for key, value in vars(args).items() if key != "handler"}
    settings.update({"device": str(device), "model": asdict(cfg), "loss_weights": asdict(weights)})
    with (run / "config.json").open("x", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "device": str(device), "train_images": len(train_pairs), "val_images": len(val_pairs),
        "trainable_parameters": sum(param.numel() for param in parameters),
        "total_parameters": sum(param.numel() for param in model.parameters()),
    }), flush=True)
    best = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = torch.zeros((), device=device)
        consistency_total = torch.zeros((), device=device)
        shifted_images = 0
        for batch in train_loader:
            images, masks = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            shifted = augment(images)
            with torch.autocast(device_type="cuda", enabled=amp):
                output = model(images if shifted is None else shifted)
                loss = segmentation_loss(output, masks, weights)
                if shifted is not None:
                    shifted_images += images.shape[0]
                    if args.consistency_weight:
                        with torch.no_grad():
                            clean = model(images).logits
                        agreement = consistency_loss(output.logits, clean)
                        loss = loss + args.consistency_weight * agreement
                        consistency_total += agreement.detach() * images.shape[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.detach() * images.shape[0]
        counts, val_loss = validate(model, val_loader, device, 0.5, weights)
        metrics = counts.scores()
        if not math.isfinite(val_loss):
            raise RuntimeError("Validation loss became non-finite.")
        improved = metrics["dice"] > best
        log = {
            "epoch": epoch, "train_loss": loss_total.item() / len(train_pairs),
            "val_loss": val_loss, **metrics, "best": improved,
        }
        if args.domain_augment:
            log["corrupted_train_images"] = shifted_images
            log["consistency_loss_on_corruptions"] = (
                consistency_total.item() / shifted_images if shifted_images and args.consistency_weight else None
            )
        with (run / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
        if improved:
            best = metrics["dice"]
            stale = 0
            save_checkpoint(checkpoint, model, args, epoch, metrics)
        else:
            stale += 1
            if stale >= args.patience:
                break
    return checkpoint


def add_named_splits(splits, entries, option):
    for entry in entries:
        label, sep, path = entry.partition("=")
        if not sep or not label or not path or label in splits or label == "total":
            raise ValueError(f"Use distinct {option} NAME=PATH values: {entry}")
        splits[label] = path


def disjoint_pairs(splits):
    seen = set()
    result = {}
    for label, source in splits.items():
        pairs = read_pairs(source)
        paths = {image for image, _ in pairs}
        if seen & paths:
            raise ValueError(f"Evaluation split {label} overlaps another split.")
        seen.update(paths)
        result[label] = pairs
    return result


def evaluate(args):
    device = select_device(args.device)
    model = restore_checkpoint(args, device)
    splits = {"test": args.test} if args.test else {}
    add_named_splits(splits, args.split or [], "--split")
    if not splits:
        raise ValueError("Supply --test PATH or --split NAME=PATH.")
    reports = {}
    combined = PixelCounts(device)
    for name, pairs in disjoint_pairs(splits).items():
        counts, _ = validate(model, data_loader(pairs, args, device), device, args.threshold)
        combined.totals += counts.totals
        reports[name] = {"images": len(pairs), **counts.scores()}
    if len(splits) > 1:
        reports["total"] = {"images": sum(row["images"] for row in reports.values()), **combined.scores()}
    print(json.dumps(reports, indent=2))
    return reports


def evaluate_ood(args):
    splits = {"id": args.id}
    add_named_splits(splits, args.ood or [], "--ood")
    corruptions = tuple(dict.fromkeys(args.corruption or []))
    severities = tuple(dict.fromkeys(args.severity))
    if len(splits) == 1 and not corruptions:
        raise ValueError("Supply at least one --ood NAME=PATH or --corruption KIND.")
    destination = Path(args.report) if args.report else None
    if destination and destination.exists():
        raise FileExistsError(f"Report path already exists: {destination}")
    pairs = disjoint_pairs(splits)
    device = select_device(args.device)
    model = restore_checkpoint(args, device)

    def score(examples, shift=None):
        return score_domain(
            model, data_loader(examples, args, device, shift=shift), device,
            args.threshold, args.small_smoke_fraction, args.flip_tta,
        )

    reference = score(pairs["id"])
    reference_report = reference.scores()
    report = {
        "settings": {
            "checkpoint": str(Path(args.checkpoint).resolve()), "size": args.size,
            "threshold": args.threshold, "mask_threshold": args.mask_threshold,
            "small_smoke_fraction": args.small_smoke_fraction, "flip_tta": args.flip_tta,
            "corruption_seed": args.corruption_seed, "splits": splits,
        },
        "id": reference_report,
        "domains": {
            label: compare_domain(reference, score(examples), reference_report)
            for label, examples in pairs.items() if label != "id"
        },
        "corruptions": {
            kind: {
                severity: compare_domain(
                    reference, score(pairs["id"], (kind, severity, args.corruption_seed)), reference_report,
                )
                for severity in severities
            }
            for kind in corruptions
        },
    }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
        print(f"Wrote OOD report to {destination}.")
    else:
        print(serialized)
    return report


@torch.inference_mode()
def predict(args):
    device = select_device(args.device)
    model = restore_checkpoint(args, device)
    source = Path(args.input)
    destination = Path(args.output)
    if source.is_dir():
        images = [(path, destination / (str(stem) + ".png")) for stem, path in by_stem(source).items()]
    else:
        if destination.suffix.lower() != ".png":
            raise ValueError("For a single image, --output must be a PNG filename.")
        images = [(source, destination)]
    if not images:
        raise ValueError(f"No supported images found in {source}.")
    for path, output in images:
        if output.exists():
            raise FileExistsError(f"Output path already exists: {output}")
        if source.is_dir() and source.resolve() in output.resolve().parents:
            raise ValueError("Choose an output directory outside the input image directory.")
    for path, output in images:
        image, size = read_image(path, args.size)
        logits = model(image.unsqueeze(0).to(device), output_size=size).logits
        mask = ((logits[0, 0].float().sigmoid() >= args.threshold) * 255).byte().cpu().numpy()
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(output)
    print(f"Wrote {len(images)} mask(s) to {destination}.")
    return [output for _, output in images]


def smoke_test(args):
    torch.set_num_threads(min(2, torch.get_num_threads()))
    with TemporaryDirectory(prefix="scd-smoke-") as work:
        root = Path(work)
        for split in ("train", "val"):
            for folder in ("images", "masks"):
                (root / split / folder).mkdir(parents=True)
            for index in range(2):
                mask = np.zeros((40, 56), dtype=np.uint8)
                offset = 2 if split == "val" else 0
                mask[8 + offset:30 + offset, 8 + 8 * index:35 + 8 * index] = 255
                rgb = np.repeat((mask // 2 + 70)[..., None], 3, axis=2)
                Image.fromarray(rgb).save(root / split / "images" / f"{index}.png")
                Image.fromarray(mask).save(root / split / "masks" / f"{index}.png")
        parser = make_parser()
        checkpoint = train(parser.parse_args([
            "train", "--backbone", "tiny", "--device", args.device, "--train", str(root / "train"),
            "--val", str(root / "val"), "--run", str(root / "run"), "--epochs", "4", "--batch-size", "1",
            "--lr", "0.002",
        ]))
        evaluate(parser.parse_args([
            "evaluate", "--checkpoint", str(checkpoint), "--test", str(root / "val"), "--device", args.device,
        ]))
        prediction = predict(parser.parse_args([
            "predict", "--checkpoint", str(checkpoint), "--input", str(root / "val" / "images" / "0.png"),
            "--output", str(root / "prediction.png"), "--device", args.device,
        ]))[0]
        with Image.open(prediction) as mask:
            if mask.size != (56, 40) or set(np.unique(mask)) - {0, 255}:
                raise RuntimeError("Smoke test produced an invalid mask.")
    print("Offline training, checkpoint loading, evaluation, and inference passed.")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train an SCD or baseline segmentation model")
    training.add_argument("--train", required=True)
    training.add_argument("--val")
    training.add_argument("--val-fraction", type=float, default=0.1)
    training.add_argument("--run", required=True)
    training.add_argument("--backbone", choices=BACKBONES, default="dinov3")
    training.add_argument("--variant", choices=VARIANTS, default="scd")
    training.add_argument("--taps", type=int, nargs=4, metavar="BLOCK")
    training.add_argument("--rank", type=int, default=8)
    training.add_argument("--experts", type=int, default=4)
    training.add_argument("--top-k", type=int, default=2)
    training.add_argument("--alpha", type=float, default=8)
    training.add_argument("--router-width", type=int, default=128)
    training.add_argument("--width", type=int, default=128)
    training.add_argument("--cue-width", type=int, default=16)
    training.add_argument("--aux-weight", type=float, default=0.4)
    training.add_argument("--boundary-weight", type=float, default=0.1)
    training.add_argument("--balance-weight", type=float, default=0.01)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--patience", type=int, default=10)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--no-amp", action="store_true")
    training.add_argument("--domain-augment", type=float, default=0.0, help="probability of corrupting a training batch")
    training.add_argument("--augment-severity", type=int, choices=range(1, 6), default=3)
    training.add_argument("--consistency-weight", type=float, default=0.2, help="clean-teacher weight on augmented batches")
    training.set_defaults(handler=train)

    evaluation = commands.add_parser("evaluate", help="evaluate one test set or named subsets")
    evaluation.add_argument("--test")
    evaluation.add_argument("--split", action="append", metavar="NAME=PATH")
    evaluation.set_defaults(handler=evaluate)

    ood = commands.add_parser("ood", help="evaluate domain shift, synthetic corruptions, and routing drift")
    ood.add_argument("--id", required=True, help="in-distribution evaluation split")
    ood.add_argument("--ood", action="append", metavar="NAME=PATH", help="out-of-distribution split")
    ood.add_argument("--corruption", action="append", choices=CORRUPTIONS)
    ood.add_argument("--severity", type=int, nargs="+", choices=range(1, 6), default=[1, 3, 5])
    ood.add_argument("--corruption-seed", type=int, default=42)
    ood.add_argument("--small-smoke-fraction", type=float, default=0.02)
    ood.add_argument("--flip-tta", action="store_true", help="average original and mirrored probabilities")
    ood.add_argument("--report", help="write the JSON report to this file")
    ood.set_defaults(handler=evaluate_ood)

    inference = commands.add_parser("predict", help="export PNG masks at the original image dimensions")
    inference.add_argument("--input", required=True)
    inference.add_argument("--output", required=True)
    inference.set_defaults(handler=predict)

    smoke = commands.add_parser("smoke-test", help="verify the entire pipeline without data or downloads")
    smoke.add_argument("--device", default="cpu")
    smoke.set_defaults(handler=smoke_test)

    for command in (training, evaluation, inference, ood):
        command.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
        command.add_argument("--size", type=int, help="side length of square encoder input")
        command.add_argument("--weights", help="local pretrained checkpoint, Transformers directory, or HF ID")
        command.add_argument("--download", action="store_true", help="allow Hugging Face downloads")
    for command in (training, evaluation, ood):
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--mask-threshold", type=int, default=0 if command is training else None)
    for command in (evaluation, inference, ood):
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--threshold", type=float, default=0.5)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    for name in ("size", "batch_size", "epochs", "patience"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    for name in ("workers", "mask_threshold", "aux_weight", "boundary_weight", "balance_weight", "weight_decay", "consistency_weight"):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative.")
    if hasattr(args, "threshold") and not 0 < args.threshold < 1:
        parser.error("--threshold must be in (0, 1).")
    if args.command == "ood":
        if not 0 < args.small_smoke_fraction < 1:
            parser.error("--small-smoke-fraction must be in (0, 1).")
        if args.corruption_seed < 0:
            parser.error("--corruption-seed must be nonnegative.")
    if args.command == "train":
        if not 0 <= args.domain_augment <= 1:
            parser.error("--domain-augment must be in [0, 1].")
        if not math.isfinite(args.lr) or args.lr <= 0:
            parser.error("--lr must be finite and positive.")
        if any((Path(args.run) / filename).exists() for filename in ("config.json", "history.jsonl", "best.pt")):
            parser.error("Choose a fresh --run directory. It already contains training results.")
    try:
        args.handler(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
