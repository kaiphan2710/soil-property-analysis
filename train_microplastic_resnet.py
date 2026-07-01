"""
Minimal runnable training/test pipeline for hyperspectral ResNet segmentation.

Run from PROJECT root:
    python train_microplastic_resnet.py --data-root microplastic --pattern "**/Merged/*_s*.npz" --epochs 1 --max-batches 2

This script is intentionally small:
    - loads dish-level NPZ files
    - builds a ResNet-based hyperspectral segmentation model
    - runs train/validation loops
    - saves a checkpoint

Expected classes:
    0 = background/soil
    1 = PP
    2 = PET
    3 = PS
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from microplastic_dataset import MicroplasticDishDataset, segmentation_collate_fn
from microplastic_resnet import HyperspectralResNetSegmenter


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_datasets(
    data_root: str,
    pattern: str,
    val_fraction: float,
    seed: int,
    normalise: str,
    overfit_one: bool,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    full_dataset = MicroplasticDishDataset.from_folder(
        data_root,
        pattern=pattern,
        task="segmentation",
        normalise=normalise,
    )

    sample_image, _ = full_dataset[0]
    in_channels = int(sample_image.shape[0])

    if overfit_one:
        one_sample = Subset(full_dataset, [0])
        return one_sample, one_sample, in_channels

    if len(full_dataset) == 1:
        return full_dataset, full_dataset, in_channels

    val_size = max(1, int(round(len(full_dataset) * val_fraction)))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )
    return train_dataset, val_dataset, in_channels


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_background: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        start_class = 1 if self.ignore_background else 0
        probs = probs[:, start_class:]
        target_one_hot = target_one_hot[:, start_class:]

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        denominator = torch.sum(probs + target_one_hot, dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        background_weight: float = 0.05,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        class_weights[0] = background_weight
        self.register_buffer("class_weights", class_weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        self.dice = DiceLoss(num_classes=num_classes, ignore_background=True)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def compute_batch_iou(logits: torch.Tensor, masks: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    ious = []
    for class_id in range(1, num_classes):
        pred_mask = preds == class_id
        true_mask = masks == class_id
        union = torch.logical_or(pred_mask, true_mask).sum().item()
        if union == 0:
            continue
        intersection = torch.logical_and(pred_mask, true_mask).sum().item()
        ious.append(intersection / union)
    return float(sum(ious) / len(ious)) if ious else 0.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    num_classes: int,
    train: bool,
    max_batches: int = 0,
):
    model.train(train)
    total_loss = 0.0
    total_iou = 0.0
    count = 0

    for batch_idx, (images, masks) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_iou += compute_batch_iou(logits.detach(), masks.detach(), num_classes)
        count += 1

    if count == 0:
        return {"loss": 0.0, "iou": 0.0}

    return {
        "loss": total_loss / count,
        "iou": total_iou / count,
    }


def main():
    parser = argparse.ArgumentParser(description="Train/test hyperspectral ResNet segmentation.")
    parser.add_argument("--data-root", default="microplastic", help="Dataset root folder")
    parser.add_argument("--pattern", default="**/Merged/*_s*.npz", help="Glob pattern under data root")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resnet", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--normalise", choices=["minmax", "snv", "none"], default="minmax")
    parser.add_argument("--max-batches", type=int, default=0, help="Limit batches per epoch for quick tests")
    parser.add_argument("--checkpoint", default="checkpoints/hyperspectral_resnet_segmenter.pt")
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument(
        "--overfit-one",
        action="store_true",
        help="Use the same first sample for train and val to verify the model can learn.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device()
    print(f"Using device: {device}")

    train_dataset, val_dataset, in_channels = build_datasets(
        data_root=args.data_root,
        pattern=args.pattern,
        val_fraction=args.val_fraction,
        seed=args.seed,
        normalise=args.normalise,
        overfit_one=args.overfit_one,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Input channels: {in_channels}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=segmentation_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=segmentation_collate_fn,
    )

    model = HyperspectralResNetSegmenter(
        in_channels=in_channels,
        num_classes=args.num_classes,
        resnet_name=args.resnet,
    ).to(device)

    criterion = CombinedSegmentationLoss(
        num_classes=args.num_classes,
        background_weight=args.background_weight,
        dice_weight=args.dice_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=args.num_classes,
            train=True,
            max_batches=args.max_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            num_classes=args.num_classes,
            train=False,
            max_batches=args.max_batches,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f}, train IoU={train_metrics['iou']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}, val IoU={val_metrics['iou']:.4f}"
        )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": in_channels,
            "num_classes": args.num_classes,
            "resnet": args.resnet,
            "class_mapping": {"background": 0, "PP": 1, "PET": 2, "PS": 3},
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
