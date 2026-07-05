"""
Run inference on one exported dish NPZ and save a predicted mask PNG.

Example:
    python predict_microplastic_resnet.py \
      --checkpoint checkpoints/hyperspectral_resnet_segmenter.pt \
      --npz microplastic/S1_S4_S7_S10_2026_06_30_14_54_02/Merged/S1_S4_S7_S10_FX10FX17_2026_06_30_14_54_02_S1.npz \
      --out prediction_s1.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from microplastic_dataset import MicroplasticDishDataset
from microplastic_resnet import HyperspectralResNetSegmenter


PALETTE = {
    0: (0, 0, 0),        # background
    1: (255, 150, 0),    # PP
    2: (255, 0, 0),      # PET
    3: (255, 0, 180),    # PS
}


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def colorize_mask(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser(description="Predict microplastic mask for one NPZ dish.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", default="prediction_mask.png")
    args = parser.parse_args()

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    dataset = MicroplasticDishDataset(
        [args.npz],
        task="segmentation",
        normalise="minmax",
    )
    image, _ = dataset[0]

    model = HyperspectralResNetSegmenter(
        in_channels=int(checkpoint["in_channels"]),
        num_classes=int(checkpoint["num_classes"]),
        resnet_name=checkpoint.get("resnet", "resnet18"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colorize_mask(pred).save(out_path)
    print(f"Saved prediction mask: {out_path}")


if __name__ == "__main__":
    main()
