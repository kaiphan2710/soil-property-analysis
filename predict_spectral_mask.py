"""
Predict a 2D mask from a dish NPZ using a trained 1D spectral classifier.

The classifier predicts each pixel spectrum independently. The resulting labels
are reshaped back to the dish image grid, then optionally cleaned with connected
component filtering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spectral_utils import PALETTE, colorize_mask, normalise_cube


def _valid_pixel_mask(cube: np.ndarray, min_percentile: float) -> np.ndarray:
    mean_image = cube.mean(axis=0)
    threshold = np.percentile(mean_image, min_percentile)
    return mean_image > threshold


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask

    try:
        from scipy import ndimage
    except Exception:
        print("scipy not available; skipping connected-component cleanup.")
        return mask

    cleaned = mask.copy()
    for class_id in sorted(PALETTE):
        if class_id == 0:
            continue
        binary = cleaned == class_id
        labels, count = ndimage.label(binary)
        for component_id in range(1, count + 1):
            component = labels == component_id
            if int(component.sum()) < min_area:
                cleaned[component] = 0
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Predict 2D mask with a 1D spectral classifier.")
    parser.add_argument("--model", required=True, help="Path to .joblib model from train_spectral_classifier.py")
    parser.add_argument("--npz", required=True, help="Dish-level NPZ to predict")
    parser.add_argument("--out", default="predictions/spectral_mask.png")
    parser.add_argument("--raw-out", default="", help="Optional raw label mask .npy output")
    parser.add_argument("--normalise", choices=["minmax", "snv", "none"], default="minmax")
    parser.add_argument("--valid-min-percentile", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--min-area", type=int, default=20)
    args = parser.parse_args()

    import joblib

    payload = joblib.load(args.model)
    model = payload["model"]

    with np.load(args.npz, allow_pickle=True) as npz_file:
        cube = npz_file["datacube"]

    cube = normalise_cube(cube, args.normalise)
    bands, height, width = cube.shape
    valid_mask = _valid_pixel_mask(cube, min_percentile=args.valid_min_percentile)

    X = cube.reshape(bands, -1).T.astype(np.float32, copy=False)
    pred_flat = np.zeros((X.shape[0],), dtype=np.uint8)
    valid_flat = valid_mask.reshape(-1)
    valid_indices = np.flatnonzero(valid_flat)

    for start in range(0, len(valid_indices), args.chunk_size):
        indices = valid_indices[start : start + args.chunk_size]
        pred_flat[indices] = model.predict(X[indices]).astype(np.uint8)

    pred = pred_flat.reshape(height, width)
    pred[~valid_mask] = 0
    pred = _remove_small_components(pred, min_area=args.min_area)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colorize_mask(pred).save(out_path)

    if args.raw_out:
        raw_path = Path(args.raw_out)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, pred)

    print(f"Saved spectral prediction mask: {out_path}")
    if args.raw_out:
        print(f"Saved raw label mask: {args.raw_out}")


if __name__ == "__main__":
    main()
