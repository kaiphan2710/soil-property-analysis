"""
Extract sample-level features for microplastic content regression.

This is the bridge from "where are the particles?" to "how much content is in
the sample?". It converts each dish-level NPZ and its annotation mask into one
CSV row of spatial + spectral features.

Typical usage:
    python extract_content_features.py \
      --data-root microplastic \
      --pattern "**/Merged/*.npz" \
      --out features/content_features_v1.csv

The produced CSV can be used with train_content_regressor.py. Until true
content labels are available, the script also writes annotation-derived proxy
targets such as pp_proxy_percent based on labelled area ratio.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from spectral_utils import (
    DEFAULT_CLASS_TO_ID,
    class_name_from_annotation,
    draw_annotation_mask,
    is_dish_npz,
    load_sidecar_annotations,
    normalise_cube,
    npz_scalar_as_str,
    read_npz_json,
)


ID_TO_CLASS = {
    0: "soil",
    1: "PP",
    2: "PET",
    3: "PS",
}


def choose_torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_detector(checkpoint_path: str):
    import torch

    from microplastic_resnet import HyperspectralResNetSegmenter

    device = choose_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HyperspectralResNetSegmenter(
        in_channels=int(checkpoint["in_channels"]),
        num_classes=int(checkpoint["num_classes"]),
        resnet_name=checkpoint.get("resnet", "resnet18"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def predict_detector_mask(cube: np.ndarray, detector_model, detector_device) -> np.ndarray:
    import torch

    image = torch.from_numpy(np.asarray(cube, dtype=np.float32).copy()).unsqueeze(0).to(detector_device)
    with torch.no_grad():
        logits = detector_model(image)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


def parse_band_indices(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def valid_pixel_mask(cube: np.ndarray, min_percentile: float) -> np.ndarray:
    mean_image = cube.mean(axis=0)
    threshold = np.percentile(mean_image, min_percentile)
    return mean_image > threshold


def load_dish_npz(path: Path, normalise: str):
    with np.load(path, allow_pickle=True) as npz_file:
        cube = npz_file["datacube"]
        annotations = read_npz_json(npz_file, "annotations_json", None)
        source_roi = read_npz_json(npz_file, "source_roi_json", {})
        name = npz_scalar_as_str(npz_file, "name") or path.stem
        export_type = npz_scalar_as_str(npz_file, "type")

    if annotations is None:
        annotations = load_sidecar_annotations(path)

    return normalise_cube(cube, normalise), annotations, source_roi, name, export_type


def build_annotation_mask(annotations: List[Dict], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        class_name = class_name_from_annotation(ann.get("name", ""))
        label = DEFAULT_CLASS_TO_ID.get(class_name)
        if label is None:
            continue
        instance_mask = draw_annotation_mask(ann, height, width).astype(bool)
        mask[instance_mask] = int(label)
    return mask


def component_stats(binary_mask: np.ndarray) -> Tuple[int, int, float]:
    """
    Return connected-component count, max area, and mean area.

    Uses scipy if installed; otherwise falls back to a simple pure-numpy flood
    fill. The masks are small dish crops, so the fallback is acceptable.
    """
    if not np.any(binary_mask):
        return 0, 0, 0.0

    try:
        from scipy import ndimage

        labels, count = ndimage.label(binary_mask)
        if count == 0:
            return 0, 0, 0.0
        areas = np.bincount(labels.ravel())[1:]
        return int(count), int(areas.max()), float(areas.mean())
    except Exception:
        visited = np.zeros(binary_mask.shape, dtype=bool)
        height, width = binary_mask.shape
        areas = []
        for y in range(height):
            for x in range(width):
                if visited[y, x] or not binary_mask[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                area = 0
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and binary_mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                areas.append(area)
        return len(areas), int(max(areas)), float(sum(areas) / len(areas))


def add_region_spectral_features(
    row: Dict[str, float],
    cube: np.ndarray,
    region_mask: np.ndarray,
    prefix: str,
    band_indices: Iterable[int],
    include_full_spectrum: bool,
):
    bands = cube.shape[0]
    selected_band_means: Dict[int, float] = {}
    selected_band_stds: Dict[int, float] = {}
    if np.any(region_mask):
        spectra = cube[:, region_mask]
        mean_spectrum = spectra.mean(axis=1)
        std_spectrum = spectra.std(axis=1)
        row[f"{prefix}_spectral_mean"] = float(mean_spectrum.mean())
        row[f"{prefix}_spectral_std"] = float(mean_spectrum.std())
        row[f"{prefix}_pixel_spectral_std_mean"] = float(std_spectrum.mean())
        for band in band_indices:
            if 0 <= band < bands:
                selected_band_means[band] = float(mean_spectrum[band])
                selected_band_stds[band] = float(std_spectrum[band])
                row[f"{prefix}_band_{band}_mean"] = selected_band_means[band]
                row[f"{prefix}_band_{band}_std"] = selected_band_stds[band]
        if include_full_spectrum:
            for band in range(bands):
                row[f"{prefix}_band_{band}_mean"] = float(mean_spectrum[band])
    else:
        row[f"{prefix}_spectral_mean"] = 0.0
        row[f"{prefix}_spectral_std"] = 0.0
        row[f"{prefix}_pixel_spectral_std_mean"] = 0.0
        for band in band_indices:
            if 0 <= band < bands:
                selected_band_means[band] = 0.0
                selected_band_stds[band] = 0.0
                row[f"{prefix}_band_{band}_mean"] = 0.0
                row[f"{prefix}_band_{band}_std"] = 0.0
        if include_full_spectrum:
            for band in range(bands):
                row[f"{prefix}_band_{band}_mean"] = 0.0

    selected_bands = sorted(selected_band_means)
    for idx, band_a in enumerate(selected_bands):
        for band_b in selected_bands[idx + 1 :]:
            mean_a = selected_band_means[band_a]
            mean_b = selected_band_means[band_b]
            row[f"{prefix}_band_{band_b}_minus_{band_a}_mean"] = float(mean_b - mean_a)
            row[f"{prefix}_band_{band_b}_div_{band_a}_mean"] = (
                float(mean_b / mean_a) if abs(mean_a) > 1e-8 else 0.0
            )


def extract_features_for_file(
    path: Path,
    normalise: str,
    valid_min_percentile: float,
    band_indices: List[int],
    include_full_spectrum: bool,
    mask_source: str = "annotation",
    detector_model=None,
    detector_device=None,
) -> Dict[str, object]:
    cube, annotations, source_roi, name, export_type = load_dish_npz(path, normalise=normalise)
    bands, height, width = cube.shape
    if mask_source == "detector":
        if detector_model is None or detector_device is None:
            raise ValueError("detector_model and detector_device are required when mask_source='detector'.")
        class_mask = predict_detector_mask(cube, detector_model, detector_device)
    else:
        class_mask = build_annotation_mask(annotations, height, width)
    valid_mask = valid_pixel_mask(cube, valid_min_percentile)
    valid_pixels = int(valid_mask.sum())

    row: Dict[str, object] = {
        "sample_id": Path(path).stem,
        "dish_name": name,
        "path": str(path),
        "type": export_type,
        "bands": int(bands),
        "height": int(height),
        "width": int(width),
        "valid_pixels": valid_pixels,
        "source_roi_name": source_roi.get("name", ""),
        "mask_source": mask_source,
    }

    microplastic_mask = (class_mask > 0) & valid_mask
    row["has_microplastic"] = int(np.any(microplastic_mask))
    row["total_microplastic_area_pixels"] = int(microplastic_mask.sum())
    row["total_microplastic_area_ratio"] = (
        float(microplastic_mask.sum() / valid_pixels) if valid_pixels else 0.0
    )
    count, max_area, mean_area = component_stats(microplastic_mask)
    row["total_count"] = count
    row["total_component_max_area"] = max_area
    row["total_component_mean_area"] = mean_area

    for class_id, class_name in ID_TO_CLASS.items():
        prefix = class_name.lower()
        if class_id == 0:
            region = (class_mask == 0) & valid_mask
        else:
            region = (class_mask == class_id) & valid_mask
        area = int(region.sum())
        row[f"{prefix}_area_pixels"] = area
        row[f"{prefix}_area_ratio"] = float(area / valid_pixels) if valid_pixels else 0.0
        comp_count, comp_max, comp_mean = component_stats(region if class_id > 0 else np.zeros_like(region))
        row[f"{prefix}_count"] = comp_count
        row[f"{prefix}_component_max_area"] = comp_max
        row[f"{prefix}_component_mean_area"] = comp_mean
        add_region_spectral_features(
            row,
            cube,
            region,
            prefix=prefix,
            band_indices=band_indices,
            include_full_spectrum=include_full_spectrum,
        )

    # Proxy targets allow the regression pipeline to be tested before lab-measured
    # content percentages are available.
    row["pp_proxy_percent"] = float(row["pp_area_ratio"]) * 100.0
    row["pet_proxy_percent"] = float(row["pet_area_ratio"]) * 100.0
    row["ps_proxy_percent"] = float(row["ps_area_ratio"]) * 100.0
    row["total_proxy_percent"] = float(row["total_microplastic_area_ratio"]) * 100.0

    return row


def main():
    parser = argparse.ArgumentParser(description="Extract regression features from dish-level NPZ files.")
    parser.add_argument("--data-root", default="microplastic")
    parser.add_argument("--pattern", default="**/Merged/*.npz")
    parser.add_argument("--out", default="features/content_features_v1.csv")
    parser.add_argument("--normalise", choices=["minmax", "snv", "none"], default="minmax")
    parser.add_argument("--valid-min-percentile", type=float, default=5.0)
    parser.add_argument("--bands", default="48,116,183", help="Comma-separated band indices for compact features.")
    parser.add_argument("--include-full-spectrum", action="store_true")
    parser.add_argument("--mask-source", choices=["annotation", "detector"], default="annotation")
    parser.add_argument("--detector-checkpoint", default="", help="Segmentation checkpoint used when --mask-source detector.")
    args = parser.parse_args()

    root = Path(args.data_root)
    paths = sorted(p for p in root.glob(args.pattern) if p.suffix.lower() == ".npz")
    paths = [p for p in paths if is_dish_npz(p)]
    if not paths:
        raise ValueError("No dish-level NPZ files found.")

    band_indices = parse_band_indices(args.bands)
    detector_model = None
    detector_device = None
    if args.mask_source == "detector":
        if not args.detector_checkpoint:
            raise ValueError("--detector-checkpoint is required when --mask-source detector.")
        detector_model, detector_device = load_detector(args.detector_checkpoint)

    rows = [
        extract_features_for_file(
            path,
            normalise=args.normalise,
            valid_min_percentile=args.valid_min_percentile,
            band_indices=band_indices,
            include_full_spectrum=args.include_full_spectrum,
            mask_source=args.mask_source,
            detector_model=detector_model,
            detector_device=detector_device,
        )
        for path in paths
    ]

    all_fields = []
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(int(row["has_microplastic"]) for row in rows)
    print(f"Saved features: {out_path}")
    print(f"Samples: {len(rows)}")
    print(f"Positive samples: {positives}")
    print(f"Negative samples: {len(rows) - positives}")


if __name__ == "__main__":
    main()
