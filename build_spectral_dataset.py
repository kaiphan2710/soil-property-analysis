"""
Build a 1D spectral dataset from dish-level hyperspectral NPZ exports.

The 2D segmentation pipeline trains on whole dish images. This script creates
the alternative spectral-learning view:

    one pixel spectrum with 633 bands -> material label

Output NPZ fields:
    X: float32 array, shape (num_samples, num_bands)
    y: int64 array, shape (num_samples,)
    class_names: object array, index-aligned with labels
    source_path: object array, source NPZ for every sampled spectrum
    pixel_yx: int32 array, shape (num_samples, 2), pixel coordinate in source dish
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from spectral_utils import (
    DEFAULT_CLASS_TO_ID,
    class_name_from_annotation,
    draw_annotation_mask,
    is_dish_npz,
    load_sidecar_annotations,
    normalise_cube,
    read_npz_json,
)


ID_TO_CLASS = {
    0: "soil",
    1: "PP",
    2: "PET",
    3: "PS",
}


def _load_cube_and_annotations(path: Path, normalise: str):
    with np.load(path, allow_pickle=True) as npz_file:
        cube = npz_file["datacube"]
        annotations = read_npz_json(npz_file, "annotations_json", None)

    if annotations is None:
        annotations = load_sidecar_annotations(path)

    cube = normalise_cube(cube, normalise)
    return cube, annotations


def _valid_pixel_mask(cube: np.ndarray, min_percentile: float) -> np.ndarray:
    """
    Remove the padded black outside-dish area from soil sampling.

    Dish crops are rectangular, but the actual Petri dish is circular and the
    area outside it is usually black. A low percentile threshold on mean
    intensity is a simple, data-format-independent way to avoid sampling that
    black border as soil.
    """
    mean_image = cube.mean(axis=0)
    threshold = np.percentile(mean_image, min_percentile)
    return mean_image > threshold


def _build_class_mask(
    annotations: List[Dict],
    height: int,
    width: int,
    class_to_id: Dict[str, int],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.int64)

    for ann in annotations:
        class_name = class_name_from_annotation(ann.get("name", ""))
        label = class_to_id.get(class_name)
        if label is None:
            continue
        instance_mask = draw_annotation_mask(ann, height, width).astype(bool)
        mask[instance_mask] = int(label)

    return mask


def _sample_indices(mask: np.ndarray, max_count: int, rng: random.Random) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return coords
    if max_count <= 0 or len(coords) <= max_count:
        return coords
    keep = rng.sample(range(len(coords)), max_count)
    return coords[np.asarray(keep, dtype=np.int64)]


def build_spectral_dataset(
    data_root: str,
    pattern: str,
    output: str,
    normalise: str,
    seed: int,
    max_soil_per_file: int,
    max_plastic_per_class_per_file: int,
    valid_min_percentile: float,
):
    rng = random.Random(seed)
    root = Path(data_root)
    paths = sorted(p for p in root.glob(pattern) if p.suffix.lower() == ".npz")
    paths = [p for p in paths if is_dish_npz(p)]

    spectra_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    source_paths: List[str] = []
    pixel_coords: List[Tuple[int, int]] = []
    per_class_counts = {class_id: 0 for class_id in ID_TO_CLASS}
    positive_files = 0
    negative_files = 0

    for path in paths:
        cube, annotations = _load_cube_and_annotations(path, normalise=normalise)
        bands, height, width = cube.shape
        class_mask = _build_class_mask(annotations, height, width, DEFAULT_CLASS_TO_ID)
        valid_mask = _valid_pixel_mask(cube, min_percentile=valid_min_percentile)
        has_plastic = bool(np.any(class_mask > 0))
        positive_files += int(has_plastic)
        negative_files += int(not has_plastic)

        for class_id in sorted(ID_TO_CLASS):
            if class_id == 0:
                candidate_mask = (class_mask == 0) & valid_mask
                max_count = max_soil_per_file
            else:
                candidate_mask = class_mask == class_id
                max_count = max_plastic_per_class_per_file

            coords = _sample_indices(candidate_mask, max_count=max_count, rng=rng)
            if len(coords) == 0:
                continue

            yy = coords[:, 0]
            xx = coords[:, 1]
            spectra = cube[:, yy, xx].T.astype(np.float32, copy=False)
            labels = np.full((len(coords),), class_id, dtype=np.int64)

            spectra_batches.append(spectra)
            label_batches.append(labels)
            per_class_counts[class_id] += int(len(coords))
            source_paths.extend([str(path)] * len(coords))
            pixel_coords.extend((int(y), int(x)) for y, x in coords)

    if not spectra_batches:
        raise ValueError("No spectra were extracted. Check data_root/pattern and annotations.")

    X = np.concatenate(spectra_batches, axis=0)
    y = np.concatenate(label_batches, axis=0)
    source_path_arr = np.asarray(source_paths, dtype=object)
    pixel_yx = np.asarray(pixel_coords, dtype=np.int32)
    class_names = np.asarray([ID_TO_CLASS[i] for i in range(len(ID_TO_CLASS))], dtype=object)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        class_names=class_names,
        source_path=source_path_arr,
        pixel_yx=pixel_yx,
        data_root=str(root),
        pattern=pattern,
        normalise=normalise,
        valid_min_percentile=float(valid_min_percentile),
    )

    print(f"Dish NPZ files: {len(paths)}")
    print(f"Positive dish files: {positive_files}")
    print(f"Negative dish files: {negative_files}")
    print(f"Saved spectral dataset: {output_path}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class counts:")
    for class_id, count in per_class_counts.items():
        print(f"  {class_id} {ID_TO_CLASS[class_id]}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Extract 1D spectral samples from dish NPZ files.")
    parser.add_argument("--data-root", default="microplastic")
    parser.add_argument("--pattern", default="**/Merged/*.npz")
    parser.add_argument("--out", default="spectral_datasets/microplastic_spectra.npz")
    parser.add_argument("--normalise", choices=["minmax", "snv", "none"], default="minmax")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-soil-per-file", type=int, default=3000)
    parser.add_argument("--max-plastic-per-class-per-file", type=int, default=10000)
    parser.add_argument(
        "--valid-min-percentile",
        type=float,
        default=5.0,
        help="Pixels below this mean-intensity percentile are ignored as outside-dish background.",
    )
    args = parser.parse_args()

    build_spectral_dataset(
        data_root=args.data_root,
        pattern=args.pattern,
        output=args.out,
        normalise=args.normalise,
        seed=args.seed,
        max_soil_per_file=args.max_soil_per_file,
        max_plastic_per_class_per_file=args.max_plastic_per_class_per_file,
        valid_min_percentile=args.valid_min_percentile,
    )


if __name__ == "__main__":
    main()
