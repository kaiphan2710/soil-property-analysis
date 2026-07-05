"""
Data loading for exported hyperspectral microplastic dish crops.

Expected input files are dish-level NPZ exports from the labelling app, for example:
    S1_S4_S7_S10_FX10FX17_..._S1.npz
    S1_S4_S7_S10_FX10FX17_..._S4.npz

Each NPZ should contain:
    datacube:          numpy array with shape (bands, height, width)
    annotations_json:  JSON list of target annotations already transferred into NPZ-local coordinates
    wavelengths:       optional wavelength array
    source_roi_json:   optional source dish metadata from the original HDR image

This file supports two training directions:
    1. Detection / instance segmentation:
       returns image + target dict containing boxes, labels, instance masks.

    2. Semantic segmentation:
       returns image + class_mask where every pixel is 0=background, 1=PP, 2=PET, 3=PS.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import re
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, random_split


TaskType = Literal["detection", "segmentation"]


DEFAULT_CLASS_TO_ID = {
    "PP": 1,
    "PET": 2,
    "PS": 3,
}


def detection_collate_fn(batch):
    """
    Required for detection-style models because each image can have a different
    number of objects.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def segmentation_collate_fn(batch):
    images, masks = zip(*batch)

    max_channels = max(image.shape[0] for image in images)
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)

    padded_images = []
    padded_masks = []
    for image, mask in zip(images, masks):
        channels, height, width = image.shape
        image_pad = torch.zeros((max_channels, max_height, max_width), dtype=image.dtype)
        mask_pad = torch.zeros((max_height, max_width), dtype=mask.dtype)
        image_pad[:channels, :height, :width] = image
        mask_pad[:height, :width] = mask
        padded_images.append(image_pad)
        padded_masks.append(mask_pad)

    return torch.stack(padded_images, dim=0), torch.stack(padded_masks, dim=0)


def find_npz_files(folder: str, pattern: str = "*.npz") -> List[Path]:
    return sorted(Path(folder).expanduser().glob(pattern))


def _npz_scalar_as_str(npz_file, key: str) -> str:
    if key not in npz_file.files:
        return ""
    value = npz_file[key]
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def _is_dish_npz(path: Path) -> bool:
    """
    Keep only dish-level ROI exports such as S1/S2/S10.

    This avoids accidentally training on individual target exports like PP/PET/PS
    or files whose folder/name prefix happens to contain S1/S4.
    """
    try:
        with np.load(path, allow_pickle=True) as npz_file:
            export_type = _npz_scalar_as_str(npz_file, "type").lower()
            export_name = _npz_scalar_as_str(npz_file, "name")
    except Exception:
        return False

    if export_type and export_type != "roi":
        return False

    if export_name and re.fullmatch(r"[sS]\d+", export_name.strip()):
        return True

    return re.search(r"_[sS]\d+\.npz$", path.name) is not None


def split_dataset(
    dataset: Dataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_dataloaders(
    folder: str,
    pattern: str = "*.npz",
    task: TaskType = "detection",
    batch_size: int = 2,
    val_fraction: float = 0.2,
    seed: int = 42,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    dataset = MicroplasticDishDataset.from_folder(
        folder=folder,
        pattern=pattern,
        task=task,
        **dataset_kwargs,
    )
    train_dataset, val_dataset = split_dataset(dataset, val_fraction=val_fraction, seed=seed)

    if task == "detection":
        collate_fn = detection_collate_fn
    else:
        collate_fn = segmentation_collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def _read_npz_json(npz_file, key: str, default):
    if key not in npz_file.files:
        return default

    raw_value = npz_file[key]
    if hasattr(raw_value, "item"):
        raw_value = raw_value.item()

    try:
        return json.loads(str(raw_value))
    except Exception:
        return default


def _load_sidecar_annotations(npz_path: Path) -> List[Dict]:
    sidecar_path = npz_path.with_suffix(".json")
    if not sidecar_path.exists():
        return []
    with sidecar_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _class_name_from_annotation(name: str) -> str:
    # Accept names like PP, PP_01, pet_piece_3.
    return name.strip().upper().split("_")[0]


def _normalise_cube(cube: np.ndarray, mode: str) -> np.ndarray:
    cube = cube.astype(np.float32, copy=False)

    if mode == "none":
        return cube

    if mode == "minmax":
        lo = np.percentile(cube, 1)
        hi = np.percentile(cube, 99)
        if hi <= lo:
            return np.zeros_like(cube, dtype=np.float32)
        cube = np.clip(cube, lo, hi)
        return (cube - lo) / (hi - lo)

    if mode == "snv":
        # Standard normal variate per pixel spectrum.
        mean = cube.mean(axis=0, keepdims=True)
        std = cube.std(axis=0, keepdims=True)
        return (cube - mean) / (std + 1e-6)

    raise ValueError(f"Unknown normalisation mode: {mode}")


def _rect_corners(x: float, y: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    local = np.array(
        [
            [0.0, 0.0],
            [w, 0.0],
            [w, h],
            [0.0, h],
        ],
        dtype=np.float32,
    )
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _annotation_to_box(annotation: Dict) -> Optional[List[float]]:
    x = float(annotation["x"])
    y = float(annotation["y"])
    w = float(annotation["w"])
    h = float(annotation["h"])
    angle = float(annotation.get("angle", 0.0))
    shape = annotation.get("shape_type", "Rectangle")

    if w <= 0 or h <= 0:
        return None

    if shape in ("Ellipse", "Circle"):
        return [x, y, x + w, y + h]

    corners = _rect_corners(x, y, w, h, angle)
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def _draw_annotation_mask(annotation: Dict, height: int, width: int) -> np.ndarray:
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    x = float(annotation["x"])
    y = float(annotation["y"])
    w = float(annotation["w"])
    h = float(annotation["h"])
    angle = float(annotation.get("angle", 0.0))
    shape = annotation.get("shape_type", "Rectangle")

    if shape in ("Ellipse", "Circle"):
        draw.ellipse([x, y, x + w, y + h], fill=1)
    else:
        corners = _rect_corners(x, y, w, h, angle)
        draw.polygon([tuple(p) for p in corners], fill=1)

    return np.asarray(mask_img, dtype=np.uint8)


class MicroplasticDishDataset(Dataset):
    """
    Dataset for dish-level hyperspectral NPZ files.

    Detection mode output:
        image:  FloatTensor (channels, height, width)
        target: dict with boxes, labels, masks, metadata

    Segmentation mode output:
        image:      FloatTensor (channels, height, width)
        class_mask: LongTensor (height, width)
    """

    def __init__(
        self,
        npz_paths: Iterable[str | Path],
        task: TaskType = "detection",
        class_to_id: Optional[Dict[str, int]] = None,
        band_indices: Optional[Sequence[int]] = None,
        normalise: str = "minmax",
        skip_unknown_classes: bool = True,
        require_dish_roi: bool = True,
    ):
        self.paths = [Path(p).expanduser() for p in npz_paths]
        self.paths = [p for p in self.paths if p.suffix.lower() == ".npz"]
        if require_dish_roi:
            self.paths = [p for p in self.paths if _is_dish_npz(p)]
        self.task = task
        self.class_to_id = class_to_id or DEFAULT_CLASS_TO_ID
        self.band_indices = list(band_indices) if band_indices is not None else None
        self.normalise = normalise
        self.skip_unknown_classes = skip_unknown_classes

        if task not in ("detection", "segmentation"):
            raise ValueError("task must be either 'detection' or 'segmentation'.")
        if not self.paths:
            raise ValueError("No .npz files were provided.")

    @classmethod
    def from_folder(
        cls,
        folder: str,
        pattern: str = "*.npz",
        **kwargs,
    ) -> "MicroplasticDishDataset":
        paths = find_npz_files(folder, pattern=pattern)
        return cls(paths, **kwargs)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image, annotations, source_roi = self._load_sample(path)

        if self.task == "segmentation":
            return image, self._build_class_mask(image, annotations)

        return image, self._build_detection_target(index, path, image, annotations, source_roi)

    def _load_sample(self, path: Path):
        with np.load(path, allow_pickle=True) as npz_file:
            cube = npz_file["datacube"]
            annotations = _read_npz_json(npz_file, "annotations_json", None)
            source_roi = _read_npz_json(npz_file, "source_roi_json", {})

        if annotations is None:
            annotations = _load_sidecar_annotations(path)

        if self.band_indices is not None:
            cube = cube[self.band_indices, :, :]

        cube = _normalise_cube(cube, self.normalise)
        image = torch.from_numpy(np.ascontiguousarray(cube)).float()
        return image, annotations, source_roi

    def _valid_annotations(self, annotations: List[Dict]):
        for ann in annotations:
            class_name = _class_name_from_annotation(ann.get("name", ""))
            label = self.class_to_id.get(class_name)
            if label is None:
                if self.skip_unknown_classes:
                    continue
                label = 0
            yield ann, label

    def _build_detection_target(
        self,
        index: int,
        path: Path,
        image: torch.Tensor,
        annotations: List[Dict],
        source_roi: Dict,
    ) -> Dict:
        _, height, width = image.shape
        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        kept_annotations: List[Dict] = []

        for ann, label in self._valid_annotations(annotations):
            box = _annotation_to_box(ann)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            masks.append(_draw_annotation_mask(ann, height, width))
            kept_annotations.append(ann)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([index], dtype=torch.long),
            "masks": (
                torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
                if masks
                else torch.zeros((0, height, width), dtype=torch.uint8)
            ),
            "path": str(path),
            "annotations": kept_annotations,
            "source_roi": source_roi,
        }
        return target

    def _build_class_mask(self, image: torch.Tensor, annotations: List[Dict]) -> torch.Tensor:
        _, height, width = image.shape
        class_mask = torch.zeros((height, width), dtype=torch.long)

        for ann, label in self._valid_annotations(annotations):
            instance_mask = _draw_annotation_mask(ann, height, width)
            class_mask[torch.from_numpy(instance_mask.copy()).bool()] = int(label)

        return class_mask


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Inspect microplastic dish NPZ data loading.")
    parser.add_argument("folder", help="Folder containing exported dish .npz files")
    parser.add_argument("--pattern", default="**/Merged/*.npz", help="Glob pattern for exported NPZ files")
    parser.add_argument("--task", choices=["detection", "segmentation"], default="detection")
    parser.add_argument("--normalise", choices=["minmax", "snv", "none"], default="minmax")
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Use only approximate RGB bands [183, 116, 48] instead of all hyperspectral bands",
    )
    args = parser.parse_args()

    _set_seed(42)
    band_indices = [183, 116, 48] if args.rgb_only else None
    dataset = MicroplasticDishDataset.from_folder(
        args.folder,
        pattern=args.pattern,
        task=args.task,
        normalise=args.normalise,
        band_indices=band_indices,
    )

    print(f"Loaded {len(dataset)} samples")
    sample = dataset[0]

    if args.task == "detection":
        image, target = sample
        print("image shape:", tuple(image.shape))
        print("boxes shape:", tuple(target["boxes"].shape))
        print("labels:", target["labels"].tolist())
        print("masks shape:", tuple(target["masks"].shape))
        print("path:", target["path"])
    else:
        image, class_mask = sample
        print("image shape:", tuple(image.shape))
        print("class mask shape:", tuple(class_mask.shape))
        print("classes in mask:", torch.unique(class_mask).tolist())


if __name__ == "__main__":
    main()
