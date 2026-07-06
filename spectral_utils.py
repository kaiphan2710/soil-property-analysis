"""Small utilities shared by the 1D spectral microplastic scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_CLASS_TO_ID = {
    "PP": 1,
    "PET": 2,
    "PS": 3,
}


PALETTE = {
    0: (0, 0, 0),        # soil/background
    1: (255, 150, 0),    # PP
    2: (255, 0, 0),      # PET
    3: (255, 0, 180),    # PS
}


def npz_scalar_as_str(npz_file, key: str) -> str:
    if key not in npz_file.files:
        return ""
    value = npz_file[key]
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


def is_dish_npz(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as npz_file:
            export_type = npz_scalar_as_str(npz_file, "type").lower()
            export_name = npz_scalar_as_str(npz_file, "name")
    except Exception:
        return False

    if export_type and export_type != "roi":
        return False

    if export_name and re.fullmatch(r"[sS]\d+", export_name.strip()):
        return True

    return re.search(r"_[sS]\d+\.npz$", path.name) is not None


def read_npz_json(npz_file, key: str, default):
    if key not in npz_file.files:
        return default
    raw_value = npz_file[key]
    if hasattr(raw_value, "item"):
        raw_value = raw_value.item()
    try:
        return json.loads(str(raw_value))
    except Exception:
        return default


def load_sidecar_annotations(npz_path: Path):
    sidecar_path = npz_path.with_suffix(".json")
    if not sidecar_path.exists():
        return []
    with sidecar_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def class_name_from_annotation(name: str) -> str:
    return name.strip().upper().split("_")[0]


def normalise_cube(cube: np.ndarray, mode: str) -> np.ndarray:
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
        mean = cube.mean(axis=0, keepdims=True)
        std = cube.std(axis=0, keepdims=True)
        return (cube - mean) / (std + 1e-6)

    raise ValueError(f"Unknown normalisation mode: {mode}")


def rect_corners(x: float, y: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    local = np.array(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]],
        dtype=np.float32,
    )
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def draw_annotation_mask(annotation: Dict, height: int, width: int) -> np.ndarray:
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
        corners = rect_corners(x, y, w, h, angle)
        draw.polygon([tuple(p) for p in corners], fill=1)

    return np.asarray(mask_img, dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color
    return Image.fromarray(rgb)
