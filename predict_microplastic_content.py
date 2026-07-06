"""
Predict microplastic content percentages for one or more dish-level NPZ files.

This script extracts the same sample-level features as extract_content_features.py
and feeds them to a trained regressor from train_content_regressor.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from extract_content_features import extract_features_for_file, load_detector, parse_band_indices
from spectral_utils import is_dish_npz
from train_content_regressor import (
    force_total_from_classes,
    inverse_target_transform,
    predict_model,
)


def build_feature_vector(row: dict, feature_columns: List[str]) -> np.ndarray:
    values = []
    for col in feature_columns:
        try:
            values.append(float(row.get(col, 0.0)))
        except Exception:
            values.append(0.0)
    return np.asarray([values], dtype=np.float32)


def target_indices(target_columns: List[str]) -> Dict[str, int]:
    mapping = {}
    for idx, name in enumerate(target_columns):
        lower = name.lower()
        if lower.startswith("pp_"):
            mapping["PP"] = idx
        elif lower.startswith("pet_"):
            mapping["PET"] = idx
        elif lower.startswith("ps_"):
            mapping["PS"] = idx
        elif lower.startswith("total_"):
            mapping["total"] = idx
    return mapping


def normalise_dish_name(value: object) -> str:
    text = str(value or "").strip()
    return text.upper() if text else ""


def estimate_uncertainty(model, model_type: str, X: np.ndarray, class_specific: bool) -> np.ndarray | None:
    if model_type != "rf":
        return None
    try:
        if class_specific:
            stds = []
            for single_model in model:
                tree_preds = np.stack([tree.predict(X) for tree in single_model.estimators_], axis=0)
                stds.append(tree_preds.std(axis=0))
            return np.stack(stds, axis=1)[0]
        tree_preds = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)
        return tree_preds.std(axis=0)[0]
    except Exception:
        return None


def postprocess_prediction(
    pred: np.ndarray,
    target_columns: List[str],
    zero_threshold: float,
    clip_nonnegative: bool,
    force_total_sum: bool,
) -> np.ndarray:
    values = np.asarray(pred, dtype=np.float32).reshape(1, -1)
    if clip_nonnegative:
        values = np.clip(values, 0.0, None)
    if force_total_sum:
        values = force_total_from_classes(values, target_columns)

    idx = target_indices(target_columns)
    total_idx = idx.get("total")
    class_idxs = [idx[key] for key in ("PP", "PET", "PS") if key in idx]
    total_value = values[0, total_idx] if total_idx is not None else float(values[0, class_idxs].sum())
    if zero_threshold > 0 and total_value < zero_threshold:
        for class_idx in class_idxs:
            values[0, class_idx] = 0.0
        if total_idx is not None:
            values[0, total_idx] = 0.0
    elif force_total_sum and total_idx is not None:
        values = force_total_from_classes(values, target_columns)
    return values[0]


def readable_row(output_row: Dict[str, object], target_columns: List[str], pred: np.ndarray) -> Dict[str, object]:
    idx = target_indices(target_columns)
    pp = float(pred[idx["PP"]]) if "PP" in idx else 0.0
    pet = float(pred[idx["PET"]]) if "PET" in idx else 0.0
    ps = float(pred[idx["PS"]]) if "PS" in idx else 0.0
    total = float(pred[idx["total"]]) if "total" in idx else pp + pet + ps
    return {
        "dish": normalise_dish_name(output_row.get("dish_name")),
        "PP_percent": round(pp, 6),
        "PET_percent": round(pet, 6),
        "PS_percent": round(ps, 6),
        "total_microplastic_percent": round(total, 6),
        "sample_id": output_row.get("sample_id", ""),
        "path": output_row.get("path", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict microplastic content for NPZ dish files.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--npz", default="", help="Single dish-level NPZ file.")
    parser.add_argument("--data-root", default="", help="Optional root for batch prediction.")
    parser.add_argument("--pattern", default="**/Merged/*.npz")
    parser.add_argument("--out", default="predictions/content_predictions.csv")
    parser.add_argument("--readable-out", default="", help="Optional compact CSV with final percentage columns.")
    parser.add_argument("--zero-threshold", type=float, default=0.0, help="Set all class percentages to 0 if total is below this percent.")
    parser.add_argument("--allow-negative", action="store_true", help="Do not clip negative predictions to 0.")
    parser.add_argument("--normalise", choices=["minmax", "snv", "none"], default="minmax")
    parser.add_argument("--valid-min-percentile", type=float, default=5.0)
    parser.add_argument("--bands", default="48,116,183")
    parser.add_argument("--include-full-spectrum", action="store_true")
    parser.add_argument("--mask-source", choices=["annotation", "detector"], default="annotation")
    parser.add_argument("--detector-checkpoint", default="", help="Segmentation checkpoint used when --mask-source detector.")
    args = parser.parse_args()

    import joblib

    payload = joblib.load(args.model)
    model = payload["model"]
    model_type = payload.get("model_type", "rf")
    target_transform = payload.get("target_transform", "none")
    class_specific = bool(payload.get("class_specific", False))
    force_total_sum = bool(payload.get("force_total_sum", True))
    feature_columns = payload["feature_columns"]
    target_columns = payload["target_columns"]

    if args.npz:
        paths = [Path(args.npz)]
    elif args.data_root:
        root = Path(args.data_root)
        paths = sorted(p for p in root.glob(args.pattern) if p.suffix.lower() == ".npz")
        paths = [p for p in paths if is_dish_npz(p)]
    else:
        raise ValueError("Provide either --npz or --data-root.")

    band_indices = parse_band_indices(args.bands)
    detector_model = None
    detector_device = None
    if args.mask_source == "detector":
        if not args.detector_checkpoint:
            raise ValueError("--detector-checkpoint is required when --mask-source detector.")
        detector_model, detector_device = load_detector(args.detector_checkpoint)

    rows = []
    readable_rows = []
    for path in paths:
        row = extract_features_for_file(
            path,
            normalise=args.normalise,
            valid_min_percentile=args.valid_min_percentile,
            band_indices=band_indices,
            include_full_spectrum=args.include_full_spectrum,
            mask_source=args.mask_source,
            detector_model=detector_model,
            detector_device=detector_device,
        )
        X = build_feature_vector(row, feature_columns)
        pred_fit = predict_model(
            model,
            model_type,
            X,
            target_columns=target_columns,
            feature_columns=feature_columns,
            class_specific=class_specific,
        )[0]
        pred = inverse_target_transform(np.asarray(pred_fit).reshape(1, -1), target_transform)[0]
        pred = postprocess_prediction(
            pred,
            target_columns=target_columns,
            zero_threshold=args.zero_threshold,
            clip_nonnegative=not args.allow_negative,
            force_total_sum=force_total_sum,
        )
        uncertainty = estimate_uncertainty(model, model_type, X, class_specific)
        output_row = {
            "sample_id": row["sample_id"],
            "dish_name": row["dish_name"],
            "path": row["path"],
        }
        for name, value in zip(target_columns, pred):
            output_row[f"pred_{name}"] = float(value)
        if uncertainty is not None:
            for name, value in zip(target_columns, uncertainty):
                output_row[f"std_{name}"] = float(value)
        rows.append(output_row)
        readable_rows.append(readable_row(output_row, target_columns, pred))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved content predictions: {out_path}")
    if args.readable_out:
        readable_path = Path(args.readable_out)
        readable_path.parent.mkdir(parents=True, exist_ok=True)
        readable_fields = ["dish", "PP_percent", "PET_percent", "PS_percent", "total_microplastic_percent", "sample_id", "path"]
        with readable_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=readable_fields)
            writer.writeheader()
            writer.writerows(readable_rows)
        print(f"Saved readable percentages: {readable_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
