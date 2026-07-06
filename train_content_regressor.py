"""
Train a sample-level regression model for microplastic content estimation.

The model maps extracted dish features to continuous content estimates:
    features -> [PP%, PET%, PS%, total%]

If real lab-measured percentages are not available yet, use:
    --target-mode proxy-area
to train against annotation-derived area-ratio proxy targets. That is useful for
pipeline validation, but it is not a substitute for true content labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


META_COLUMNS = {
    "sample_id",
    "dish_name",
    "path",
    "type",
    "source_roi_name",
    "mask_source",
}

PROXY_TARGETS = ["pp_proxy_percent", "pet_proxy_percent", "ps_proxy_percent", "total_proxy_percent"]
REAL_TARGETS = ["pp_percent", "pet_percent", "ps_percent", "total_percent"]
CLASS_TARGET_PREFIXES = ("pp", "pet", "ps")


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def merge_targets(feature_rows: List[Dict[str, str]], target_csv: str) -> List[Dict[str, str]]:
    target_rows = read_csv_rows(target_csv)
    by_id = {row["sample_id"]: row for row in target_rows}
    merged = []
    for row in feature_rows:
        sample_id = row["sample_id"]
        if sample_id not in by_id:
            continue
        combined = dict(row)
        combined.update(by_id[sample_id])
        merged.append(combined)
    if not merged:
        raise ValueError("No feature rows matched target CSV by sample_id.")
    return merged


def is_compact_feature(name: str) -> bool:
    compact_tokens = (
        "area_ratio",
        "_count",
        "component_",
        "spectral_mean",
        "spectral_std",
        "pixel_spectral_std_mean",
        "_minus_",
        "_div_",
        "valid_pixels",
        "bands",
        "height",
        "width",
        "has_microplastic",
    )
    return any(token in name for token in compact_tokens)


def build_arrays(rows: List[Dict[str, str]], target_columns: List[str], feature_set: str = "all"):
    excluded = set(META_COLUMNS) | set(PROXY_TARGETS) | set(REAL_TARGETS)
    candidate_columns = [key for key in rows[0].keys() if key not in excluded]
    if feature_set == "compact":
        candidate_columns = [col for col in candidate_columns if is_compact_feature(col)]
    feature_columns = []
    for col in candidate_columns:
        values = [parse_float(row.get(col, "")) for row in rows]
        if any(math.isfinite(v) and v != 0.0 for v in values):
            feature_columns.append(col)

    X = np.asarray([[parse_float(row.get(col, "")) for col in feature_columns] for row in rows], dtype=np.float32)
    y = np.asarray([[parse_float(row.get(col, "")) for col in target_columns] for row in rows], dtype=np.float32)
    return X, y, feature_columns


def build_model(model_name: str, seed: int):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def target_transform(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "log1p":
        return np.log1p(np.clip(y, 0.0, None))
    raise ValueError(f"Unknown target transform: {mode}")


def inverse_target_transform(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "log1p":
        return np.expm1(y)
    raise ValueError(f"Unknown target transform: {mode}")


def force_total_from_classes(y_pred: np.ndarray, target_columns: List[str]) -> np.ndarray:
    pred = np.array(y_pred, dtype=np.float32, copy=True)
    lower_cols = [col.lower() for col in target_columns]
    try:
        pp_idx = lower_cols.index(next(col for col in lower_cols if col.startswith("pp_")))
        pet_idx = lower_cols.index(next(col for col in lower_cols if col.startswith("pet_")))
        ps_idx = lower_cols.index(next(col for col in lower_cols if col.startswith("ps_")))
    except StopIteration:
        return pred

    total_idx = None
    for idx, col in enumerate(lower_cols):
        if col.startswith("total_"):
            total_idx = idx
            break
    if total_idx is not None:
        pred[:, total_idx] = pred[:, pp_idx] + pred[:, pet_idx] + pred[:, ps_idx]
    return pred


def fit_area_baseline(
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: List[str],
    target_columns: List[str],
) -> Dict[str, object]:
    coeffs: Dict[str, float] = {}
    for prefix, target_name in zip(CLASS_TARGET_PREFIXES, target_columns[:3]):
        feature_name = f"{prefix}_area_ratio"
        if feature_name not in feature_columns:
            coeffs[target_name] = 0.0
            continue
        feature_idx = feature_columns.index(feature_name)
        target_idx = target_columns.index(target_name)
        area = X[:, feature_idx]
        target = y[:, target_idx]
        denom = float(np.dot(area, area))
        coeffs[target_name] = float(np.dot(area, target) / denom) if denom > 1e-12 else 0.0
    return {"area_coefficients": coeffs}


def predict_area_baseline(
    area_payload: Dict[str, object],
    X: np.ndarray,
    feature_columns: List[str],
    target_columns: List[str],
) -> np.ndarray:
    coeffs = area_payload.get("area_coefficients", {})
    pred = np.zeros((X.shape[0], len(target_columns)), dtype=np.float32)
    for prefix, target_name in zip(CLASS_TARGET_PREFIXES, target_columns[:3]):
        feature_name = f"{prefix}_area_ratio"
        if feature_name not in feature_columns:
            continue
        feature_idx = feature_columns.index(feature_name)
        target_idx = target_columns.index(target_name)
        pred[:, target_idx] = X[:, feature_idx] * float(coeffs.get(target_name, 0.0))
    return force_total_from_classes(pred, target_columns)


def fit_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    target_columns: List[str],
    feature_columns: List[str],
    class_specific: bool,
):
    if model_name == "area":
        return fit_area_baseline(X, y, feature_columns, target_columns)
    if class_specific:
        return [build_model(model_name, seed + idx).fit(X, y[:, idx]) for idx in range(y.shape[1])]
    model = build_model(model_name, seed)
    model.fit(X, y)
    return model


def predict_model(
    model,
    model_name: str,
    X: np.ndarray,
    target_columns: List[str],
    feature_columns: List[str],
    class_specific: bool,
) -> np.ndarray:
    if model_name == "area":
        return predict_area_baseline(model, X, feature_columns, target_columns)
    if class_specific:
        columns = [single_model.predict(X) for single_model in model]
        return np.stack(columns, axis=1).astype(np.float32)
    return np.asarray(model.predict(X), dtype=np.float32)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, target_columns: List[str]) -> Dict[str, object]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics: Dict[str, object] = {}
    metrics["mae_mean"] = float(mean_absolute_error(y_true, y_pred))
    metrics["rmse_mean"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        metrics["r2_mean"] = float(r2_score(y_true, y_pred))
    except Exception:
        metrics["r2_mean"] = None

    per_target = {}
    for idx, name in enumerate(target_columns):
        per_target[name] = {
            "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))),
        }
    metrics["per_target"] = per_target
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train microplastic content regressor.")
    parser.add_argument("--features", default="features/content_features_v1.csv")
    parser.add_argument("--targets", default="", help="Optional CSV with sample_id and real percentage targets.")
    parser.add_argument("--target-mode", choices=["real", "proxy-area"], default="proxy-area")
    parser.add_argument("--model", choices=["rf", "mlp", "area"], default="rf")
    parser.add_argument("--feature-set", choices=["all", "compact"], default="all")
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="none")
    parser.add_argument("--class-specific", action="store_true", help="Train one regressor per target.")
    parser.add_argument("--loo", action="store_true", help="Use leave-one-out validation before fitting final model.")
    parser.add_argument("--force-total-sum", action="store_true", default=True)
    parser.add_argument("--out", default="checkpoints/content_regressor_v1.joblib")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import joblib
    from sklearn.model_selection import LeaveOneOut, train_test_split

    rows = read_csv_rows(args.features)
    if args.target_mode == "real":
        if not args.targets:
            raise ValueError("--targets is required when --target-mode real.")
        rows = merge_targets(rows, args.targets)
        target_columns = REAL_TARGETS
    else:
        target_columns = PROXY_TARGETS

    X, y, feature_columns = build_arrays(rows, target_columns, feature_set=args.feature_set)
    if len(rows) < 2:
        raise ValueError("Need at least two samples to train a regressor.")

    y_fit = target_transform(y, args.target_transform)

    metrics = {}
    if args.loo and len(rows) >= 3:
        loo = LeaveOneOut()
        predictions = np.zeros_like(y, dtype=np.float32)
        for fold, (train_idx, val_idx) in enumerate(loo.split(X)):
            fold_model = fit_model(
                args.model,
                X[train_idx],
                y_fit[train_idx],
                seed=args.seed + fold,
                target_columns=target_columns,
                feature_columns=feature_columns,
                class_specific=args.class_specific,
            )
            pred_fit = predict_model(
                fold_model,
                args.model,
                X[val_idx],
                target_columns=target_columns,
                feature_columns=feature_columns,
                class_specific=args.class_specific,
            )
            pred = inverse_target_transform(pred_fit, args.target_transform)
            pred = np.clip(pred, 0.0, None)
            if args.force_total_sum:
                pred = force_total_from_classes(pred, target_columns)
            predictions[val_idx] = pred
        metrics = evaluate(y, predictions, target_columns)
        train_samples = len(X)
        val_samples = len(X)
    elif args.test_size > 0 and len(rows) >= 4:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_fit,
            test_size=args.test_size,
            random_state=args.seed,
        )
        model = fit_model(
            args.model,
            X_train,
            y_train,
            seed=args.seed,
            target_columns=target_columns,
            feature_columns=feature_columns,
            class_specific=args.class_specific,
        )
        pred_fit = predict_model(
            model,
            args.model,
            X_val,
            target_columns=target_columns,
            feature_columns=feature_columns,
            class_specific=args.class_specific,
        )
        pred = inverse_target_transform(pred_fit, args.target_transform)
        pred = np.clip(pred, 0.0, None)
        if args.force_total_sum:
            pred = force_total_from_classes(pred, target_columns)
        metrics = evaluate(inverse_target_transform(y_val, args.target_transform), pred, target_columns)
        train_samples = len(X_train)
        val_samples = len(X_val)
    else:
        model = fit_model(
            args.model,
            X,
            y_fit,
            seed=args.seed,
            target_columns=target_columns,
            feature_columns=feature_columns,
            class_specific=args.class_specific,
        )
        pred_fit = predict_model(
            model,
            args.model,
            X,
            target_columns=target_columns,
            feature_columns=feature_columns,
            class_specific=args.class_specific,
        )
        pred = inverse_target_transform(pred_fit, args.target_transform)
        pred = np.clip(pred, 0.0, None)
        if args.force_total_sum:
            pred = force_total_from_classes(pred, target_columns)
        metrics = evaluate(y, pred, target_columns)
        train_samples = len(X)
        val_samples = 0

    final_model = fit_model(
        args.model,
        X,
        y_fit,
        seed=args.seed,
        target_columns=target_columns,
        feature_columns=feature_columns,
        class_specific=args.class_specific,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_model,
        "model_type": args.model,
        "target_mode": args.target_mode,
        "target_transform": args.target_transform,
        "feature_set": args.feature_set,
        "class_specific": bool(args.class_specific),
        "force_total_sum": bool(args.force_total_sum),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "features_csv": args.features,
        "targets_csv": args.targets,
    }
    joblib.dump(payload, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "target_mode": args.target_mode,
                "target_transform": args.target_transform,
                "feature_set": args.feature_set,
                "class_specific": bool(args.class_specific),
                "validation": "leave_one_out" if args.loo else ("holdout" if val_samples else "train_fit"),
                "samples": len(rows),
                "train_samples": train_samples,
                "val_samples": val_samples,
                "num_features": len(feature_columns),
                "target_columns": target_columns,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Samples: {len(rows)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Train samples: {train_samples}")
    print(f"Val samples: {val_samples}")
    print(json.dumps(metrics, indent=2))
    print(f"Saved regressor: {out_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
