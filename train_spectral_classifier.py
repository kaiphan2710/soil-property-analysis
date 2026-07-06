"""
Train a 1D spectral classifier for hyperspectral microplastic pixels.

Input is produced by build_spectral_dataset.py:
    X: (num_samples, 633)
    y: 0=soil, 1=PP, 2=PET, 3=PS

This is a baseline for the supervisor's suggested 1D/regression-style direction:
the model learns material signatures from spectra rather than spatial 2D shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _balanced_subsample(X: np.ndarray, y: np.ndarray, max_per_class: int, seed: int):
    if max_per_class <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    keep_indices = []
    for class_id in sorted(np.unique(y)):
        indices = np.flatnonzero(y == class_id)
        if len(indices) > max_per_class:
            indices = rng.choice(indices, size=max_per_class, replace=False)
        keep_indices.append(indices)

    keep = np.concatenate(keep_indices)
    rng.shuffle(keep)
    return X[keep], y[keep]


def _build_model(model_name: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    if model_name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=512,
                        learning_rate_init=1e-3,
                        max_iter=250,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train 1D spectral microplastic classifier.")
    parser.add_argument("--dataset", default="spectral_datasets/microplastic_spectra.npz")
    parser.add_argument("--model", choices=["rf", "mlp"], default="rf")
    parser.add_argument("--out", default="checkpoints/spectral_classifier.joblib")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=20000,
        help="Balance memory/time by keeping at most this many spectra per class. Use 0 for all.",
    )
    args = parser.parse_args()

    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"].astype(np.float32, copy=False)
    y = data["y"].astype(np.int64, copy=False)
    class_names = [str(x) for x in data["class_names"].tolist()]

    X, y = _balanced_subsample(X, y, max_per_class=args.max_per_class, seed=args.seed)

    stratify = y if min(np.bincount(y, minlength=len(class_names))) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    model = _build_model(args.model, seed=args.seed)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    report = classification_report(
        y_val,
        pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_val, pred, labels=list(range(len(class_names))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "model_type": args.model,
        "class_names": class_names,
        "dataset": args.dataset,
    }
    joblib.dump(payload, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "dataset": args.dataset,
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
                "class_names": class_names,
                "class_counts": {class_names[i]: int((y == i).sum()) for i in range(len(class_names))},
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Loaded spectra: {X.shape}")
    print(f"Train samples: {len(y_train)}")
    print(f"Val samples: {len(y_val)}")
    print(report)
    print("Confusion matrix:")
    print(matrix)
    print(f"Saved model: {out_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
