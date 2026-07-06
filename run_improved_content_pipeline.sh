#!/usr/bin/env bash
set -euo pipefail

# Run from project root:
#   cd /home/sae2026/soil_scan_project/soil-property-analysis
#   bash run_improved_content_pipeline.sh

mkdir -p features checkpoints predictions logs

echo "1) Extract annotation-based content features"
python extract_content_features.py \
  --data-root microplastic \
  --pattern "**/Merged/*.npz" \
  --out features/content_features_v2_annotation.csv \
  --normalise minmax \
  --mask-source annotation

echo "2) Train improved random-forest content regressor with leave-one-out validation"
python train_content_regressor.py \
  --features features/content_features_v2_annotation.csv \
  --targets content_targets_real_mixed_v1.csv \
  --target-mode real \
  --model rf \
  --feature-set compact \
  --target-transform log1p \
  --class-specific \
  --loo \
  --out checkpoints/content_regressor_rf_real_v2.joblib \
  2>&1 | tee logs/train_content_regressor_rf_real_v2.log

echo "3) Predict readable percentages from annotation masks"
python predict_microplastic_content.py \
  --model checkpoints/content_regressor_rf_real_v2.joblib \
  --data-root microplastic/S1_S4_S7_S10_2026_06_30_14_54_02 \
  --pattern "Merged/*.npz" \
  --out predictions/content_predictions_rf_real_v2.csv \
  --readable-out predictions/content_predictions_rf_real_v2_readable.csv \
  --zero-threshold 0.5 \
  --mask-source annotation

echo "4) Optional: predict readable percentages through the old detection network"
echo "   Set DETECTOR_CHECKPOINT to your segmentation checkpoint path, then rerun this block."
if [[ -n "${DETECTOR_CHECKPOINT:-}" ]]; then
  python predict_microplastic_content.py \
    --model checkpoints/content_regressor_rf_real_v2.joblib \
    --data-root microplastic/S1_S4_S7_S10_2026_06_30_14_54_02 \
    --pattern "Merged/*.npz" \
    --out predictions/content_predictions_rf_real_v2_detector.csv \
    --readable-out predictions/content_predictions_rf_real_v2_detector_readable.csv \
    --zero-threshold 0.5 \
    --mask-source detector \
    --detector-checkpoint "$DETECTOR_CHECKPOINT"
fi

echo "Done."
