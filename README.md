# Motorcycle Vibration → Subjective Rating Prediction (Hierarchical Physical LSTM)

Predicts subjective vibration ratings (1–10 scale, in 0.25 steps) for motorcycle touchpoints from HIP accelerometer data, using full RPM-wise vibration curves fed into a hierarchical LSTM — no hand-engineered summary features, no auxiliary scalar inputs.

This is the full-curve, no-mirror variant of the pipeline: it consumes raw RPM-vs-acceleration curves directly (rather than mean/max/RMS/P95 summaries) and removes the Mirror position entirely, since mirror features were found to be redundant.

## What it does

- Loads HIP sensor workbooks across three engine classes (125–160cc, 200–250cc, 350+cc) and a compiled subjective ratings workbook.
- Maps each subjective touchpoint to its domain-expert-specified sensor(s):

| Position | Sensor(s) |
|---|---|
| Handlebar | HG |
| Rider step | RFP LH, RFP RH |
| Rider Seat | Rider seat foam |
| Petrol tank | Thigh LH, Thigh RH |
| Pillion Seat | Pillion seat frame |
| Pillion step | PFP LH, PFP RH |
| Grab handle | Grab |

- Builds one training sample per **vehicle × position × RPM band × rating**, where each sample is a full resampled RPM curve (not a summary statistic) across every sensor/order/direction channel (1st order, 2nd order, 0–400 Hz broadband; lateral, longitudinal, vertical, and resultant).
- Feeds each sample through a **hierarchical LSTM**:
  1. Each physically coherent *sensor–order* group (e.g. `thigh_lh__order_1`) is processed as an RPM sequence by its own LSTM branch.
  2. Branch outputs for a sensor are combined into a sensor-level representation.
  3. Sensor-level representations are concatenated and passed through a head MLP to produce the final rating.
- Trains one independent model per subjective position.
- Validates with **leave-one-rated-vehicle-out (LOVO)** cross-validation, evaluated separately per position (and reported by engine class too).
- Reports raw, clipped (1–10), and nearest-0.25-rounded prediction errors (MAE, RMSE, R², within-tolerance rates).
- Logs epoch-wise train/validation loss and MAE for diagnosing under/overfitting.
- Supports scoring brand-new, unrated vehicles from a saved model bundle without retraining.

## Why hierarchical LSTM over summary features

Rather than collapsing each RPM sweep into a handful of statistics, the model keeps the full curve shape and lets an LSTM learn the RPM-dependent pattern per sensor group. The hierarchy (sensor-order → sensor → position) is enforced architecturally so the model never directly mixes physically unrelated channels (e.g. 1st-order vibration with 2nd-order, or left with right side) at the first stage of representation learning.

## Requirements

```bash
pip install pandas numpy scikit-learn openpyxl torch
```

## Usage

```bash
python P4_hierarchical_lstm_all_cc_no_mirror.py \
  --data-dir "path/to/All_CC" \
  --out-dir "path/to/model_outputs_hierarchical_lstm" \
  --epochs 500 \
  --patience 60
```

Key optional arguments:

| Flag | Default | Description |
|---|---|---|
| `--n-points` | 35 | Number of resampled points per RPM curve |
| `--batch-size` | 16 | Training batch size |
| `--learning-rate` | 0.001 | AdamW learning rate |
| `--weight-decay` | 0.01 | AdamW weight decay |
| `--dropout` | 0.25 | Dropout used throughout the model |
| `--order-features` | 8 | LSTM hidden size per sensor-order branch |
| `--lstm-layers` | 1 | LSTM layers per branch |
| `--bidirectional-lstm` | off | Use bidirectional LSTM branches |
| `--sensor-features` | 16 | Feature size per sensor after combining branches |
| `--head-features` | 32 | Hidden units in the final rating head |
| `--seed` | 42 | Random seed |
| `--skip-validation` | off | Skip LOVO cross-validation and go straight to final model training |
| `--device` | auto | `auto` / `cpu` / `cuda` |

## Data expectations

- **HIP workbooks**: one `.xlsm` per engine-class batch, with `V1`–`V7` sheets, each containing an RPM column and sensor/order/direction-labeled acceleration columns starting near row 13.
- **Subjective ratings workbook**: one sheet per engine class, with position/RPM-band labels and per-vehicle rating columns, mapped via the `VEHICLES` table in the script.
- Do **not** use a random row split for validation — vehicles must be split as whole groups (leave-one-vehicle-out), since neural nets can otherwise overfit to a small number of independent vehicles. Always compare against Ridge/other baselines using the same LOVO split.

## Outputs (written to `--out-dir`)

- `parsed_ratings.csv` — cleaned subjective ratings (Mirror excluded)
- `curve_dataset_index.csv` — index of every built curve sample
- `skipped_training_rows.csv` — rows dropped and why
- `curve_channel_manifest.json` — channel names per position
- `hier_lstm_lovo_predictions.csv` / `hier_lstm_lovo_metrics_by_position.csv` / `..._by_position_and_engine_class.csv` — LOVO validation results
- `hier_lstm_lovo_epoch_history.csv` / `hier_lstm_final_epoch_history.csv` — per-epoch train/val loss and MAE
- `hier_lstm_final_model_summary.csv` — final per-position training summary
- `hier_lstm_models_by_position.pt` — frozen model bundle (state dicts, scalers, channel manifests, config) for inference on new vehicles
- `run_config.json` — full run configuration

## Scoring a new, unrated vehicle

```python
import torch
from pathlib import Path
from P4_hierarchical_lstm_all_cc_no_mirror import predict_workbook

bundle = torch.load("hier_lstm_models_by_position.pt", weights_only=False)
predictions, skipped = predict_workbook(
    path=Path("NewVehicle_HIP_DATA.xlsm"),
    models=bundle["models"],
)
```

This reconstructs each position's frozen architecture from its saved config and state dict — no retraining, no leakage, deterministic reproduction of the trained model. Since this variant uses no auxiliary scalar inputs, only the vibration curves are needed to score a new vehicle (no engine-cc or RPM-band context required beyond what's in the curve itself).

## Notes

- Mirror position is intentionally excluded — its features were found redundant.
- Only full RPM-wise curves are used as input; there are no auxiliary scalar features in this version (unlike earlier Ridge-fusion variants of this pipeline).
- Positions with fewer than 3 rated vehicles are skipped for both LOVO validation and final training, since LOVO requires enough independent groups to be meaningful.
