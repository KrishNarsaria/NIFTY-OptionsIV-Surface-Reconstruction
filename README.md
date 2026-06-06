# NIFTY50 Options IV Surface Reconstruction

This project reconstructs missing implied volatility (IV) values for a NIFTY50 options dataset and generates a submission file for the missing entries.

The core script is [Model.py](Model.py), which builds a causal ensemble of interpolation and extrapolation methods to fill missing option IVs across call and put strikes.

## Overview

The model uses:

- Akima interpolation with damped wing extrapolation
- Total variance interpolation with PCHIP/linear fallback
- Log-space wing extrapolation
- Quadratic polynomial fit
- Linear slope extrapolation
- Leave-one-out dynamic weighting across the five base learners
- Strictly causal time-series fallbacks after row-wise reconstruction

The pipeline is designed to avoid future leakage when filling the IV surface over time.

## Files

- [Model.py](Model.py): main reconstruction pipeline
- [dataset.csv](dataset.csv): input dataset with missing IV values
- [filled_dataset.csv](filled_dataset.csv): reconstructed dataset output
- [submission.csv](submission.csv): submission file containing only originally missing entries
- [submission-converter.ipynb](submission-converter.ipynb): notebook present in the project workspace

## Input Format

`dataset.csv` is expected to contain:

- `datetime`
- `underlying_price`
- option IV columns named like `NIFTY27JAN2625200CE` or `NIFTY27JAN2623800PE`

From the current dataset:

- Rows: 975
- Columns: 30
- Option columns: call and put strikes around the January 27, 2026 expiry

## Method

For each timestamp:

1. Parse time-to-expiry from `datetime` relative to the fixed expiry `2026-01-27 15:30:00`.
2. Separate call and put surfaces.
3. Reconstruct missing values strike-wise using an ensemble of five methods.
4. Compute leave-one-out errors on observed strikes and convert them into dynamic method weights.
5. Fill any remaining gaps using causal forward fill, then causal expanding-mean fallback.
6. Clip all IVs to the configured bounds `[0.001, 8.0]`.

After reconstruction:

- `filled_dataset.csv` stores the full completed dataset.
- `submission.csv` stores only values that were missing in the original file, with IDs formatted as:

```text
<datetime>||<option_column>