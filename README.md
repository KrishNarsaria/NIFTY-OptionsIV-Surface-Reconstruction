## Repository
[https://github.com/KrishNarsaria/FinClub-Project2](https://github.com/KrishNarsaria/FinClub-Project2)
# NIFTY50 Options IV Surface Reconstruction

**Author:** Krish Narsaria (ID: 24114052)  
**Institution:** Indian Institute of Technology (IIT) Roorkee  
**Course:** Computer Science and Engineering (CSE-3Y)  
**Duration:** May-June 2026

## Overview

This project reconstructs missing implied volatility (IV) values for a NIFTY50 options dataset using an **Autonomous Causal Adaptive Ensemble** approach. The core script [Model.py](Model.py) employs a sophisticated blend of five interpolation and extrapolation methods with leave-one-out (LOO) dynamic weighting to fill missing option IVs across call and put strikes.

The pipeline is designed to maintain causality and avoid future leakage when filling the IV surface over time.

## Problem Statement

Options trading requires accurate implied volatility surfaces for pricing and risk management. Missing IV values in historical datasets need to be reconstructed in a principled way that respects temporal causality and market microstructure (strike structure, time-to-expiry dynamics).

## Key Features

- **5-Method Ensemble Architecture:**
  - Akima interpolation with damped wing extrapolation
  - Total variance interpolation with PCHIP/linear fallback
  - Log-space wing extrapolation
  - Quadratic polynomial fit
  - Linear slope extrapolation

- **Leave-One-Out (LOO) Dynamic Weighting:** Methods are weighted based on their cross-validation performance on known strikes
- **Strictly Causal Time-Series Fallbacks:** Forward-fill and expanding-mean strategies respect temporal ordering
- **Robust IV Clipping:** Output constrained to realistic bounds [0.001, 8.0]

## Files

| File | Purpose |
|------|---------|
| [Model.py](Model.py) | Main reconstruction pipeline |
| [dataset.csv](dataset.csv) | Input dataset with missing IV values |
| [filled_dataset.csv](filled_dataset.csv) | Complete reconstructed dataset output |
| [submission.csv](submission.csv) | Submission file (only originally missing entries) |
| [submission-converter.ipynb](submission-converter.ipynb) | Notebook utility for format conversion |
| [requirements.txt](requirements.txt) | Project dependencies |

## Installation

### Requirements
- Python 3.8+
- Dependencies listed in [requirements.txt](requirements.txt)

### Setup
```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline
```bash
python Model.py
```

This will:
1. Read `dataset.csv`
2. Reconstruct all missing IV values
3. Output `filled_dataset.csv` (complete dataset)
4. Output `submission.csv` (only originally missing values)

## Input Format

### Dataset Structure
`dataset.csv` should contain:

- **`datetime`** — timestamp for each observation
- **`underlying_price`** — NIFTY50 spot price at observation time
- **Option IV Columns** — named as `NIFTY<DD><MMM><YYYY><STRIKE><CE/PE>`
  - Example: `NIFTY27JAN2625200CE` (Call), `NIFTY27JAN2623800PE` (Put)

### Current Dataset Statistics

| Metric | Value |
|--------|-------|
| Rows | 975 |
| Columns | 30 |
| Expiry Date | January 27, 2026 15:30:00 |
| Strike Type | Calls and Puts around ATM |
| Time Horizon | Multiple timestamps leading to expiry |

## Method

### Algorithm Overview

For each timestamp, the pipeline processes call and put surfaces independently:

1. **Time-to-Expiry Calculation:** Parse time-to-expiry from `datetime` relative to fixed expiry `2026-01-27 15:30:00`
2. **Surface Separation:** Separate call and put IV columns
3. **Row-wise Reconstruction:** For each option surface at each timestamp:
   - Identify known (non-missing) and missing IV values
   - Sort strikes in log-space relative to spot price
   - Compute LOO weights across five base methods
   - Generate predictions from each method
   - Blend predictions using dynamic weights
4. **Causality-Respecting Fallbacks:**
   - Forward-fill missing values using temporally available data only
   - Apply expanding-mean fallback for any remaining gaps
5. **Final Clipping:** Enforce IV bounds [0.001, 8.0]

### Five Base Methods

| Method | Approach | Strengths |
|--------|----------|-----------|
| **Akima Wing** | Akima spline + damped quadratic wings | Smooth interpolation, adaptive extrapolation |
| **Total Variance** | Convert to variance space, interpolate, convert back | Respects variance dynamics |
| **Log Wing** | Log-space extrapolation with PCHIP | Handles extreme strikes |
| **Polynomial Fit** | Quadratic or linear polynomial | Simple, stable |
| **Linear Slope** | Piecewise linear with slope preservation | Fast, robust |

### Leave-One-Out Weighting

For each strike surface:
- Train all five methods on $n-1$ points (omitting one strike)
- Predict the omitted strike
- Record squared error
- Average MSE across all $n$ left-out observations
- Weight inversely proportional to LOO error: $w_i = \frac{1/e_i}{\sum_j 1/e_j}$

### Output Format

**filled_dataset.csv** contains the complete dataset with all missing IVs filled.

**submission.csv** contains only originally missing entries in the format:
```
id,value
<datetime>||<option_column>,<filled_iv_value>
...
```

Example:
```
2026-01-01 09:15:00||NIFTY27JAN2625200CE,0.2145
2026-01-01 09:15:00||NIFTY27JAN2623800PE,0.1923
```

## Design Principles

- **Causality:** No future information is used when filling past values
- **Adaptivity:** Method weights adjust based on local surface characteristics
- **Robustness:** Multiple fallback strategies handle edge cases (flat surfaces, sparse data)
- **Efficiency:** Vectorized numpy operations for fast computation

## Configuration

Key parameters in [Model.py](Model.py):

```python
IV_LO, IV_HI = 0.001, 8.0          # IV bounds
IV_PRIOR = 0.20                     # Fallback prior (20% IV)
EXPIRY_DT = "2026-01-27 15:30:00"   # Fixed option expiry
DAYS_PER_YEAR = 365.25              # Days per year for time calc
```

## Performance Considerations

- **Time Complexity:** O(n × m × k) where n = timestamps, m = strikes, k = ensemble size
- **Memory:** ~150 MB for typical dataset (1000 rows × 30 strikes)
- **Runtime:** ~5-10 seconds for full reconstruction

## Future Enhancements

1. Machine learning-based weight optimization (XGBoost, neural networks)
2. Volatility smile/skew parameterization (SVI, SSVI models)
3. Multi-expiry surface interpolation
4. Parallel processing for large datasets
5. Confidence intervals on reconstructed values

## License

Project submission for IIT Roorkee FinClub (May-June 2026).