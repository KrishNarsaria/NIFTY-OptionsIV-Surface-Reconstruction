"""
=============================================================================
Project:      NIFTY50 Options IV Surface Reconstruction
Description:  Autonomous Causal Adaptive Ensemble for Implied Volatility (IV)
              Surface Reconstruction. Implements Leave-One-Out (LOO) dynamic 
              weighting across 5 base learners with strictly causal time-series 
              fallbacks .

Author:       Krish Narsaria (ID: 24114052)
Duration:     May-June 2026
Course:       Computer Science and Engineering (CSE-3Y)
Institution:  Indian Institute of Technology (IIT) Roorkee

=============================================================================
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, interp1d, PchipInterpolator

warnings.filterwarnings('ignore')

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH   = os.path.join(BASE_DIR, "dataset.csv")
OUTPUT_FILLED  = os.path.join(BASE_DIR, "filled_dataset.csv")
OUTPUT_SUB     = os.path.join(BASE_DIR, "submission.csv")
SEPARATOR      = "||"
IV_LO, IV_HI   = 0.001, 8.0
IV_PRIOR       = 0.20
EXPIRY_DT      = pd.to_datetime("2026-01-27 15:30:00")
DAYS_PER_YEAR  = 365.25

def extract_strike(col_name):
    match = re.search(r'(\d{5})(CE|PE)', col_name)
    return int(match.group(1)) if match else 0

def method_akima_wing(xk_log, yk, xm_log, min_x, max_x, y0, yn):
    if len(xk_log) >= 3:
        akima = Akima1DInterpolator(xk_log, yk)
    else:
        akima = interp1d(xk_log, yk, kind='linear', fill_value='extrapolate')
    
    n_pts = min(3, len(xk_log))
    
    left_x = xk_log[:n_pts]
    left_y = yk[:n_pts]
    if len(left_x) >= 2:
        dx_left = left_x[1:] - min_x
        dy_left = left_y[1:] - y0
        s_left = dy_left / dx_left
        poly_left = np.polyfit(dx_left, s_left, 1) if len(dx_left) >= 2 else [0, s_left[0]]
        slope_L = poly_left[1]
        convexity_L = max(poly_left[0], 0.0)
    else:
        slope_L, convexity_L = 0.0, 0.0
    
    right_x = xk_log[-n_pts:]
    right_y = yk[-n_pts:]
    if len(right_x) >= 2:
        dx_right = right_x[:-1] - max_x
        dy_right = right_y[:-1] - yn
        s_right = dy_right / dx_right
        poly_right = np.polyfit(dx_right, s_right, 1) if len(dx_right) >= 2 else [0, s_right[0]]
        slope_R = poly_right[1]
        convexity_R = max(poly_right[0], 0.0)
    else:
        slope_R, convexity_R = 0.0, 0.0
    
    preds = np.zeros_like(xm_log)
    for i, xl in enumerate(xm_log):
        if min_x <= xl <= max_x:
            preds[i] = akima(xl)
        elif xl > max_x:
            dx = xl - max_x
            damp = 1.0 / (1.0 + 2.0 * abs(dx))
            preds[i] = yn + slope_R * dx + convexity_R * dx**2 * damp
        else:
            dx = xl - min_x
            damp = 1.0 / (1.0 + 2.0 * abs(dx))
            preds[i] = y0 + slope_L * dx + convexity_L * dx**2 * damp
            
    return np.clip(preds, IV_LO, IV_HI)

def method_totvar(xk_log, yk, xm_log, tte_years, min_x, max_x):
    w_obs = (yk ** 2) * tte_years
    if len(xk_log) >= 3:
        pchip = PchipInterpolator(xk_log, w_obs, extrapolate=True)
        w_pred = pchip(xm_log)
    else:
        f = interp1d(xk_log, w_obs, kind='linear', fill_value='extrapolate')
        w_pred = f(xm_log)
        
    w_pred = np.maximum(w_pred, 1e-8)
    iv_pred = np.sqrt(w_pred / tte_years)
    return np.clip(iv_pred, IV_LO, IV_HI)

def method_logwing(xk_log, yk, xm_log, min_x, max_x, y0, yn):
    if len(xk_log) >= 3:
        pchip = PchipInterpolator(xk_log, yk, extrapolate=False)
    else:
        pchip = None
    
    if len(xk_log) >= 2:
        left_slope = (np.log(yk[1]) - np.log(yk[0])) / (xk_log[1] - xk_log[0] + 1e-6)
        right_slope = (np.log(yk[-1]) - np.log(yk[-2])) / (xk_log[-1] - xk_log[-2] + 1e-6)
    else:
        left_slope = right_slope = 0.0
    
    preds = np.zeros_like(xm_log)
    for i, xl in enumerate(xm_log):
        if pchip is not None and min_x <= xl <= max_x:
            pred = pchip(xl)
            if not np.isnan(pred):
                preds[i] = pred
                continue
        if xl > max_x:
            preds[i] = np.exp(np.log(yn) + right_slope * (xl - max_x))
        else:
            preds[i] = np.exp(np.log(y0) + left_slope * (xl - min_x))
            
    return np.clip(preds, IV_LO, IV_HI)

def method_poly2(xk_log, yk, xm_log):
    if len(xk_log) < 3:
        coeff = np.polyfit(xk_log, yk, 1)
        pred = np.polyval(coeff, xm_log)
    else:
        coeff = np.polyfit(xk_log, yk, 2)
        pred = np.polyval(coeff, xm_log)
    return np.clip(pred, IV_LO, IV_HI)

def method_linear_slope(xk_log, yk, xm_log, min_x, max_x, y0, yn):
    if len(xk_log) < 2:
        return np.full_like(xm_log, np.mean(yk) if len(yk) else 0.2)
    
    left_slope = (yk[1] - yk[0]) / (xk_log[1] - xk_log[0] + 1e-6)
    right_slope = (yk[-1] - yk[-2]) / (xk_log[-1] - xk_log[-2] + 1e-6)
    
    preds = np.zeros_like(xm_log)
    for i, xl in enumerate(xm_log):
        if min_x <= xl <= max_x:
            preds[i] = np.interp(xl, xk_log, yk)
        elif xl > max_x:
            preds[i] = yn + right_slope * (xl - max_x)
        else:
            preds[i] = y0 + left_slope * (xl - min_x)
    return np.clip(preds, IV_LO, IV_HI)

def calc_loo_weights(xk_log, yk, tte_years, min_x, max_x, y0, yn):
    n = len(xk_log)
    if n < 3:
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    errors = []
    methods = [
        ("akima", lambda x, y, xp, t, mn, mx, y0, yn: method_akima_wing(x, y, xp, mn, mx, y0, yn)),
        ("totvar", lambda x, y, xp, t, mn, mx, y0, yn: method_totvar(x, y, xp, t, mn, mx)),
        ("logwing", lambda x, y, xp, t, mn, mx, y0, yn: method_logwing(x, y, xp, mn, mx, y0, yn)),
        ("poly2", lambda x, y, xp, t, mn, mx, y0, yn: method_poly2(x, y, xp)),
        ("slope", lambda x, y, xp, t, mn, mx, y0, yn: method_linear_slope(x, y, xp, mn, mx, y0, yn))
    ]
    
    def loo_error(func):
        sq_err = []
        for i in range(n):
            train_x = np.delete(xk_log, i)
            train_y = np.delete(yk, i)
            test_x = xk_log[i]
            test_y = yk[i]
            
            if len(train_x) == 0:
                pred = test_y
            else:
                train_min, train_max = train_x[0], train_x[-1]
                train_y0, train_yn = train_y[0], train_y[-1]
                pred = func(train_x, train_y, np.array([test_x]), tte_years,
                            train_min, train_max, train_y0, train_yn)[0]
            sq_err.append((test_y - pred)**2)
        return np.mean(sq_err)
    
    for name, func in methods:
        try:
            err = loo_error(func)
            errors.append(max(err, 1e-8))
        except:
            errors.append(1e8)
    
    inv_err = 1.0 / np.array(errors)
    return inv_err / inv_err.sum()

def process_single_row(row_ivs, strikes, spot, tte_min, tte_years):
    k_log = np.log(strikes / spot)
    iv = row_ivs.copy()
    known = ~np.isnan(iv)
    missing = np.isnan(iv)
    
    if known.sum() == 0 or missing.sum() == 0:
        return row_ivs
    
    xk = k_log[known]
    yk = iv[known]
    xm = k_log[missing]
    
    sort_idx = np.argsort(xk)
    xk, yk = xk[sort_idx], yk[sort_idx]
    
    min_x, max_x = xk[0], xk[-1]
    y0, yn = yk[0], yk[-1]
    
    weights = calc_loo_weights(xk, yk, tte_years, min_x, max_x, y0, yn)
    
    pred_akima = method_akima_wing(xk, yk, xm, min_x, max_x, y0, yn)
    pred_totvar = method_totvar(xk, yk, xm, tte_years, min_x, max_x)
    pred_logwing = method_logwing(xk, yk, xm, min_x, max_x, y0, yn)
    pred_poly2 = method_poly2(xk, yk, xm)
    pred_slope = method_linear_slope(xk, yk, xm, min_x, max_x, y0, yn)
    
    blended = (weights[0] * pred_akima +
               weights[1] * pred_totvar +
               weights[2] * pred_logwing +
               weights[3] * pred_poly2 +
               weights[4] * pred_slope)
    
    filled = iv.copy()
    filled[missing] = np.clip(blended, IV_LO, IV_HI)
    return filled

def causal_forward_fill(mat):
    filled = mat.copy()
    for j in range(filled.shape[1]):
        last = np.nan
        for i in range(filled.shape[0]):
            if not np.isnan(filled[i, j]):
                last = filled[i, j]
            elif not np.isnan(last):
                filled[i, j] = last
    return filled

def causal_expanding_mean_fallback(mat, prior):
    filled = mat.copy()
    
    for j in range(filled.shape[1]):
        sum_vals = 0.0
        count = 0
        for i in range(filled.shape[0]):
            if not np.isnan(filled[i, j]):
                sum_vals += filled[i, j]
                count += 1
            elif np.isnan(filled[i, j]) and count > 0:
                filled[i, j] = sum_vals / count

    for i in range(filled.shape[0]):
        row = filled[i, :]
        if np.isnan(row).any():
            row_median = np.nanmedian(row)
            fallback_val = row_median if not np.isnan(row_median) else prior
            row[np.isnan(row)] = fallback_val
            filled[i, :] = row
            
    return filled

def run_pipeline(df_input, ce_cols, pe_cols, ce_strikes, pe_strikes, tte_min_arr, tte_years_arr, spot_arr, option_cols):
    df_out = df_input.copy()
    
    for cols, strikes in [(ce_cols, ce_strikes), (pe_cols, pe_strikes)]:
        data = df_out[cols].values.copy()
        for i in range(data.shape[0]):
            row = data[i, :]
            if np.isnan(row).all():
                continue
            spot = spot_arr[i]
            if np.isnan(spot) or spot <= 0:
                continue
            
            tte_min = tte_min_arr[i]
            tte_years = tte_years_arr[i]
            filled_row = process_single_row(row, strikes, spot, tte_min, tte_years)
            data[i, :] = filled_row
        df_out[cols] = data
    
    df_out[option_cols] = causal_forward_fill(df_out[option_cols].values)
    df_out[option_cols] = causal_expanding_mean_fallback(df_out[option_cols].values, IV_PRIOR)
    df_out[option_cols] = df_out[option_cols].clip(IV_LO, IV_HI)
    
    return df_out

def main():
    df_raw = pd.read_csv(DATASET_PATH)
    df_working = df_raw.copy()
    
    df_working['datetime_parsed'] = pd.to_datetime(df_working['datetime'], format="mixed")
    tte_min = (EXPIRY_DT - df_working['datetime_parsed']).dt.total_seconds() / 60.0
    df_working['tte_min'] = tte_min.clip(lower=1.0)
    df_working['tte_years'] = df_working['tte_min'] / (60 * 24 * DAYS_PER_YEAR)
    
    df_working['underlying_price'] = df_working['underlying_price'].ffill()
    
    if pd.isna(df_working['underlying_price'].iloc[0]):
        first_valid = df_working['underlying_price'].first_valid_index()
        if first_valid is not None and first_valid == 0:
            pass 
        else:
            row0 = df_working.iloc[0]
            best_diff = float('inf')
            inferred_spot = np.nan
            
            ce_tmp = {extract_strike(c): c for c in df_working.columns if c.endswith('CE')}
            pe_tmp = {extract_strike(c): c for c in df_working.columns if c.endswith('PE')}
            common_strikes = set(ce_tmp.keys()).intersection(set(pe_tmp.keys()))
            
            for k in common_strikes:
                c_val = pd.to_numeric(row0[ce_tmp[k]], errors='coerce')
                p_val = pd.to_numeric(row0[pe_tmp[k]], errors='coerce')
                if not np.isnan(c_val) and not np.isnan(p_val):
                    diff = abs(c_val - p_val)
                    if diff < best_diff:
                        best_diff = diff
                        inferred_spot = float(k)
            
            if not np.isnan(inferred_spot):
                df_working.loc[0, 'underlying_price'] = inferred_spot
            else:
                all_strikes = [extract_strike(c) for c in (list(ce_tmp.keys()) + list(pe_tmp.keys()))]
                df_working.loc[0, 'underlying_price'] = float(np.median(all_strikes))
            
            df_working['underlying_price'] = df_working['underlying_price'].ffill()
    
    option_cols = [c for c in df_working.columns 
                   if c not in ['datetime', 'underlying_price', 'datetime_parsed', 'tte_min', 'tte_years']]
    for col in option_cols:
        df_working[col] = pd.to_numeric(df_working[col], errors='coerce')
    
    ce_cols = sorted([c for c in option_cols if c.endswith('CE')], key=extract_strike)
    pe_cols = sorted([c for c in option_cols if c.endswith('PE')], key=extract_strike)
    
    ce_strikes = np.array([extract_strike(c) for c in ce_cols])
    pe_strikes = np.array([extract_strike(c) for c in pe_cols])
    spot_arr = df_working['underlying_price'].values
    tte_min_arr = df_working['tte_min'].values
    tte_years_arr = df_working['tte_years'].values
    
    df_filled = run_pipeline(df_working, ce_cols, pe_cols, ce_strikes, pe_strikes,
                             tte_min_arr, tte_years_arr, spot_arr, option_cols)
    
    df_filled.drop(columns=['datetime_parsed', 'tte_min', 'tte_years'], inplace=True, errors='ignore')
    df_filled.to_csv(OUTPUT_FILLED, index=False)
    
    submission_rows = []
    for col in option_cols:
        was_missing = df_raw[col].isna()
        if not was_missing.any():
            continue
        for idx in df_raw.index[was_missing]:
            dt = df_raw.loc[idx, 'datetime']
            uid = f"{dt}{SEPARATOR}{col}"
            val = df_filled.loc[idx, col]
            submission_rows.append({"id": uid, "value": float(val)})
    
    submission = pd.DataFrame(submission_rows, columns=['id', 'value'])
    submission = submission.sort_values('id').reset_index(drop=True)
    submission.to_csv(OUTPUT_SUB, index=False)
    
    print(f"Submission generated: {OUTPUT_SUB} ({len(submission)} rows)")

if __name__ == "__main__":
    main()
