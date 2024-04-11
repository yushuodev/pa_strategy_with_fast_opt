import numpy as np
import pandas as pd
from itertools import product
import vectorbt as vbt
from numba import njit, prange
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import time
import os

from sls import high_low_sl
from bt_nb import fast_inf_nb, fast_eval_score_nb, labels_nb

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path, index_col='timestamp', parse_dates=True)
    return df['high'].values, df['low'].values, df['close'].values, df['long_signals'].values

def calculates_sls(highs, lows, closes, sl_params):
    sl_combinations = list(product(*sl_params.values()))
    results = Parallel(n_jobs=-1)(
        delayed(high_low_sl)(highs, lows, closes, *sl_comb)
        for sl_comb in tqdm(sl_combinations, desc="Calculating SL and Labels"))
    precomputed_results = dict(zip(sl_combinations, results))
    return precomputed_results

def rolling_walk_forward_optimization(signals, highs, lows, closes, pre_computed_values, ans, optimization_window, steps):
    n = len(closes)
    wfo_results = []
    for start_idx in tqdm(range(0, n - optimization_window + 1, steps), desc="wfo"):
        result = opt_window(start_idx, signals, highs, lows, closes, pre_computed_values, ans, optimization_window, steps)
        wfo_results.append(result)

    return wfo_results

def opt_window(start_idx, signals, highs, lows, closes, pre_computed_values, ans, optimization_window, steps):
    signal_in = signals[start_idx:start_idx + optimization_window]
    highs_in = highs[start_idx:start_idx + optimization_window]
    lows_in = lows[start_idx:start_idx + optimization_window]
    closes_in = closes[start_idx:start_idx + optimization_window]
    tasks = []

    for sl_params, sls in pre_computed_values.items():
        long_sl = sls[0][start_idx:start_idx + optimization_window]
        for odd in range(20,110,10):
            tasks.append((signal_in, highs_in, lows_in, closes_in, long_sl, odd, sl_params, 0.2))
    results = Parallel(n_jobs=-1)(
        delayed(calculate_for_combination)(*task) for task in tasks
    )
    # print('opt_finish')
    max_score = -100
    for score, dd, position, odd, sl_params in results:
        if score > max_score:
            max_score = score
            best_sl = sl_params
            best_odd = odd
            best_dd = dd
            best_position = position
    
    signal_out = signals[start_idx + optimization_window:start_idx + optimization_window + steps]
    winandlose_out = ans[best_sl][(best_odd,)][start_idx + optimization_window:start_idx + optimization_window + steps]
    out_pnl, out_dd, out_pf = fast_inf_nb(signal_out, winandlose_out, best_odd, best_position, sig_th = 40)
    return start_idx, best_sl, best_odd, max_score, best_dd, best_position, out_pnl, out_dd, out_pf

def calculate_for_combination(signal_in, highs_in, lows_in, closes_in, long_sl, odd, sl_params, dd_limit):
    winandlose_in = labels_nb(closes_in, long_sl, highs_in, lows_in, odd)
    # print("finish")
    score, dd, pf, position = fast_eval_score_nb(signal_in, winandlose_in, odd, sig_th = 40, max_iter=20, dd_limit=dd_limit)
    return score, dd, position, odd, sl_params

if __name__ == "__main__":

    csv_file_path = "eth_sig.csv"
    highs, lows, closes, signals = load_data(csv_file_path)

    sl_params = {
    'length': np.arange(12, 360, 8),
    'mult': np.arange(2, 10, 1),
    'tighter': np.arange(2, 10, 1)
    }
    label_params = {
        'odd': np.arange(20,110,10)
    }

    if os.path.isfile('pre_computed_values.pkl'):
        pre_computed_values = joblib.load('pre_computed_values.pkl')
    else:
        pre_computed_values = calculates_sls(highs, lows, closes, sl_params)
        joblib.dump(pre_computed_values,'pre_computed_values.pkl')
    
    if os.path.isfile('ans.pkl'):
        ans = joblib.load('ans.pkl')
    
    # for debunging
    # highs = highs[:8000]
    # lows = lows[:8000]
    # closes = closes[:8000]
    # signals = signals[:8000]

    out_window = [400]
    in_window = [2000, 4000, 8000, 10000]

    for ow in out_window:
        for iw in in_window:
            results = rolling_walk_forward_optimization(signals, highs, lows, closes,
                                                         pre_computed_values, ans, optimization_window=iw, steps=ow)
            joblib.dump(results,'_result.pkl')
            df_results = pd.DataFrame(results, columns=['StartIdx', 'best_sl', 'best_odd', 'MaxScore', 'DD', 'best_position', 'out_pnl', 'out_dd', 'out_pf'])
            timestr = time.strftime("%Y%m%d-%H%M%S")
            csv_file_path = f'optimization_results_{iw}_{ow}_{timestr}.csv'
            df_results.to_csv(csv_file_path, index=False)