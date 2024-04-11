from numba import njit, prange
import numpy as np

@njit
def fast_eval_score_nb(signals, results, odd, sig_th, max_iter=20, dd_limit=0.2):
    scores = np.zeros(max_iter)
    mdds = np.zeros(max_iter)
    num = signals.shape[0]
    
    for p in range(max_iter):
        lag_pnl = np.zeros(num, dtype=np.float32)
        pnl = 1.0
        running_max = 0.0
        max_dd = 0.000001
        profits = 0.0
        losses = 0.000001
        position = 0.001*(p+1)
        # assume 10x leverage and total fee is 0.1%
        fee = position * 0.001 * 10
        for i in range(1,len(signals)):
            pnl = pnl + lag_pnl[i]
            if signals[i] > sig_th: #open order
                if results[i] == 0:
                    losses += pnl * (position + fee)
                    pnl *= (1 - position - fee)
                elif results[i] > 0:
                    lag_pnl[i+results[i]-1] += pnl * ((odd / 10 + 1) * position)
                    profits += pnl * ((odd / 10 ) * position - fee)
                    pnl = pnl * (1 - position - fee)
                elif results[i] <0:
                    lag_pnl[-1] += pnl * (((-results[i]) + 1) * position)
                    profits += pnl * ((-results[i]) * position - fee)
                    pnl = pnl * (1 - position - fee)
            if pnl > running_max:
                running_max = pnl
            elif (running_max - pnl)/running_max > max_dd:
                max_dd = (running_max - pnl)/running_max
            
            pf = profits/losses
        
            if max_dd < dd_limit:
                scores[p] = pnl/(max_dd)
                mdds[p] = max_dd

    max_score_idx = np.argmax(scores)
    return scores[max_score_idx], mdds[max_score_idx], pf, 0.001*(max_score_idx+1)

@njit
def fast_inf_nb(signals, results, odd, position, sig_th):
    num = signals.shape[0]
    lag_pnl = np.zeros(num, dtype=np.float32)
    pnl = 1.0
    running_max = 0.0
    max_dd = 0.0
    profits = 0.0
    losses = 0.000001
    for i in range(1,len(signals)):
        pnl = pnl + lag_pnl[i]
        if signals[i] > sig_th: #open order
            if results[i] == 0:
                pnl *= (1 - position)
                losses += pnl * (position)
            elif results[i] > 0:
                if i+results[i] > num:
                    lag_pnl[-1] += pnl * ((odd / 10 + 1) * position)
                else:
                    lag_pnl[i+results[i]-1] += pnl * ((odd / 10 + 1) * position)
                pnl = pnl * (1 - position)
                profits += pnl * ((odd / 10 ) * position)
            elif results[i] <0:
                lag_pnl[-1] += pnl * (((-results[i]) + 1) * position)
                pnl = pnl * (1 - position)
                profits += pnl * ((-results[i]) * position)
        if pnl > running_max:
            running_max = pnl
        elif (running_max - pnl)/running_max > max_dd:
            max_dd = (running_max - pnl)/running_max
    pf = profits/losses
    return pnl, max_dd, pf

@njit
def mdd_nb(balance_array):
    running_max = balance_array[0]
    max_drawdown = 0
    for i in range(1, len(balance_array)):
        if balance_array[i] > running_max:
            running_max = balance_array[i]
        elif (running_max - balance_array[i])/running_max > max_drawdown:
            max_drawdown = (running_max - balance_array[i])/running_max
    return max_drawdown

@njit
def labels_nb(closes, long_sls, highs, lows, n):
    num_rows = len(closes)
    results = np.zeros(num_rows, dtype=np.int32)
    for i in range(num_rows):
        entry_price = closes[i]
        long_sl = long_sls[i]
        long_sl_distance = entry_price - long_sl
        for j in range(i+1, num_rows):
            if lows[j] <= long_sl:
                result = 0
                break
            if highs[j] >= (entry_price + n / 10 * long_sl_distance):
                result = j - i
                break
            if j == num_rows-1:
                if closes[j] < entry_price:
                    result = 0
                    break
                else:
                    result = -((closes[j] - entry_price)//long_sl_distance)
                    break
        results[i] = result
    return results

@njit
def labels_short_nb(closes, short_sls, highs, lows, n):
    num_rows = len(closes)
    results = np.zeros(num_rows, dtype=np.int32)
    for i in range(num_rows):
        entry_price = closes[i]
        short_sl = short_sls[i]
        short_sl_distance = entry_price - short_sl
        for j in range(i+1, num_rows):
            if highs[j] >= short_sl:
                result = 0
                break
            if lows[j] <= (entry_price + n / 10 * short_sl_distance):
                result = j - i
                break
            if j == num_rows-1:
                if closes[j] > entry_price:
                    result = 0
                    break
                else:
                    result = -((entry_price - closes[j])//short_sl_distance)
                    break
        results[i] = result
    return results