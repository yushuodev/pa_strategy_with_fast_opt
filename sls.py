import vectorbt as vbt
from numba import njit
import numpy as np

@njit
def high_low_sl_np_part(atr, min1, min2, min3, max1, max2, max3, close, mult=2, tighter=7):
    mult = mult/10
    tighter = tighter/100
    v = np.multiply(atr, mult)
    long_sl_1 = np.subtract(min1,v)
    long_sl_2 = np.subtract(min2,v)
    long_sl_3 = np.subtract(min3,v)

    short_sl_1 = np.add(max1,v)
    short_sl_2 = np.add(max2,v)
    short_sl_3 = np.add(max3,v)

    long_sl_4  = np.multiply(close,1-tighter)
    short_sl_4 = np.multiply(close,1+tighter)
    long_sl_target  = np.where(long_sl_1>long_sl_4,long_sl_1
                                ,np.where(long_sl_2>long_sl_4,long_sl_2
                                ,np.where(long_sl_3>long_sl_4,long_sl_3,long_sl_4)
                                         )
                               )
    short_sl_target = np.where(short_sl_1<short_sl_4,short_sl_1
                            ,np.where(short_sl_2<short_sl_4,short_sl_2
                            ,np.where(short_sl_3<short_sl_4,short_sl_3,short_sl_4)
                                        )
                            )
    return long_sl_target, short_sl_target

def high_low_sl(high, low, close, length, mult, tighter):
    atr = vbt.IndicatorFactory.from_talib("ATR").run(high, low, close, timeperiod = length).real.to_numpy()
    min1 = vbt.IndicatorFactory.from_talib("MIN").run(low, timeperiod = length).real.to_numpy()
    min2 = vbt.IndicatorFactory.from_talib("MIN").run(low, timeperiod = length//2).real.to_numpy()
    min3 = vbt.IndicatorFactory.from_talib("MIN").run(low, timeperiod = length//4).real.to_numpy()

    max1 = vbt.IndicatorFactory.from_talib("MAX").run(high, timeperiod = length).real.to_numpy()
    max2 = vbt.IndicatorFactory.from_talib("MAX").run(high, timeperiod = length//2).real.to_numpy()
    max3 = vbt.IndicatorFactory.from_talib("MAX").run(high, timeperiod = length//4).real.to_numpy()
    return high_low_sl_np_part(atr, min1, min2, min3, max1, max2, max3, close, mult, tighter)