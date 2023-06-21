import numpy as np
import yfinance as yf
import pandas as pd
import math
from datetime import datetime


def local_vol(returns, K):
    bi_pwrs = np.abs(returns * returns.shift(1)).dropna()
    bi_pwr_window = np.lib.stride_tricks.sliding_window_view(bi_pwrs, K)
    local_vol_est = [math.sqrt(sum(x)/(K-2)) for x in bi_pwr_window]
    df = pd.DataFrame(local_vol_est, index=bi_pwrs.index[K-1:])[0]
    return df


def detect_jumps(returns, freq, tol=0.01):
    """
    :param returns: natural log returns of price observations
    :param freq: int, number of price observations per day
    :param tol: confidence level for jump detection
    :return:
    """

    # Define constants for Mykland test statistic
    K = math.ceil((252*freq) ** (1/2))
    n = len(returns) + 1
    c = (2 / math.pi) ** (1/2)
    c_n = ((2 * math.log(n)) ** (1/2)) / c - \
          (math.log(math.pi) + math.log(math.log(n)))/2/c/((2*math.log(n)) ** (1/2))

    s_n = 1/c/((2*math.log(n))**(1/2))
    l_threshold = -math.log(-math.log(1-tol))

    """
    Use ratio of log return: local volatility for each timestep to detect jumps by comparing test statistic to the 
    maximum  expected value to arise from diffusion actions alone, at a specified confidence threshold.
    """
    local_vol_est = local_vol(returns=returns, K=K)
    ret_sub = returns[local_vol_est.index]
    statistic = (np.abs(ret_sub / local_vol_est) - c_n) / s_n
    jump_df = ret_sub.loc[statistic > l_threshold]
    return jump_df


if __name__ == '__main__':
    prices = yf.download(['TSLA'], start=datetime(2015, 1, 1), end=datetime(2023, 6, 1))['Adj Close']
    log_rets = np.log(prices / prices.shift(1)).dropna()
    test = detect_jumps(returns=log_rets, freq=1, tol=0.01)
    print(test)