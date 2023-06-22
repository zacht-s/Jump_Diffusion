import math
import scipy.stats as sps
import numpy as np
import pandas as pd
import yfinance as yf
import time
from datetime import datetime
from scipy.optimize import minimize


# Second Order Approximation to the cumulative distribution function for a Log-Normal Jump Diffusion model
def log_normal_cdf(u_j, sigma_j, intensity, x1, x2, d_t, u_ld, sigma_d):
    """
    :param u_j: Mean Jump Size
    :param sigma_j: Volatility of Jump Size
    :param intensity: Jump Intensity (Expected Number of Jumps Annually)
    :param x1: Lower limit of integration for CDF in return space
    :param x2: Upper limit of integration for CDF in return space
    :param d_t: Time increment in years between price observations
    :param u_ld: Mean of Diffusion Process
    :param sigma_d: Volatility of Diffusion Process
    :return:
    """
    n0_mean = u_ld * d_t
    n0_sd = (d_t * sigma_d ** 2) ** (1 / 2)
    n_0 = math.exp(-intensity * d_t) * (intensity * d_t) ** 0 / math.factorial(0) * \
        (sps.norm.cdf(x=x2, loc=n0_mean, scale=n0_sd) - sps.norm.cdf(x=x1, loc=n0_mean, scale=n0_sd))

    n1_mean = u_ld * d_t + u_j
    n1_sd = (d_t * sigma_d ** 2 + sigma_j ** 2) ** (1 / 2)
    n_1 = math.exp(-intensity * d_t) * (intensity * d_t) ** 1 / math.factorial(1) * \
          (sps.norm.cdf(x=x2, loc=n1_mean, scale=n1_sd) - sps.norm.cdf(x=x1, loc=n1_mean, scale=n1_sd))

    n2_mean = u_ld * d_t + 2 * u_j
    n2_sd = (d_t * sigma_d ** 2 + 2 * sigma_j ** 2) ** (1 / 2)
    n_2 = math.exp(-intensity * d_t) * (intensity * d_t) ** 2 / math.factorial(2) * \
          (sps.norm.cdf(x=x2, loc=n2_mean, scale=n2_sd) - sps.norm.cdf(x=x1, loc=n2_mean, scale=n2_sd))

    denom = 0
    for i in range(3):
        denom += math.exp(-intensity * d_t) * (intensity * d_t) ** i / math.factorial(i)

    return (n_0 + n_1 + n_2) / denom


# Define function for computing the distribution function for a Kou Double Exponential Jump Diffusion model
def double_exp_cdf(u1, u2, p1, intensity, x1, x2, d_t, u_ld, sigma_d):
    """

    :param u1: Mean positive jump size
    :param u2: Mean negative jump size
    :param p1: Probability of a positive jump given a jump occurs
    :param intensity: Expected number of jumps per year
    :param x1: Lower limit of integration for CDF in return space
    :param x2: Upper limit of integration for CDF in return space
    :param d_t: Time increment in years between price observations
    :param u_ld: Mean of Diffusion Process
    :param sigma_d: Volatility of Diffusion Process
    :return:
    """

    p2 = 1 - p1
    u = u_ld * d_t
    sigma = math.sqrt(sigma_d ** 2 * d_t)

    v1 = u - 0.5 * sigma ** 2 / u1
    v2 = u + 0.5 * sigma ** 2 / u2

    p11 = (p1 / u1) ** 2
    p22 = (p2 / u2) ** 2
    p12 = 2 * p1 * p2 / (u1 + u2)

    z1 = (x1 - u) / sigma
    z2 = (x2 - u) / sigma

    # Error handling for exponential terms
    try:
        math.exp((x2 - v1) / u1)
    except OverflowError:
        #print('Prevent OverFlow Error 1')
        u1 = p1 / 100

    try:
        math.exp((x1 - v1) / u1)
    except OverflowError:
        #print('Prevent OverFlow Error 2')
        u1 = p1 / 100

    try:
        math.exp(-(x1 - v2) / u2)
    except OverflowError:
        #print('Prevent OverFlow Error 3')
        u2 = p2 / 100

    try:
        math.exp(-(x2 - v2) / u2)
    except OverflowError:
        #print('Prevent OverFlow Error 4')
        u2 = p2 / 100

    norm_diff = sps.norm.cdf(x=x2, loc=u, scale=sigma) - sps.norm.cdf(x=x1, loc=u, scale=sigma)
    px2_v1 = math.exp((x2 - v1) / u1) * sps.norm.cdf(x=-x2, loc=-u + sigma ** 2 / u1, scale=sigma)
    px1_v1 = math.exp((x1 - v1) / u1) * sps.norm.cdf(x=-x1, loc=-u + sigma ** 2 / u1, scale=sigma)
    px1_v2 = math.exp(-(x1 - v2) / u2) * sps.norm.cdf(x=x1, loc=u + sigma ** 2 / u2, scale=sigma)
    px2_v2 = math.exp(-(x2 - v2) / u2) * sps.norm.cdf(x=x2, loc=u + sigma ** 2 / u2, scale=sigma)

    n_0 = math.exp(-intensity * d_t) * (intensity * d_t) ** 0 / math.factorial(0) * norm_diff

    n_1 = math.exp(-intensity * d_t) * (intensity * d_t) ** 1 / math.factorial(1) * (
            norm_diff + p1 * (px2_v1 - px1_v1) + p2 * (px1_v2 - px2_v2))

    n_2 = math.exp(-intensity * d_t) * (intensity * d_t) ** 2 / math.factorial(2) * (norm_diff + \
            u1 * ((p12 + p11 * (u - sigma ** 2 / u1 + u1 - x2)) * px2_v1 - (p12 + p11 * (u - sigma ** 2 / u1 + u1 - x1)) * px1_v1) + \
            u2 * ((p12 - p22 * (u + sigma ** 2 / u2 - u2 - x1)) * px1_v2 - (p12 - p22 * (u + sigma ** 2 / u2 - u2 - x2)) * px2_v2) + \
            sigma / math.sqrt(2 * math.pi) * (u2 * p22 - u1 * p11) * (math.exp(-(z1 ** 2 / 2)) - math.exp(-(z2 ** 2 / 2))))

    denom = 0
    for i in range(3):
        denom += math.exp(-intensity * d_t) * (intensity * d_t) ** i / math.factorial(i)

    return (n_0 + n_1 + n_2) / denom


def mjd_mle_obj(params, returns, d_t):
    """
    :param params: List of parameters to be optimized in scipy.optimize.minimize [u_j, sigma_j, intensity]
    :param returns: Asset of returns to use in MLE calibration
    :param d_t: Timestep
    :return: Objective function for calibration (max log_likelihood)
    """

    # Split off input parameters from initial guess vector
    u_j, sigma_j, intensity = params[0], params[1], params[2]
    n = len(returns)

    # Match 1st and 2nd moments of JD model to sampled returns
    u_ld = (np.mean(returns) - u_j * intensity * d_t) / d_t
    sigma_d2 = max(((np.std(returns) ** 2 - (sigma_j ** 2 + u_j ** 2) * intensity * d_t) / d_t), 0)
    sigma_d = max(sigma_d2 ** (1/2), 0.001)

    # Bin Returns into histogram and compute log likelihood across all bins
    ret_hist, ret_bins = np.histogram(returns, bins=100, density=False)
    log_like = 0
    for x in range(len(ret_hist)):
        x1, x2 = ret_bins[x], ret_bins[x+1]

        cdf = log_normal_cdf(u_j, sigma_j, intensity, x1, x2, d_t, u_ld, sigma_d)
        if cdf == 0:
            bin_log_like = 0
        else:
            bin_log_like = math.log(cdf * n) * ret_hist[x]
        log_like += bin_log_like

    # Change to negative log_like for scipy.optimize.minimize()
    return -log_like / n


def mjd_mle_obj_fix_intensity(params, returns, d_t, intensity):
    """
    :param params: List of parameters to be optimized in scipy.optimize.minimize [u_j, sigma_j]
    :param returns: Asset of returns to use in MLE calibration
    :param d_t: Timestep
    :return: Objective function for calibration (max log_likelihood)
    """

    # Split off input parameters from initial guess vector
    u_j, sigma_j = params[0], params[1]
    n = len(returns)

    # Match 1st and 2nd moments of JD model to sampled returns
    u_ld = (np.mean(returns) - u_j * intensity * d_t) / d_t
    sigma_d2 = max(((np.std(returns) ** 2 - (sigma_j ** 2 + u_j ** 2) * intensity * d_t) / d_t), 0)
    sigma_d = max(sigma_d2 ** (1/2), 0.001)

    # Bin Returns into histogram and compute log likelihood across all bins
    ret_hist, ret_bins = np.histogram(returns, bins=100, density=False)
    log_like = 0
    for x in range(len(ret_hist)):
        x1, x2 = ret_bins[x], ret_bins[x+1]

        cdf = log_normal_cdf(u_j, sigma_j, intensity, x1, x2, d_t, u_ld, sigma_d)
        if cdf == 0:
            bin_log_like = 0
        else:
            bin_log_like = math.log(cdf * n) * ret_hist[x]
        log_like += bin_log_like

    # Change to negative log_like for scipy.optimize.minimize()
    return -log_like / n


def dejd_mle_obj(params, returns, d_t):
    """
    :param params: List of parameters to be optimized in scipy.optimize.minimize [u1, u2, p1, intensity]
    :param returns: Asset of returns to use in MLE calibration
    :param d_t: Timestep
    :return: Objective function for calibration (max log_likelihood)
    """

    # Split off input parameters from initial guess vector
    u1, u2, p1, intensity = params[0], params[1], params[2], params[3]
    n = len(returns)

    # Calculate expected jump size and standard deviation of jump size given DEJD parameters
    u_j = -p1 * u1 + (1-p1) * u2
    sigma_j = math.sqrt(p1 * ((u_j + u1) ** 2 + u1 ** 2) + (1-p1) * ((u_j - u2) ** 2 + u2 ** 2))

    # Match 1st and 2nd moments of JD model to sampled returns
    u_ld = (np.mean(returns) - u_j * intensity * d_t) / d_t
    sigma_d2 = max(((np.std(returns) ** 2 - (sigma_j ** 2 + u_j ** 2) * intensity * d_t) / d_t), 0)
    sigma_d = max(sigma_d2 ** (1 / 2), 0.001)

    # Bin Returns into histogram and compute log likelihood across all bins
    ret_hist, ret_bins = np.histogram(returns, bins=100, density=False)
    log_like = 0
    for x in range(len(ret_hist)):
        x1, x2 = ret_bins[x], ret_bins[x+1]

        cdf = double_exp_cdf(u1, u2, p1, intensity, x1, x2, d_t, u_ld, sigma_d)

        if cdf <= 0:
            bin_log_like = 0
        else:
            bin_log_like = math.log(cdf * n) * ret_hist[x]
        log_like += bin_log_like

        # Change to negative log_like for scipy.optimize.minimize()
        return -log_like / n


def dejd_mle_obj_fix_intensity(params, returns, d_t, intensity):
    """
    :param params: List of parameters to be optimized in scipy.optimize.minimize [u1, u2, p1]
    :param returns: Asset of returns to use in MLE calibration
    :param d_t: Timestep
    :return: Objective function for calibration (max log_likelihood)
    """

    # Split off input parameters from initial guess vector
    u1, u2, p1 = params[0], params[1], params[2]
    n = len(returns)

    # Calculate expected jump size and standard deviation of jump size given DEJD parameters
    u_j = -p1 * u1 + (1-p1) * u2
    sigma_j = math.sqrt(p1 * ((u_j + u1) ** 2 + u1 ** 2) + (1-p1) * ((u_j - u2) ** 2 + u2 ** 2))

    # Match 1st and 2nd moments of JD model to sampled returns
    u_ld = (np.mean(returns) - u_j * intensity * d_t) / d_t
    sigma_d2 = max(((np.std(returns) ** 2 - (sigma_j ** 2 + u_j ** 2) * intensity * d_t) / d_t), 0)
    sigma_d = max(sigma_d2 ** (1 / 2), 0.001)

    # Bin Returns into histogram and compute log likelihood across all bins
    ret_hist, ret_bins = np.histogram(returns, bins=100, density=False)
    log_like = 0
    for x in range(len(ret_hist)):
        x1, x2 = ret_bins[x], ret_bins[x+1]

        cdf = double_exp_cdf(u1, u2, p1, intensity, x1, x2, d_t, u_ld, sigma_d)

        if cdf <= 0:
            bin_log_like = 0
        else:
            bin_log_like = math.log(cdf * n) * ret_hist[x]
        log_like += bin_log_like

        # Change to negative log_like for scipy.optimize.minimize()
        return -log_like / n


class MJD_Calibration:
    def __init__(self, returns, d_t):
        self.returns = returns
        self.d_t = d_t

    def calibrate(self, guess=np.array([0, 0.012, 100]), display_status=False):
        """
        :param guess: Optional np array for initial guess vector for [u_j, sigma_j, intensity] parameters
        :param display_status: Boolean, toggles print statements to show when calibration starts/ends
        :return: optimal [u_j, sigma_j, intensity] vector from Max Log Likelihood
        """
        start = time.time()
        if display_status:
            print('')
            print('Begin Log Normal JD Model Calibration (Free Intensity)')

        fitted_mjd_params = minimize(fun=mjd_mle_obj, x0=guess, args=(self.returns, self.d_t),
                                     method='Nelder-Mead', bounds=[(None, None), (0, None), (1, 20000)])

        end = time.time()
        if display_status:
            print(f'Calibration Complete, Execution Time: {round(end - start, 2)} seconds')
            print('')

        return pd.DataFrame(fitted_mjd_params.x, index=['u_j', 'sigma_j', 'intensity'])[0]

    def calibrate_fix_intensity(self, intensity, guess=np.array([0, 0.012]), display_status=False):
        """
        :param intensity: Expected number of jumps per year
        :param guess: Optional np array for initial guess vector for [u_j, sigma_j] parameters
        :param display_status: Boolean, toggles print statements to show when calibration starts/ends
        :return: optimal [u_j, sigma_j] vector from Max Log Likelihood
        """
        start = time.time()
        if display_status:
            print('')
            print(f'Begin Log Normal JD Model Calibration (Fixed Intensity of {intensity})')

        fitted_mjd_params = minimize(fun=mjd_mle_obj_fix_intensity, x0=guess, args=(self.returns, self.d_t, intensity),
                                     method='Nelder-Mead', bounds=[(None, None), (0, None)])
        end = time.time()
        if display_status:
            print(f'Calibration Complete, Execution Time: {round(end - start, 2)} seconds')
            print('')

        return pd.DataFrame(fitted_mjd_params.x, index=['u_j', 'sigma_j'])[0]


class DEJD_Calibration:
    def __init__(self, returns, d_t):
        self.returns = returns
        self.d_t = d_t

    def calibrate(self, guess=np.array([0.06, 0.06, 0.5, 100]), display_status=False):
        """
        :param guess: Optional np array for initial guess vector for [u1, u2, p1, intensity] parameters
        :param display_status: Boolean, toggles print statements to show when calibration starts/ends
        :return: optimal [u1, u2, p1, intensity] vector from Max Log Likelihood
        """
        start = time.time()
        if display_status:
            print('')
            print('Begin Double Exponential JD Model Calibration (Free Intensity)')

        fitted_dejd_params = minimize(fun=dejd_mle_obj, x0=guess, args=(self.returns, self.d_t),
                                      method='Nelder-Mead',
                                      bounds=[(0.0001, None), (0.0001, None), (0.1, 0.9), (1, 20000)])

        end = time.time()
        if display_status:
            print(f'Calibration Complete, Execution Time: {round(end - start, 2)} seconds')
            print('')

        return pd.DataFrame(fitted_dejd_params.x, index=['u1', 'u2', 'p1', 'intensity'])[0]

    def calibrate_fix_intensity(self, intensity, guess=np.array([0.06, 0.06, 0.5]), display_status=False):
        """
        :param intensity: Expected number of jumps per year
        :param guess: Optional np array for initial guess vector for [u1, u2, p1] parameters
        :param display_status: Boolean, toggles print statements to show when calibration starts/ends
        :return: optimal [u1, u2, p1] vector from Max Log Likelihood
        """
        start = time.time()
        if display_status:
            print('')
            print(f'Begin Double Exponential JD Model Calibration (Fixed Intensity of {intensity})')

        fitted_dejd_params = minimize(fun=dejd_mle_obj_fix_intensity, x0=guess, args=(self.returns, self.d_t, intensity),
                                      method='Nelder-Mead', bounds=[(0.0001, None), (0.0001, None), (0.1, 0.9)])

        end = time.time()
        if display_status:
            print(f'Calibration Complete, Execution Time: {round(end - start, 2)} seconds')
            print('')

        return pd.DataFrame(fitted_dejd_params.x, index=['u1', 'u2', 'p1'])[0]


if __name__ == '__main__':

    prices = yf.download('TSLA', start=datetime(2015, 1, 1), end=datetime(2021, 1, 1))['Adj Close']
    returns = np.log(prices / prices.shift(1)).dropna()

    test4 = MJD_Calibration(returns=returns, d_t=1/252)
    print(test4.calibrate(display_status=True))
    print(test4.calibrate_fix_intensity(intensity=75, display_status=True))

    test5 = DEJD_Calibration(returns=returns, d_t=1/252)
    print(test5.calibrate(display_status=True))
    print(test5.calibrate_fix_intensity(intensity=150, display_status=True))
