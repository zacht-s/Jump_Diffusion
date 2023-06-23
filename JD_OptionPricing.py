# Code adapted from Dr. Yves J Hilpisch, "Derivatives Analytics with Python" (2014).
import math
import numpy as np
from scipy.integrate import quad


def mjd_characteristic_function(z, ttm, r, sigma_d, intensity, u_j, sigma_j):
    """
    Valuation of European call option in Merton's Lognormal Jump Diffusion model (1976) via
    Lewis (2001) Fourier-based approach: characteristic function.
    :param z: Transformation of log(St). z = log(St) - 0.5i
    :param ttm: Option Time to Maturity, years
    :param r: Risk Free Interest Rate
    :param sigma_d: Asset Diffusive Volatility
    :param intensity: Expected Number of Jumps per Year
    :param u_j: Expected Jump Magnitude
    :param sigma_j: Volatility of Jump Magnitude
    :return:
    """

    omega = r - 0.5 * sigma_d ** 2 - intensity * (np.exp(u_j + 0.5 * sigma_j ** 2) - 1)
    value = np.exp((1j * z * omega - 0.5 * z ** 2 * sigma_d ** 2 +
                    intensity * (np.exp(1j * z * u_j - z ** 2 * sigma_j ** 2 * 0.5) - 1)) * ttm)
    return value


def mjd_pricing_integral(z, s0, k, ttm, r, sigma_d, intensity, u_j, sigma_j):
    """
    Valuation of European call option in Merton's Lognormal Jump Diffusion model (1976) via
    Lewis (2001) Fourier-based approach: integration function.
    :param z: Integration Limit, in log(St) space
    :param s0: Current Stock Price
    :param k: Option Strike Price
    :param ttm: Option Time to Maturity, years
    :param r: Risk Free Interest Rate
    :param sigma_d: Asset Diffusive Volatility
    :param intensity: Expected Number of Jumps per Year
    :param u_j: Expected Jump Magnitude
    :param sigma_j: Volatility of Jump Magnitude
    :return:
    """
    char = mjd_characteristic_function(z - 0.5 * 1j, ttm, r, sigma_d, intensity, u_j, sigma_j)
    value = 1 / (z ** 2 + 0.25) * (np.exp(1j * z * math.log(s0 / k)) * char).real
    return value


def dejd_characteristic_function(z, ttm, r, sigma_d, intensity, kappa, eta):
    """
    Valuation of European call option in Merton's Lognormal Jump Diffusion model (1976) via
    Lewis (2001) Fourier-based approach: characteristsic function.
    :param z: Transformation of log(St). z = log(St) - 0.5i
    :param ttm: Option Time to Maturity, years
    :param r: Risk Free Interest Rate
    :param sigma_d: Asset Diffusive Volatility
    :param intensity: Expected Number of Jumps per Year
    :param kappa: Mean of Double Exponential Jump Diffusion Process
    :param eta: Volatility (sd) of Double Exponential Jump Diffusion Process * SQRT(2)
    :return:
    """
    omega = r - 0.5 * sigma_d ** 2 - intensity * (np.exp(kappa) - 1)
    value = np.exp((1j * z * omega - 0.5 * z ** 2 * sigma_d ** 2 +
                    intensity * (np.exp(1j * z * kappa) * (1 - eta ** 2) / (1 + z ** 2 * eta ** 2) - 1)) * ttm)
    return value


def dejd_pricing_integral(z, s0, k, ttm, r, sigma_d, intensity, kappa, eta):
    """
    Valuation of European call option in Merton's Lognormal Jump Diffusion model (1976) via
    Lewis (2001) Fourier-based approach: integration function.
    :param z: Integration Limit, in log(St) space
    :param s0: Current Stock Price
    :param k: Option Strike Price
    :param ttm: Option Time to Maturity, years
    :param r: Risk Free Interest Rate
    :param sigma_d: Asset Diffusive Volatility
    :param intensity: Expected Number of Jumps per Year
    :param kappa: Mean of Double Exponential Jump Diffusion Process
    :param eta: Volatility (sd) of Double Exponential Jump Diffusion Process * SQRT(2)
    :return:
    """
    char = dejd_characteristic_function(z - 0.5 * 1j, ttm, r, sigma_d, intensity, kappa, eta)
    value = 1 / (z ** 2 + 0.25) * (np.exp(1j * z * math.log(s0 / k)) * char).real
    return value


class EuropeanOptionJD:
    def __init__(self, s0, k, ttm, r, type):
        """
        :param s0: Current Stock Price
        :param k: Option Strike Price
        :param ttm: Option Time to Maturity, years
        :param r: Risk Free Interest Rate
        :param type: call or put, for option kind
        """
        self.s0 = s0
        self.k = k
        self.ttm = ttm
        self.r = r
        self.type = type

        if self.type.lower() not in ['call', 'put']:
            raise TypeError('type must be either "call" or "put"')

    def mjd_price(self, sigma_d, intensity, u_j, sigma_j):
        """
        :param sigma_d: Asset Diffusive Volatility
        :param intensity: Expected Number of Jumps per Year
        :param u_j: Expected Jump Magnitude
        :param sigma_j: Volatility of Jump Magnitude
        :return: Price of Option under log-normal jump diffusion
        """
        int_value = quad(lambda z: mjd_pricing_integral(z, self.s0, self.k, self.ttm, self.r,
                                                        sigma_d, intensity, u_j, sigma_j), 0, 50, limit=250)[0]
        if self.type.lower() == 'call':
            option_price = self.s0 - np.exp(-self.r * self.ttm) * math.sqrt(self.s0 * self.k) / math.pi * int_value
        else:
            option_price = self.k * np.exp(-self.r * self.ttm) - \
                           np.exp(-self.r * self.ttm) * math.sqrt(self.s0 * self.k) / math.pi * int_value

        return round(option_price, 3)

    def dejd_price(self, u1, u2, p1, intensity, u_ld, sigma_d, d_t=1/252):
        """
        :param u1: Mean negative jump size
        :param u2: Mean positive jump size
        :param p1: Probability of a positive jump given a jump occurs
        :param intensity: Expected number of jumps per year
        :param u_ld: Mean of Diffusion Process
        :param sigma_d: Asset Diffusive Volatility
        :param d_t: Timestep between Price Observations
        :return: Price of Option under double exponential jump diffusion
        """

        u_j = -p1 * u1 + (1-p1) * u2
        sigma_j = p1 * ((u_j + u1) ** 2 + u1 ** 2) + (1 - p1) * ((u_j - u2) ** 2 + u2 ** 2)

        kappa = u_ld * d_t + u_j * intensity * d_t
        eta = math.sqrt((sigma_d**2 + intensity * (sigma_j**2 + u_j**2)) * d_t / 2)

        int_value = quad(lambda z: dejd_pricing_integral(z, self.s0, self.k, self.ttm, self.r,
                                                         sigma_d, intensity, kappa, eta), 0, 50, limit=250)[0]
        if self.type.lower() == 'call':
            option_price = self.s0 - np.exp(-self.r * self.ttm) * math.sqrt(self.s0 * self.k) / math.pi * int_value
        else:
            option_price = self.k * np.exp(-self.r * self.ttm) - np.exp(-self.r * self.ttm) * \
                    math.sqrt(self.s0 * self.k) / math.pi * int_value

        return round(option_price, 3)


if __name__ == '__main__':
    # Here the passed MJD and DEJD parameters are somewhat made up, not expecting to be equal

    option_test = EuropeanOptionJD(s0=100, k=100, ttm=1, r=0.05, type='call')
    temp1 = option_test.mjd_price(sigma_d=0.4, intensity=100, u_j=-0.2, sigma_j=0.1)

    option_test = EuropeanOptionJD(s0=100, k=100, ttm=1, r=0.05, type='put')
    temp2 = option_test.mjd_price(sigma_d=0.4, intensity=100, u_j=-0.2, sigma_j=0.1)

    option_test = EuropeanOptionJD(s0=100, k=100, ttm=1, r=0.05, type='call')
    temp3 = option_test.dejd_price(u1=0.2874, u2=0.1431, p1=0.7969, intensity=100, u_ld=0.13818, sigma_d=0.4)

    option_test = EuropeanOptionJD(s0=100, k=100, ttm=1, r=0.05, type='put')
    temp4 = option_test.dejd_price(u1=0.2874, u2=0.1431, p1=0.7969, intensity=100, u_ld=0.13818, sigma_d=0.4)

    print(f'MJD Call Price: {temp1},        MJD Put Price: {temp2}')
    print(f'DEJD Call Price: {temp3},       DEJD Put Price: {temp4}')





