# linearModelExample.py
import numpy as np
from scipy import stats

def SXY(x, y):
    return np.sum((x - x.mean()) * (y - y.mean()))

def SXX(x):
    return SXY(x, x)

np.random.seed(0)
x = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
b0, b1 = 10, 0.5
sigma = 0.3
y_from_x = b0 + b1 * x
y_noise = np.random.normal(loc = 0, scale = sigma, size = x.size)
y_actual = y_from_x + y_noise
y = y_actual.round(1) # to get the data in the report

SXX_value = SXX(x)
SXY_value = SXY(x, y)
x_mean, y_mean = x.mean(), y.mean()

b1 = SXY_value / SXX_value
b0 = y_mean - b1 * x_mean
residuals = y - (b0 + b1 * x)
SSres = np.sum(residuals ** 2)
MSres = SSres / x.size

print('\nX mean and Y mean')
print(x_mean, y_mean)

print('\nSXY, SXX')
print(SXY_value, SXX_value)

print('\nb0, b1')
print(b0, b1)

norm_0025 = stats.norm.ppf(0.025)
norm_0975 = stats.norm.ppf(0.975)
t_0975 = stats.t.ppf(0.975, df = x.size - 2)
t_0025 = stats.t.ppf(0.025, df = x.size - 2)
chi2_0975 = stats.chi2.ppf(0.975, df = x.size - 2)
chi2_0025 = stats.chi2.ppf(0.025, df = x.size - 2)

print('\nSlope b1 CI with sigma')
normalError_b1 = np.sqrt(sigma**2 / SXX_value)
leftValueWithSigma = b1 - norm_0975 * normalError_b1
rightValueWithSigma = b1 - norm_0025 * normalError_b1
print(leftValueWithSigma, rightValueWithSigma)

print('\nSlope b1 CI without sigma')
se_b1 = np.sqrt(MSres / SXX_value)
leftValueWithoutSigma = b1 - t_0975 * se_b1
rightValueWithoutSigma = b1 - t_0025 * se_b1
print(leftValueWithoutSigma, rightValueWithoutSigma)

print('\nIntercept b0 CI with sigma')
denominator = (1 / x.size + x_mean**2 / SXX_value)
normalError_b0 = np.sqrt(sigma**2 / denominator)
leftValueWithSigma = b0 - norm_0975 * normalError_b0
rightValueWithSigma = b0 - norm_0025 * normalError_b0
print(leftValueWithSigma, rightValueWithSigma)

print('\nIntercept b0 CI without sigma')
se_b0 = np.sqrt(MSres / denominator)
leftValueWithoutSigma = b0 - t_0975 * se_b0
rightValueWithoutSigma = b0 - t_0025 * se_b0
print(leftValueWithoutSigma, rightValueWithoutSigma)

print(f'\nCI of sigma^2 = {sigma**2}')
leftValueOfSigma = SSres / chi2_0975
rightValueOfSigma = SSres / chi2_0025
print(leftValueOfSigma, rightValueOfSigma)
