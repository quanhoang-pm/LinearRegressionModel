# samples_generator.py
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats


sigma = 0.3
intercept = 10 # b0
slope = 0.5 # b1
nSamples = 100
nTrials = 10**4 # 10**4

x = np.linspace(10, 20, num = nSamples, endpoint = True)
x_input = x.reshape(-1, 1)
y_from_x = slope * x + intercept

np.random.seed(0)
slopeValuesList = []
interceptValuesList = []
SSresValuesList = []
for _ in range(nTrials):
    y_noise = np.random.normal(loc = 0, scale = sigma, size = x.size)
    y = y_from_x + y_noise
    reg = LinearRegression().fit(x_input, y)
    y_predict = reg.predict(x_input)
    SSres = np.sum((y - y_predict)**2)
    slopeValuesList.append(reg.coef_[0])
    interceptValuesList.append(reg.intercept_)
    SSresValuesList.append(SSres)


SxxValue = np.sum((x - x.mean())**2)
theoreticalInterceptVariance = sigma**2 * (1 / nSamples + x.mean()**2 / SxxValue)
theoreticalInterceptExpectedValue = intercept
print('\nIntercept')
print('Mean / Variance')
print(f'Theoretical:    {theoreticalInterceptExpectedValue:7.4f}   {theoreticalInterceptVariance:7.4f}')
print(f'Practical:      {np.mean(interceptValuesList):7.4f}   {np.var(interceptValuesList):7.4f}')


theoreticalSlopeVariance = sigma**2 / SxxValue
theoreticalSlopeExpectedValue = slope
print('\nSlope')
print('Mean / Variance')
print(f'Theoretical:    {theoreticalSlopeExpectedValue:7.4f}   {theoreticalSlopeVariance:7.7f}')
print(f'Practical:      {np.mean(slopeValuesList):7.4f}   {np.var(slopeValuesList):7.7f}')

print('\nSigma')
CI_ratio = 0.95
low = stats.chi2.ppf((1 - CI_ratio) / 2, nSamples - 2)
high = stats.chi2.ppf((1 + CI_ratio) / 2, nSamples - 2)
sigmaSquared = sigma**2

counterTrue = 0
ratiosList = []
for i, SSres in enumerate(SSresValuesList):
    if SSres / high <= sigmaSquared <= SSres / low:
        counterTrue += 1
    ratiosList.append(counterTrue / (i + 1))
print(f'Practical {counterTrue}')
print(f'Expected {len(SSresValuesList) * CI_ratio}')
